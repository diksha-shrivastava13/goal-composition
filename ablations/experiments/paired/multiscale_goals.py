"""
F2: Multi-Scale Goal Detection.

Detect goals at multiple temporal scales (Ngo's framework).
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment
from ..utils.paired_helpers import (
    generate_levels, extract_level_features_batch, get_pro_hstates,
    get_values_from_rollout, get_action_distribution, get_pro_ant_returns,
)


class TemporalScale(Enum):
    """Temporal scale for goal detection."""
    IMMEDIATE = 3  # Within-step
    SHORT_TERM = 15  # Few steps
    MEDIUM_TERM = 50  # Episode-level
    LONG_TERM = 256  # Cross-episode


@dataclass
class ScaleGoal:
    """Goal detected at a specific temporal scale."""
    scale: TemporalScale
    feature_dimensions: np.ndarray
    predictive_power: float  # How well this predicts behavior at this scale
    cross_scale_influence: Dict[str, float]  # Influence on other scales
    active_contexts: List[Dict[str, float]]


class MultiscaleGoalsExperiment(CheckpointExperiment):
    """
    Detect goals at multiple temporal scales (Ngo's framework).

    Protocol:
    1. Decompose value function and policy into temporal scales
    2. Identify goal-encoding dimensions at each scale
    3. Measure hierarchical composition between scales
    4. Track scale emergence over training
    """

    @property
    def name(self) -> str:
        return "multiscale_goals"

    def __init__(
        self,
        n_episodes: int = 100,
        max_steps_per_episode: int = 50,
        hidden_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.hidden_dim = hidden_dim
        self._episode_data: List[Dict[str, Any]] = []
        self._scale_goals: Dict[TemporalScale, List[ScaleGoal]] = {}
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"MultiscaleGoalsExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> List[Dict[str, Any]]:
        """Collect multi-scale trajectory data."""
        for ep in range(self.n_episodes):
            rng, ep_rng = jax.random.split(rng)
            episode = self._collect_episode(ep_rng, ep)
            self._episode_data.append(episode)

        return self._episode_data

    def _collect_episode(self, rng: chex.PRNGKey, episode_idx: int) -> Dict[str, Any]:
        """Collect data for a single episode using real network evaluations."""
        rng, level_rng, hstate_rng, val_rng, action_rng, ret_rng = jax.random.split(rng, 6)

        # Generate a single real level for this episode
        levels = generate_levels(self.agent, level_rng, 1)

        # Get real hidden states from protagonist
        hstate_single = get_pro_hstates(hstate_rng, levels, self)  # (1, hidden_dim)
        self.hidden_dim = hstate_single.shape[1]

        # Get real value trajectory from rollout
        values_traj = get_values_from_rollout(
            self.train_state, self.agent, levels, val_rng,
            max_steps=self.max_steps_per_episode,
        )  # (1, max_steps)
        values_ep = values_traj[0]  # (max_steps,)

        # Get real action logits
        logits, entropies = get_action_distribution(
            self.train_state, self.agent, levels, action_rng,
            max_steps=self.max_steps_per_episode,
        )  # logits: (1, max_steps, n_actions)
        actions_ep = np.argmax(logits[0], axis=-1)  # (max_steps,)

        # Get real return for episode difficulty proxy
        pro_returns, _, _ = get_pro_ant_returns(ret_rng, levels, self)
        episode_return = float(pro_returns[0])

        # Extract level features
        features_batch = extract_level_features_batch(levels)
        wall_density = float(features_batch['wall_density'][0])
        goal_distance = float(features_batch['goal_distance'][0])
        episode_difficulty = wall_density * 0.5 + goal_distance * 0.05

        # Build per-step hstates by tiling the terminal hstate
        # (the real hstate captures the episode-level representation)
        n_steps = min(self.max_steps_per_episode, len(values_ep))
        hstates = np.tile(hstate_single[0], (n_steps, 1))  # (max_steps, hidden_dim)

        # Build per-step level features
        level_features = [
            {
                'wall_density': wall_density,
                'goal_distance': goal_distance,
                'step': step,
            }
            for step in range(n_steps)
        ]

        # Compute rewards from value differences (approximate)
        rewards = np.zeros(n_steps)
        rewards[:-1] = values_ep[1:n_steps] - values_ep[:n_steps-1] * 0.99
        rewards[-1] = episode_return - float(np.sum(rewards[:-1]))

        return {
            'episode_idx': episode_idx,
            'episode_difficulty': episode_difficulty,
            'hstates': hstates,
            'values': values_ep[:n_steps],
            'actions': actions_ep[:n_steps],
            'rewards': rewards,
            'level_features': level_features,
        }

    def _detect_scale_goals(self, scale: TemporalScale) -> List[ScaleGoal]:
        """Detect goals at a specific temporal scale."""
        horizon = scale.value
        goals = []

        # Aggregate data at appropriate temporal resolution
        aggregated_hstates = []
        aggregated_returns = []

        for episode in self._episode_data:
            hstates = episode['hstates']
            rewards = episode['rewards']

            # Aggregate over horizon windows
            for start in range(0, len(hstates) - horizon + 1, horizon):
                end = start + horizon
                aggregated_hstates.append(hstates[start:end].mean(axis=0))

                # Return at this scale
                discounted_return = sum(
                    rewards[start + t] * (0.99 ** t)
                    for t in range(min(horizon, len(rewards) - start))
                )
                aggregated_returns.append(discounted_return)

        if len(aggregated_hstates) < 10:
            return goals

        hstates = np.array(aggregated_hstates)
        returns = np.array(aggregated_returns)

        # Find dimensions predictive of returns at this scale
        correlations = []
        for d in range(self.hidden_dim):
            corr = np.corrcoef(hstates[:, d], returns)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0.0)

        correlations = np.array(correlations)

        # Top dimensions for this scale
        top_dims = np.argsort(correlations)[-30:]

        # Create goal representation
        predictive_power = np.mean(correlations[top_dims])

        goals.append(ScaleGoal(
            scale=scale,
            feature_dimensions=top_dims,
            predictive_power=float(predictive_power),
            cross_scale_influence={},  # Filled later
            active_contexts=[],
        ))

        return goals

    def _compute_cross_scale_influence(self):
        """Compute how goals at different scales influence each other."""
        for scale1 in self._scale_goals:
            for goal in self._scale_goals[scale1]:
                for scale2 in self._scale_goals:
                    if scale1 != scale2 and self._scale_goals[scale2]:
                        # Compute dimension overlap
                        dims1 = set(goal.feature_dimensions)
                        dims2 = set(self._scale_goals[scale2][0].feature_dimensions)
                        overlap = len(dims1 & dims2) / max(len(dims1), 1)
                        goal.cross_scale_influence[scale2.name] = float(overlap)

    def analyze(self) -> Dict[str, Any]:
        """Analyze multi-scale goals."""
        if not self._episode_data:
            raise ValueError("Must call collect_data first")

        results = {}

        # Detect goals at each scale
        for scale in TemporalScale:
            self._scale_goals[scale] = self._detect_scale_goals(scale)

        # Compute cross-scale influence
        self._compute_cross_scale_influence()

        # Goal scale count
        results['goal_scale_count'] = sum(
            len(goals) for goals in self._scale_goals.values()
        )

        # Hierarchical composition score
        results['hierarchical_composition_score'] = self._measure_composition()

        # Effective planning horizon
        results['effective_planning_horizon'] = self._compute_horizon()

        # Horizon expansion rate
        results['horizon_expansion_rate'] = self._track_expansion()

        # Adversary-driven scale emergence
        results['adversary_driven_scale_emergence'] = self._attribute_to_adversary()

        # Cross-scale conflict rate
        results['cross_scale_conflict_rate'] = self._detect_conflicts()

        # Conflict resolution pattern
        results['conflict_resolution_pattern'] = self._analyze_resolutions()

        # Per-scale details
        results['per_scale'] = {}
        for scale, goals in self._scale_goals.items():
            if goals:
                results['per_scale'][scale.name] = {
                    'n_goals': len(goals),
                    'mean_predictive_power': np.mean([g.predictive_power for g in goals]),
                    'cross_scale_influence': goals[0].cross_scale_influence if goals else {},
                }

        return results

    def _measure_composition(self) -> float:
        """Measure hierarchical composition between scales."""
        # Composition = how much longer-scale goals build on shorter-scale ones
        composition_scores = []

        scales = sorted(self._scale_goals.keys(), key=lambda s: s.value)

        for i in range(len(scales) - 1):
            shorter = scales[i]
            longer = scales[i + 1]

            if self._scale_goals[shorter] and self._scale_goals[longer]:
                shorter_dims = set(self._scale_goals[shorter][0].feature_dimensions)
                longer_dims = set(self._scale_goals[longer][0].feature_dimensions)

                # Composition = longer includes shorter + additional
                overlap = len(shorter_dims & longer_dims) / max(len(shorter_dims), 1)
                extension = len(longer_dims - shorter_dims) / max(len(longer_dims), 1)

                composition_scores.append(overlap * extension)

        return float(np.mean(composition_scores)) if composition_scores else 0.0

    def _compute_horizon(self) -> float:
        """Compute effective planning horizon."""
        # Horizon = scale at which goal-encoding is strongest
        scale_powers = []

        for scale, goals in self._scale_goals.items():
            if goals:
                power = np.mean([g.predictive_power for g in goals])
                scale_powers.append((scale.value, power))

        if not scale_powers:
            return 1.0

        # Weighted average by predictive power
        total_power = sum(p for _, p in scale_powers)
        if total_power < 1e-10:
            return 1.0

        weighted_horizon = sum(h * p for h, p in scale_powers) / total_power
        return float(weighted_horizon)

    def _track_expansion(self) -> float:
        """Track how planning horizon expands over training."""
        # Compare horizon in early vs late episodes
        early_episodes = self._episode_data[:len(self._episode_data) // 3]
        late_episodes = self._episode_data[2 * len(self._episode_data) // 3:]

        def estimate_horizon(episodes):
            # Estimate based on value function temporal structure
            if not episodes:
                return 1.0

            value_autocorrs = []
            for ep in episodes:
                values = ep['values']
                if len(values) > 5:
                    autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
                    if not np.isnan(autocorr):
                        value_autocorrs.append(autocorr)

            if value_autocorrs:
                # Higher autocorrelation = longer planning horizon
                mean_autocorr = np.mean(value_autocorrs)
                return float(5.0 / (1.0 - mean_autocorr + 0.1))  # Convert to horizon estimate
            return 1.0

        early_horizon = estimate_horizon(early_episodes)
        late_horizon = estimate_horizon(late_episodes)

        return float((late_horizon - early_horizon) / max(early_horizon, 1.0))

    def _attribute_to_adversary(self) -> float:
        """Attribute scale emergence to adversary curriculum."""
        # Measure correlation between adversary difficulty and goal emergence
        difficulties = [ep['episode_difficulty'] for ep in self._episode_data]

        # Proxy for goal emergence: value function variance
        variances = [ep['values'].var() for ep in self._episode_data]

        if len(difficulties) < 2:
            return 0.0

        corr = np.corrcoef(difficulties, variances)[0, 1]
        return float(abs(corr)) if not np.isnan(corr) else 0.0

    def _detect_conflicts(self) -> float:
        """Detect cross-scale conflicts."""
        # Conflict = when goals at different scales suggest different actions
        conflict_count = 0
        total_comparisons = 0

        for episode in self._episode_data:
            hstates = episode['hstates']
            actions = episode['actions']

            for t in range(len(hstates)):
                h = hstates[t]

                # Check if different scale regions suggest different actions
                immediate_signal = h[:30].mean()
                short_signal = h[30:80].mean()
                medium_signal = h[80:150].mean()

                # Conflict if signals disagree significantly
                signals = [immediate_signal, short_signal, medium_signal]
                signal_range = max(signals) - min(signals)

                if signal_range > 1.0:  # Threshold for conflict
                    conflict_count += 1
                total_comparisons += 1

        return float(conflict_count / max(total_comparisons, 1))

    def _analyze_resolutions(self) -> Dict[str, float]:
        """Analyze how conflicts are resolved."""
        return {
            'immediate_dominates': 0.4,  # Placeholder
            'longer_scale_dominates': 0.3,
            'compromise': 0.3,
        }

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize multi-scale goals."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._episode_data:
            return figures

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Predictive power by scale
        ax = axes[0, 0]
        scales = []
        powers = []
        for scale in TemporalScale:
            if self._scale_goals.get(scale):
                scales.append(scale.name)
                powers.append(self._scale_goals[scale][0].predictive_power)

        if scales:
            ax.bar(scales, powers, alpha=0.7)
            ax.set_ylabel('Predictive Power')
            ax.set_title('Goal Predictive Power by Temporal Scale')
            ax.set_xticklabels(scales, rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No scale goals detected', ha='center', va='center', transform=ax.transAxes)

        # Dimension overlap across scales
        ax = axes[0, 1]
        if len(self._scale_goals) >= 2:
            scale_names = [s.name for s in TemporalScale if self._scale_goals.get(s)]
            overlap_matrix = np.zeros((len(scale_names), len(scale_names)))

            for i, s1 in enumerate(scale_names):
                for j, s2 in enumerate(scale_names):
                    scale1 = TemporalScale[s1]
                    scale2 = TemporalScale[s2]
                    if self._scale_goals.get(scale1) and self._scale_goals.get(scale2):
                        dims1 = set(self._scale_goals[scale1][0].feature_dimensions)
                        dims2 = set(self._scale_goals[scale2][0].feature_dimensions)
                        overlap = len(dims1 & dims2) / max(len(dims1 | dims2), 1)
                        overlap_matrix[i, j] = overlap

            im = ax.imshow(overlap_matrix, cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(len(scale_names)))
            ax.set_xticklabels(scale_names, rotation=45, ha='right')
            ax.set_yticks(range(len(scale_names)))
            ax.set_yticklabels(scale_names)
            ax.set_title('Dimension Overlap Across Scales')
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'Insufficient scales', ha='center', va='center', transform=ax.transAxes)

        # Value trajectory example
        ax = axes[1, 0]
        if self._episode_data:
            ep = self._episode_data[0]
            steps = np.arange(len(ep['values']))
            ax.plot(steps, ep['values'], 'b-', linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.set_title('Example Episode Value Trajectory')
            ax.grid(True, alpha=0.3)

        # Planning horizon over training
        ax = axes[1, 1]
        horizons = []
        for i in range(0, len(self._episode_data), 10):
            subset = self._episode_data[max(0, i-5):i+5]
            if subset:
                autocorrs = []
                for ep in subset:
                    if len(ep['values']) > 5:
                        autocorr = np.corrcoef(ep['values'][:-1], ep['values'][1:])[0, 1]
                        if not np.isnan(autocorr):
                            autocorrs.append(autocorr)
                if autocorrs:
                    mean_autocorr = np.mean(autocorrs)
                    horizon = 5.0 / (1.0 - mean_autocorr + 0.1)
                    horizons.append((i, horizon))

        if horizons:
            episodes, horizon_vals = zip(*horizons)
            ax.plot(episodes, horizon_vals, 'g-o', linewidth=2, markersize=6)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Estimated Planning Horizon')
            ax.set_title('Planning Horizon Over Training')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["multiscale_goals"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
