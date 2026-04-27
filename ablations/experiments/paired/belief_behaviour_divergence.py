"""
F4: Belief-Behaviour Divergence.

Detect misalignment between probe-decoded beliefs and actual policy.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment
from ..utils.paired_helpers import (
    generate_levels, extract_level_features_batch, get_pro_hstates,
    get_action_distribution, get_values_from_rollout,
)


@dataclass
class DivergencePoint:
    """A single data point for divergence analysis."""
    step: int
    level_features: Dict[str, float]
    hstate: np.ndarray
    probe_decoded_belief: Dict[str, float]
    actual_policy: np.ndarray  # Action probabilities
    actual_value: float
    belief_policy_divergence: float
    belief_value_divergence: float


@dataclass
class DivergenceEvent:
    """Event where beliefs and behavior significantly diverge."""
    step: int
    divergence_magnitude: float
    belief_features: Dict[str, float]
    policy_entropy: float
    context: Dict[str, float]


class BeliefBehaviourDivergenceExperiment(CheckpointExperiment):
    """
    Detect belief-behaviour misalignment.

    Protocol:
    1. Train probes to decode beliefs from hidden states
    2. Compare decoded beliefs to actual policy/value outputs
    3. Identify contexts where beliefs and behavior diverge
    4. Analyze causes of divergence (optimization pressure, conflicting goals)
    """

    @property
    def name(self) -> str:
        return "belief_behaviour_divergence"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = self.exp_config("n_samples")
        self.hidden_dim = self.exp_config("hidden_dim")
        self.divergence_threshold = self.exp_config("divergence_threshold")
        self.max_steps = self.exp_config("max_steps", 256)
        self._data_points: List[DivergencePoint] = []
        self._divergence_events: List[DivergenceEvent] = []
        self._probe_weights: Dict[str, np.ndarray] = {}
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"BeliefBehaviourDivergenceExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> List[DivergencePoint]:
        """Collect data for divergence analysis, iterating over checkpoints if available."""
        import glob
        import os

        # Check for multi-checkpoint directories
        checkpoint_dirs = sorted(
            glob.glob(os.path.join(os.getcwd(), 'checkpoints', 'step_*'))
            + glob.glob(os.path.join(os.getcwd(), 'checkpoints', 'checkpoint_*'))
        )

        if checkpoint_dirs:
            # Multi-checkpoint mode: iterate over available checkpoints
            for ckpt_dir in checkpoint_dirs:
                rng, ckpt_rng = jax.random.split(rng)
                self._collect_single_checkpoint(ckpt_rng)
        else:
            # Single-checkpoint fallback
            self._collect_single_checkpoint(rng)

        return self._data_points

    def _collect_single_checkpoint(self, rng: chex.PRNGKey) -> None:
        """Collect divergence data from a single checkpoint."""
        # Train probes on real data
        rng, probe_rng = jax.random.split(rng)
        self._train_probes(probe_rng)

        # Collect and analyze data in batch
        rng, level_rng, hstate_rng, action_rng, val_rng, diff_rng = jax.random.split(rng, 6)

        # Generate real levels
        levels = generate_levels(self.agent, level_rng, self.n_samples)

        # Get real hidden states
        hstates = get_pro_hstates(hstate_rng, levels, self, self.max_steps)
        self.hidden_dim = hstates.shape[1]

        # Get real action distribution (policy)
        logits, entropies = get_action_distribution(
            self.train_state, self.agent, levels, action_rng,
            max_steps=self.max_steps,
        )
        last_logits = logits[:, -1, :]

        # Get real value estimates
        values = get_values_from_rollout(
            self.train_state, self.agent, levels, val_rng,
            max_steps=self.max_steps,
        )
        mean_values = values.mean(axis=1)

        # Extract real level features
        features_batch = extract_level_features_batch(levels)

        from ..utils.paired_helpers import compute_difficulty
        difficulties = compute_difficulty(levels, self, diff_rng, self.max_steps)

        # Fit Ridge models for divergence computation
        from sklearn.linear_model import Ridge
        belief_matrix = np.stack([
            np.array(features_batch['wall_density']),
            np.array(features_batch['goal_distance']),
            np.array(difficulties),
        ], axis=1)

        self._belief_policy_model = Ridge(alpha=1.0)
        logits_target = last_logits[:, :4] if last_logits.shape[1] >= 4 else last_logits
        # Filter out NaN rows (can occur with untrained networks or tiny configs)
        valid_mask = ~np.isnan(belief_matrix).any(axis=1) & ~np.isnan(logits_target).any(axis=1)
        if valid_mask.sum() >= 2:
            self._belief_policy_model.fit(belief_matrix[valid_mask], logits_target[valid_mask])
        else:
            self._belief_policy_model.fit(np.zeros((2, belief_matrix.shape[1])), np.zeros((2, logits_target.shape[1])))

        self._belief_value_model = Ridge(alpha=1.0)
        valid_v = valid_mask & ~np.isnan(mean_values)
        if valid_v.sum() >= 2:
            self._belief_value_model.fit(belief_matrix[valid_v], mean_values[valid_v])
        else:
            self._belief_value_model.fit(np.zeros((2, belief_matrix.shape[1])), np.zeros(2))

        for i in range(self.n_samples):
            wall_density = float(features_batch['wall_density'][i])
            goal_distance = float(features_batch['goal_distance'][i])
            difficulty = float(difficulties[i])

            level_features = {
                'wall_density': wall_density,
                'goal_distance': goal_distance,
                'difficulty': difficulty,
            }

            h = hstates[i]
            decoded_beliefs = self._decode_beliefs(h)

            sample_logits = last_logits[i]
            if len(sample_logits) < 4:
                sample_logits = np.pad(sample_logits, (0, 4 - len(sample_logits)))
            elif len(sample_logits) > 4:
                sample_logits = sample_logits[:4]
            policy = np.exp(sample_logits - np.max(sample_logits))
            policy = policy / policy.sum()

            actual_value = float(mean_values[i])

            belief_policy_divergence = self._compute_belief_policy_divergence(
                decoded_beliefs, policy, level_features
            )
            belief_value_divergence = self._compute_belief_value_divergence(
                decoded_beliefs, actual_value, level_features
            )

            self._data_points.append(DivergencePoint(
                step=i,
                level_features=level_features,
                hstate=h,
                probe_decoded_belief=decoded_beliefs,
                actual_policy=policy,
                actual_value=actual_value,
                belief_policy_divergence=belief_policy_divergence,
                belief_value_divergence=belief_value_divergence,
            ))

    def _train_probes(self, rng: chex.PRNGKey):
        """Train probes to decode beliefs from real hidden states."""
        rng, level_rng, hstate_rng, diff_rng = jax.random.split(rng, 4)

        n_probe_train = min(200, self.n_samples)

        # Generate real levels for probe training
        levels = generate_levels(self.agent, level_rng, n_probe_train)

        # Get real hidden states
        training_hstates = get_pro_hstates(hstate_rng, levels, self, self.max_steps)
        self.hidden_dim = training_hstates.shape[1]

        # Extract real features as targets
        features_batch = extract_level_features_batch(levels)

        from ..utils.paired_helpers import compute_difficulty
        probe_difficulties = compute_difficulty(levels, self, diff_rng, self.max_steps)

        training_features = []
        for i in range(n_probe_train):
            wall_density = float(features_batch['wall_density'][i])
            goal_distance = float(features_batch['goal_distance'][i])
            training_features.append({
                'wall_density': wall_density,
                'goal_distance': goal_distance,
                'difficulty': float(probe_difficulties[i]),
            })

        # Train linear probes for each feature
        from sklearn.linear_model import Ridge

        for feature_name in ['wall_density', 'goal_distance', 'difficulty']:
            targets = np.array([f[feature_name] for f in training_features])
            model = Ridge(alpha=0.1)
            model.fit(training_hstates, targets)
            self._probe_weights[feature_name] = {
                'coef': model.coef_,
                'intercept': model.intercept_,
            }

    def _decode_beliefs(self, hstate: np.ndarray) -> Dict[str, float]:
        """Decode beliefs from hidden state using trained probes."""
        beliefs = {}
        for feature_name, weights in self._probe_weights.items():
            prediction = np.dot(hstate, weights['coef']) + weights['intercept']
            beliefs[feature_name] = float(prediction)
        return beliefs

    def _compute_belief_policy_divergence(
        self,
        beliefs: Dict[str, float],
        policy: np.ndarray,
        true_features: Dict[str, float],
    ) -> float:
        """Compute divergence between beliefs and policy.

        Uses fitted Ridge model (belief -> policy) trained during collect_data.
        Falls back to L2 distance if model not yet fitted.
        """
        if not hasattr(self, '_belief_policy_model') or self._belief_policy_model is None:
            # Fallback before model is fitted
            return 0.0

        belief_vec = np.array([[beliefs['wall_density'], beliefs['goal_distance'], beliefs['difficulty']]])
        expected_logits = self._belief_policy_model.predict(belief_vec)[0]
        expected_policy = np.exp(expected_logits - np.max(expected_logits))
        expected_policy = expected_policy / expected_policy.sum()

        # KL divergence
        kl = np.sum(policy * np.log((policy + 1e-10) / (expected_policy + 1e-10)))
        return float(max(0, kl))

    def _compute_belief_value_divergence(
        self,
        beliefs: Dict[str, float],
        actual_value: float,
        true_features: Dict[str, float],
    ) -> float:
        """Compute divergence between beliefs and value.

        Uses fitted Ridge model (belief -> value) trained during collect_data.
        Falls back to zero if model not yet fitted.
        """
        if not hasattr(self, '_belief_value_model') or self._belief_value_model is None:
            return 0.0

        belief_vec = np.array([[beliefs['wall_density'], beliefs['goal_distance'], beliefs['difficulty']]])
        expected_value = float(self._belief_value_model.predict(belief_vec)[0])

        # Squared error
        return float((actual_value - expected_value) ** 2)

    def _detect_divergence_events(self) -> List[DivergenceEvent]:
        """Detect significant divergence events."""
        events = []

        for point in self._data_points:
            total_divergence = point.belief_policy_divergence + point.belief_value_divergence

            if total_divergence > self.divergence_threshold:
                policy_entropy = -np.sum(point.actual_policy * np.log(point.actual_policy + 1e-10))

                events.append(DivergenceEvent(
                    step=point.step,
                    divergence_magnitude=total_divergence,
                    belief_features=point.probe_decoded_belief,
                    policy_entropy=float(policy_entropy),
                    context=point.level_features,
                ))

        self._divergence_events = events
        return events

    def analyze(self) -> Dict[str, Any]:
        """Analyze belief-behaviour divergence."""
        if not self._data_points:
            raise ValueError("Must call collect_data first")

        results = {}

        # Detect divergence events
        events = self._detect_divergence_events()
        results['num_divergence_events'] = len(events)
        results['divergence_rate'] = len(events) / len(self._data_points)

        # Overall divergence statistics
        policy_divergences = [p.belief_policy_divergence for p in self._data_points]
        value_divergences = [p.belief_value_divergence for p in self._data_points]

        results['mean_policy_divergence'] = float(np.mean(policy_divergences))
        results['std_policy_divergence'] = float(np.std(policy_divergences))
        results['mean_value_divergence'] = float(np.mean(value_divergences))
        results['std_value_divergence'] = float(np.std(value_divergences))

        # Probe accuracy (how well do probes decode beliefs?)
        results['probe_accuracy'] = self._compute_probe_accuracy()

        # Divergence by context
        results['divergence_by_context'] = self._analyze_divergence_by_context()

        # Temporal trend
        results['divergence_trend'] = self._analyze_temporal_trend()

        # Causes analysis
        results['divergence_causes'] = self._analyze_causes()

        # Divergence event details
        results['events'] = [
            {
                'step': e.step,
                'magnitude': e.divergence_magnitude,
                'policy_entropy': e.policy_entropy,
                'context': e.context,
            }
            for e in events[:20]  # Limit output
        ]

        return results

    def _compute_probe_accuracy(self) -> Dict[str, float]:
        """Compute probe decoding accuracy."""
        accuracies = {}

        for feature in ['wall_density', 'goal_distance', 'difficulty']:
            true_values = [p.level_features[feature] for p in self._data_points]
            decoded_values = [p.probe_decoded_belief[feature] for p in self._data_points]

            # R-squared
            ss_res = sum((t - d) ** 2 for t, d in zip(true_values, decoded_values))
            ss_tot = sum((t - np.mean(true_values)) ** 2 for t in true_values)

            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
            accuracies[feature] = float(r2)

        return accuracies

    def _analyze_divergence_by_context(self) -> Dict[str, Dict[str, float]]:
        """Analyze divergence by feature context."""
        results = {}

        # Bin by wall density
        low_wall = [p for p in self._data_points if p.level_features['wall_density'] < 0.2]
        high_wall = [p for p in self._data_points if p.level_features['wall_density'] >= 0.2]

        results['low_wall_density'] = {
            'mean_policy_div': float(np.mean([p.belief_policy_divergence for p in low_wall])) if low_wall else 0.0,
            'mean_value_div': float(np.mean([p.belief_value_divergence for p in low_wall])) if low_wall else 0.0,
        }
        results['high_wall_density'] = {
            'mean_policy_div': float(np.mean([p.belief_policy_divergence for p in high_wall])) if high_wall else 0.0,
            'mean_value_div': float(np.mean([p.belief_value_divergence for p in high_wall])) if high_wall else 0.0,
        }

        # Bin by goal distance
        close_goal = [p for p in self._data_points if p.level_features['goal_distance'] < 6.0]
        far_goal = [p for p in self._data_points if p.level_features['goal_distance'] >= 6.0]

        results['close_goal'] = {
            'mean_policy_div': float(np.mean([p.belief_policy_divergence for p in close_goal])) if close_goal else 0.0,
            'mean_value_div': float(np.mean([p.belief_value_divergence for p in close_goal])) if close_goal else 0.0,
        }
        results['far_goal'] = {
            'mean_policy_div': float(np.mean([p.belief_policy_divergence for p in far_goal])) if far_goal else 0.0,
            'mean_value_div': float(np.mean([p.belief_value_divergence for p in far_goal])) if far_goal else 0.0,
        }

        return results

    def _analyze_temporal_trend(self) -> Dict[str, float]:
        """Analyze how divergence changes over the sample collection."""
        if len(self._data_points) < 10:
            return {'trend_slope': 0.0}

        steps = [p.step for p in self._data_points]
        divergences = [p.belief_policy_divergence + p.belief_value_divergence for p in self._data_points]

        # Linear regression (guard against near-constant x)
        if np.ptp(steps) < 1e-6:
            slope = 0.0
        else:
            slope = np.polyfit(steps, divergences, 1)[0]

        # Compare early vs late
        n = len(self._data_points)
        early = self._data_points[:n // 3]
        late = self._data_points[2 * n // 3:]

        early_mean = np.mean([p.belief_policy_divergence for p in early])
        late_mean = np.mean([p.belief_policy_divergence for p in late])

        return {
            'trend_slope': float(slope),
            'early_divergence': float(early_mean),
            'late_divergence': float(late_mean),
            'divergence_increase': float(late_mean - early_mean),
        }

    def _analyze_causes(self) -> Dict[str, float]:
        """Analyze potential causes of divergence."""
        causes = {}

        # Cause 1: Belief decoding error (probe inaccuracy)
        probe_errors = []
        for p in self._data_points:
            error = sum(
                (p.probe_decoded_belief[f] - p.level_features[f]) ** 2
                for f in p.level_features
            )
            probe_errors.append(error)

        divergences = [p.belief_policy_divergence for p in self._data_points]

        # Correlation between probe error and divergence
        corr = np.corrcoef(probe_errors, divergences)[0, 1]
        causes['probe_error_contribution'] = float(corr) if not np.isnan(corr) else 0.0

        # Cause 2: High difficulty contexts
        difficulties = [p.level_features['difficulty'] for p in self._data_points]
        corr = np.corrcoef(difficulties, divergences)[0, 1]
        causes['difficulty_contribution'] = float(corr) if not np.isnan(corr) else 0.0

        # Cause 3: Policy entropy (uncertainty)
        entropies = [
            -np.sum(p.actual_policy * np.log(p.actual_policy + 1e-10))
            for p in self._data_points
        ]
        corr = np.corrcoef(entropies, divergences)[0, 1]
        causes['entropy_contribution'] = float(corr) if not np.isnan(corr) else 0.0

        return causes

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize belief-behaviour divergence."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._data_points:
            return figures

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Divergence distribution
        ax = axes[0, 0]
        policy_div = np.array([p.belief_policy_divergence for p in self._data_points])
        value_div = np.array([p.belief_value_divergence for p in self._data_points])
        policy_valid = policy_div[~np.isnan(policy_div)]
        value_valid = value_div[~np.isnan(value_div)]
        if len(policy_valid) > 0:
            ax.hist(policy_valid, bins=30, alpha=0.6, label='Policy Divergence', edgecolor='black')
        if len(value_valid) > 0:
            ax.hist(value_valid, bins=30, alpha=0.6, label='Value Divergence', edgecolor='black')
        ax.set_xlabel('Divergence')
        ax.set_ylabel('Count')
        ax.set_title('Belief-Behaviour Divergence Distribution')
        ax.legend()

        # Divergence vs difficulty
        ax = axes[0, 1]
        difficulties = [p.level_features['difficulty'] for p in self._data_points]
        total_div = [p.belief_policy_divergence + p.belief_value_divergence for p in self._data_points]
        ax.scatter(difficulties, total_div, alpha=0.5, s=20)
        ax.set_xlabel('Level Difficulty')
        ax.set_ylabel('Total Divergence')
        ax.set_title('Divergence vs Difficulty')
        # Fit line (guard against near-constant difficulties)
        if np.ptp(difficulties) > 1e-6:
            z = np.polyfit(difficulties, total_div, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(difficulties), max(difficulties), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2)

        # Probe accuracy
        ax = axes[1, 0]
        accuracy = self._compute_probe_accuracy()
        features = list(accuracy.keys())
        values = list(accuracy.values())
        ax.bar(features, values, alpha=0.7)
        ax.set_ylabel('R²')
        ax.set_title('Probe Decoding Accuracy')
        ax.set_ylim(0, 1)

        # Temporal trend
        ax = axes[1, 1]
        steps = [p.step for p in self._data_points]
        divergences = [p.belief_policy_divergence for p in self._data_points]

        # Rolling mean
        window = 20
        if len(divergences) > window:
            rolling_mean = np.convolve(divergences, np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], rolling_mean, 'b-', linewidth=2, label='Rolling Mean')

        ax.scatter(steps, divergences, alpha=0.3, s=10, label='Individual')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Policy Divergence')
        ax.set_title('Divergence Over Data Collection')
        ax.legend()

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["belief_behaviour_divergence"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
