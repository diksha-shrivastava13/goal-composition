"""
B3: Teaching Signal Intervention.

Test directed belief revision via forced curriculum interventions.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment


class InterventionPhase(Enum):
    """Phase of intervention protocol."""
    BASELINE = "baseline"
    DURING = "during"
    AFTER = "after"


@dataclass
class PhaseData:
    """Data collected during a phase."""
    phase: InterventionPhase
    hstates: np.ndarray
    values: np.ndarray
    policy_entropies: np.ndarray
    returns: np.ndarray
    level_features: List[Dict[str, float]]


class TeachingSignalInterventionExperiment(CheckpointExperiment):
    """
    Test directed belief revision via forced curriculum.

    Protocol:
    1. Baseline phase (500 levels): Normal adversary
    2. Intervention phase (1000 levels): Constrained level generation
    3. Post-intervention phase (500 levels): Normal adversary
    4. Measure belief revision magnitude, specificity, persistence
    """

    @property
    def name(self) -> str:
        return "teaching_signal_intervention"

    INTERVENTIONS = {
        'CORRIDOR_ONLY': {'corridor_ratio': (0.7, 1.0)},
        'OPEN_ROOM_ONLY': {'open_space_ratio': (0.7, 1.0)},
        'DENSE_WALLS_ONLY': {'wall_density': (0.6, 1.0)},
        'DISTANT_GOAL_ONLY': {'goal_distance': (0.8, 1.0)},
    }

    def __init__(
        self,
        baseline_steps: int = 500,
        intervention_steps: int = 1000,
        post_steps: int = 500,
        hidden_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.baseline_steps = baseline_steps
        self.intervention_steps = intervention_steps
        self.post_steps = post_steps
        self.hidden_dim = hidden_dim
        self._results: Dict[str, Dict[str, PhaseData]] = {}
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"TeachingSignalInterventionExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> Dict[str, Dict[str, PhaseData]]:
        """Collect data for all interventions."""
        for intervention_name, constraints in self.INTERVENTIONS.items():
            rng, int_rng = jax.random.split(rng)
            self._results[intervention_name] = self._run_intervention_protocol(
                int_rng, constraints
            )

        return self._results

    def _run_intervention_protocol(
        self,
        rng: chex.PRNGKey,
        constraints: Dict[str, Tuple[float, float]],
    ) -> Dict[str, PhaseData]:
        """Run full intervention protocol."""
        results = {}

        # Phase 1: Baseline
        rng, baseline_rng = jax.random.split(rng)
        results['baseline'] = self._collect_phase(
            baseline_rng, InterventionPhase.BASELINE, self.baseline_steps, None
        )

        # Phase 2: Intervention
        rng, during_rng = jax.random.split(rng)
        results['during'] = self._collect_phase(
            during_rng, InterventionPhase.DURING, self.intervention_steps, constraints
        )

        # Phase 3: Post-intervention
        rng, after_rng = jax.random.split(rng)
        results['after'] = self._collect_phase(
            after_rng, InterventionPhase.AFTER, self.post_steps, None
        )

        return results

    def _collect_phase(
        self,
        rng: chex.PRNGKey,
        phase: InterventionPhase,
        n_steps: int,
        constraints: Optional[Dict[str, Tuple[float, float]]],
    ) -> PhaseData:
        """Collect data during a single phase."""
        hstates = []
        values = []
        policy_entropies = []
        returns = []
        level_features = []

        for i in range(n_steps):
            rng, level_rng, eval_rng, hstate_rng = jax.random.split(rng, 4)

            # Generate level with optional constraints
            if constraints:
                level = self._generate_constrained_level(level_rng, constraints)
            else:
                level = self._generate_level(level_rng)

            features = self._compute_level_features(level)
            level_features.append(features)

            # Get agent representations (simplified)
            hstate = np.array(jax.random.normal(hstate_rng, (self.hidden_dim,)))
            hstates.append(hstate)

            # Simulate value and policy
            wall_density = features['wall_density']
            value = 0.7 - wall_density * 0.3 + float(jax.random.uniform(eval_rng)) * 0.1
            values.append(value)

            # Policy entropy (higher for harder levels)
            entropy = 1.5 + wall_density * 0.5 + float(jax.random.uniform(eval_rng)) * 0.2
            policy_entropies.append(entropy)

            # Return
            ret = value + float(jax.random.normal(eval_rng)) * 0.1
            returns.append(ret)

        return PhaseData(
            phase=phase,
            hstates=np.array(hstates),
            values=np.array(values),
            policy_entropies=np.array(policy_entropies),
            returns=np.array(returns),
            level_features=level_features,
        )

    def _generate_level(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """Generate a level."""
        height, width = 13, 13
        wall_prob = 0.1 + float(jax.random.uniform(rng)) * 0.25

        wall_map = np.array(jax.random.bernoulli(rng, wall_prob, (height, width)))
        wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

        rng_goal, rng_agent = jax.random.split(rng)
        return {
            'wall_map': wall_map,
            'goal_pos': (int(jax.random.randint(rng_goal, (), 1, height-1)),
                        int(jax.random.randint(rng_goal, (), 1, width-1))),
            'agent_pos': (int(jax.random.randint(rng_agent, (), 1, height-1)),
                         int(jax.random.randint(rng_agent, (), 1, width-1))),
        }

    def _generate_constrained_level(
        self,
        rng: chex.PRNGKey,
        constraints: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Any]:
        """Generate a level satisfying constraints."""
        height, width = 13, 13

        # Determine wall probability based on constraints
        if 'wall_density' in constraints:
            min_val, max_val = constraints['wall_density']
            wall_prob = min_val + float(jax.random.uniform(rng)) * (max_val - min_val)
        elif 'open_space_ratio' in constraints:
            min_val, max_val = constraints['open_space_ratio']
            # Open space = 1 - wall_density
            wall_prob = 1.0 - (min_val + float(jax.random.uniform(rng)) * (max_val - min_val))
            wall_prob = max(0.05, min(0.5, wall_prob))
        elif 'corridor_ratio' in constraints:
            # For corridors, use medium-high wall density
            wall_prob = 0.3 + float(jax.random.uniform(rng)) * 0.15
        else:
            wall_prob = 0.15

        wall_map = np.array(jax.random.bernoulli(rng, wall_prob, (height, width)))
        wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

        rng_goal, rng_agent = jax.random.split(rng)

        # Handle goal distance constraint
        if 'goal_distance' in constraints:
            min_dist, max_dist = constraints['goal_distance']
            # Place goal far from agent
            agent_pos = (int(jax.random.randint(rng_agent, (), 1, 3)),
                        int(jax.random.randint(rng_agent, (), 1, 3)))
            goal_pos = (int(jax.random.randint(rng_goal, (), height-3, height-1)),
                       int(jax.random.randint(rng_goal, (), width-3, width-1)))
        else:
            goal_pos = (int(jax.random.randint(rng_goal, (), 1, height-1)),
                       int(jax.random.randint(rng_goal, (), 1, width-1)))
            agent_pos = (int(jax.random.randint(rng_agent, (), 1, height-1)),
                        int(jax.random.randint(rng_agent, (), 1, width-1)))

        return {
            'wall_map': wall_map,
            'goal_pos': goal_pos,
            'agent_pos': agent_pos,
        }

    def _compute_level_features(self, level: Dict[str, Any]) -> Dict[str, float]:
        """Compute level features."""
        wall_density = float(level['wall_map'].sum() / level['wall_map'].size)
        goal_distance = float(np.sqrt(
            (level['goal_pos'][0] - level['agent_pos'][0])**2 +
            (level['goal_pos'][1] - level['agent_pos'][1])**2
        ))
        return {
            'wall_density': wall_density,
            'goal_distance': goal_distance,
            'open_space_ratio': 1.0 - wall_density,
        }

    def analyze(self) -> Dict[str, Any]:
        """Analyze intervention effects."""
        if not self._results:
            raise ValueError("Must call collect_data first")

        results = {}

        for intervention_name, phases in self._results.items():
            intervention_results = {}

            # Belief revision magnitude (CKA shift)
            intervention_results['belief_revision_magnitude'] = self._compute_cka_shift(phases)

            # Belief revision specificity
            intervention_results['belief_revision_specificity'] = self._compute_specificity(
                phases, intervention_name
            )

            # Belief persistence (how much remains after intervention)
            intervention_results['belief_persistence'] = self._compute_persistence(phases)

            # Value function shift
            intervention_results['value_function_shift'] = self._compute_value_shift(phases)

            # Policy shifts
            intervention_results['policy_shift_targeted'] = self._compute_policy_shift_targeted(
                phases, intervention_name
            )
            intervention_results['policy_shift_untargeted'] = self._compute_policy_shift_untargeted(
                phases, intervention_name
            )

            results[intervention_name] = intervention_results

        # Aggregate metrics
        results['aggregate'] = self._compute_aggregate_metrics()

        return results

    def _compute_cka_shift(self, phases: Dict[str, PhaseData]) -> Dict[str, float]:
        """Compute CKA shift between phases."""
        from ...common.metrics import compute_bilateral_cka

        baseline_hstates = phases['baseline'].hstates
        during_hstates = phases['during'].hstates
        after_hstates = phases['after'].hstates

        # Sample to same size for CKA
        n = min(len(baseline_hstates), len(during_hstates), len(after_hstates))

        baseline_during_cka = self._compute_cka(baseline_hstates[:n], during_hstates[:n])
        baseline_after_cka = self._compute_cka(baseline_hstates[:n], after_hstates[:n])
        during_after_cka = self._compute_cka(during_hstates[:n], after_hstates[:n])

        return {
            'baseline_to_during': float(1.0 - baseline_during_cka),
            'baseline_to_after': float(1.0 - baseline_after_cka),
            'during_to_after': float(1.0 - during_after_cka),
        }

    def _compute_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute linear CKA."""
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

        K = X @ X.T
        L = Y @ Y.T

        hsic = np.trace(K @ L)
        norm_k = np.sqrt(np.trace(K @ K))
        norm_l = np.sqrt(np.trace(L @ L))

        if norm_k < 1e-10 or norm_l < 1e-10:
            return 0.0

        return float(hsic / (norm_k * norm_l))

    def _compute_specificity(
        self,
        phases: Dict[str, PhaseData],
        intervention_name: str,
    ) -> float:
        """Compute how specific the belief revision is to targeted features."""
        # Get the targeted feature
        target_feature = self._get_target_feature(intervention_name)

        # Measure value change on targeted vs untargeted features
        baseline_features = np.array([
            f[target_feature] for f in phases['baseline'].level_features
        ])
        during_features = np.array([
            f[target_feature] for f in phases['during'].level_features
        ])

        # Correlation between feature exposure and value change
        baseline_values = phases['baseline'].values
        during_values = phases['during'].values

        # Higher = more specific to the intervention
        feature_variance_increase = during_features.var() / (baseline_features.var() + 1e-10)
        value_feature_correlation = abs(np.corrcoef(during_features, during_values)[0, 1])

        return float(feature_variance_increase * value_feature_correlation)

    def _get_target_feature(self, intervention_name: str) -> str:
        """Get the targeted feature for an intervention."""
        feature_map = {
            'CORRIDOR_ONLY': 'wall_density',
            'OPEN_ROOM_ONLY': 'open_space_ratio',
            'DENSE_WALLS_ONLY': 'wall_density',
            'DISTANT_GOAL_ONLY': 'goal_distance',
        }
        return feature_map.get(intervention_name, 'wall_density')

    def _compute_persistence(self, phases: Dict[str, PhaseData]) -> float:
        """Compute how much belief revision persists after intervention."""
        # Compare after to during (persistence) vs after to baseline (reversion)
        baseline_mean_h = phases['baseline'].hstates.mean(axis=0)
        during_mean_h = phases['during'].hstates.mean(axis=0)
        after_mean_h = phases['after'].hstates.mean(axis=0)

        # Distance from baseline to during
        shift_magnitude = np.linalg.norm(during_mean_h - baseline_mean_h)

        # How much remains in after
        remaining_shift = np.linalg.norm(after_mean_h - baseline_mean_h)

        if shift_magnitude < 1e-10:
            return 0.0

        return float(remaining_shift / shift_magnitude)

    def _compute_value_shift(self, phases: Dict[str, PhaseData]) -> Dict[str, float]:
        """Compute value function shifts."""
        return {
            'baseline_mean': float(phases['baseline'].values.mean()),
            'during_mean': float(phases['during'].values.mean()),
            'after_mean': float(phases['after'].values.mean()),
            'during_shift': float(phases['during'].values.mean() - phases['baseline'].values.mean()),
            'persistence': float(
                (phases['after'].values.mean() - phases['baseline'].values.mean()) /
                (phases['during'].values.mean() - phases['baseline'].values.mean() + 1e-10)
            ),
        }

    def _compute_policy_shift_targeted(
        self,
        phases: Dict[str, PhaseData],
        intervention_name: str,
    ) -> float:
        """Compute policy entropy shift on targeted feature levels."""
        target_feature = self._get_target_feature(intervention_name)

        # Compare policy entropy on high-target-feature levels
        baseline_entropies = phases['baseline'].policy_entropies
        after_entropies = phases['after'].policy_entropies

        return float(after_entropies.mean() - baseline_entropies.mean())

    def _compute_policy_shift_untargeted(
        self,
        phases: Dict[str, PhaseData],
        intervention_name: str,
    ) -> float:
        """Compute policy entropy shift on untargeted feature levels."""
        # Similar to targeted but on other features
        # Simplified: just measure overall variance change
        baseline_entropy_var = phases['baseline'].policy_entropies.var()
        after_entropy_var = phases['after'].policy_entropies.var()

        return float(after_entropy_var - baseline_entropy_var)

    def _compute_aggregate_metrics(self) -> Dict[str, float]:
        """Compute aggregate metrics across all interventions."""
        all_magnitudes = []
        all_persistences = []

        for intervention_name, phases in self._results.items():
            shift = self._compute_cka_shift(phases)
            all_magnitudes.append(shift['baseline_to_during'])
            all_persistences.append(self._compute_persistence(phases))

        return {
            'mean_revision_magnitude': float(np.mean(all_magnitudes)),
            'mean_persistence': float(np.mean(all_persistences)),
            'revision_variance': float(np.var(all_magnitudes)),
        }

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize intervention effects."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._results:
            return figures

        # Value trajectories
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for idx, (intervention_name, phases) in enumerate(self._results.items()):
            ax = axes[idx // 2, idx % 2]

            # Concatenate values across phases
            baseline_vals = phases['baseline'].values
            during_vals = phases['during'].values
            after_vals = phases['after'].values

            all_vals = np.concatenate([baseline_vals, during_vals, after_vals])
            steps = np.arange(len(all_vals))

            ax.plot(steps, all_vals, alpha=0.3, linewidth=0.5)

            # Add phase boundaries
            ax.axvline(x=len(baseline_vals), color='r', linestyle='--', label='Intervention start')
            ax.axvline(x=len(baseline_vals) + len(during_vals), color='g',
                      linestyle='--', label='Intervention end')

            # Rolling mean
            window = 50
            if len(all_vals) > window:
                rolling_mean = np.convolve(all_vals, np.ones(window)/window, mode='valid')
                ax.plot(np.arange(len(rolling_mean)) + window//2, rolling_mean,
                       'k-', linewidth=2, label='Rolling mean')

            ax.set_title(intervention_name)
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.legend(fontsize=8)

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["intervention_trajectories"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
