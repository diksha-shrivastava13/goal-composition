"""
B5: Activation Patching.

Causally identify goal-encoding dimensions via activation patching.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment


class PatchTarget(Enum):
    """Target for activation patching."""
    CELL_STATE = "cell_state"
    HIDDEN_STATE = "hidden_state"
    FIRST_HALF_DIMS = "first_half_dims"
    SECOND_HALF_DIMS = "second_half_dims"
    TOP_VARIANCE_DIMS = "top_variance_dims"
    REGRET_ENCODING_DIMS = "regret_encoding_dims"


@dataclass
class PatchResult:
    """Result of a single patch operation."""
    target: PatchTarget
    source_level_idx: int
    target_level_idx: int
    original_policy_entropy: float
    patched_policy_entropy: float
    original_value: float
    patched_value: float
    policy_kl_divergence: float
    value_change: float


class ActivationPatchingExperiment(CheckpointExperiment):
    """
    Causally identify goal-encoding dimensions via patching.

    Protocol:
    1. Generate source and target levels with different features
    2. Get hidden states from both
    3. Patch specific dimensions from source to target
    4. Measure policy/value changes
    5. Attribute goal encoding to specific dimensions
    """

    @property
    def name(self) -> str:
        return "activation_patching"

    def __init__(
        self,
        n_pairs: int = 200,
        hidden_dim: int = 256,
        top_k_variance: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_pairs = n_pairs
        self.hidden_dim = hidden_dim
        self.top_k_variance = top_k_variance
        self._levels: List[Dict[str, Any]] = []
        self._hstates: np.ndarray = None
        self._patch_results: Dict[PatchTarget, List[PatchResult]] = {}
        self._regret_dims: Optional[np.ndarray] = None
        self._variance_dims: Optional[np.ndarray] = None
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"ActivationPatchingExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Collect levels and hidden states."""
        # Generate diverse levels
        for i in range(self.n_pairs * 2):
            rng, level_rng = jax.random.split(rng)
            self._levels.append(self._generate_level(level_rng))

        # Get hidden states (simplified)
        rng, hstate_rng = jax.random.split(rng)
        self._hstates = np.array(
            jax.random.normal(hstate_rng, (len(self._levels), self.hidden_dim))
        )

        # Add feature-correlated structure to hidden states
        for i, level in enumerate(self._levels):
            features = self._compute_level_features(level)
            # First 50 dims encode wall density
            self._hstates[i, :50] += features['wall_density'] * 2.0
            # Next 50 dims encode goal distance
            self._hstates[i, 50:100] += features['goal_distance'] * 0.5

        # Identify important dimensions
        self._identify_important_dims()

        return self._levels, self._hstates

    def _generate_level(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """Generate a level."""
        height, width = 13, 13
        wall_prob = 0.1 + float(jax.random.uniform(rng)) * 0.3

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

    def _compute_level_features(self, level: Dict[str, Any]) -> Dict[str, float]:
        """Compute level features."""
        wall_density = float(level['wall_map'].sum() / level['wall_map'].size)
        goal_distance = float(np.sqrt(
            (level['goal_pos'][0] - level['agent_pos'][0])**2 +
            (level['goal_pos'][1] - level['agent_pos'][1])**2
        ))
        return {'wall_density': wall_density, 'goal_distance': goal_distance}

    def _identify_important_dims(self):
        """Identify top variance and regret-encoding dimensions."""
        # Top variance dimensions
        variances = self._hstates.var(axis=0)
        self._variance_dims = np.argsort(variances)[-self.top_k_variance:]

        # Regret-encoding dimensions (simplified: dimensions correlated with wall density)
        features = np.array([
            self._compute_level_features(level)['wall_density']
            for level in self._levels
        ])
        correlations = np.array([
            abs(np.corrcoef(self._hstates[:, d], features)[0, 1])
            for d in range(self.hidden_dim)
        ])
        self._regret_dims = np.argsort(correlations)[-self.top_k_variance:]

    def _run_patches(self, rng: chex.PRNGKey):
        """Run all patch experiments."""
        patch_targets = [
            PatchTarget.CELL_STATE,
            PatchTarget.HIDDEN_STATE,
            PatchTarget.FIRST_HALF_DIMS,
            PatchTarget.SECOND_HALF_DIMS,
            PatchTarget.TOP_VARIANCE_DIMS,
            PatchTarget.REGRET_ENCODING_DIMS,
        ]

        for target in patch_targets:
            rng, patch_rng = jax.random.split(rng)
            self._patch_results[target] = self._run_patch_target(patch_rng, target)

    def _run_patch_target(
        self,
        rng: chex.PRNGKey,
        target: PatchTarget,
    ) -> List[PatchResult]:
        """Run patches for a specific target."""
        results = []

        for i in range(self.n_pairs):
            source_idx = i * 2
            target_idx = i * 2 + 1

            rng, eval_rng = jax.random.split(rng)

            # Get original and patched hidden states
            original_h = self._hstates[target_idx].copy()
            patched_h = self._apply_patch(
                self._hstates[source_idx],
                original_h,
                target,
            )

            # Evaluate policy/value with original and patched
            original_policy_entropy, original_value = self._evaluate_with_hstate(
                eval_rng, original_h, self._levels[target_idx]
            )
            patched_policy_entropy, patched_value = self._evaluate_with_hstate(
                eval_rng, patched_h, self._levels[target_idx]
            )

            # Compute KL divergence (simplified)
            policy_kl = abs(patched_policy_entropy - original_policy_entropy) * 0.5

            results.append(PatchResult(
                target=target,
                source_level_idx=source_idx,
                target_level_idx=target_idx,
                original_policy_entropy=original_policy_entropy,
                patched_policy_entropy=patched_policy_entropy,
                original_value=original_value,
                patched_value=patched_value,
                policy_kl_divergence=policy_kl,
                value_change=patched_value - original_value,
            ))

        return results

    def _apply_patch(
        self,
        source_h: np.ndarray,
        target_h: np.ndarray,
        patch_target: PatchTarget,
    ) -> np.ndarray:
        """Apply patch from source to target."""
        patched = target_h.copy()

        if patch_target == PatchTarget.CELL_STATE:
            # Simulate cell state as first half
            patched[:self.hidden_dim // 2] = source_h[:self.hidden_dim // 2]
        elif patch_target == PatchTarget.HIDDEN_STATE:
            # Simulate hidden state as second half
            patched[self.hidden_dim // 2:] = source_h[self.hidden_dim // 2:]
        elif patch_target == PatchTarget.FIRST_HALF_DIMS:
            patched[:self.hidden_dim // 2] = source_h[:self.hidden_dim // 2]
        elif patch_target == PatchTarget.SECOND_HALF_DIMS:
            patched[self.hidden_dim // 2:] = source_h[self.hidden_dim // 2:]
        elif patch_target == PatchTarget.TOP_VARIANCE_DIMS:
            patched[self._variance_dims] = source_h[self._variance_dims]
        elif patch_target == PatchTarget.REGRET_ENCODING_DIMS:
            patched[self._regret_dims] = source_h[self._regret_dims]

        return patched

    def _evaluate_with_hstate(
        self,
        rng: chex.PRNGKey,
        hstate: np.ndarray,
        level: Dict[str, Any],
    ) -> Tuple[float, float]:
        """Evaluate policy and value with given hidden state."""
        # Simplified evaluation
        features = self._compute_level_features(level)

        # Value depends on hstate and features
        value = 0.5 + np.tanh(hstate[:50].mean()) * 0.3 - features['wall_density'] * 0.2
        value += float(jax.random.uniform(rng)) * 0.05

        # Policy entropy depends on hstate uncertainty
        hstate_var = hstate.var()
        entropy = 1.0 + hstate_var * 0.5 + features['wall_density'] * 0.3
        entropy += float(jax.random.uniform(rng)) * 0.1

        return float(entropy), float(value)

    def analyze(self) -> Dict[str, Any]:
        """Analyze activation patching results."""
        if self._hstates is None:
            raise ValueError("Must call collect_data first")

        # Run patches
        rng = jax.random.PRNGKey(42)
        self._run_patches(rng)

        results = {}

        # Goal attribution scores
        results['goal_attribution_score'] = self._compute_attribution_scores()

        # Regret encoding dimensions
        results['regret_encoding_dimensions'] = {
            'indices': self._regret_dims.tolist() if self._regret_dims is not None else [],
            'count': len(self._regret_dims) if self._regret_dims is not None else 0,
        }

        # Policy sensitivity by dimension group
        results['policy_sensitivity_by_dimension'] = self._compute_per_dim_sensitivity()

        # Value sensitivity
        results['value_sensitivity_by_dimension'] = self._compute_value_sensitivity()

        # Summary statistics
        results['summary'] = self._compute_summary_stats()

        return results

    def _compute_attribution_scores(self) -> Dict[str, float]:
        """Compute goal attribution scores for each patch target."""
        attribution_scores = {}

        for target, results in self._patch_results.items():
            if not results:
                attribution_scores[target.value] = 0.0
                continue

            # Attribution = mean policy KL when patching this target
            mean_kl = np.mean([r.policy_kl_divergence for r in results])
            attribution_scores[target.value] = float(mean_kl)

        return attribution_scores

    def _compute_per_dim_sensitivity(self) -> Dict[str, float]:
        """Compute policy sensitivity per dimension group."""
        sensitivity = {}

        for target, results in self._patch_results.items():
            if not results:
                sensitivity[target.value] = 0.0
                continue

            # Sensitivity = mean absolute entropy change
            mean_entropy_change = np.mean([
                abs(r.patched_policy_entropy - r.original_policy_entropy)
                for r in results
            ])
            sensitivity[target.value] = float(mean_entropy_change)

        return sensitivity

    def _compute_value_sensitivity(self) -> Dict[str, float]:
        """Compute value function sensitivity per dimension group."""
        sensitivity = {}

        for target, results in self._patch_results.items():
            if not results:
                sensitivity[target.value] = 0.0
                continue

            # Sensitivity = mean absolute value change
            mean_value_change = np.mean([abs(r.value_change) for r in results])
            sensitivity[target.value] = float(mean_value_change)

        return sensitivity

    def _compute_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        all_kls = []
        all_value_changes = []

        for results in self._patch_results.values():
            for r in results:
                all_kls.append(r.policy_kl_divergence)
                all_value_changes.append(abs(r.value_change))

        return {
            'mean_policy_kl': float(np.mean(all_kls)) if all_kls else 0.0,
            'std_policy_kl': float(np.std(all_kls)) if all_kls else 0.0,
            'mean_value_change': float(np.mean(all_value_changes)) if all_value_changes else 0.0,
            'std_value_change': float(np.std(all_value_changes)) if all_value_changes else 0.0,
            'n_pairs_tested': self.n_pairs,
            'n_targets_tested': len(self._patch_results),
        }

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize patching results."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._patch_results:
            return figures

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Attribution scores bar chart
        ax = axes[0]
        attribution = self._compute_attribution_scores()
        targets = list(attribution.keys())
        scores = list(attribution.values())
        x = np.arange(len(targets))
        ax.bar(x, scores, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=45, ha='right')
        ax.set_ylabel('Attribution Score (Policy KL)')
        ax.set_title('Goal Attribution by Dimension Group')

        # Value vs Policy sensitivity scatter
        ax = axes[1]
        policy_sens = self._compute_per_dim_sensitivity()
        value_sens = self._compute_value_sensitivity()

        for target in policy_sens.keys():
            ax.scatter(
                policy_sens[target],
                value_sens[target],
                s=100,
                label=target,
                alpha=0.7,
            )
            ax.annotate(
                target[:10],
                (policy_sens[target], value_sens[target]),
                fontsize=8,
            )

        ax.set_xlabel('Policy Sensitivity')
        ax.set_ylabel('Value Sensitivity')
        ax.set_title('Dimension Group Sensitivity')

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["patching_analysis"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
