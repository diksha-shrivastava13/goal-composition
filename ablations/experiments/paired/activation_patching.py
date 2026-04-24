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
from ..utils.paired_helpers import (
    generate_levels,
    extract_level_features_batch,
    get_pro_hstates,
    levels_to_dicts,
)


class PatchTarget(Enum):
    """Target for activation patching."""
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

    def __init__(self, **kwargs):
        """
        Args:
            n_pairs: Number of source/target pairs for patching.
            hidden_dim: Initial hidden dimension estimate (updated from data).
            top_k_variance: Max number of top-variance dimensions to patch.
                Capped at hidden_dim // 4 to avoid patching too large a fraction.
        """
        super().__init__(**kwargs)
        self.n_pairs = self.exp_config("n_pairs")
        self.hidden_dim = self.exp_config("hidden_dim")
        self.top_k_variance = self.exp_config("top_k_variance")
        self.max_steps = self.exp_config("max_steps", 256)
        self._levels: List[Dict[str, Any]] = []
        self._hstates: np.ndarray = None
        self._patch_results: Dict[PatchTarget, List[PatchResult]] = {}
        self._regret_dims: Optional[np.ndarray] = None
        self._variance_dims: Optional[np.ndarray] = None
        self._n_pairs_skipped: Dict[PatchTarget, int] = {}
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"ActivationPatchingExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Collect levels and hidden states using real network evaluations."""
        n_total = self.n_pairs * 2
        rng, level_rng, h_rng = jax.random.split(rng, 3)

        # Generate real levels
        levels_pytree = generate_levels(self.agent, level_rng, n_total)

        # Convert to list of dicts for downstream compatibility
        self._levels = levels_to_dicts(levels_pytree, n_total)

        # Get real protagonist hidden states
        self._hstates = get_pro_hstates(h_rng, levels_pytree, self, self.max_steps)

        # Update hidden_dim from actual data
        self.hidden_dim = self._hstates.shape[1]

        # Identify important dimensions
        self._identify_important_dims()

        return self._levels, self._hstates

    def _compute_level_features(self, level: Dict[str, Any]) -> Dict[str, float]:
        """Compute level features."""
        wall_density = float(level['wall_map'].sum() / level['wall_map'].size)
        goal_pos = level['goal_pos']
        agent_pos = level['agent_pos']
        goal_distance = float(np.sqrt(
            (goal_pos[0] - agent_pos[0])**2 +
            (goal_pos[1] - agent_pos[1])**2
        ))
        return {'wall_density': wall_density, 'goal_distance': goal_distance}

    def _identify_important_dims(self):
        """Identify top variance and regret-encoding dimensions."""
        # Adaptive top_k: cap at hidden_dim // 4
        effective_top_k = min(self.top_k_variance, self.hidden_dim // 4)

        # Top variance dimensions
        variances = self._hstates.var(axis=0)
        self._variance_dims = np.argsort(variances)[-effective_top_k:]

        # Regret-encoding dimensions (simplified: dimensions correlated with wall density)
        features = np.array([
            self._compute_level_features(level)['wall_density']
            for level in self._levels
        ])
        correlations = np.array([
            abs(np.corrcoef(self._hstates[:, d], features)[0, 1])
            for d in range(self.hidden_dim)
        ])
        self._regret_dims = np.argsort(correlations)[-effective_top_k:]

    def _run_patches(self, rng: chex.PRNGKey):
        """Run all patch experiments."""
        patch_targets = [
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
        n_skipped = 0

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

            # Skip if forward pass failed (NaN sentinel from _evaluate_with_hstate)
            if (np.isnan(original_policy_entropy) or np.isnan(original_value)
                    or np.isnan(patched_policy_entropy) or np.isnan(patched_value)):
                n_skipped += 1
                continue

            # Entropy difference as KL proxy (exact KL needs full distributions)
            policy_kl = abs(patched_policy_entropy - original_policy_entropy)

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

        self._n_pairs_skipped[target] = n_skipped
        skip_rate = n_skipped / self.n_pairs if self.n_pairs > 0 else 0
        if skip_rate > 0.2:
            import logging
            logging.getLogger(__name__).warning(
                f"Activation patching {target.value}: {n_skipped}/{self.n_pairs} "
                f"pairs skipped due to NaN ({skip_rate:.0%})"
            )

        return results

    def _apply_patch(
        self,
        source_h: np.ndarray,
        target_h: np.ndarray,
        patch_target: PatchTarget,
    ) -> np.ndarray:
        """Apply patch from source to target."""
        patched = target_h.copy()

        if patch_target == PatchTarget.HIDDEN_STATE:
            patched[:] = source_h[:]
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
        """Evaluate policy entropy and value using real network forward pass."""
        try:
            # Generate a proper level from agent and reset to get correct obs format
            rng, level_rng, obs_rng = jax.random.split(rng, 3)
            level_obj = self.agent.sample_random_level(level_rng)

            # Override wall_map from the level dict
            if 'wall_map' in level:
                wall_map = jnp.array(level['wall_map'])
                level_obj = level_obj.replace(wall_map=wall_map)
            if 'goal_pos' in level:
                level_obj = level_obj.replace(goal_pos=jnp.array(level['goal_pos']))
            if 'agent_pos' in level:
                level_obj = level_obj.replace(agent_pos=jnp.array(level['agent_pos']))

            # Use the env to get proper observation format
            obs, _ = self.agent.env.reset_env_to_level(obs_rng, level_obj, self.agent.env_params)

            # Add batch and sequence dims: (H, W, C) -> (1, 1, H, W, C)
            obs_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs)
            done_batch = jnp.zeros((1, 1), dtype=bool)

            # Reshape flat hstate vector into LSTM carry format
            half = len(hstate) // 2
            h_c = jnp.array(hstate[:half]).reshape(1, -1)
            h_h = jnp.array(hstate[half:]).reshape(1, -1)
            hstate_tree = (h_c, h_h)

            outputs = self.train_state.apply_fn(
                self.train_state.params,
                (obs_batch, done_batch),
                hstate_tree,
            )

            if len(outputs) == 4:
                _, pi, value, _ = outputs
            else:
                _, pi, value = outputs

            entropy = float(pi.entropy()[0, 0]) if hasattr(pi, 'entropy') else float('nan')
            val = float(value[0, 0])
            return entropy, val

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Forward pass failed in _evaluate_with_hstate: {e}. Returning NaN."
            )
            return float('nan'), float('nan')

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

        # NaN skip statistics
        results['nan_skip_stats'] = {
            target.value: {
                'n_pairs_skipped': n_skipped,
                'skip_rate': n_skipped / self.n_pairs if self.n_pairs > 0 else 0,
            }
            for target, n_skipped in self._n_pairs_skipped.items()
        }

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
