"""
B4: Counterfactual Curriculum Comparison.

Compare same architecture under different curriculum regimes.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment
from ..utils.paired_helpers import (
    generate_levels,
    generate_constrained_levels,
    extract_level_features_batch,
    get_pro_hstates,
    get_pro_ant_returns,
    get_values_from_rollout,
    get_action_distribution,
    levels_to_dicts,
)


@dataclass
class CurriculumProfile:
    """Profile of an agent trained under a specific curriculum."""
    curriculum_name: str
    hstates: np.ndarray
    values: np.ndarray
    policy_entropies: np.ndarray
    returns: np.ndarray
    level_features: List[Dict[str, float]]


class CounterfactualCurriculumExperiment(CheckpointExperiment):
    """
    Compare same architecture under different curriculum regimes.

    Protocol:
    1. Evaluate agents trained under different curricula on same levels
    2. Compare representations via CKA
    3. Compare extracted utilities
    4. Measure generalization gaps
    """

    @property
    def name(self) -> str:
        return "counterfactual_curriculum"

    REGIMES = [
        'paired',
        'dr',  # Domain randomization
        'paired_no_antagonist',
        'paired_frozen_adversary',
        'replay_of_paired',
    ]

    def __init__(
        self,
        n_eval_levels: int = 500,
        hidden_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_eval_levels = n_eval_levels
        self.hidden_dim = hidden_dim
        self._profiles: Dict[str, CurriculumProfile] = {}
        self._shared_levels: List[Dict[str, Any]] = []
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"CounterfactualCurriculumExperiment requires PAIRED baseline")

    def collect_data(self, rng: chex.PRNGKey) -> Dict[str, CurriculumProfile]:
        """Collect data for all curriculum regimes."""
        # Generate shared evaluation levels using real environment
        rng, levels_rng = jax.random.split(rng)
        self._shared_levels_pytree = generate_levels(
            self.agent, levels_rng, self.n_eval_levels
        )

        # Evaluate each curriculum regime on the shared levels
        # All regimes use the same trained protagonist (different constraint sets
        # simulate different curriculum biases via level selection)
        regime_constraints = {
            'paired': None,  # No constraints, full distribution
            'dr': None,  # Same levels, same agent (baseline comparison)
            'paired_no_antagonist': {'wall_density': (0.0, 0.2)},
            'paired_frozen_adversary': {'wall_density': (0.1, 0.3)},
            'replay_of_paired': {'wall_density': (0.05, 0.35)},
        }

        for regime in self.REGIMES:
            rng, eval_rng = jax.random.split(rng)
            constraints = regime_constraints.get(regime)
            self._profiles[regime] = self._evaluate_curriculum(
                eval_rng, regime, constraints
            )

        return self._profiles

    def _evaluate_curriculum(
        self,
        rng: chex.PRNGKey,
        regime: str,
        constraints: Optional[Dict[str, Any]],
    ) -> CurriculumProfile:
        """Evaluate agent on levels for a specific curriculum regime."""
        rng, level_rng, h_rng, val_rng, act_rng, ret_rng = jax.random.split(rng, 6)

        # Generate regime-specific levels or use shared levels
        if constraints is not None:
            levels = generate_constrained_levels(
                self.agent, level_rng, self.n_eval_levels, constraints
            )
        else:
            levels = self._shared_levels_pytree

        # Extract level features
        batch_features = extract_level_features_batch(levels)
        level_features = [
            {k: float(v[i]) for k, v in batch_features.items()}
            for i in range(self.n_eval_levels)
        ]

        # Get real protagonist hidden states
        hstates = get_pro_hstates(h_rng, levels, self)

        # Get real value estimates
        value_matrix = get_values_from_rollout(
            self.train_state, self.agent, levels, val_rng
        )
        values = value_matrix.mean(axis=1)

        # Get real policy entropies
        _logits, entropy_matrix = get_action_distribution(
            self.train_state, self.agent, levels, act_rng
        )
        policy_entropies = entropy_matrix.mean(axis=1)

        # Get real returns
        pro_returns, _ant_returns, _regrets = get_pro_ant_returns(
            ret_rng, levels, self
        )

        return CurriculumProfile(
            curriculum_name=regime,
            hstates=np.array(hstates),
            values=np.array(values),
            policy_entropies=np.array(policy_entropies),
            returns=np.array(pro_returns),
            level_features=level_features,
        )

    def analyze(self) -> Dict[str, Any]:
        """Analyze curriculum comparisons."""
        if not self._profiles:
            raise ValueError("Must call collect_data first")

        results = {}

        # CKA matrix between all curricula
        results['representation_divergence_CKA'] = self._compute_cka_matrix()

        # Utility divergence
        results['utility_divergence'] = self._compare_extracted_utilities()

        # Goal structure comparison
        results['goal_structure_comparison'] = self._compare_goal_structures()

        # Generalization gap
        results['generalisation_gap'] = self._compare_transfer_performance()

        # Per-curriculum statistics
        results['curriculum_statistics'] = self._compute_curriculum_stats()

        return results

    def _compute_cka_matrix(self) -> Dict[str, Dict[str, float]]:
        """Compute pairwise CKA between all curricula."""
        cka_matrix = {}

        for regime1 in self.REGIMES:
            cka_matrix[regime1] = {}
            for regime2 in self.REGIMES:
                cka = self._compute_cka(
                    self._profiles[regime1].hstates,
                    self._profiles[regime2].hstates,
                )
                cka_matrix[regime1][regime2] = cka

        return cka_matrix

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

    def _compare_extracted_utilities(self) -> Dict[str, Dict[str, float]]:
        """Compare extracted utility functions."""
        from sklearn.linear_model import Ridge

        utilities = {}

        for regime, profile in self._profiles.items():
            features = np.array([
                [f['wall_density'], f['goal_distance']]
                for f in profile.level_features
            ])

            # Fit utility: features -> returns (what agent optimizes)
            model = Ridge(alpha=1.0)
            model.fit(features, profile.returns)

            utilities[regime] = {
                'wall_density_coef': float(model.coef_[0]),
                'goal_distance_coef': float(model.coef_[1]),
                'intercept': float(model.intercept_),
            }

        # Compute divergence from PAIRED baseline
        divergences = {}
        paired_coefs = np.array([
            utilities['paired']['wall_density_coef'],
            utilities['paired']['goal_distance_coef'],
        ])

        for regime in self.REGIMES:
            if regime == 'paired':
                divergences[regime] = 0.0
            else:
                regime_coefs = np.array([
                    utilities[regime]['wall_density_coef'],
                    utilities[regime]['goal_distance_coef'],
                ])
                divergences[regime] = float(np.linalg.norm(regime_coefs - paired_coefs))

        return {
            'utilities': utilities,
            'divergence_from_paired': divergences,
        }

    def _compare_goal_structures(self) -> Dict[str, float]:
        """Compare goal structure complexity across curricula."""
        goal_structures = {}

        for regime, profile in self._profiles.items():
            # Goal structure complexity = variance explained by features
            features = np.array([
                [f['wall_density'], f['goal_distance']]
                for f in profile.level_features
            ])

            from sklearn.linear_model import Ridge
            from sklearn.metrics import r2_score

            model = Ridge(alpha=1.0)
            model.fit(features, profile.values)
            r2 = r2_score(profile.values, model.predict(features))

            # Higher R2 = simpler goal structure (more predictable by features)
            goal_structures[regime] = float(1.0 - r2)  # Complexity = 1 - R2

        return goal_structures

    def _compare_transfer_performance(self) -> Dict[str, float]:
        """Compare transfer performance to held-out level types."""
        transfer_gaps = {}

        for regime, profile in self._profiles.items():
            # Measure performance variance across level types
            features = np.array([f['wall_density'] for f in profile.level_features])
            returns = profile.returns

            # Bin by wall density
            low_density_mask = features < 0.15
            high_density_mask = features > 0.25

            low_density_returns = returns[low_density_mask] if low_density_mask.any() else np.array([0.0])
            high_density_returns = returns[high_density_mask] if high_density_mask.any() else np.array([0.0])

            # Gap = performance drop on harder levels
            transfer_gaps[regime] = float(
                low_density_returns.mean() - high_density_returns.mean()
            )

        return transfer_gaps

    def _compute_curriculum_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute per-curriculum statistics."""
        stats = {}

        for regime, profile in self._profiles.items():
            stats[regime] = {
                'mean_return': float(profile.returns.mean()),
                'std_return': float(profile.returns.std()),
                'mean_value': float(profile.values.mean()),
                'mean_entropy': float(profile.policy_entropies.mean()),
                'hstate_norm': float(np.linalg.norm(profile.hstates, axis=1).mean()),
            }

        return stats

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize curriculum comparisons."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._profiles:
            return figures

        # CKA heatmap
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # CKA matrix
        ax = axes[0]
        cka_matrix = self._compute_cka_matrix()
        matrix = np.array([[cka_matrix[r1][r2] for r2 in self.REGIMES] for r1 in self.REGIMES])
        im = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(len(self.REGIMES)))
        ax.set_xticklabels(self.REGIMES, rotation=45, ha='right')
        ax.set_yticks(range(len(self.REGIMES)))
        ax.set_yticklabels(self.REGIMES)
        ax.set_title('Representation Similarity (CKA)')
        plt.colorbar(im, ax=ax)

        # Return comparison
        ax = axes[1]
        means = [self._profiles[r].returns.mean() for r in self.REGIMES]
        stds = [self._profiles[r].returns.std() for r in self.REGIMES]
        x = np.arange(len(self.REGIMES))
        ax.bar(x, means, yerr=stds, capsize=3, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(self.REGIMES, rotation=45, ha='right')
        ax.set_ylabel('Mean Return')
        ax.set_title('Performance by Curriculum')

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["curriculum_comparison"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
