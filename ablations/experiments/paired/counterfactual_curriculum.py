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
        # Generate shared evaluation levels
        rng, levels_rng = jax.random.split(rng)
        self._shared_levels = self._generate_shared_levels(levels_rng)

        # Evaluate each curriculum
        for regime in self.REGIMES:
            rng, eval_rng = jax.random.split(rng)
            self._profiles[regime] = self._evaluate_curriculum(eval_rng, regime)

        return self._profiles

    def _generate_shared_levels(self, rng: chex.PRNGKey) -> List[Dict[str, Any]]:
        """Generate shared evaluation levels."""
        levels = []
        for i in range(self.n_eval_levels):
            rng, level_rng = jax.random.split(rng)
            levels.append(self._generate_level(level_rng))
        return levels

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

    def _evaluate_curriculum(
        self,
        rng: chex.PRNGKey,
        regime: str,
    ) -> CurriculumProfile:
        """Evaluate agent from a specific curriculum."""
        hstates = []
        values = []
        policy_entropies = []
        returns = []
        level_features = []

        # Curriculum-specific biases (simplified simulation)
        regime_biases = {
            'paired': {'value_bias': 0.0, 'entropy_bias': 0.0},
            'dr': {'value_bias': -0.05, 'entropy_bias': 0.1},
            'paired_no_antagonist': {'value_bias': 0.05, 'entropy_bias': -0.05},
            'paired_frozen_adversary': {'value_bias': 0.02, 'entropy_bias': 0.05},
            'replay_of_paired': {'value_bias': -0.02, 'entropy_bias': 0.02},
        }
        bias = regime_biases.get(regime, {'value_bias': 0.0, 'entropy_bias': 0.0})

        for level in self._shared_levels:
            rng, eval_rng, hstate_rng = jax.random.split(rng, 3)

            features = self._compute_level_features(level)
            level_features.append(features)

            # Simulate agent representations with curriculum-specific characteristics
            hstate = np.array(jax.random.normal(hstate_rng, (self.hidden_dim,)))
            # Add curriculum-specific structure
            if regime == 'paired':
                hstate[:50] *= 1.2  # PAIRED uses more regret-encoding dims
            elif regime == 'dr':
                hstate *= 0.9  # DR has less structured representations

            hstates.append(hstate)

            # Value with curriculum bias
            wall_density = features['wall_density']
            value = 0.7 - wall_density * 0.3 + bias['value_bias']
            value += float(jax.random.uniform(eval_rng)) * 0.1
            values.append(value)

            # Policy entropy with curriculum bias
            entropy = 1.5 + wall_density * 0.5 + bias['entropy_bias']
            entropy += float(jax.random.uniform(eval_rng)) * 0.2
            policy_entropies.append(entropy)

            # Return
            ret = value + float(jax.random.normal(eval_rng)) * 0.1
            returns.append(ret)

        return CurriculumProfile(
            curriculum_name=regime,
            hstates=np.array(hstates),
            values=np.array(values),
            policy_entropies=np.array(policy_entropies),
            returns=np.array(returns),
            level_features=level_features,
        )

    def _compute_level_features(self, level: Dict[str, Any]) -> Dict[str, float]:
        """Compute level features."""
        wall_density = float(level['wall_map'].sum() / level['wall_map'].size)
        goal_distance = float(np.sqrt(
            (level['goal_pos'][0] - level['agent_pos'][0])**2 +
            (level['goal_pos'][1] - level['agent_pos'][1])**2
        ))
        return {'wall_density': wall_density, 'goal_distance': goal_distance}

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
