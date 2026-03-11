"""
C2: Antagonist Audit.

Test whether antagonist is a general agent or degenerate specialist.
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
    get_pro_ant_returns,
    get_action_distribution,
)


@dataclass
class AuditResult:
    """Result of antagonist audit on a level type."""
    level_type: str
    n_levels: int
    antagonist_mean_return: float
    antagonist_std_return: float
    protagonist_mean_return: float
    protagonist_std_return: float
    gap: float


class AntagonistAuditExperiment(CheckpointExperiment):
    """
    Test whether antagonist is a general agent or degenerate specialist.

    Protocol:
    1. Evaluate antagonist on training distribution levels
    2. Evaluate antagonist on held-out level types
    3. Compare to protagonist on same levels
    4. Measure strategy diversity and degeneration
    """

    @property
    def name(self) -> str:
        return "antagonist_audit"

    LEVEL_TYPES = {
        'training_distribution': {},
        'low_wall_density': {'wall_density': (0.0, 0.1)},
        'high_wall_density': {'wall_density': (0.35, 0.5)},
        'small_goal_distance': {'goal_distance': (0.0, 3.0)},
        'large_goal_distance': {'goal_distance': (8.0, 15.0)},
        'extreme_combination': {'wall_density': (0.3, 0.5), 'goal_distance': (10.0, 15.0)},
    }

    def __init__(
        self,
        n_levels_per_type: int = 100,
        hidden_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_levels_per_type = n_levels_per_type
        self.hidden_dim = hidden_dim
        self._audit_results: Dict[str, AuditResult] = {}
        self._strategy_data: List[Dict[str, Any]] = []
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"AntagonistAuditExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> Dict[str, AuditResult]:
        """Collect audit data for all level types using real rollouts."""
        for level_type, constraints in self.LEVEL_TYPES.items():
            rng, type_rng = jax.random.split(rng)
            self._audit_results[level_type] = self._audit_level_type(
                type_rng, level_type, constraints
            )

        return self._audit_results

    def _audit_level_type(
        self,
        rng: chex.PRNGKey,
        level_type: str,
        constraints: Dict[str, tuple],
    ) -> AuditResult:
        """Audit antagonist on a specific level type using real rollouts."""
        rng, gen_rng, eval_rng, entropy_rng = jax.random.split(rng, 4)

        # Generate levels (constrained or unconstrained)
        if constraints:
            levels = generate_constrained_levels(
                self.agent, gen_rng, self.n_levels_per_type, constraints
            )
        else:
            levels = generate_levels(self.agent, gen_rng, self.n_levels_per_type)

        # Extract features for all levels
        batch_features = extract_level_features_batch(levels)

        # Get real protagonist and antagonist returns via rollouts
        pro_returns_arr, ant_returns_arr, _ = get_pro_ant_returns(
            eval_rng, levels, self
        )

        # Get real action entropies for antagonist
        ant_ts = getattr(self.train_state, 'ant_train_state', self.train_state)
        _, ant_entropies = get_action_distribution(
            ant_ts, self.agent, levels, entropy_rng
        )
        # Mean entropy per episode (across time steps)
        mean_entropies_per_level = ant_entropies.mean(axis=1)

        # Build per-level strategy data
        for i in range(self.n_levels_per_type):
            level_features = {
                'wall_density': float(batch_features['wall_density'][i]),
                'goal_distance': float(batch_features['goal_distance'][i]),
            }
            self._strategy_data.append({
                'level_type': level_type,
                'ant_return': float(ant_returns_arr[i]),
                'pro_return': float(pro_returns_arr[i]),
                'level_features': level_features,
                'ant_action_entropy': float(mean_entropies_per_level[i]),
            })

        return AuditResult(
            level_type=level_type,
            n_levels=self.n_levels_per_type,
            antagonist_mean_return=float(np.mean(ant_returns_arr)),
            antagonist_std_return=float(np.std(ant_returns_arr)),
            protagonist_mean_return=float(np.mean(pro_returns_arr)),
            protagonist_std_return=float(np.std(pro_returns_arr)),
            gap=float(np.mean(ant_returns_arr) - np.mean(pro_returns_arr)),
        )

    def analyze(self) -> Dict[str, Any]:
        """Analyze antagonist audit results."""
        if not self._audit_results:
            raise ValueError("Must call collect_data first")

        results = {}

        # Generalization metrics
        results['antagonist_generalisation'] = self._compute_generalisation()

        # Gap analysis
        results['antagonist_vs_protagonist_gap'] = self._compute_gap_analysis()

        # Strategy diversity
        results['antagonist_strategy_entropy'] = self._compute_strategy_diversity()

        # Degeneration score
        results['antagonist_degeneration_score'] = self._measure_degeneration()

        # Per-level-type results
        results['per_level_type'] = {
            lt: {
                'ant_mean': r.antagonist_mean_return,
                'ant_std': r.antagonist_std_return,
                'pro_mean': r.protagonist_mean_return,
                'pro_std': r.protagonist_std_return,
                'gap': r.gap,
            }
            for lt, r in self._audit_results.items()
        }

        return results

    def _compute_generalisation(self) -> Dict[str, float]:
        """Compute generalization metrics."""
        training_result = self._audit_results.get('training_distribution')
        if not training_result:
            return {}

        training_return = training_result.antagonist_mean_return

        generalisation_gaps = {}
        for level_type, result in self._audit_results.items():
            if level_type != 'training_distribution':
                gap = training_return - result.antagonist_mean_return
                generalisation_gaps[level_type] = float(gap)

        return {
            'training_performance': float(training_return),
            'generalisation_gaps': generalisation_gaps,
            'mean_generalisation_gap': float(np.mean(list(generalisation_gaps.values()))),
        }

    def _compute_gap_analysis(self) -> Dict[str, float]:
        """Analyze gap between antagonist and protagonist."""
        gaps = {lt: r.gap for lt, r in self._audit_results.items()}

        return {
            'per_level_type': gaps,
            'mean_gap': float(np.mean(list(gaps.values()))),
            'gap_variance': float(np.var(list(gaps.values()))),
            'gap_on_training': float(gaps.get('training_distribution', 0.0)),
        }

    def _compute_strategy_diversity(self) -> float:
        """Compute antagonist strategy diversity."""
        if not self._strategy_data:
            return 0.0

        entropies = [d['ant_action_entropy'] for d in self._strategy_data]
        # Higher mean entropy = more diverse strategies
        return float(np.mean(entropies))

    def _measure_degeneration(self) -> float:
        """Measure antagonist degeneration (specialization to training distribution)."""
        training_result = self._audit_results.get('training_distribution')
        if not training_result:
            return 0.0

        # Degeneration = how much worse on held-out vs training
        held_out_returns = []
        for level_type, result in self._audit_results.items():
            if level_type != 'training_distribution':
                held_out_returns.append(result.antagonist_mean_return)

        if not held_out_returns:
            return 0.0

        training_return = training_result.antagonist_mean_return
        held_out_mean = np.mean(held_out_returns)

        # Normalize: 0 = no degeneration, 1 = complete degeneration
        degeneration = (training_return - held_out_mean) / (training_return + 1e-10)
        return float(max(0.0, min(1.0, degeneration)))

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize audit results."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._audit_results:
            return figures

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Performance comparison
        ax = axes[0]
        level_types = list(self._audit_results.keys())
        x = np.arange(len(level_types))
        width = 0.35

        ant_means = [self._audit_results[lt].antagonist_mean_return for lt in level_types]
        pro_means = [self._audit_results[lt].protagonist_mean_return for lt in level_types]
        ant_stds = [self._audit_results[lt].antagonist_std_return for lt in level_types]
        pro_stds = [self._audit_results[lt].protagonist_std_return for lt in level_types]

        ax.bar(x - width/2, ant_means, width, yerr=ant_stds, label='Antagonist',
               capsize=3, alpha=0.8)
        ax.bar(x + width/2, pro_means, width, yerr=pro_stds, label='Protagonist',
               capsize=3, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(level_types, rotation=45, ha='right')
        ax.set_ylabel('Mean Return')
        ax.set_title('Antagonist vs Protagonist by Level Type')
        ax.legend()

        # Gap visualization
        ax = axes[1]
        gaps = [self._audit_results[lt].gap for lt in level_types]
        colors = ['g' if g > 0 else 'r' for g in gaps]
        ax.bar(x, gaps, color=colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(level_types, rotation=45, ha='right')
        ax.set_ylabel('Gap (Ant - Pro)')
        ax.set_title('Antagonist Advantage by Level Type')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["antagonist_audit"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
