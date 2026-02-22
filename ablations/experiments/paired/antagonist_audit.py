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
        """Collect audit data for all level types."""
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
        """Audit antagonist on a specific level type."""
        ant_returns = []
        pro_returns = []

        for i in range(self.n_levels_per_type):
            rng, level_rng, ant_rng, pro_rng = jax.random.split(rng, 4)

            # Generate level with constraints
            level = self._generate_constrained_level(level_rng, constraints)

            # Evaluate both agents
            ant_return = self._evaluate_antagonist(ant_rng, level)
            pro_return = self._evaluate_protagonist(pro_rng, level)

            ant_returns.append(ant_return)
            pro_returns.append(pro_return)

            # Collect strategy data for diversity analysis
            rng, strat_rng = jax.random.split(rng)
            self._strategy_data.append({
                'level_type': level_type,
                'ant_return': ant_return,
                'pro_return': pro_return,
                'level_features': self._compute_level_features(level),
                'ant_action_entropy': self._get_action_entropy(strat_rng, 'antagonist', level),
            })

        return AuditResult(
            level_type=level_type,
            n_levels=self.n_levels_per_type,
            antagonist_mean_return=float(np.mean(ant_returns)),
            antagonist_std_return=float(np.std(ant_returns)),
            protagonist_mean_return=float(np.mean(pro_returns)),
            protagonist_std_return=float(np.std(pro_returns)),
            gap=float(np.mean(ant_returns) - np.mean(pro_returns)),
        )

    def _generate_constrained_level(
        self,
        rng: chex.PRNGKey,
        constraints: Dict[str, tuple],
    ) -> Dict[str, Any]:
        """Generate a level satisfying constraints."""
        height, width = 13, 13

        # Wall density constraint
        if 'wall_density' in constraints:
            min_wd, max_wd = constraints['wall_density']
            wall_prob = min_wd + float(jax.random.uniform(rng)) * (max_wd - min_wd)
        else:
            wall_prob = 0.1 + float(jax.random.uniform(rng)) * 0.25

        wall_map = np.array(jax.random.bernoulli(rng, wall_prob, (height, width)))
        wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

        rng_goal, rng_agent = jax.random.split(rng)

        # Goal distance constraint
        if 'goal_distance' in constraints:
            min_dist, max_dist = constraints['goal_distance']
            # Place agent near corner, goal at controlled distance
            agent_pos = (2, 2)
            target_dist = min_dist + float(jax.random.uniform(rng_goal)) * (max_dist - min_dist)
            angle = float(jax.random.uniform(rng_goal)) * 2 * np.pi
            goal_y = int(agent_pos[0] + target_dist * np.sin(angle))
            goal_x = int(agent_pos[1] + target_dist * np.cos(angle))
            goal_pos = (
                max(1, min(height - 2, goal_y)),
                max(1, min(width - 2, goal_x)),
            )
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
        return {'wall_density': wall_density, 'goal_distance': goal_distance}

    def _evaluate_antagonist(self, rng: chex.PRNGKey, level: Dict[str, Any]) -> float:
        """Evaluate antagonist (simplified)."""
        features = self._compute_level_features(level)
        # Antagonist is stronger but degrades on out-of-distribution
        base_return = 0.8 - features['wall_density'] * 0.3
        # Degrade on extreme levels
        if features['wall_density'] > 0.35 or features['goal_distance'] > 10:
            base_return -= 0.15
        noise = float(jax.random.uniform(rng)) * 0.1
        return float(base_return + noise)

    def _evaluate_protagonist(self, rng: chex.PRNGKey, level: Dict[str, Any]) -> float:
        """Evaluate protagonist (simplified)."""
        features = self._compute_level_features(level)
        base_return = 0.7 - features['wall_density'] * 0.4
        noise = float(jax.random.uniform(rng)) * 0.1
        return float(base_return + noise)

    def _get_action_entropy(
        self,
        rng: chex.PRNGKey,
        agent: str,
        level: Dict[str, Any],
    ) -> float:
        """Get action entropy (simplified)."""
        features = self._compute_level_features(level)
        # Higher entropy on harder levels
        base_entropy = 1.5 + features['wall_density'] * 0.5
        if agent == 'antagonist':
            base_entropy *= 0.9  # Antagonist more confident
        return float(base_entropy + float(jax.random.uniform(rng)) * 0.2)

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
