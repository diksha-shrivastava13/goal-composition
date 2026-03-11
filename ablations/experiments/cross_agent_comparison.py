"""
Cross-Agent Prediction Loss Comparison.

Core ablation experiment: Compare prediction/probe loss learning curves
across all agent architectures to test whether curriculum awareness
can emerge without explicit information access.

Expected ordering (if emergent awareness is possible):
- next_env_prediction: Upper bound (explicit access)
- persistent_lstm: Test case (can memory enable emergence?)
- context_vector: Test case (can compressed context help?)
- episodic_memory: Test case (can episodic retrieval help?)
- accel_probe: Lower bound (only weight encoding)

This experiment:
1. Loads checkpoints at matched training steps
2. Computes prediction/probe loss on standardized test levels
3. Tracks learning curves across training
4. Performs statistical comparison between agents
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
import chex

from .base import CheckpointExperiment
from .utils.agent_aware_loss import (
    compute_agent_prediction_loss,
    detect_agent_type,
    compute_random_baseline_loss,
    compute_information_gain,
    compute_normalized_loss,
)


@dataclass
class ComparisonData:
    """Container for cross-agent comparison data."""
    # Per-level losses
    losses: np.ndarray = field(default_factory=lambda: np.array([]))

    # Level metadata
    level_features: Dict[str, np.ndarray] = field(default_factory=dict)

    # Agent info
    agent_type: str = ""
    training_step: int = 0

    # Loss components (for detailed analysis)
    wall_losses: np.ndarray = field(default_factory=lambda: np.array([]))
    goal_losses: np.ndarray = field(default_factory=lambda: np.array([]))
    agent_pos_losses: np.ndarray = field(default_factory=lambda: np.array([]))
    agent_dir_losses: np.ndarray = field(default_factory=lambda: np.array([]))


class CrossAgentComparisonExperiment(CheckpointExperiment):
    """
    Compare prediction loss across agent types at matched training steps.

    This is the core ablation experiment for testing whether curriculum
    awareness can emerge without explicit information access.

    Key outputs:
    - Learning curves: loss vs training step per agent
    - Convergence analysis: final loss comparison
    - Statistical tests: significance of differences between agents
    - Information gain: improvement over random baseline
    """

    @property
    def name(self) -> str:
        return "cross_agent_comparison"

    def __init__(
        self,
        n_test_levels: int = 200,
        wall_density_range: Tuple[float, float] = (0.05, 0.35),
        seed: int = 42,
        **kwargs,
    ):
        """
        Initialize cross-agent comparison experiment.

        Args:
            n_test_levels: Number of test levels for evaluation
            wall_density_range: Range of wall densities to sample
            seed: Random seed for reproducible test levels
        """
        super().__init__(**kwargs)
        self.n_test_levels = n_test_levels
        self.wall_density_range = wall_density_range
        self.seed = seed

        self._data: Optional[ComparisonData] = None
        self._results: Dict[str, Any] = {}

    def collect_data(self, rng: chex.PRNGKey) -> ComparisonData:
        """
        Collect prediction loss on standardized level set.

        Uses the same test levels across all agents for fair comparison.
        """
        self._data = ComparisonData()

        # Get agent info
        self._data.agent_type = detect_agent_type(self.agent)
        if hasattr(self.train_state, 'training_step'):
            self._data.training_step = int(self.train_state.training_step)
        elif hasattr(self.train_state, 'step'):
            self._data.training_step = int(self.train_state.step)
        else:
            self._data.training_step = 0

        # Generate fixed set of test levels (reproducible with seed)
        test_levels = self._generate_test_levels(
            jax.random.PRNGKey(self.seed),
            n_levels=self.n_test_levels
        )

        # Compute loss on each level
        losses = []
        wall_losses = []
        goal_losses = []
        agent_pos_losses = []
        agent_dir_losses = []
        level_features = {
            'wall_density': [],
            'goal_distance': [],
            'goal_x': [],
            'goal_y': [],
        }

        for i, level in enumerate(test_levels):
            rng, eval_rng = jax.random.split(rng)

            loss, metrics = compute_agent_prediction_loss(
                self.agent,
                self.train_state,
                level,
                eval_rng,
            )

            losses.append(loss)

            # Store component losses if available
            wall_losses.append(metrics.get('wall_loss', loss / 4))
            goal_losses.append(metrics.get('goal_loss', loss / 4))
            agent_pos_losses.append(metrics.get('agent_pos_loss', loss / 4))
            agent_dir_losses.append(metrics.get('agent_dir_loss', loss / 4))

            # Store level features
            level_features['wall_density'].append(level['wall_density'])
            goal_pos = level['goal_pos']
            agent_pos = level['agent_pos']
            goal_dist = abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])
            level_features['goal_distance'].append(goal_dist)
            level_features['goal_x'].append(goal_pos[1])
            level_features['goal_y'].append(goal_pos[0])

        self._data.losses = np.array(losses)
        self._data.wall_losses = np.array(wall_losses)
        self._data.goal_losses = np.array(goal_losses)
        self._data.agent_pos_losses = np.array(agent_pos_losses)
        self._data.agent_dir_losses = np.array(agent_dir_losses)
        self._data.level_features = {k: np.array(v) for k, v in level_features.items()}

        return self._data

    def _generate_test_levels(
        self,
        rng: chex.PRNGKey,
        n_levels: int,
    ) -> List[Dict[str, Any]]:
        """
        Generate fixed set of test levels for standardized evaluation.

        Uses deterministic generation to ensure same levels across agents.
        """
        levels = []
        height, width = 13, 13

        for i in range(n_levels):
            rng, level_rng = jax.random.split(rng)

            # Stratified wall density sampling
            density_min, density_max = self.wall_density_range
            wall_prob = density_min + (i / n_levels) * (density_max - density_min)

            # Generate walls
            rng_walls, rng_goal, rng_agent, rng_dir = jax.random.split(level_rng, 4)
            wall_map = np.array(jax.random.bernoulli(rng_walls, wall_prob, (height, width)))
            wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

            # Random goal and agent positions
            goal_pos = (
                int(jax.random.randint(rng_goal, (), 1, height - 1)),
                int(jax.random.randint(rng_goal, (), 1, width - 1)),
            )
            agent_pos = (
                int(jax.random.randint(rng_agent, (), 1, height - 1)),
                int(jax.random.randint(rng_agent, (), 1, width - 1)),
            )
            agent_dir = int(jax.random.randint(rng_dir, (), 0, 4))

            levels.append({
                'wall_map': wall_map,
                'wall_density': wall_map.sum() / (height * width),
                'goal_pos': goal_pos,
                'agent_pos': agent_pos,
                'agent_dir': agent_dir,
            })

        return levels

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze prediction loss patterns.

        Computes:
        - Summary statistics (mean, std, median)
        - Loss by difficulty (wall density bins)
        - Comparison to random baseline
        - Component-wise analysis
        """
        if self._data is None:
            raise ValueError("Must call collect_data before analyze")

        results = {}

        # Basic statistics
        results['summary'] = {
            'mean_loss': float(np.mean(self._data.losses)),
            'std_loss': float(np.std(self._data.losses)),
            'median_loss': float(np.median(self._data.losses)),
            'min_loss': float(np.min(self._data.losses)),
            'max_loss': float(np.max(self._data.losses)),
            'agent_type': self._data.agent_type,
            'training_step': self._data.training_step,
            'n_levels': len(self._data.losses),
        }

        # Per-level losses for cross-agent statistical tests
        results['per_level_losses'] = self._data.losses.tolist()

        # Random baseline comparison
        random_baseline = compute_random_baseline_loss()
        mean_loss = float(np.mean(self._data.losses))
        results['baseline_comparison'] = {
            'random_baseline': random_baseline,
            'information_gain': compute_information_gain(mean_loss, random_baseline),
            'normalized_accuracy': compute_normalized_loss(mean_loss, random_baseline),
            'relative_improvement': (random_baseline - mean_loss) / random_baseline,
        }

        # Loss by difficulty (wall density terciles)
        results['by_difficulty'] = self._analyze_by_difficulty()

        # Component analysis
        results['components'] = self._analyze_components()

        # Loss distribution
        results['distribution'] = {
            'percentiles': {
                '10': float(np.percentile(self._data.losses, 10)),
                '25': float(np.percentile(self._data.losses, 25)),
                '50': float(np.percentile(self._data.losses, 50)),
                '75': float(np.percentile(self._data.losses, 75)),
                '90': float(np.percentile(self._data.losses, 90)),
            },
        }

        self._results = results
        return results

    def _analyze_by_difficulty(self) -> Dict[str, Any]:
        """Analyze loss by level difficulty (wall density)."""
        wall_densities = self._data.level_features['wall_density']
        losses = self._data.losses

        # Tercile boundaries
        terciles = np.percentile(wall_densities, [33, 66])

        easy_mask = wall_densities < terciles[0]
        medium_mask = (wall_densities >= terciles[0]) & (wall_densities < terciles[1])
        hard_mask = wall_densities >= terciles[1]

        results = {}

        if easy_mask.sum() > 0:
            results['easy'] = {
                'mean_loss': float(np.mean(losses[easy_mask])),
                'std_loss': float(np.std(losses[easy_mask])),
                'n_samples': int(easy_mask.sum()),
                'wall_density_range': f"< {terciles[0]:.3f}",
            }

        if medium_mask.sum() > 0:
            results['medium'] = {
                'mean_loss': float(np.mean(losses[medium_mask])),
                'std_loss': float(np.std(losses[medium_mask])),
                'n_samples': int(medium_mask.sum()),
                'wall_density_range': f"{terciles[0]:.3f} - {terciles[1]:.3f}",
            }

        if hard_mask.sum() > 0:
            results['hard'] = {
                'mean_loss': float(np.mean(losses[hard_mask])),
                'std_loss': float(np.std(losses[hard_mask])),
                'n_samples': int(hard_mask.sum()),
                'wall_density_range': f"> {terciles[1]:.3f}",
            }

        # Correlation with difficulty
        if len(wall_densities) > 10:
            corr = np.corrcoef(wall_densities, losses)[0, 1]
            results['difficulty_correlation'] = float(corr) if np.isfinite(corr) else 0.0

        return results

    def _analyze_components(self) -> Dict[str, Any]:
        """Analyze individual loss components."""
        return {
            'wall_loss': {
                'mean': float(np.mean(self._data.wall_losses)),
                'std': float(np.std(self._data.wall_losses)),
            },
            'goal_loss': {
                'mean': float(np.mean(self._data.goal_losses)),
                'std': float(np.std(self._data.goal_losses)),
            },
            'agent_pos_loss': {
                'mean': float(np.mean(self._data.agent_pos_losses)),
                'std': float(np.std(self._data.agent_pos_losses)),
            },
            'agent_dir_loss': {
                'mean': float(np.mean(self._data.agent_dir_losses)),
                'std': float(np.std(self._data.agent_dir_losses)),
            },
        }

    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data for cross-agent comparison."""
        if not self._results:
            raise ValueError("Must call analyze before visualize")

        viz_data = {
            'summary': self._results.get('summary', {}),
            'baseline_comparison': self._results.get('baseline_comparison', {}),
            'by_difficulty': self._results.get('by_difficulty', {}),
            'components': self._results.get('components', {}),
            'distribution': self._results.get('distribution', {}),

            # Raw data for plotting
            'losses': self._data.losses.tolist(),
            'wall_densities': self._data.level_features['wall_density'].tolist(),
        }

        return viz_data

    @staticmethod
    def compare_agents(
        results_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare results across multiple agents.

        This is a static method that takes results from multiple runs
        and performs cross-agent comparison with statistical significance tests.

        Args:
            results_list: List of results dicts from different agents

        Returns:
            Comparative analysis results
        """
        from scipy import stats

        if len(results_list) < 2:
            return {'error': 'Need at least 2 agents to compare'}

        comparison = {
            'agents': [],
            'rankings': {},
            'statistical_tests': {},
        }

        # Collect summary stats from each agent
        per_level_losses = []  # For pairwise tests
        for i, result in enumerate(results_list):
            summary = result.get('summary', {})
            comparison['agents'].append({
                'index': i,
                'agent_type': summary.get('agent_type', f'agent_{i}'),
                'training_step': summary.get('training_step', 0),
                'mean_loss': summary.get('mean_loss', float('inf')),
                'std_loss': summary.get('std_loss', 0),
            })
            # Collect per-level losses for statistical tests
            level_losses = result.get('per_level_losses', [])
            per_level_losses.append(np.array(level_losses) if level_losses else None)

        # Rank by mean loss (lower is better)
        sorted_agents = sorted(comparison['agents'], key=lambda x: x['mean_loss'])
        comparison['rankings']['by_mean_loss'] = [
            a['agent_type'] for a in sorted_agents
        ]

        # Statistical significance tests
        stat_tests = {}

        # Pairwise t-tests between agents (if per-level data available)
        has_per_level = all(p is not None and len(p) > 1 for p in per_level_losses)
        if has_per_level:
            pairwise = {}
            for i in range(len(per_level_losses)):
                for j in range(i + 1, len(per_level_losses)):
                    agent_i = comparison['agents'][i]['agent_type']
                    agent_j = comparison['agents'][j]['agent_type']
                    t_stat, p_value = stats.ttest_ind(per_level_losses[i], per_level_losses[j])
                    pairwise[f'{agent_i}_vs_{agent_j}'] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant_at_005': bool(p_value < 0.05),
                        'significant_at_001': bool(p_value < 0.01),
                        'effect_size_cohens_d': float(
                            (np.mean(per_level_losses[i]) - np.mean(per_level_losses[j]))
                            / np.sqrt((np.var(per_level_losses[i]) + np.var(per_level_losses[j])) / 2 + 1e-10)
                        ),
                    }
            stat_tests['pairwise_t_tests'] = pairwise

            # One-way ANOVA across all agents
            f_stat, anova_p = stats.f_oneway(*per_level_losses)
            stat_tests['anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(anova_p),
                'significant_at_005': bool(anova_p < 0.05),
                'n_groups': len(per_level_losses),
            }
        else:
            stat_tests['note'] = 'Per-level loss data not available; only ranking by mean loss.'

        comparison['statistical_tests'] = stat_tests

        # Add expected ordering note
        expected_ordering = [
            'next_env_prediction',  # Upper bound
            'episodic_memory',
            'context_vector',
            'persistent_lstm',
            'accel_probe',  # Lower bound
        ]
        comparison['expected_ordering'] = expected_ordering
        comparison['note'] = (
            "Expected: next_env_prediction (best) > persistent/context/episodic > accel_probe (worst). "
            "If probe agents match next_env_prediction, curriculum awareness can emerge without explicit access."
        )

        return comparison

    @staticmethod
    def aggregate_learning_curves(
        results_by_step: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Aggregate results across training steps to create learning curves.

        Args:
            results_by_step: Dict mapping training_step to results

        Returns:
            Learning curve data
        """
        steps = sorted(results_by_step.keys())
        mean_losses = []
        std_losses = []
        info_gains = []

        for step in steps:
            result = results_by_step[step]
            summary = result.get('summary', {})
            baseline = result.get('baseline_comparison', {})

            mean_losses.append(summary.get('mean_loss', float('inf')))
            std_losses.append(summary.get('std_loss', 0))
            info_gains.append(baseline.get('information_gain', 0))

        return {
            'training_steps': steps,
            'mean_losses': mean_losses,
            'std_losses': std_losses,
            'information_gains': info_gains,
            'agent_type': results_by_step[steps[0]].get('summary', {}).get('agent_type', 'unknown'),
        }
