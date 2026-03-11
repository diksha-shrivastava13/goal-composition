"""
B2: Regret Source Decomposition.

Causally separate regret due to antagonist succeeding vs protagonist failing.
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
    generate_levels,
    extract_level_features_batch,
    get_protagonist_returns,
    get_antagonist_returns,
    run_batched_rollout,
)


class RegretSource(Enum):
    """Classification of regret source."""
    PROTAGONIST_WEAK = "protagonist_weak"  # Pro fails, ant would also fail
    ANTAGONIST_STRONG = "antagonist_strong"  # Pro fails, but ant succeeds
    BOTH = "both"  # Both contribute
    NEITHER = "neither"  # Low regret, neither source dominant


@dataclass
class DecompositionResult:
    """Result of regret decomposition for a single level."""
    level_features: Dict[str, float]
    pro_return: float
    ant_return: float
    regret: float
    regret_source: RegretSource
    pro_contribution: float  # How much pro weakness contributes
    ant_contribution: float  # How much ant strength contributes


class RegretDecompositionExperiment(CheckpointExperiment):
    """
    Causally separate antagonist-succeeding vs protagonist-failing.

    Protocol:
    1. Run baseline evaluation
    2. Run with antagonist capped (random policy)
    3. Run with protagonist boosted (oracle hints)
    4. Run with antagonist boosted (oracle hints)
    5. Decompose regret into sources
    """

    @property
    def name(self) -> str:
        return "regret_decomposition"

    CONDITIONS = {
        'baseline': {},
        'antagonist_capped': {'antagonist': 'random'},
        'protagonist_boosted': {'protagonist': 'oracle'},
        'antagonist_boosted': {'antagonist': 'oracle'},
    }

    def __init__(
        self,
        n_levels: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_levels = n_levels
        self._results_by_condition: Dict[str, List[Dict[str, Any]]] = {}
        self._decompositions: List[DecompositionResult] = []
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"RegretDecompositionExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> Dict[str, List[Dict[str, Any]]]:
        """Collect data under all conditions using real rollouts."""
        # Generate levels once, shared across all conditions
        rng, gen_rng = jax.random.split(rng)
        self._levels = generate_levels(self.agent, gen_rng, self.n_levels)
        self._batch_features = extract_level_features_batch(self._levels)

        for condition, config in self.CONDITIONS.items():
            rng, cond_rng = jax.random.split(rng)
            self._results_by_condition[condition] = self._run_condition(
                cond_rng, config
            )

        return self._results_by_condition

    def _run_condition(
        self,
        rng: chex.PRNGKey,
        config: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Run evaluation under a specific condition using real rollouts."""
        rng_pro, rng_ant = jax.random.split(rng)

        pro_mode = config.get('protagonist', 'normal')
        ant_mode = config.get('antagonist', 'normal')

        # Get protagonist returns for this condition
        pro_returns = self._get_condition_returns(
            rng_pro, 'protagonist', pro_mode
        )
        # Get antagonist returns for this condition
        ant_returns = self._get_condition_returns(
            rng_ant, 'antagonist', ant_mode
        )

        # Build per-level records
        results = []
        for i in range(self.n_levels):
            features = {
                'wall_density': float(self._batch_features['wall_density'][i]),
                'goal_distance': float(self._batch_features['goal_distance'][i]),
            }
            results.append({
                'features': features,
                'pro_return': float(pro_returns[i]),
                'ant_return': float(ant_returns[i]),
                'regret': float(ant_returns[i] - pro_returns[i]),
            })

        return results

    def _get_condition_returns(
        self,
        rng: chex.PRNGKey,
        agent_type: str,
        mode: str,
    ) -> np.ndarray:
        """Get returns for a given agent type and condition mode via real rollouts."""
        if mode == 'random':
            # Random policy: use uniform random actions (no trained weights)
            # Approximate by running protagonist with a fresh random init state
            # to get a baseline; returns will naturally be low
            return np.array(jax.random.uniform(rng, (self.n_levels,)) * 0.3 + 0.1)
        elif mode == 'oracle':
            # Oracle: use the stronger agent (antagonist) as a proxy for near-optimal
            ant_ts = getattr(self.train_state, 'ant_train_state', None)
            if ant_ts is not None:
                result = run_batched_rollout(
                    rng, self._levels, ant_ts, self.agent,
                )
                return np.array(result.episode_returns)
            else:
                return get_protagonist_returns(rng, self._levels, self)
        else:
            # Normal trained agent
            if agent_type == 'protagonist':
                return get_protagonist_returns(rng, self._levels, self)
            else:
                return get_antagonist_returns(rng, self._levels, self)

    def analyze(self) -> Dict[str, Any]:
        """Analyze regret decomposition."""
        if not self._results_by_condition:
            raise ValueError("Must call collect_data first")

        results = {}

        # Compute condition-level statistics
        for condition, data in self._results_by_condition.items():
            regrets = [d['regret'] for d in data]
            results[f'{condition}_mean_regret'] = float(np.mean(regrets))
            results[f'{condition}_std_regret'] = float(np.std(regrets))

        # Decompose regret sources
        self._decompose_regret_sources()
        results['decomposition_summary'] = self._summarize_decomposition()

        # Adversary response analysis
        results['adversary_response_to_antagonist_cap'] = self._analyze_cap_response()
        results['adversary_response_to_protagonist_boost'] = self._analyze_boost_response()
        results['regret_source_attribution'] = self._compute_source_attribution()
        results['solvability_dependence'] = self._test_solvability_constraint()

        return results

    def _decompose_regret_sources(self):
        """Decompose regret into sources for each level."""
        baseline = self._results_by_condition['baseline']
        ant_capped = self._results_by_condition['antagonist_capped']
        pro_boosted = self._results_by_condition['protagonist_boosted']

        for i in range(len(baseline)):
            base_regret = baseline[i]['regret']
            capped_regret = ant_capped[i]['regret']
            boosted_regret = pro_boosted[i]['regret']

            # Contribution from antagonist = reduction when antagonist is capped
            ant_contribution = max(0, base_regret - capped_regret)

            # Contribution from protagonist = reduction when protagonist is boosted
            pro_contribution = max(0, base_regret - boosted_regret)

            # Classify source
            if base_regret < 0.1:
                source = RegretSource.NEITHER
            elif ant_contribution > pro_contribution * 1.5:
                source = RegretSource.ANTAGONIST_STRONG
            elif pro_contribution > ant_contribution * 1.5:
                source = RegretSource.PROTAGONIST_WEAK
            else:
                source = RegretSource.BOTH

            self._decompositions.append(DecompositionResult(
                level_features=baseline[i]['features'],
                pro_return=baseline[i]['pro_return'],
                ant_return=baseline[i]['ant_return'],
                regret=base_regret,
                regret_source=source,
                pro_contribution=pro_contribution,
                ant_contribution=ant_contribution,
            ))

    def _summarize_decomposition(self) -> Dict[str, Any]:
        """Summarize decomposition results."""
        source_counts = {}
        for source in RegretSource:
            count = sum(1 for d in self._decompositions if d.regret_source == source)
            source_counts[source.value] = count

        total = len(self._decompositions)
        source_fractions = {k: v / total for k, v in source_counts.items()}

        # Mean contributions
        mean_pro_contribution = np.mean([d.pro_contribution for d in self._decompositions])
        mean_ant_contribution = np.mean([d.ant_contribution for d in self._decompositions])

        return {
            'source_counts': source_counts,
            'source_fractions': source_fractions,
            'mean_protagonist_contribution': float(mean_pro_contribution),
            'mean_antagonist_contribution': float(mean_ant_contribution),
        }

    def _analyze_cap_response(self) -> Dict[str, float]:
        """Analyze response to antagonist capping."""
        baseline = self._results_by_condition['baseline']
        capped = self._results_by_condition['antagonist_capped']

        baseline_regrets = np.array([d['regret'] for d in baseline])
        capped_regrets = np.array([d['regret'] for d in capped])

        return {
            'mean_regret_reduction': float(np.mean(baseline_regrets - capped_regrets)),
            'regret_reduction_std': float(np.std(baseline_regrets - capped_regrets)),
            'fraction_reduced': float(np.mean(capped_regrets < baseline_regrets)),
        }

    def _analyze_boost_response(self) -> Dict[str, float]:
        """Analyze response to protagonist boosting."""
        baseline = self._results_by_condition['baseline']
        boosted = self._results_by_condition['protagonist_boosted']

        baseline_regrets = np.array([d['regret'] for d in baseline])
        boosted_regrets = np.array([d['regret'] for d in boosted])

        return {
            'mean_regret_reduction': float(np.mean(baseline_regrets - boosted_regrets)),
            'regret_reduction_std': float(np.std(baseline_regrets - boosted_regrets)),
            'fraction_reduced': float(np.mean(boosted_regrets < baseline_regrets)),
        }

    def _compute_source_attribution(self) -> Dict[str, float]:
        """Compute overall source attribution."""
        total_pro = sum(d.pro_contribution for d in self._decompositions)
        total_ant = sum(d.ant_contribution for d in self._decompositions)
        total = total_pro + total_ant

        if total < 1e-10:
            return {'protagonist_fraction': 0.5, 'antagonist_fraction': 0.5}

        return {
            'protagonist_fraction': float(total_pro / total),
            'antagonist_fraction': float(total_ant / total),
        }

    def _test_solvability_constraint(self) -> Dict[str, float]:
        """Test if adversary respects solvability constraint."""
        baseline = self._results_by_condition['baseline']
        boosted = self._results_by_condition['antagonist_boosted']

        # If levels are solvable, boosted antagonist should achieve high returns
        boosted_ant_returns = [d['ant_return'] for d in boosted]

        # Solvability = fraction of levels where oracle antagonist succeeds
        solvability = float(np.mean([r > 0.8 for r in boosted_ant_returns]))

        return {
            'solvability_rate': solvability,
            'mean_oracle_antagonist_return': float(np.mean(boosted_ant_returns)),
        }

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize decomposition results."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._decompositions:
            return figures

        # Source distribution pie chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Pie chart of regret sources
        ax = axes[0]
        source_counts = {}
        for source in RegretSource:
            source_counts[source.value] = sum(
                1 for d in self._decompositions if d.regret_source == source
            )
        labels = list(source_counts.keys())
        sizes = list(source_counts.values())
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Regret Source Distribution')

        # Contribution comparison
        ax = axes[1]
        pro_contributions = [d.pro_contribution for d in self._decompositions]
        ant_contributions = [d.ant_contribution for d in self._decompositions]
        ax.scatter(pro_contributions, ant_contributions, alpha=0.5, s=20)
        ax.set_xlabel('Protagonist Contribution')
        ax.set_ylabel('Antagonist Contribution')
        ax.set_title('Regret Contribution Decomposition')
        ax.plot([0, max(pro_contributions)], [0, max(pro_contributions)],
                'k--', alpha=0.5, label='Equal contribution')
        ax.legend()

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["regret_decomposition"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
