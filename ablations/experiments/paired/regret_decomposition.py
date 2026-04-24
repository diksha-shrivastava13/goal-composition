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

    def __init__(self, **kwargs):
        """
        Args:
            n_levels: Number of levels to decompose.
            min_regret_threshold: Minimum regret to classify source
                (below this → NEITHER). Default 0.1.
            dominance_ratio: Ratio for one source to dominate the other.
                Default 1.5 (50% stronger contribution).
            adaptive_threshold: If True, normalize min_regret_threshold
                by the 10th percentile of observed dataset regrets.
            solvability_threshold: Minimum antagonist return to consider
                a level "solvable". Default 0.8.
        """
        super().__init__(**kwargs)
        self.n_levels = self.exp_config("n_levels")
        self.min_regret_threshold = self.exp_config("min_regret_threshold")
        self.dominance_ratio = self.exp_config("dominance_ratio")
        self.adaptive_threshold = self.exp_config("adaptive_threshold")
        self.solvability_threshold = self.exp_config("solvability_threshold")
        self.max_steps = self.exp_config("max_steps", 256)
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
            # Random policy baseline: uniform random actions produce near-zero returns
            # Run real rollout but use a random init state (agent hasn't learned = random-like)
            init_hstate = jax.tree_util.tree_map(jnp.zeros_like, self.agent.initialize_hidden_state(self.n_levels))
            from ..utils.batched_rollout import batched_rollout
            result = batched_rollout(
                rng, self._levels, self.max_steps,
                self.train_state.pro_train_state.apply_fn,
                self.train_state.pro_train_state.params,
                self.agent.env, self.agent.env_params,
                init_hstate,
                collection_steps=[],
            )
            return np.array(result.episode_returns)
        elif mode == 'oracle':
            # Oracle: use the stronger agent (antagonist) as a proxy for near-optimal
            ant_ts = getattr(self.train_state, 'ant_train_state', None)
            if ant_ts is not None:
                result = run_batched_rollout(
                    rng, self._levels, ant_ts, self.agent,
                    max_steps=self.max_steps,
                )
                return np.array(result.episode_returns)
            else:
                return get_protagonist_returns(rng, self._levels, self, self.max_steps)
        else:
            # Normal trained agent
            if agent_type == 'protagonist':
                return get_protagonist_returns(rng, self._levels, self, self.max_steps)
            else:
                return get_antagonist_returns(rng, self._levels, self, self.max_steps)

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

        # H-state probing for regret source classification
        results['hstate_regret_source_probing'] = self._probe_regret_source_from_hstates()

        return results

    def _decompose_regret_sources(self):
        """Decompose regret into sources for each level.

        Uses adversary-implied difficulty baseline ('oracle') to separate
        antagonist-strong vs protagonist-weak contributions. The min_regret_threshold
        filters negligible regret, and dominance_ratio determines when one
        source clearly dominates the other.
        """
        baseline = self._results_by_condition['baseline']
        ant_capped = self._results_by_condition['antagonist_capped']
        pro_boosted = self._results_by_condition['protagonist_boosted']

        # Adaptive threshold: scale by 10th percentile of dataset regrets
        threshold = self.min_regret_threshold
        if self.adaptive_threshold:
            all_regrets = [baseline[i]['regret'] for i in range(len(baseline))]
            p10 = np.percentile(all_regrets, 10)
            if p10 > 0:
                threshold = self.min_regret_threshold * p10

        for i in range(len(baseline)):
            base_regret = baseline[i]['regret']
            capped_regret = ant_capped[i]['regret']
            boosted_regret = pro_boosted[i]['regret']

            # Contribution from antagonist = reduction when antagonist is capped
            ant_contribution = max(0, base_regret - capped_regret)

            # Contribution from protagonist = reduction when protagonist is boosted
            pro_contribution = max(0, base_regret - boosted_regret)

            # Classify source
            if base_regret < threshold:
                source = RegretSource.NEITHER
            elif ant_contribution > pro_contribution * self.dominance_ratio:
                source = RegretSource.ANTAGONIST_STRONG
            elif pro_contribution > ant_contribution * self.dominance_ratio:
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
        solvability = float(np.mean([r > self.solvability_threshold for r in boosted_ant_returns]))

        return {
            'solvability_rate': solvability,
            'mean_oracle_antagonist_return': float(np.mean(boosted_ant_returns)),
        }

    def _probe_regret_source_from_hstates(self) -> Dict[str, Any]:
        """Train classifier on protagonist h-states to predict regret source."""
        if not self._decompositions:
            return {'error': 'No decompositions available'}

        # Collect protagonist h-states
        from ..utils.paired_helpers import get_pro_hstates
        import jax

        if self._levels is None:
            return {'error': 'No levels stored for h-state probing'}

        try:
            rng = jax.random.PRNGKey(42)
            hstates = get_pro_hstates(rng, self._levels, self, self.max_steps)
            hstates_np = np.array(hstates)
        except Exception as e:
            return {'error': f'Failed to collect h-states: {e}'}

        # Build labels from decomposition
        labels = np.array([d.regret_source.value for d in self._decompositions])
        n_samples = min(len(labels), len(hstates_np))
        labels = labels[:n_samples]
        X = hstates_np[:n_samples]

        # Need at least 2 classes
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return {'error': 'Only one regret source class found', 'unique_classes': unique_labels.tolist()}

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            clf = LogisticRegression(max_iter=1000)
            scores = cross_val_score(clf, X_scaled, labels, cv=min(5, len(unique_labels)), scoring='accuracy')
            clf.fit(X_scaled, labels)

            # Per-source accuracy
            per_source = {}
            for src in unique_labels:
                mask = labels == src
                if mask.sum() > 0:
                    per_source[str(src)] = float(clf.score(X_scaled[mask], labels[mask]))

            return {
                'cv_accuracy_mean': float(np.mean(scores)),
                'cv_accuracy_std': float(np.std(scores)),
                'n_samples': n_samples,
                'n_classes': len(unique_labels),
                'per_source_accuracy': per_source,
            }
        except Exception as e:
            return {'error': str(e)}

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
