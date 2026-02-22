"""
Behavioral Coupling Experiment.

Tracks correlation between predictive signal strength and task performance
over training. Analyzes temporal dynamics to understand whether predictive
signals and performance co-vary or exhibit temporal relationships.

Note: This is EXPLORATORY analysis. Granger causality tests have known
limitations with non-stationary training data.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import chex

from .base import TrainingTimeExperiment
from .utils.time_series_analysis import (
    compute_predictive_signal_strength,
    compute_rolling_correlation,
    compute_granger_causality,
    detect_change_points,
)


@dataclass
class CouplingTimeSeries:
    """Container for time series data collected during training."""
    steps: List[int] = field(default_factory=list)
    probe_losses: List[float] = field(default_factory=list)
    task_performance: List[float] = field(default_factory=list)  # Solve rate or mean return
    signal_strengths: List[float] = field(default_factory=list)
    policy_entropies: List[float] = field(default_factory=list)
    value_estimates: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert lists to numpy arrays."""
        return {
            'steps': np.array(self.steps),
            'probe_losses': np.array(self.probe_losses),
            'task_performance': np.array(self.task_performance),
            'signal_strengths': np.array(self.signal_strengths),
            'policy_entropies': np.array(self.policy_entropies),
            'value_estimates': np.array(self.value_estimates),
            'gradient_norms': np.array(self.gradient_norms),
        }


class BehavioralCouplingExperiment(TrainingTimeExperiment):
    """
    Track coupling between predictive signals and task performance.

    Collects data during training via hooks and analyzes:
    - Signal strength over time (from probe loss)
    - Performance over time (solve rate, returns)
    - Rolling correlation between signals and performance
    - Granger causality (with appropriate caveats)
    - Change point detection (exploratory)
    """

    @property
    def name(self) -> str:
        return "behavioral_coupling"

    def __init__(
        self,
        collection_interval: int = 100,
        probe_n_samples: int = 50,
        rolling_window: int = 20,
        granger_max_lag: int = 10,
        random_baseline_samples: int = 100,
        **kwargs,
    ):
        """
        Initialize behavioral coupling experiment.

        Args:
            collection_interval: Steps between data collection
            probe_n_samples: Number of samples for probe loss estimation
            rolling_window: Window size for rolling correlation
            granger_max_lag: Maximum lag for Granger causality tests
            random_baseline_samples: Samples for computing random baseline
        """
        super().__init__(**kwargs)
        self.collection_interval = collection_interval
        self.probe_n_samples = probe_n_samples
        self.rolling_window = rolling_window
        self.granger_max_lag = granger_max_lag
        self.random_baseline_samples = random_baseline_samples

        self._timeseries = CouplingTimeSeries()
        self._random_baseline: Optional[float] = None
        self._results: Dict[str, Any] = {}

    def training_hook(
        self,
        train_state: Any,
        metrics: Dict[str, Any],
        step: int,
    ) -> Dict[str, Any]:
        """
        Hook called during training to collect coupling data.

        Args:
            train_state: Current training state
            metrics: Training metrics from this step
            step: Current training step

        Returns:
            Dict with collected metrics (for logging)
        """
        if step % self.collection_interval != 0:
            return {}

        # Store train state for analysis
        self.train_state = train_state

        # Extract performance metrics
        solve_rate = metrics.get('solve_rate', metrics.get('success_rate', 0.0))
        mean_return = metrics.get('mean_return', metrics.get('episode_return', 0.0))
        policy_entropy = metrics.get('policy_entropy', metrics.get('entropy', 0.0))
        value_mean = metrics.get('value_mean', metrics.get('v_mean', 0.0))
        grad_norm = metrics.get('grad_norm', metrics.get('gradient_norm', 0.0))

        # Compute probe loss if we have a probe available
        probe_loss = self._estimate_probe_loss()

        # Compute signal strength relative to random baseline
        if self._random_baseline is None:
            self._random_baseline = self._compute_random_baseline()

        signal_strength = compute_predictive_signal_strength(
            probe_loss,
            self._random_baseline
        )

        # Store time series data
        self._timeseries.steps.append(step)
        self._timeseries.probe_losses.append(probe_loss)
        self._timeseries.task_performance.append(solve_rate if solve_rate > 0 else mean_return)
        self._timeseries.signal_strengths.append(signal_strength)
        self._timeseries.policy_entropies.append(policy_entropy)
        self._timeseries.value_estimates.append(value_mean)
        self._timeseries.gradient_norms.append(grad_norm)

        return {
            'coupling/probe_loss': probe_loss,
            'coupling/signal_strength': signal_strength,
            'coupling/task_performance': solve_rate if solve_rate > 0 else mean_return,
        }

    def _estimate_probe_loss(self) -> float:
        """
        Estimate actual probe/prediction loss on current hidden states.

        Uses agent-aware loss computation to get the actual prediction loss
        (from probe for probe-based agents, from prediction head for
        next_env_prediction agent).
        """
        if not hasattr(self, 'train_state') or self.train_state is None:
            return 1.0  # Max loss if no state available

        try:
            import jax
            import jax.numpy as jnp
            from .utils.agent_aware_loss import compute_agent_prediction_loss

            rng = jax.random.PRNGKey(0)
            losses = []

            for i in range(self.probe_n_samples):
                rng, sample_rng, level_rng = jax.random.split(rng, 3)

                # Generate a random level for evaluation
                level = self._generate_random_level(level_rng)

                # Compute actual prediction/probe loss
                loss, _ = compute_agent_prediction_loss(
                    self.agent,
                    self.train_state,
                    level,
                    sample_rng,
                )
                losses.append(loss)

            return float(np.mean(losses)) if losses else 1.0

        except Exception:
            return 1.0

    def _generate_random_level(self, rng) -> Dict[str, Any]:
        """Generate a random level for probe loss evaluation."""
        import jax

        height, width = 13, 13
        wall_prob = 0.1 + float(jax.random.uniform(rng)) * 0.2

        wall_map = np.array(jax.random.bernoulli(rng, wall_prob, (height, width)))
        wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

        rng_goal, rng_agent = jax.random.split(rng)
        goal_pos = (
            int(jax.random.randint(rng_goal, (), 1, height - 1)),
            int(jax.random.randint(rng_goal, (), 1, width - 1)),
        )
        agent_pos = (
            int(jax.random.randint(rng_agent, (), 1, height - 1)),
            int(jax.random.randint(rng_agent, (), 1, width - 1)),
        )

        return {
            'wall_map': wall_map,
            'wall_density': wall_map.sum() / (height * width),
            'goal_pos': goal_pos,
            'agent_pos': agent_pos,
            'agent_dir': 0,
        }

    def _compute_random_baseline(self) -> float:
        """Compute random baseline probe loss."""
        # For random baseline, we assume maximum uncertainty
        # This is 1.0 for normalized losses
        return 1.0

    def collect_data(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """
        Collect final coupling data.

        This is called after training is complete to finalize data collection.
        Most data is collected via training_hook during training.
        """
        return self._timeseries.to_arrays()

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze coupling between signals and performance.

        Performs:
        1. Rolling correlation analysis
        2. Granger causality (with caveats)
        3. Change point detection (exploratory)
        4. Summary statistics
        """
        data = self._timeseries.to_arrays()

        if len(data['steps']) < self.rolling_window:
            return {
                'error': 'Insufficient data points for analysis',
                'n_points': len(data['steps']),
                'required': self.rolling_window,
            }

        results = {}

        # 1. Overall correlation
        results['overall_correlation'] = self._compute_overall_correlation(data)

        # 2. Rolling correlation
        results['rolling_correlation'] = self._compute_rolling_analysis(data)

        # 3. Granger causality (with strong caveats)
        results['granger_analysis'] = self._compute_granger_analysis(data)

        # 4. Change point detection
        results['change_points'] = self._detect_change_points(data)

        # 5. Summary statistics
        results['summary'] = self._compute_summary_stats(data)

        # 6. Caveats (important!)
        results['caveats'] = self._get_analysis_caveats()

        self._results = results
        return results

    def _compute_overall_correlation(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute overall correlation between signals and performance."""
        from scipy import stats

        signal = data['signal_strengths']
        performance = data['task_performance']

        # Remove NaN/Inf
        valid = np.isfinite(signal) & np.isfinite(performance)
        if valid.sum() < 3:
            return {'error': 'Insufficient valid data points'}

        signal_valid = signal[valid]
        perf_valid = performance[valid]

        # Pearson correlation
        r_pearson, p_pearson = stats.pearsonr(signal_valid, perf_valid)

        # Spearman correlation (more robust to non-linearity)
        r_spearman, p_spearman = stats.spearmanr(signal_valid, perf_valid)

        return {
            'pearson_r': float(r_pearson),
            'pearson_p': float(p_pearson),
            'spearman_r': float(r_spearman),
            'spearman_p': float(p_spearman),
            'n_samples': int(valid.sum()),
        }

    def _compute_rolling_analysis(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute rolling correlation over training."""
        signal = data['signal_strengths']
        performance = data['task_performance']
        steps = data['steps']

        rolling_corr = compute_rolling_correlation(
            signal,
            performance,
            window=self.rolling_window
        )

        # Find periods of strong correlation
        strong_corr_threshold = 0.5
        strong_corr_periods = []
        in_period = False
        period_start = None

        for i, corr in enumerate(rolling_corr):
            if abs(corr) > strong_corr_threshold and not in_period:
                in_period = True
                period_start = i + self.rolling_window
            elif abs(corr) <= strong_corr_threshold and in_period:
                in_period = False
                if period_start is not None:
                    strong_corr_periods.append({
                        'start_step': int(steps[period_start]) if period_start < len(steps) else 0,
                        'end_step': int(steps[min(i + self.rolling_window, len(steps) - 1)]),
                    })

        return {
            'rolling_correlations': rolling_corr.tolist(),
            'rolling_steps': steps[self.rolling_window:].tolist(),
            'mean_rolling_corr': float(np.nanmean(rolling_corr)),
            'std_rolling_corr': float(np.nanstd(rolling_corr)),
            'strong_correlation_periods': strong_corr_periods,
        }

    def _compute_granger_analysis(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Compute Granger causality tests (with strong caveats).

        WARNING: Granger causality has severe limitations for training data:
        - Non-stationarity violates assumptions
        - Shared confounds (training progress) dominate
        - Granger causality != true causation
        """
        signal = data['signal_strengths']
        performance = data['task_performance']

        results = {
            'warning': 'Granger causality results should be interpreted with extreme caution',
            'caveats': [
                'Training time series are non-stationary',
                'Shared confounds (training progress) likely dominate',
                'Granger causality does NOT imply true causation',
                'Use curriculum interventions for causal evidence',
            ],
        }

        try:
            # Test: does signal Granger-cause performance?
            granger_result = compute_granger_causality(
                signal,
                performance,
                max_lag=self.granger_max_lag
            )
            results['signal_to_performance'] = granger_result

            # Test: does performance Granger-cause signal?
            granger_reverse = compute_granger_causality(
                performance,
                signal,
                max_lag=self.granger_max_lag
            )
            results['performance_to_signal'] = granger_reverse

        except Exception as e:
            results['error'] = str(e)

        return results

    def _detect_change_points(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Detect potential change points in time series.

        WARNING: These are EXPLORATORY diagnostics, not confirmed transitions.
        """
        results = {
            'warning': 'Change points are exploratory diagnostics, not confirmed transitions',
        }

        # Detect change points in signal strength
        try:
            signal_cps = detect_change_points(
                data['signal_strengths'],
                penalty=1.0
            )
            results['signal_change_points'] = [
                int(data['steps'][min(cp, len(data['steps']) - 1)])
                for cp in signal_cps
            ]
        except Exception as e:
            results['signal_change_points_error'] = str(e)

        # Detect change points in performance
        try:
            perf_cps = detect_change_points(
                data['task_performance'],
                penalty=1.0
            )
            results['performance_change_points'] = [
                int(data['steps'][min(cp, len(data['steps']) - 1)])
                for cp in perf_cps
            ]
        except Exception as e:
            results['performance_change_points_error'] = str(e)

        # Check for coinciding change points
        if 'signal_change_points' in results and 'performance_change_points' in results:
            sig_cps = set(results['signal_change_points'])
            perf_cps = set(results['performance_change_points'])

            # Find change points within 5 steps of each other
            coinciding = []
            tolerance = 500  # Steps
            for s_cp in sig_cps:
                for p_cp in perf_cps:
                    if abs(s_cp - p_cp) <= tolerance:
                        coinciding.append({
                            'signal_step': s_cp,
                            'performance_step': p_cp,
                            'gap': abs(s_cp - p_cp),
                        })

            results['coinciding_change_points'] = coinciding

        return results

    def _compute_summary_stats(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute summary statistics for the coupling analysis."""
        return {
            'n_data_points': len(data['steps']),
            'training_steps_covered': int(data['steps'][-1] - data['steps'][0]) if len(data['steps']) > 1 else 0,
            'signal_strength': {
                'mean': float(np.nanmean(data['signal_strengths'])),
                'std': float(np.nanstd(data['signal_strengths'])),
                'min': float(np.nanmin(data['signal_strengths'])),
                'max': float(np.nanmax(data['signal_strengths'])),
            },
            'task_performance': {
                'mean': float(np.nanmean(data['task_performance'])),
                'std': float(np.nanstd(data['task_performance'])),
                'min': float(np.nanmin(data['task_performance'])),
                'max': float(np.nanmax(data['task_performance'])),
            },
            'probe_loss': {
                'mean': float(np.nanmean(data['probe_losses'])),
                'std': float(np.nanstd(data['probe_losses'])),
                'final': float(data['probe_losses'][-1]) if len(data['probe_losses']) > 0 else 0.0,
            },
            'policy_entropy': {
                'mean': float(np.nanmean(data['policy_entropies'])),
                'final': float(data['policy_entropies'][-1]) if len(data['policy_entropies']) > 0 else 0.0,
            },
        }

    def _get_analysis_caveats(self) -> List[str]:
        """Return list of analysis caveats."""
        return [
            "This is EXPLORATORY analysis, not confirmatory",
            "Granger causality does NOT prove true causation",
            "Training data violates stationarity assumptions",
            "Shared confounds (training progress) likely dominate correlations",
            "Change points are smoothing artifacts until independently validated",
            "Use curriculum interventions (Experiment 6) for causal evidence",
            "Correlation does not imply the agent 'knows' or 'models' anything",
        ]

    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data for behavioral coupling."""
        if not self._results:
            raise ValueError("Must call analyze before visualize")

        data = self._timeseries.to_arrays()

        viz_data = {
            'time_series': {
                'steps': data['steps'].tolist(),
                'signal_strengths': data['signal_strengths'].tolist(),
                'task_performance': data['task_performance'].tolist(),
                'policy_entropies': data['policy_entropies'].tolist(),
                'probe_losses': data['probe_losses'].tolist(),
            },
        }

        # Rolling correlation plot
        if 'rolling_correlation' in self._results:
            rc = self._results['rolling_correlation']
            viz_data['rolling_correlation'] = {
                'steps': rc.get('rolling_steps', []),
                'correlations': rc.get('rolling_correlations', []),
            }

        # Change points
        if 'change_points' in self._results:
            cp = self._results['change_points']
            viz_data['change_points'] = {
                'signal': cp.get('signal_change_points', []),
                'performance': cp.get('performance_change_points', []),
                'coinciding': cp.get('coinciding_change_points', []),
            }

        # Summary stats
        if 'summary' in self._results:
            viz_data['summary'] = self._results['summary']

        return viz_data

    def compare_agents(
        self,
        other_timeseries: CouplingTimeSeries,
        agent_name: str = "other",
    ) -> Dict[str, Any]:
        """
        Compare coupling patterns between this agent and another.

        Args:
            other_timeseries: Time series data from another agent
            agent_name: Name of the other agent for labeling

        Returns:
            Dict with comparative analysis
        """
        self_data = self._timeseries.to_arrays()
        other_data = other_timeseries.to_arrays()

        results = {}

        # Compare overall correlations
        self_corr = np.corrcoef(self_data['signal_strengths'], self_data['task_performance'])[0, 1]
        other_corr = np.corrcoef(other_data['signal_strengths'], other_data['task_performance'])[0, 1]

        results['correlation_comparison'] = {
            'self': float(self_corr) if np.isfinite(self_corr) else 0.0,
            agent_name: float(other_corr) if np.isfinite(other_corr) else 0.0,
            'difference': float(self_corr - other_corr) if np.isfinite(self_corr) and np.isfinite(other_corr) else 0.0,
        }

        # Compare signal strength trajectories
        # Interpolate to common steps for comparison
        common_steps = np.intersect1d(self_data['steps'], other_data['steps'])
        if len(common_steps) > 0:
            self_signals = np.interp(common_steps, self_data['steps'], self_data['signal_strengths'])
            other_signals = np.interp(common_steps, other_data['steps'], other_data['signal_strengths'])

            results['signal_trajectory_comparison'] = {
                'self_mean': float(np.mean(self_signals)),
                f'{agent_name}_mean': float(np.mean(other_signals)),
                'correlation': float(np.corrcoef(self_signals, other_signals)[0, 1]),
            }

        return results
