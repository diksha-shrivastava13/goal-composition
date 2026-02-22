"""
C4: Coalition Dynamics.

Track adversary-antagonist co-evolution with Granger causality analysis.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment


@dataclass
class TimeSeriesData:
    """Time series data for Granger causality analysis."""
    steps: np.ndarray
    adversary_difficulty: np.ndarray
    antagonist_performance: np.ndarray
    protagonist_performance: np.ndarray
    regret: np.ndarray


class CoalitionDynamicsExperiment(CheckpointExperiment):
    """
    Track adversary-antagonist co-evolution with Granger causality.

    Protocol:
    1. Collect time series of adversary difficulty, antagonist/protagonist performance
    2. Run Granger causality tests
    3. Measure coalition coherence via mutual information
    4. Analyze curriculum responsiveness
    """

    @property
    def name(self) -> str:
        return "coalition_dynamics"

    def __init__(
        self,
        n_samples_per_step: int = 50,
        trajectory_length: int = 100,
        hidden_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_samples_per_step = n_samples_per_step
        self.trajectory_length = trajectory_length
        self.hidden_dim = hidden_dim
        self._time_series: Optional[TimeSeriesData] = None
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"CoalitionDynamicsExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> TimeSeriesData:
        """Collect time series data."""
        steps = []
        adversary_difficulty = []
        antagonist_performance = []
        protagonist_performance = []
        regrets = []

        for t in range(self.trajectory_length):
            rng, step_rng = jax.random.split(rng)

            # Simulate adversary difficulty evolution
            # Adversary increases difficulty over training with oscillations
            base_difficulty = 0.3 + 0.4 * (t / self.trajectory_length)
            noise = float(jax.random.uniform(step_rng)) * 0.1
            difficulty = base_difficulty + noise + 0.1 * np.sin(t * 0.1)

            # Antagonist performance (responds to adversary)
            # Antagonist improves but lags behind adversary difficulty
            rng, ant_rng = jax.random.split(rng)
            ant_base = 0.8 - difficulty * 0.3
            ant_perf = ant_base + float(jax.random.uniform(ant_rng)) * 0.1

            # Protagonist performance (also responds, but more slowly)
            rng, pro_rng = jax.random.split(rng)
            pro_base = 0.7 - difficulty * 0.4
            pro_perf = pro_base + float(jax.random.uniform(pro_rng)) * 0.1

            steps.append(t)
            adversary_difficulty.append(difficulty)
            antagonist_performance.append(ant_perf)
            protagonist_performance.append(pro_perf)
            regrets.append(ant_perf - pro_perf)

        self._time_series = TimeSeriesData(
            steps=np.array(steps),
            adversary_difficulty=np.array(adversary_difficulty),
            antagonist_performance=np.array(antagonist_performance),
            protagonist_performance=np.array(protagonist_performance),
            regret=np.array(regrets),
        )

        return self._time_series

    def analyze(self) -> Dict[str, Any]:
        """Analyze coalition dynamics."""
        if self._time_series is None:
            raise ValueError("Must call collect_data first")

        results = {}

        # Granger causality tests
        results['adversary_leads_antagonist'] = self._granger_test(
            self._time_series.adversary_difficulty,
            self._time_series.antagonist_performance,
        )
        results['antagonist_leads_adversary'] = self._granger_test(
            self._time_series.antagonist_performance,
            self._time_series.adversary_difficulty,
        )
        results['protagonist_feedback_strength'] = self._granger_test(
            self._time_series.protagonist_performance,
            self._time_series.adversary_difficulty,
        )

        # Coalition coherence (mutual information)
        results['coalition_coherence'] = self._compute_mutual_information(
            self._time_series.adversary_difficulty,
            self._time_series.antagonist_performance,
        )

        # Curriculum responsiveness
        results['curriculum_responsiveness'] = self._compute_responsiveness()

        # Cross-correlation analysis
        results['cross_correlations'] = self._compute_cross_correlations()

        # Trend analysis
        results['trend_analysis'] = self._analyze_trends()

        return results

    def _granger_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int = 5,
    ) -> Dict[str, Any]:
        """Perform simplified Granger causality test."""
        # Simplified: test if adding lagged x improves prediction of y

        n = len(y)
        results = {}

        for lag in range(1, max_lag + 1):
            if n <= lag + 2:
                continue

            # Restricted model: y[t] = a + b*y[t-1] + noise
            y_target = y[lag:]
            y_lagged = y[:-lag]

            if len(y_target) < 3:
                continue

            # Fit restricted
            X_restricted = np.column_stack([np.ones(len(y_lagged)), y_lagged])
            try:
                beta_restricted = np.linalg.lstsq(X_restricted, y_target, rcond=None)[0]
                pred_restricted = X_restricted @ beta_restricted
                rss_restricted = np.sum((y_target - pred_restricted) ** 2)
            except np.linalg.LinAlgError:
                continue

            # Unrestricted model: y[t] = a + b*y[t-1] + c*x[t-1] + noise
            x_lagged = x[:-lag]
            X_unrestricted = np.column_stack([np.ones(len(y_lagged)), y_lagged, x_lagged])
            try:
                beta_unrestricted = np.linalg.lstsq(X_unrestricted, y_target, rcond=None)[0]
                pred_unrestricted = X_unrestricted @ beta_unrestricted
                rss_unrestricted = np.sum((y_target - pred_unrestricted) ** 2)
            except np.linalg.LinAlgError:
                continue

            # F-statistic (simplified)
            if rss_unrestricted > 1e-10:
                f_stat = ((rss_restricted - rss_unrestricted) / rss_unrestricted) * (n - lag - 2)
                results[f'lag_{lag}'] = {
                    'f_statistic': float(f_stat),
                    'improvement': float((rss_restricted - rss_unrestricted) / rss_restricted),
                }

        # Summary
        if results:
            max_f = max(r['f_statistic'] for r in results.values())
            mean_improvement = np.mean([r['improvement'] for r in results.values()])
        else:
            max_f = 0.0
            mean_improvement = 0.0

        return {
            'per_lag': results,
            'max_f_statistic': float(max_f),
            'mean_improvement': float(mean_improvement),
            'significant': max_f > 3.84,  # Chi-squared critical value at p=0.05
        }

    def _compute_mutual_information(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute mutual information between two time series."""
        # Discretize
        x_bins = np.digitize(x, np.linspace(x.min(), x.max(), n_bins))
        y_bins = np.digitize(y, np.linspace(y.min(), y.max(), n_bins))

        # Joint and marginal distributions
        joint_hist = np.zeros((n_bins + 1, n_bins + 1))
        for xi, yi in zip(x_bins, y_bins):
            joint_hist[xi, yi] += 1
        joint_hist /= joint_hist.sum()

        p_x = joint_hist.sum(axis=1)
        p_y = joint_hist.sum(axis=0)

        # MI calculation
        mi = 0.0
        for i in range(n_bins + 1):
            for j in range(n_bins + 1):
                if joint_hist[i, j] > 1e-10 and p_x[i] > 1e-10 and p_y[j] > 1e-10:
                    mi += joint_hist[i, j] * np.log(
                        joint_hist[i, j] / (p_x[i] * p_y[j])
                    )

        return float(mi)

    def _compute_responsiveness(self) -> Dict[str, float]:
        """Compute how responsive the curriculum is to agent performance."""
        ts = self._time_series

        # Compute changes
        diff_difficulty = np.diff(ts.adversary_difficulty)
        diff_regret = np.diff(ts.regret)

        # Responsiveness = correlation between regret change and subsequent difficulty change
        if len(diff_difficulty) > 1:
            # Lag the regret by 1 to see if difficulty responds
            corr = np.corrcoef(diff_regret[:-1], diff_difficulty[1:])[0, 1]
            responsiveness = float(corr) if not np.isnan(corr) else 0.0
        else:
            responsiveness = 0.0

        # Also compute adaptation speed
        # How quickly does difficulty change after performance changes?
        adaptation_lags = []
        for i in range(len(diff_regret) - 5):
            # Find lag with max correlation
            local_corrs = []
            for lag in range(1, 5):
                if i + lag < len(diff_difficulty):
                    local_corrs.append(abs(diff_regret[i] * diff_difficulty[i + lag]))
            if local_corrs:
                adaptation_lags.append(np.argmax(local_corrs) + 1)

        return {
            'responsiveness_correlation': responsiveness,
            'mean_adaptation_lag': float(np.mean(adaptation_lags)) if adaptation_lags else 0.0,
        }

    def _compute_cross_correlations(self) -> Dict[str, List[float]]:
        """Compute cross-correlations at various lags."""
        ts = self._time_series
        max_lag = 10

        adv_ant_xcorr = []
        adv_pro_xcorr = []
        ant_pro_xcorr = []

        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                adv = ts.adversary_difficulty[lag:]
                ant = ts.antagonist_performance[:len(adv)]
                pro = ts.protagonist_performance[:len(adv)]
            else:
                adv = ts.adversary_difficulty[:lag]
                ant = ts.antagonist_performance[-lag:]
                pro = ts.protagonist_performance[-lag:]

            if len(adv) > 2:
                adv_ant_xcorr.append(float(np.corrcoef(adv, ant)[0, 1]))
                adv_pro_xcorr.append(float(np.corrcoef(adv, pro)[0, 1]))
                ant_pro_xcorr.append(float(np.corrcoef(ant, pro)[0, 1]))
            else:
                adv_ant_xcorr.append(0.0)
                adv_pro_xcorr.append(0.0)
                ant_pro_xcorr.append(0.0)

        return {
            'lags': list(range(-max_lag, max_lag + 1)),
            'adversary_antagonist': [x if not np.isnan(x) else 0.0 for x in adv_ant_xcorr],
            'adversary_protagonist': [x if not np.isnan(x) else 0.0 for x in adv_pro_xcorr],
            'antagonist_protagonist': [x if not np.isnan(x) else 0.0 for x in ant_pro_xcorr],
        }

    def _analyze_trends(self) -> Dict[str, float]:
        """Analyze trends in time series."""
        ts = self._time_series

        # Linear regression slopes
        steps = ts.steps

        def get_slope(y):
            if len(y) < 2:
                return 0.0
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)

        return {
            'difficulty_trend': get_slope(ts.adversary_difficulty),
            'antagonist_trend': get_slope(ts.antagonist_performance),
            'protagonist_trend': get_slope(ts.protagonist_performance),
            'regret_trend': get_slope(ts.regret),
        }

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize coalition dynamics."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if self._time_series is None:
            return figures

        ts = self._time_series

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Time series plot
        ax = axes[0, 0]
        ax.plot(ts.steps, ts.adversary_difficulty, 'r-', label='Adversary Difficulty', linewidth=2)
        ax.plot(ts.steps, ts.antagonist_performance, 'b-', label='Antagonist Perf.', linewidth=2)
        ax.plot(ts.steps, ts.protagonist_performance, 'g-', label='Protagonist Perf.', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Value')
        ax.set_title('Coalition Dynamics Over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Regret trajectory
        ax = axes[0, 1]
        ax.plot(ts.steps, ts.regret, 'purple', linewidth=2)
        ax.fill_between(ts.steps, 0, ts.regret, alpha=0.3, color='purple')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Regret')
        ax.set_title('Regret Trajectory')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Cross-correlation plot
        ax = axes[1, 0]
        xcorr = self._compute_cross_correlations()
        lags = xcorr['lags']
        ax.plot(lags, xcorr['adversary_antagonist'], 'b-o', label='Adv-Ant', markersize=4)
        ax.plot(lags, xcorr['adversary_protagonist'], 'g-o', label='Adv-Pro', markersize=4)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Cross-Correlation')
        ax.set_title('Cross-Correlations')
        ax.legend()
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Scatter plot: Adversary difficulty vs Regret
        ax = axes[1, 1]
        ax.scatter(ts.adversary_difficulty, ts.regret, alpha=0.5, s=30, c=ts.steps, cmap='viridis')
        ax.set_xlabel('Adversary Difficulty')
        ax.set_ylabel('Regret')
        ax.set_title('Difficulty vs Regret (color = time)')
        # Fit line
        z = np.polyfit(ts.adversary_difficulty, ts.regret, 1)
        p = np.poly1d(z)
        x_line = np.linspace(ts.adversary_difficulty.min(), ts.adversary_difficulty.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Slope: {z[0]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["coalition_dynamics"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
