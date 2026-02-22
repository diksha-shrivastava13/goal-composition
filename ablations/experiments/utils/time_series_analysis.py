"""
Time series analysis utilities for behavioral coupling experiments.

Contains functions for computing predictive signal strength, rolling correlations,
Granger causality tests, and change point detection.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


def compute_predictive_signal_strength(
    probe_loss: np.ndarray,
    random_baseline_loss: float,
) -> np.ndarray:
    """
    Compute predictive signal strength from probe loss.

    Signal strength = 1 - (probe_loss / random_baseline_loss)
    0 = no predictive content, 1 = perfect prediction

    Args:
        probe_loss: Probe loss values, shape (n_samples,)
        random_baseline_loss: Loss of random predictor

    Returns:
        Signal strength, shape (n_samples,)
    """
    probe_loss = np.asarray(probe_loss)

    # Clip to avoid negative signal strength
    ratio = np.clip(probe_loss / random_baseline_loss, 0, 1)
    signal_strength = 1 - ratio

    return signal_strength


def compute_rolling_correlation(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling Pearson correlation between two time series.

    Args:
        x: First time series
        y: Second time series
        window_size: Rolling window size

    Returns:
        (correlations, valid_indices) where correlations are computed
        for indices [window_size-1, ...]
    """
    x = np.asarray(x)
    y = np.asarray(y)

    assert len(x) == len(y), "Time series must have same length"

    n = len(x)
    correlations = np.zeros(n - window_size + 1)

    for i in range(len(correlations)):
        x_window = x[i:i+window_size]
        y_window = y[i:i+window_size]

        if np.std(x_window) > 0 and np.std(y_window) > 0:
            corr = np.corrcoef(x_window, y_window)[0, 1]
        else:
            corr = 0.0

        correlations[i] = corr

    indices = np.arange(window_size - 1, n)
    return correlations, indices


def compute_granger_causality(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 10,
    significance_level: float = 0.05,
) -> Dict[str, any]:
    """
    Compute Granger causality test: does x Granger-cause y?

    WARNING: This is for exploratory use only. Granger causality ≠ true causality.
    Training time series violate stationarity assumptions.

    Args:
        x: Potential cause time series
        y: Potential effect time series
        max_lag: Maximum lag to test
        significance_level: Significance level for test

    Returns:
        Dict with test results and caveats
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return {"error": "statsmodels not installed", "install": "pip install statsmodels"}

    x = np.asarray(x)
    y = np.asarray(y)

    # Prepare data for statsmodels (y, x format)
    data = np.column_stack([y, x])

    try:
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    except Exception as e:
        return {"error": str(e)}

    # Extract p-values for F-test at each lag
    p_values = {}
    for lag in range(1, max_lag + 1):
        if lag in results:
            p_values[f"lag_{lag}"] = results[lag][0]["ssr_ftest"][1]

    # Find minimum p-value (strongest evidence)
    if p_values:
        min_lag = min(p_values, key=p_values.get)
        min_p = p_values[min_lag]
        significant = min_p < significance_level
    else:
        min_lag = None
        min_p = 1.0
        significant = False

    return {
        "p_values_by_lag": p_values,
        "best_lag": min_lag,
        "min_p_value": float(min_p),
        "significant": significant,
        "caveats": [
            "Granger causality ≠ true causality",
            "Training time series may violate stationarity",
            "Shared confounds (training progress) likely dominate",
            "Use only as exploratory diagnostic, not evidence",
        ],
    }


def detect_change_points(
    signal: np.ndarray,
    method: str = "pelt",
    n_breakpoints: Optional[int] = None,
    penalty: Optional[float] = None,
) -> Dict[str, any]:
    """
    Detect change points in a time series.

    Note: This is EXPLORATORY. Do not claim "phase transitions" without
    independent validation.

    Args:
        signal: Time series, shape (n_samples,)
        method: "pelt" (penalized), "binseg" (binary segmentation), or "window"
        n_breakpoints: Number of breakpoints (for binseg)
        penalty: Penalty for adding breakpoints (for pelt)

    Returns:
        Dict with detected change points and caveats
    """
    try:
        import ruptures as rpt
    except ImportError:
        return {"error": "ruptures not installed", "install": "pip install ruptures"}

    signal = np.asarray(signal).reshape(-1, 1)

    if method == "pelt":
        algo = rpt.Pelt(model="rbf").fit(signal)
        if penalty is None:
            penalty = np.log(len(signal)) * signal.var()
        breakpoints = algo.predict(pen=penalty)
    elif method == "binseg":
        algo = rpt.Binseg(model="rbf").fit(signal)
        if n_breakpoints is None:
            n_breakpoints = 3
        breakpoints = algo.predict(n_bkps=n_breakpoints)
    elif method == "window":
        algo = rpt.Window(width=50, model="rbf").fit(signal)
        if penalty is None:
            penalty = np.log(len(signal)) * signal.var()
        breakpoints = algo.predict(pen=penalty)
    else:
        return {"error": f"Unknown method: {method}"}

    # Remove the final point (always included by ruptures)
    breakpoints = [b for b in breakpoints if b < len(signal)]

    return {
        "change_points": breakpoints,
        "n_segments": len(breakpoints) + 1,
        "method": method,
        "caveats": [
            "Change points are EXPLORATORY, not 'detected phases'",
            "Results depend on hyperparameters",
            "Independent validation required before claiming phase transitions",
        ],
    }


def compute_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 50,
) -> Dict[str, any]:
    """
    Compute cross-correlation between two time series at various lags.

    Positive lag means x leads y.

    Args:
        x: First time series
        y: Second time series
        max_lag: Maximum lag to compute

    Returns:
        Dict with correlation at each lag
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Normalize
    x = (x - x.mean()) / (x.std() + 1e-10)
    y = (y - y.mean()) / (y.std() + 1e-10)

    correlations = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # y leads x
            corr = np.correlate(x[:lag], y[-lag:])[0] / (len(x) + lag)
        elif lag > 0:
            # x leads y
            corr = np.correlate(x[lag:], y[:-lag])[0] / (len(x) - lag)
        else:
            corr = np.correlate(x, y)[0] / len(x)
        correlations[lag] = float(corr)

    # Find lag with max correlation
    max_lag_val = max(correlations, key=lambda k: abs(correlations[k]))
    max_corr = correlations[max_lag_val]

    return {
        "correlations": correlations,
        "max_correlation": float(max_corr),
        "max_correlation_lag": max_lag_val,
        "x_leads": max_lag_val > 0,
        "y_leads": max_lag_val < 0,
    }


def compute_phase_portrait(
    signal_strength: np.ndarray,
    performance: np.ndarray,
) -> Dict[str, any]:
    """
    Compute phase portrait of (signal_strength, performance) trajectory.

    Useful for visualizing joint dynamics.

    Args:
        signal_strength: Predictive signal over training
        performance: Task performance over training

    Returns:
        Dict with trajectory data and summary statistics
    """
    signal_strength = np.asarray(signal_strength)
    performance = np.asarray(performance)

    # Smooth for visualization
    window = min(10, len(signal_strength) // 5)
    if window > 1:
        signal_smooth = np.convolve(signal_strength, np.ones(window)/window, mode='valid')
        perf_smooth = np.convolve(performance, np.ones(window)/window, mode='valid')
    else:
        signal_smooth = signal_strength
        perf_smooth = performance

    # Compute velocity (rate of change)
    signal_velocity = np.diff(signal_smooth)
    perf_velocity = np.diff(perf_smooth)

    # Quadrant analysis
    n_points = len(signal_smooth)
    signal_median = np.median(signal_smooth)
    perf_median = np.median(perf_smooth)

    quadrants = {
        "high_signal_high_perf": 0,
        "high_signal_low_perf": 0,
        "low_signal_high_perf": 0,
        "low_signal_low_perf": 0,
    }

    for s, p in zip(signal_smooth, perf_smooth):
        if s >= signal_median and p >= perf_median:
            quadrants["high_signal_high_perf"] += 1
        elif s >= signal_median and p < perf_median:
            quadrants["high_signal_low_perf"] += 1
        elif s < signal_median and p >= perf_median:
            quadrants["low_signal_high_perf"] += 1
        else:
            quadrants["low_signal_low_perf"] += 1

    for k in quadrants:
        quadrants[k] /= n_points

    return {
        "trajectory": {
            "signal": signal_smooth.tolist(),
            "performance": perf_smooth.tolist(),
        },
        "velocity": {
            "signal": signal_velocity.tolist(),
            "performance": perf_velocity.tolist(),
        },
        "quadrant_proportions": quadrants,
        "correlation": float(np.corrcoef(signal_smooth, perf_smooth)[0, 1]),
    }
