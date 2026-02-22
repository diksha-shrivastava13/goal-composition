"""
Calibration utilities for value function analysis.

Contains functions for multi-point calibration, branch-conditioned ECE,
and value gradient analysis.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import chex


def compute_multi_point_calibration(
    values_over_time: np.ndarray,
    returns_from_point: np.ndarray,
    timesteps: List[int] = None,
    n_bins: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Compute calibration metrics at multiple timesteps within episodes.

    Args:
        values_over_time: Value predictions, shape (n_episodes, max_steps)
        returns_from_point: Returns from each point, shape (n_episodes, max_steps)
        timesteps: Timesteps to analyze (default: [1, 10, 50, 100, 200])
        n_bins: Number of bins for ECE computation

    Returns:
        Dict mapping timestep to calibration metrics
    """
    if timesteps is None:
        timesteps = [1, 10, 50, 100, 200]

    values_over_time = np.asarray(values_over_time)
    returns_from_point = np.asarray(returns_from_point)

    results = {}
    for t in timesteps:
        if t >= values_over_time.shape[1]:
            continue

        v_t = values_over_time[:, t]
        g_t = returns_from_point[:, t]

        # Filter out invalid (NaN or masked) values
        valid = ~np.isnan(v_t) & ~np.isnan(g_t)
        if valid.sum() < 10:
            continue

        v_t = v_t[valid]
        g_t = g_t[valid]

        # ECE
        ece = _compute_ece(v_t, g_t, n_bins)

        # Correlation
        corr = np.corrcoef(v_t, g_t)[0, 1] if len(v_t) > 1 else 0.0

        # MAE
        mae = np.mean(np.abs(v_t - g_t))

        # Overconfidence: cases where V >> G
        overconfident_mask = (v_t - g_t) > 0.1
        overconfidence_rate = float(overconfident_mask.mean())

        results[f"t={t}"] = {
            "ece": float(ece),
            "correlation": float(corr),
            "mae": float(mae),
            "overconfidence_rate": overconfidence_rate,
            "n_samples": int(valid.sum()),
        }

    return results


def compute_branch_conditioned_ece(
    values: np.ndarray,
    returns: np.ndarray,
    branches: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Compute ECE separately for each curriculum branch.

    Args:
        values: Value predictions, shape (n_episodes,)
        returns: Actual returns, shape (n_episodes,)
        branches: Branch indices (0=DR, 1=Replay, 2=Mutate), shape (n_episodes,)
        n_bins: Number of bins

    Returns:
        Dict with ECE per branch and comparisons
    """
    values = np.asarray(values)
    returns = np.asarray(returns)
    branches = np.asarray(branches)

    branch_names = {0: "DR", 1: "Replay", 2: "Mutate"}
    results = {}

    for branch_id, name in branch_names.items():
        mask = branches == branch_id
        if mask.sum() < 10:
            continue

        v = values[mask]
        r = returns[mask]

        ece = _compute_ece(v, r, n_bins)
        corr = np.corrcoef(v, r)[0, 1] if len(v) > 1 else 0.0
        mae = np.mean(np.abs(v - r))

        results[name] = {
            "ece": float(ece),
            "correlation": float(corr),
            "mae": float(mae),
            "n_samples": int(mask.sum()),
        }

    # Comparisons
    if "Replay" in results and "DR" in results:
        results["replay_vs_dr_ece_diff"] = (
            results["Replay"]["ece"] - results["DR"]["ece"]
        )
        results["replay_better_calibrated"] = results["replay_vs_dr_ece_diff"] < 0

    return results


def compute_value_gradient(
    train_state,
    obs: chex.ArrayTree,
    hstate: chex.ArrayTree,
    goal_positions: np.ndarray,
) -> Dict[str, float]:
    """
    Compute gradient of value w.r.t. goal distance: ∂V/∂(goal_distance).

    This tests whether the value function encodes goal-directed behavior:
    V should increase as goal distance decreases (i.e., negative gradient).

    Args:
        train_state: Agent train state with apply_fn
        obs: Observations
        hstate: Hidden state
        goal_positions: Goal positions for each sample, shape (n, 2)

    Returns:
        Dict with gradient statistics
    """
    # This requires JAX differentiation through the value function
    # We approximate by computing correlation between V and goal distance

    def get_value(params, obs_single, hstate_single, done):
        obs_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs_single)
        done_batch = jnp.array([[done]])
        _, _, value = train_state.apply_fn(params, (obs_batch, done_batch), hstate_single)
        return value[0, 0]

    # For actual gradient, we'd need to make goal_distance differentiable
    # Here we provide a proxy: correlation analysis
    return {
        "note": "Full gradient computation requires differentiable goal_distance",
        "use_correlation_as_proxy": True,
    }


def compute_temporal_consistency(
    values_over_time: np.ndarray,
    gamma: float = 0.995,
) -> Dict[str, float]:
    """
    Check temporal consistency of value estimates within episodes.

    Value should roughly follow: V(s_t) ≈ r_t + γ * V(s_{t+1})
    Large deviations suggest value function instability or goal changes.

    Args:
        values_over_time: Value predictions, shape (n_episodes, max_steps)
        gamma: Discount factor

    Returns:
        Dict with consistency metrics
    """
    values_over_time = np.asarray(values_over_time)
    n_episodes, max_steps = values_over_time.shape

    # Compute expected decrease due to discounting
    expected_decrease = []
    actual_decrease = []
    td_violations = []  # Cases where V increases unexpectedly

    for ep in range(n_episodes):
        v = values_over_time[ep]
        for t in range(max_steps - 1):
            if np.isnan(v[t]) or np.isnan(v[t + 1]):
                continue

            expected = v[t] * gamma  # V should decrease by γ each step (approx)
            actual = v[t + 1]

            expected_decrease.append(v[t] - expected)
            actual_decrease.append(v[t] - actual)

            # TD violation: value increases when it shouldn't
            if actual > v[t] + 0.1:  # Threshold for unexpected increase
                td_violations.append(1)
            else:
                td_violations.append(0)

    if len(expected_decrease) == 0:
        return {"error": "Insufficient data"}

    return {
        "mean_expected_decrease": float(np.mean(expected_decrease)),
        "mean_actual_decrease": float(np.mean(actual_decrease)),
        "decrease_correlation": float(np.corrcoef(expected_decrease, actual_decrease)[0, 1]),
        "td_violation_rate": float(np.mean(td_violations)),
        "n_transitions": len(expected_decrease),
    }


def _compute_ece(values: np.ndarray, returns: np.ndarray, n_bins: int) -> float:
    """Compute Expected Calibration Error."""
    if len(values) == 0:
        return 0.0

    v_min, v_max = values.min(), values.max()
    if v_max - v_min < 1e-6:
        return float(np.abs(values.mean() - returns.mean()))

    bin_edges = np.linspace(v_min - 1e-5, v_max + 1e-5, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (values >= bin_edges[i]) & (values < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_conf = values[mask].mean()
            bin_acc = returns[mask].mean()
            bin_size = mask.sum()
            ece += bin_size * np.abs(bin_conf - bin_acc)

    return float(ece / len(values))


def compute_calibration_by_difficulty(
    values: np.ndarray,
    returns: np.ndarray,
    difficulties: np.ndarray,
    n_difficulty_bins: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Compute calibration metrics stratified by level difficulty.

    Args:
        values: Value predictions
        returns: Actual returns
        difficulties: Difficulty scores (e.g., regret, wall density)
        n_difficulty_bins: Number of difficulty bins

    Returns:
        Dict with calibration per difficulty bin
    """
    values = np.asarray(values)
    returns = np.asarray(returns)
    difficulties = np.asarray(difficulties)

    # Bin by difficulty percentiles
    percentiles = np.percentile(difficulties, np.linspace(0, 100, n_difficulty_bins + 1))
    results = {}

    for i in range(n_difficulty_bins):
        mask = (difficulties >= percentiles[i]) & (difficulties < percentiles[i + 1])
        if mask.sum() < 5:
            continue

        v = values[mask]
        r = returns[mask]

        ece = _compute_ece(v, r, 10)
        corr = np.corrcoef(v, r)[0, 1] if len(v) > 1 else 0.0

        bin_name = f"difficulty_{i+1}/{n_difficulty_bins}"
        results[bin_name] = {
            "ece": float(ece),
            "correlation": float(corr),
            "mean_difficulty": float(difficulties[mask].mean()),
            "n_samples": int(mask.sum()),
        }

    return results
