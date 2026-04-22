"""
Calibration utilities for value function analysis.

Contains functions for multi-point calibration, branch-conditioned ECE,
and value gradient analysis.
"""

from typing import Dict, List, Optional, Tuple
import logging
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
    observations: List,
    hstate: chex.ArrayTree,
    goal_positions: np.ndarray,
    agent_positions: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute value and gradient norm per observation, then correlate with goal distance.

    Tests whether the value function encodes goal-directed behavior:
    V should increase as goal distance decreases (negative correlation).

    Args:
        train_state: Agent train state with params and apply_fn.
        observations: List of Obs objects, one per goal position.
        hstate: Initial hidden state for forward pass.
        goal_positions: (n, 2) or (n,) array of goal coordinates.
        agent_positions: (n, 2) or (n,) array of agent start positions.
            If None, extracted from observations when possible.

    Returns:
        Dict with correlation metrics and summary stats.
    """
    params = train_state.params
    apply_fn = train_state.apply_fn

    values = []
    gradients = []

    for obs in observations:
        try:
            def value_fn(image):
                obs_batch = type(obs)(image[None, None, ...], obs.agent_dir[None, None, ...])
                done_batch = jnp.zeros((1, 1), dtype=bool)
                _, _, value = apply_fn(params, (obs_batch, done_batch), hstate)
                return value[0, 0]

            grad = jax.grad(value_fn)(obs.image)
            gradients.append(float(jnp.linalg.norm(grad)))
            values.append(float(value_fn(obs.image)))
        except Exception:
            gradients.append(0.0)
            values.append(0.0)

    values = np.array(values)
    gradients = np.array(gradients)

    # Compute goal distances from actual agent positions
    agent_pos_fallback_used = False
    if goal_positions.ndim == 1:
        if agent_positions is not None:
            goal_distances = np.abs(goal_positions - np.asarray(agent_positions))
        else:
            goal_distances = np.abs(goal_positions)
    else:
        if agent_positions is not None:
            agent_pos = np.asarray(agent_positions)
        else:
            # Try to extract from observations; fall back to origin with warning
            logging.getLogger(__name__).warning(
                "Agent positions unavailable in calibration; falling back to origin (0,0)"
            )
            agent_pos = np.zeros_like(goal_positions)
            agent_pos_fallback_used = True
        goal_distances = np.sqrt(np.sum((goal_positions - agent_pos) ** 2, axis=-1))

    # Correlation between value and goal distance
    if len(values) > 2 and np.std(values) > 1e-10 and np.std(goal_distances) > 1e-10:
        value_goal_corr = float(np.corrcoef(values, goal_distances)[0, 1])
    else:
        value_goal_corr = 0.0

    # Correlation between gradient norm and goal distance
    if len(gradients) > 2 and np.std(gradients) > 1e-10 and np.std(goal_distances) > 1e-10:
        grad_goal_corr = float(np.corrcoef(gradients, goal_distances)[0, 1])
    else:
        grad_goal_corr = 0.0

    return {
        "value_goal_distance_correlation": value_goal_corr,
        "gradient_goal_distance_correlation": grad_goal_corr,
        "mean_gradient_norm": float(np.mean(gradients)),
        "mean_value": float(np.mean(values)),
        "negative_gradient": value_goal_corr < -0.1,
        "agent_pos_fallback_used": agent_pos_fallback_used,
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
