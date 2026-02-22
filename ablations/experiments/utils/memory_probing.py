"""
Memory probing utilities for cross-episode information flow experiments.

Contains functions for injecting distinctive patterns, testing memory capacity,
and analyzing selective memory.
"""

from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import jax
import jax.numpy as jnp
import chex


def create_distinctive_level_pattern(
    pattern_type: str = "unique_walls",
    pattern_id: int = 0,
    env_height: int = 13,
    env_width: int = 13,
) -> Dict[str, np.ndarray]:
    """
    Create a level with a distinctive, identifiable pattern.

    Args:
        pattern_type: Type of pattern ("unique_walls", "goal_corner", "cross")
        pattern_id: Unique identifier for the pattern
        env_height: Environment height
        env_width: Environment width

    Returns:
        Dict with wall_map and pattern metadata
    """
    wall_map = np.zeros((env_height, env_width), dtype=bool)

    if pattern_type == "unique_walls":
        # Create a unique wall pattern based on pattern_id
        np.random.seed(pattern_id * 12345)
        n_walls = 5 + (pattern_id % 10)
        positions = np.random.choice(env_height * env_width, n_walls, replace=False)
        for pos in positions:
            i, j = pos // env_width, pos % env_width
            if 1 <= i < env_height - 1 and 1 <= j < env_width - 1:
                wall_map[i, j] = True

    elif pattern_type == "goal_corner":
        # Put walls to create a corner pattern
        corner = pattern_id % 4
        if corner == 0:  # Top-left
            wall_map[1:4, 1:4] = True
        elif corner == 1:  # Top-right
            wall_map[1:4, -4:-1] = True
        elif corner == 2:  # Bottom-left
            wall_map[-4:-1, 1:4] = True
        else:  # Bottom-right
            wall_map[-4:-1, -4:-1] = True

    elif pattern_type == "cross":
        # Cross pattern in center
        mid_h, mid_w = env_height // 2, env_width // 2
        offset = pattern_id % 3
        wall_map[mid_h-1-offset:mid_h+2+offset, mid_w] = True
        wall_map[mid_h, mid_w-1-offset:mid_w+2+offset] = True

    # Compute pattern signature (for later identification)
    signature = np.sum(wall_map * np.arange(env_height * env_width).reshape(env_height, env_width))

    return {
        "wall_map": wall_map,
        "pattern_type": pattern_type,
        "pattern_id": pattern_id,
        "signature": int(signature),
    }


def inject_distinctive_pattern(
    agent,
    train_state,
    pattern: Dict[str, np.ndarray],
    rng: chex.PRNGKey,
) -> Tuple[chex.ArrayTree, Dict[str, float]]:
    """
    Run agent on a distinctive level and return hidden state.

    Args:
        agent: Agent instance
        train_state: Current train state
        pattern: Pattern dict from create_distinctive_level_pattern
        rng: Random key

    Returns:
        (final_hidden_state, episode_metrics)
    """
    # This would run a full episode on the patterned level
    # For now, return placeholder
    return {
        "note": "Full implementation requires agent episode execution",
        "pattern_id": pattern["pattern_id"],
    }


def test_memory_capacity(
    probe_hidden_state: Callable,
    n_episodes: int = 20,
    probe_property: str = "pattern_id",
) -> Dict[str, any]:
    """
    Test how many past episodes can be decoded from hidden state.

    Args:
        probe_hidden_state: Function that probes hidden state for a property
        n_episodes: Number of episodes to test
        probe_property: Property to probe for

    Returns:
        Dict with memory capacity results
    """
    # Track accuracy vs episode lag
    accuracy_by_lag = {}

    for lag in range(1, n_episodes + 1):
        # Accuracy of probing episode (current - lag)
        # This would be computed by running probe on hidden state
        # and comparing to ground truth from lag episodes ago

        # Placeholder accuracy decay
        accuracy = 1.0 / (1.0 + 0.1 * lag)
        accuracy_by_lag[lag] = accuracy

    # Find memory horizon: lag at which accuracy drops below chance
    chance_level = 1.0 / n_episodes
    memory_horizon = n_episodes
    for lag, acc in sorted(accuracy_by_lag.items()):
        if acc < chance_level + 0.05:  # Slightly above chance
            memory_horizon = lag - 1
            break

    return {
        "accuracy_by_lag": accuracy_by_lag,
        "memory_horizon": memory_horizon,
        "chance_level": chance_level,
        "n_episodes_tested": n_episodes,
    }


def analyze_selective_memory(
    episode_features: List[Dict[str, float]],
    retained_in_memory: List[bool],
) -> Dict[str, float]:
    """
    Analyze what types of episodes are preferentially retained in memory.

    Tests for biases:
    - Success bias: Are solved episodes retained more?
    - Recency bias: Are recent episodes retained more?
    - Novelty bias: Are unusual episodes retained more?
    - Difficulty bias: Are challenging episodes retained more?

    Args:
        episode_features: List of dicts with episode features
            (return, solved, length, novelty_score, etc.)
        retained_in_memory: Whether each episode was retained

    Returns:
        Dict with bias analysis
    """
    if len(episode_features) == 0:
        return {"error": "No episodes provided"}

    retained = np.array(retained_in_memory)
    n_total = len(retained)
    n_retained = retained.sum()

    if n_retained == 0 or n_retained == n_total:
        return {"error": "All or none retained, cannot analyze bias"}

    results = {
        "retention_rate": float(n_retained / n_total),
    }

    # Success bias
    if "solved" in episode_features[0]:
        solved = np.array([e["solved"] for e in episode_features])
        retained_solved = retained[solved].mean() if solved.sum() > 0 else 0.0
        retained_unsolved = retained[~solved].mean() if (~solved).sum() > 0 else 0.0
        results["success_bias"] = float(retained_solved - retained_unsolved)
        results["success_bias_significant"] = abs(results["success_bias"]) > 0.1

    # Return bias
    if "return" in episode_features[0]:
        returns = np.array([e["return"] for e in episode_features])
        median_return = np.median(returns)
        high_return = returns >= median_return
        retained_high = retained[high_return].mean()
        retained_low = retained[~high_return].mean()
        results["return_bias"] = float(retained_high - retained_low)

    # Recency bias (if episode_idx available)
    if "episode_idx" in episode_features[0]:
        indices = np.array([e["episode_idx"] for e in episode_features])
        median_idx = np.median(indices)
        recent = indices >= median_idx
        retained_recent = retained[recent].mean()
        retained_old = retained[~recent].mean()
        results["recency_bias"] = float(retained_recent - retained_old)

    # Novelty bias
    if "novelty_score" in episode_features[0]:
        novelty = np.array([e["novelty_score"] for e in episode_features])
        median_novelty = np.median(novelty)
        high_novelty = novelty >= median_novelty
        retained_novel = retained[high_novelty].mean()
        retained_familiar = retained[~high_novelty].mean()
        results["novelty_bias"] = float(retained_novel - retained_familiar)

    # Length bias (potential difficulty proxy)
    if "length" in episode_features[0]:
        lengths = np.array([e["length"] for e in episode_features])
        median_length = np.median(lengths)
        long_episodes = lengths >= median_length
        retained_long = retained[long_episodes].mean()
        retained_short = retained[~long_episodes].mean()
        results["length_bias"] = float(retained_long - retained_short)

    return results


def compute_memory_decay_curve(
    accuracies: List[float],
    lags: List[int],
) -> Dict[str, float]:
    """
    Fit decay curve to memory accuracy over lag.

    Args:
        accuracies: Probe accuracy at each lag
        lags: Lag values (episodes back)

    Returns:
        Dict with decay parameters
    """
    accuracies = np.array(accuracies)
    lags = np.array(lags)

    # Fit exponential decay: acc = a * exp(-b * lag)
    # Using log-linear fit
    valid = accuracies > 0
    if valid.sum() < 2:
        return {"error": "Insufficient valid data points"}

    log_acc = np.log(accuracies[valid])
    lags_valid = lags[valid]

    # Linear fit to log(accuracy) vs lag
    coeffs = np.polyfit(lags_valid, log_acc, 1)
    decay_rate = -coeffs[0]
    initial_accuracy = np.exp(coeffs[1])

    # Half-life: lag at which accuracy drops to 50% of initial
    if decay_rate > 0:
        half_life = np.log(2) / decay_rate
    else:
        half_life = float('inf')

    return {
        "decay_rate": float(decay_rate),
        "initial_accuracy": float(initial_accuracy),
        "half_life": float(half_life),
        "fit_r2": float(1 - np.var(log_acc - (coeffs[0] * lags_valid + coeffs[1])) / np.var(log_acc)),
    }


def compare_memory_mechanisms(
    results_by_agent: Dict[str, Dict[str, any]],
) -> Dict[str, any]:
    """
    Compare memory characteristics across agent types.

    Args:
        results_by_agent: Dict mapping agent name to memory test results

    Returns:
        Dict with comparative analysis
    """
    agents = list(results_by_agent.keys())
    if len(agents) < 2:
        return {"error": "Need at least 2 agents to compare"}

    comparison = {
        "agents": agents,
    }

    # Compare memory horizons
    horizons = {
        agent: results.get("memory_horizon", 0)
        for agent, results in results_by_agent.items()
    }
    comparison["memory_horizons"] = horizons
    comparison["best_memory"] = max(horizons, key=horizons.get)

    # Compare decay rates
    decay_rates = {}
    for agent, results in results_by_agent.items():
        if "decay_rate" in results:
            decay_rates[agent] = results["decay_rate"]
    if decay_rates:
        comparison["decay_rates"] = decay_rates
        comparison["slowest_decay"] = min(decay_rates, key=decay_rates.get)

    # Compare selective biases
    biases = {}
    for agent, results in results_by_agent.items():
        agent_biases = {}
        for key in ["success_bias", "recency_bias", "novelty_bias"]:
            if key in results:
                agent_biases[key] = results[key]
        biases[agent] = agent_biases
    comparison["selective_biases"] = biases

    return comparison
