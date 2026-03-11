"""Shared helpers for PAIRED experiments.

Provides real network-based data collection replacing simulated/placeholder patterns.
All functions use the actual trained networks (protagonist, antagonist, adversary)
via batched_rollout or direct forward passes.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import jax
import jax.numpy as jnp
import chex
import logging

logger = logging.getLogger(__name__)


def generate_levels(agent, rng: chex.PRNGKey, n: int):
    """Generate n random levels using the agent's environment.

    Returns:
        levels: Batched Level pytree with leading dim n
    """
    level_rngs = jax.random.split(rng, n)
    levels = jax.vmap(agent.sample_random_level)(level_rngs)
    jax.block_until_ready(levels)
    return levels


def generate_constrained_levels(
    agent,
    rng: chex.PRNGKey,
    n: int,
    constraints: Dict[str, Tuple[float, float]],
) -> Any:
    """Generate n levels satisfying constraints via rejection sampling.

    Args:
        agent: Agent with sample_random_level
        rng: Random key
        n: Number of levels to generate
        constraints: Dict of feature_name -> (min_val, max_val)

    Returns:
        levels: Batched Level pytree satisfying constraints
    """
    # Generate more levels than needed, filter
    oversample = max(n * 5, 100)
    rng, gen_rng = jax.random.split(rng)
    candidates = generate_levels(agent, gen_rng, oversample)

    wall_maps = np.array(candidates.wall_map)
    goal_positions = np.array(candidates.goal_pos)
    agent_positions = np.array(candidates.agent_pos)

    # Compute features for filtering
    wall_densities = wall_maps.mean(axis=(1, 2))
    if goal_positions.ndim > 1:
        goal_distances = np.sqrt(np.sum((goal_positions - agent_positions) ** 2, axis=-1))
    else:
        goal_distances = np.abs(goal_positions - agent_positions).astype(float)

    # Normalize goal distance to [0, 1]
    max_dist = np.sqrt(13**2 + 13**2)
    goal_distances_norm = goal_distances / max_dist

    # Apply constraints
    mask = np.ones(oversample, dtype=bool)
    feature_map = {
        'wall_density': wall_densities,
        'goal_distance': goal_distances_norm,
        'open_space_ratio': 1.0 - wall_densities,
        'corridor_ratio': wall_densities,  # Approximate
        'dense_walls': wall_densities,
    }
    for feat_name, (min_val, max_val) in constraints.items():
        feat_key = feat_name.lower().replace(' ', '_')
        if feat_key in feature_map:
            vals = feature_map[feat_key]
            mask &= (vals >= min_val) & (vals <= max_val)

    valid_indices = np.where(mask)[0]
    if len(valid_indices) < n:
        # Not enough, use all valid + random fill
        extra_needed = n - len(valid_indices)
        rng, extra_rng = jax.random.split(rng)
        extra_indices = np.array(jax.random.randint(extra_rng, (extra_needed,), 0, oversample))
        selected = np.concatenate([valid_indices, extra_indices])[:n]
    else:
        selected = valid_indices[:n]

    # Index into the Level pytree
    return jax.tree.map(lambda x: x[selected], candidates)


def extract_level_features_batch(levels) -> Dict[str, np.ndarray]:
    """Extract features from batched Level pytree.

    Returns:
        Dict with arrays of shape (n,) for each feature.
    """
    wall_maps = np.array(levels.wall_map)
    goal_positions = np.array(levels.goal_pos)
    agent_positions = np.array(levels.agent_pos)

    wall_densities = wall_maps.mean(axis=tuple(range(1, wall_maps.ndim)))
    if goal_positions.ndim > 1:
        goal_distances = np.sqrt(np.sum((goal_positions - agent_positions) ** 2, axis=-1))
    else:
        goal_distances = np.abs(goal_positions - agent_positions).astype(float)

    return {
        'wall_density': wall_densities,
        'goal_distance': goal_distances,
        'open_space_ratio': 1.0 - wall_densities,
    }


def extract_level_features_single(level_dict: Dict[str, Any]) -> Dict[str, float]:
    """Extract features from a single level dict."""
    wall_map = np.array(level_dict['wall_map'])
    wall_density = float(wall_map.sum() / wall_map.size)
    goal_pos = level_dict['goal_pos']
    agent_pos = level_dict['agent_pos']
    goal_distance = float(np.sqrt(
        (goal_pos[0] - agent_pos[0])**2 + (goal_pos[1] - agent_pos[1])**2
    ))
    return {
        'wall_density': wall_density,
        'goal_distance': goal_distance,
        'open_space_ratio': 1.0 - wall_density,
    }


def run_batched_rollout(
    rng: chex.PRNGKey,
    levels,
    train_state,
    agent,
    *,
    max_steps: int = 256,
    collect_values: bool = False,
    collect_actions: bool = False,
    collect_entropies: bool = False,
    collect_logits: bool = False,
    return_final_hstate: bool = False,
):
    """Run batched rollout with a train_state on levels.

    Works for protagonist, antagonist, or any agent sub-state.

    Returns:
        RolloutResult with episode_returns, episode_solved, etc.
    """
    from .batched_rollout import batched_rollout

    n = jax.tree.leaves(levels)[0].shape[0]
    init_hstate = agent.initialize_hidden_state(n)

    return batched_rollout(
        rng, levels, max_steps,
        train_state.apply_fn, train_state.params,
        agent.env, agent.env_params,
        init_hstate,
        collect_values=collect_values,
        collect_actions=collect_actions,
        collect_entropies=collect_entropies,
        collect_logits=collect_logits,
        collection_steps=[-1],
        return_final_hstate=return_final_hstate,
    )


def get_protagonist_returns(rng, levels, experiment, max_steps=256):
    """Get real protagonist returns on levels."""
    result = run_batched_rollout(
        rng, levels, experiment.train_state, experiment.agent,
        max_steps=max_steps,
    )
    return np.array(result.episode_returns)


def get_antagonist_returns(rng, levels, experiment, max_steps=256):
    """Get real antagonist returns on levels."""
    ant_ts = getattr(experiment.train_state, 'ant_train_state', None)
    if ant_ts is None:
        return get_protagonist_returns(rng, levels, experiment, max_steps)
    result = run_batched_rollout(
        rng, levels, ant_ts, experiment.agent,
        max_steps=max_steps,
    )
    return np.array(result.episode_returns)


def get_pro_ant_returns(rng, levels, experiment, max_steps=256):
    """Get both protagonist and antagonist returns on levels.

    Returns:
        (pro_returns, ant_returns, regrets) - all np.ndarray of shape (n,)
    """
    rng_pro, rng_ant = jax.random.split(rng)
    pro_returns = get_protagonist_returns(rng_pro, levels, experiment, max_steps)
    ant_returns = get_antagonist_returns(rng_ant, levels, experiment, max_steps)
    regrets = ant_returns - pro_returns
    return pro_returns, ant_returns, regrets


def get_real_hstates(
    rng: chex.PRNGKey,
    levels,
    train_state,
    agent,
    max_steps: int = 256,
) -> np.ndarray:
    """Get real hidden states by running agent on levels.

    Returns hidden states at the END of each episode (terminal hstate).
    Shape: (n, hidden_dim) where hidden_dim = 2 * lstm_features (c + h concatenated).
    """
    result = run_batched_rollout(
        rng, levels, train_state, agent,
        max_steps=max_steps,
        return_final_hstate=True,
    )

    if result.final_hstate is not None:
        # LSTM hstate is (carry, hidden) each of shape (n, features)
        leaves = jax.tree.leaves(result.final_hstate)
        # Concatenate all leaves and flatten per-env
        parts = [np.array(l).reshape(len(np.array(l)), -1) for l in leaves]
        return np.concatenate(parts, axis=-1)
    else:
        # Fallback: use terminal hstate snapshots
        if result.hstates_by_step and "-1" in result.hstates_by_step:
            return np.array(result.hstates_by_step["-1"])
        raise RuntimeError("Could not extract hidden states from rollout")


def get_pro_hstates(rng, levels, experiment, max_steps=256):
    """Get protagonist hidden states on levels."""
    return get_real_hstates(rng, levels, experiment.train_state, experiment.agent, max_steps)


def get_ant_hstates(rng, levels, experiment, max_steps=256):
    """Get antagonist hidden states on levels."""
    ant_ts = getattr(experiment.train_state, 'ant_train_state', None)
    if ant_ts is None:
        return get_pro_hstates(rng, levels, experiment, max_steps)
    return get_real_hstates(rng, levels, ant_ts, experiment.agent, max_steps)


def get_action_distribution(
    train_state,
    agent,
    levels,
    rng: chex.PRNGKey,
    max_steps: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get real action logits and entropies from rollout.

    Returns:
        (logits, entropies) - logits shape (n, max_steps, n_actions), entropies shape (n, max_steps)
    """
    result = run_batched_rollout(
        rng, levels, train_state, agent,
        max_steps=max_steps,
        collect_logits=True,
        collect_entropies=True,
    )
    return np.array(result.logits), np.array(result.entropies)


def get_values_from_rollout(
    train_state,
    agent,
    levels,
    rng: chex.PRNGKey,
    max_steps: int = 256,
) -> np.ndarray:
    """Get real value estimates from rollout. Shape (n, max_steps)."""
    result = run_batched_rollout(
        rng, levels, train_state, agent,
        max_steps=max_steps,
        collect_values=True,
    )
    return np.array(result.values)


def compute_bfs_path_length(level_dict: Dict[str, Any]) -> int:
    """BFS shortest path length from agent_pos to goal_pos."""
    from collections import deque
    wall_map = np.array(level_dict['wall_map'])
    h, w = wall_map.shape
    start = tuple(int(x) for x in level_dict['agent_pos'])
    goal = tuple(int(x) for x in level_dict['goal_pos'])

    if start == goal:
        return 0

    visited = set()
    visited.add(start)
    queue = deque([(start, 0)])

    while queue:
        (y, x), dist = queue.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited and not wall_map[ny, nx]:
                if (ny, nx) == goal:
                    return dist + 1
                visited.add((ny, nx))
                queue.append(((ny, nx), dist + 1))

    return -1  # No path found


def levels_to_dicts(levels, n: int) -> List[Dict[str, Any]]:
    """Convert batched Level pytree to list of plain dicts."""
    wall_maps = np.array(levels.wall_map)
    goal_positions = np.array(levels.goal_pos)
    agent_positions = np.array(levels.agent_pos)

    result = []
    for i in range(n):
        d = {
            'wall_map': wall_maps[i],
            'goal_pos': tuple(int(x) for x in goal_positions[i])
                if goal_positions.ndim > 1
                else (int(goal_positions[i]),),
            'agent_pos': tuple(int(x) for x in agent_positions[i])
                if agent_positions.ndim > 1
                else (int(agent_positions[i]),),
        }
        result.append(d)
    return result
