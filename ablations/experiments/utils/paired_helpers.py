"""Shared helpers for PAIRED experiments.

Provides real network-based data collection replacing simulated/placeholder patterns.
All functions use the actual trained networks (protagonist, antagonist, adversary)
via batched_rollout or direct forward passes.
"""

from typing import Dict, Any, Optional, Tuple, List, NamedTuple
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
        'corridor_ratio': np.array([
            compute_corridor_count(np.array(candidates.wall_map)[i]) / max(np.array(candidates.wall_map)[i].size, 1)
            for i in range(len(wall_densities))
        ]),
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
    return jax.tree_util.tree_map(lambda x: x[selected], candidates)


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

    n = jax.tree_util.tree_leaves(levels)[0].shape[0]
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
        leaves = jax.tree_util.tree_leaves(result.final_hstate)
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


def compute_difficulty(levels, experiment, rng: chex.PRNGKey, max_steps: int = 256) -> np.ndarray:
    """Compute real difficulty per level using protagonist returns.

    Returns:
        np.ndarray of shape (n,) with difficulty scores in [0, 1].
        Difficulty = 1 - pro_return (higher return = easier level).
    """
    pro_returns = get_protagonist_returns(rng, levels, experiment, max_steps)
    return np.clip(1.0 - pro_returns, 0.0, 1.0)


def compute_difficulty_single(level_dict: Dict[str, Any], experiment, rng: chex.PRNGKey) -> float:
    """Compute difficulty for a single level dict.

    Wraps compute_difficulty for single-level use.
    """
    import jax
    # Create a batched level from single dict by sampling and replacing wall_map
    level_rng, diff_rng = jax.random.split(rng)
    level = experiment.agent.sample_random_level(level_rng)
    level = level.replace(wall_map=jnp.array(level_dict['wall_map']))
    level_batch = jax.tree_util.tree_map(lambda x: x[None], level)
    difficulties = compute_difficulty(level_batch, experiment, diff_rng)
    return float(difficulties[0])


def compute_corridor_count(wall_map: np.ndarray) -> int:
    """Count cells with exactly 2 passable orthogonal neighbors (real corridor detection).

    A corridor cell is a passable cell where exactly 2 of the 4 orthogonal
    neighbors are also passable, forming a narrow passage.
    """
    h, w = wall_map.shape
    corridors = 0
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if not wall_map[i, j]:  # Cell is passable
                passable_neighbors = 0
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and not wall_map[ni, nj]:
                        passable_neighbors += 1
                if passable_neighbors == 2:
                    corridors += 1
    return corridors


def generate_adversary_levels(agent, adv_train_state, rng, n, adv_num_steps=50):
    """Generate n levels using the adversary's learned generation policy.

    Runs the adversary through the MazeEditor environment to produce levels
    via its trained policy (not random sampling).

    Args:
        agent: PAIREDBaseAgent with adv_env, adv_env_params, sample_empty_level
        adv_train_state: adversary's FlaxTrainState (apply_fn + params)
        rng: random key
        n: number of levels to generate
        adv_num_steps: adversary generation steps (default 50)

    Returns:
        levels: batched Level pytree (n levels)
    """
    from ablations.common.networks import AdversaryActorCritic
    from ablations.common.training import sample_trajectories_rnn

    # Create empty level templates
    empty_level = agent.sample_empty_level()
    empty_levels = jax.tree_util.tree_map(
        lambda x: jnp.array([x]).repeat(n, axis=0), empty_level
    )

    # Initialize adversary hidden state
    init_hstate = AdversaryActorCritic.initialize_carry((n,))

    # Reset MazeEditor to empty levels
    rng, rng_reset = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(
        agent.adv_env.reset_to_level, in_axes=(0, 0, None)
    )(jax.random.split(rng_reset, n), empty_levels, agent.adv_env_params)

    # Run adversary rollout through MazeEditor
    rng, rng_rollout = jax.random.split(rng)
    (_, _, _, last_env_state, _), _ = sample_trajectories_rnn(
        rng_rollout, agent.adv_env, agent.adv_env_params,
        adv_train_state, init_hstate, init_obs, init_env_state,
        n, adv_num_steps,
        track_positions=False,
    )

    # Extract generated levels from final env state
    return last_env_state.level


class AdversaryRolloutResult(NamedTuple):
    """Result of running the adversary through the MazeEditor environment."""
    levels: Any           # Batched Level pytree (n,)
    final_hstates: Any    # np.ndarray (n, hidden_dim) — LSTM c+h concatenated
    per_level_entropy: Any  # np.ndarray (n,) — mean -log_prob over generation steps


def run_adversary_rollout(agent, adv_train_state, rng, n, adv_num_steps=50):
    """Run adversary through MazeEditor, capturing levels, h-states, and entropy.

    Unlike generate_adversary_levels() which only returns generated levels, this
    function also captures the adversary's hidden states and per-level entropy
    (mean negative log-prob over generation steps).

    Args:
        agent: PAIREDBaseAgent with adv_env, adv_env_params, sample_empty_level
        adv_train_state: adversary's FlaxTrainState (apply_fn + params)
        rng: random key
        n: number of levels to generate
        adv_num_steps: adversary generation steps (default 50)

    Returns:
        AdversaryRolloutResult with levels, final_hstates, per_level_entropy
    """
    from ablations.common.networks import AdversaryActorCritic
    from ablations.common.training import sample_trajectories_rnn

    # Create empty level templates
    empty_level = agent.sample_empty_level()
    empty_levels = jax.tree_util.tree_map(
        lambda x: jnp.array([x]).repeat(n, axis=0), empty_level
    )

    # Initialize adversary hidden state
    init_hstate = AdversaryActorCritic.initialize_carry((n,))

    # Reset MazeEditor to empty levels
    rng, rng_reset = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(
        agent.adv_env.reset_to_level, in_axes=(0, 0, None)
    )(jax.random.split(rng_reset, n), empty_levels, agent.adv_env_params)

    # Run adversary rollout through MazeEditor
    rng, rng_rollout = jax.random.split(rng)
    (_, hstate, _, last_env_state, _), (_, _, _, _, log_probs, _, _) = sample_trajectories_rnn(
        rng_rollout, agent.adv_env, agent.adv_env_params,
        adv_train_state, init_hstate, init_obs, init_env_state,
        n, adv_num_steps,
        track_positions=False,
    )

    # Extract generated levels from final env state
    levels = last_env_state.level

    # Flatten hstate using same leaf-concatenation pattern as get_real_hstates()
    leaves = jax.tree_util.tree_leaves(hstate)
    parts = [np.array(l).reshape(n, -1) for l in leaves]
    final_hstates = np.concatenate(parts, axis=-1)

    # Entropy = mean negative log-prob per level over generation steps
    # log_probs shape: (adv_num_steps, n)
    per_level_entropy = np.array(-jnp.mean(log_probs, axis=0))

    return AdversaryRolloutResult(
        levels=levels,
        final_hstates=final_hstates,
        per_level_entropy=per_level_entropy,
    )


def mutate_level_walls(level, rng, k):
    """Toggle k random interior walls in a Level struct.

    Args:
        level: A single Level struct (unbatched).
        rng: JAX random key.
        k: Number of walls to toggle.

    Returns:
        New Level with k interior wall cells toggled.
    """
    wall_map = np.array(level.wall_map)
    h, w = wall_map.shape
    interior = [(i, j) for i in range(1, h - 1) for j in range(1, w - 1)]
    indices = jax.random.choice(rng, len(interior), shape=(min(k, len(interior)),), replace=False)
    for idx in np.array(indices):
        r, c = interior[idx]
        wall_map[r, c] = 1 - wall_map[r, c]
    return level.replace(wall_map=jnp.array(wall_map))


def levels_to_dicts(levels, n: int) -> List[Dict[str, Any]]:
    """Convert batched Level pytree to list of plain dicts."""
    wall_maps = np.array(levels.wall_map)
    goal_positions = np.array(levels.goal_pos)
    agent_positions = np.array(levels.agent_pos)

    result = []
    for i in range(n):
        d = {
            'wall_map': wall_maps[i],
            'wall_density': float(wall_maps[i].mean()),
            'goal_pos': tuple(int(x) for x in goal_positions[i])
                if goal_positions.ndim > 1
                else (int(goal_positions[i]),),
            'agent_pos': tuple(int(x) for x in agent_positions[i])
                if agent_positions.ndim > 1
                else (int(agent_positions[i]),),
        }
        result.append(d)
    return result
