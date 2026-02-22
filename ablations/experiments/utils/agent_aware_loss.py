"""
Agent-aware loss computation utilities.

Provides unified interface for computing prediction/probe loss
across all 5 agent architectures.

Agent types and their loss computation:
- accel_probe: ActorCritic + separate probe -> compute_probe_loss()
- persistent_lstm: ActorCriticPersistent + probe -> compute_probe_loss()
- context_vector: ActorCriticWithContext + probe -> compute_probe_loss()
- episodic_memory: ActorCriticWithContext + probe -> compute_probe_loss()
- next_env_prediction: ActorCriticWithCurriculumPrediction -> compute_curriculum_prediction_loss()
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np

import jax
import jax.numpy as jnp

from ablations.common.metrics import (
    compute_probe_loss,
    compute_curriculum_prediction_loss,
    compute_random_baselines,
)
from ablations.common.types import DEFAULT_ENV_HEIGHT, DEFAULT_ENV_WIDTH


def detect_agent_type(agent) -> str:
    """
    Detect agent type from network class name.

    Args:
        agent: Agent instance

    Returns:
        One of: "next_env_prediction", "persistent_lstm", "context_vector",
                "episodic_memory", "accel_probe"
    """
    network_name = agent.get_actor_critic_class().__name__

    if "CurriculumPrediction" in network_name:
        return "next_env_prediction"
    elif "Persistent" in network_name:
        return "persistent_lstm"
    elif "Context" in network_name:
        # Could be context_vector or episodic_memory
        # Check for episodic memory attribute
        if hasattr(agent, 'use_episodic_memory') and agent.use_episodic_memory:
            return "episodic_memory"
        # Check class name hints
        agent_class_name = agent.__class__.__name__.lower()
        if "episodic" in agent_class_name:
            return "episodic_memory"
        return "context_vector"
    else:
        return "accel_probe"


def uses_prediction_head(agent) -> bool:
    """
    Check if agent uses integrated prediction head vs separate probe.

    Args:
        agent: Agent instance

    Returns:
        True if agent uses prediction head (next_env_prediction agent)
    """
    return detect_agent_type(agent) == "next_env_prediction"


def create_observation_from_level(level: Dict[str, Any]) -> Any:
    """
    Create observation object from level dict.

    Args:
        level: Dict with wall_map, goal_pos, agent_pos, agent_dir

    Returns:
        Observation object with image and agent_dir attributes
    """
    height = width = DEFAULT_ENV_HEIGHT
    if 'wall_map' in level:
        wall_map = level['wall_map']
        if hasattr(wall_map, 'shape'):
            height, width = wall_map.shape
        else:
            wall_map = np.array(wall_map)
            height, width = wall_map.shape
    else:
        wall_map = np.zeros((height, width))

    # Create 3-channel image: walls (R), goal (G), agent (B)
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:, :, 0] = np.array(wall_map).astype(np.float32)

    # Goal position
    goal_pos = level['goal_pos']
    if hasattr(goal_pos, '__len__') and len(goal_pos) >= 2:
        goal_y, goal_x = int(goal_pos[0]), int(goal_pos[1])
        if 0 <= goal_y < height and 0 <= goal_x < width:
            image[goal_y, goal_x] = [0, 1, 0]

    # Agent position
    agent_pos = level['agent_pos']
    if hasattr(agent_pos, '__len__') and len(agent_pos) >= 2:
        agent_y, agent_x = int(agent_pos[0]), int(agent_pos[1])
        if 0 <= agent_y < height and 0 <= agent_x < width:
            image[agent_y, agent_x, 2] = 1.0

    # Create observation namedtuple-like object
    class Obs:
        def __init__(self, img, direction):
            self.image = img
            self.agent_dir = direction

    agent_dir = level.get('agent_dir', 0)
    if hasattr(agent_dir, '__len__'):
        agent_dir = int(agent_dir[0]) if len(agent_dir) > 0 else 0
    else:
        agent_dir = int(agent_dir)

    return Obs(jnp.array(image), jnp.array([agent_dir]))


def create_level_object(level: Dict[str, Any]):
    """
    Create Level dataclass from dict for loss computation.

    Args:
        level: Dict with wall_map, goal_pos, agent_pos, agent_dir

    Returns:
        Level-like object with required fields as arrays
    """
    from dataclasses import dataclass

    @dataclass
    class LevelObj:
        wall_map: jnp.ndarray
        goal_pos: jnp.ndarray
        agent_pos: jnp.ndarray
        agent_dir: jnp.ndarray

    # Handle wall_map
    if 'wall_map' in level:
        wall_map = jnp.array(level['wall_map'])
    else:
        wall_map = jnp.zeros((DEFAULT_ENV_HEIGHT, DEFAULT_ENV_WIDTH))

    # Handle goal_pos
    goal_pos = level['goal_pos']
    if not hasattr(goal_pos, 'shape'):
        goal_pos = jnp.array(goal_pos)
    else:
        goal_pos = jnp.array(goal_pos)

    # Handle agent_pos
    agent_pos = level['agent_pos']
    if not hasattr(agent_pos, 'shape'):
        agent_pos = jnp.array(agent_pos)
    else:
        agent_pos = jnp.array(agent_pos)

    # Handle agent_dir
    agent_dir = level.get('agent_dir', 0)
    if hasattr(agent_dir, '__len__'):
        agent_dir = int(agent_dir[0]) if len(agent_dir) > 0 else 0
    else:
        agent_dir = int(agent_dir)
    agent_dir = jnp.array(agent_dir)

    return LevelObj(
        wall_map=wall_map,
        goal_pos=goal_pos,
        agent_pos=agent_pos,
        agent_dir=agent_dir,
    )


def extract_curriculum_features(level: Dict[str, Any]) -> jnp.ndarray:
    """
    Extract curriculum features for next_env_prediction agent.

    Args:
        level: Dict with wall_map, goal_pos, agent_pos, agent_dir

    Returns:
        1D array of curriculum features
    """
    height, width = DEFAULT_ENV_HEIGHT, DEFAULT_ENV_WIDTH

    # Handle wall_map
    if 'wall_map' in level:
        wall_map = np.array(level['wall_map'])
        height, width = wall_map.shape
    else:
        wall_map = np.zeros((height, width))

    grid_size = height * width

    # Flatten wall map
    wall_flat = wall_map.flatten().astype(np.float32)

    # Goal position one-hot
    goal_pos = level['goal_pos']
    goal_y, goal_x = int(goal_pos[0]), int(goal_pos[1])
    goal_onehot = np.zeros(grid_size, dtype=np.float32)
    goal_idx = goal_y * width + goal_x
    if 0 <= goal_idx < grid_size:
        goal_onehot[goal_idx] = 1.0

    # Agent position one-hot
    agent_pos = level['agent_pos']
    agent_y, agent_x = int(agent_pos[0]), int(agent_pos[1])
    agent_onehot = np.zeros(grid_size, dtype=np.float32)
    agent_idx = agent_y * width + agent_x
    if 0 <= agent_idx < grid_size:
        agent_onehot[agent_idx] = 1.0

    # Direction one-hot
    agent_dir = level.get('agent_dir', 0)
    if hasattr(agent_dir, '__len__'):
        agent_dir = int(agent_dir[0]) if len(agent_dir) > 0 else 0
    else:
        agent_dir = int(agent_dir)
    dir_onehot = np.zeros(4, dtype=np.float32)
    if 0 <= agent_dir < 4:
        dir_onehot[agent_dir] = 1.0

    return jnp.concatenate([wall_flat, goal_onehot, agent_onehot, dir_onehot])


def compute_agent_prediction_loss(
    agent,
    train_state,
    level: Dict[str, Any],
    rng: jax.random.PRNGKey,
    curriculum_features: Optional[jnp.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute prediction/probe loss for any agent type.

    This is the main entry point for computing the actual prediction loss
    across all 5 agent architectures.

    Args:
        agent: Agent instance
        train_state: Current train state with params
        level: Level dict with wall_map, goal_pos, agent_pos, agent_dir
        rng: Random key
        curriculum_features: Pre-computed features for next_env_prediction agent

    Returns:
        (total_loss, component_losses_dict)
    """
    agent_type = detect_agent_type(agent)

    # Create observation from level
    obs = create_observation_from_level(level)
    hstate = agent.initialize_carry(rng, batch_dims=(1,))
    obs_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs)
    done_batch = jnp.zeros((1, 1), dtype=bool)

    try:
        if agent_type == "next_env_prediction":
            # Get predictions from integrated prediction head
            if curriculum_features is None:
                curriculum_features = extract_curriculum_features(level)

            # Forward pass with curriculum prediction
            try:
                _, _, _, predictions = train_state.apply_fn(
                    train_state.params,
                    (obs_batch, done_batch),
                    hstate,
                    curriculum_features=curriculum_features[None, :],  # Add batch dim
                    predict_curriculum=True,
                )
            except TypeError:
                # Fallback: try without extra kwargs
                outputs = train_state.apply_fn(
                    train_state.params,
                    (obs_batch, done_batch),
                    hstate,
                )
                if len(outputs) == 4:
                    _, _, _, predictions = outputs
                else:
                    # Cannot get predictions
                    random_baselines = compute_random_baselines()
                    return float(random_baselines['total_loss']), {'error': 'no_predictions'}

            if predictions is None:
                random_baselines = compute_random_baselines()
                return float(random_baselines['total_loss']), {'error': 'predictions_none'}

            # Compute prediction loss
            level_obj = create_level_object(level)
            loss, metrics = compute_curriculum_prediction_loss(predictions, level_obj)

            return float(loss), {k: float(v) for k, v in metrics.items()}

        else:
            # Probe-based agents (accel_probe, persistent_lstm, context_vector, episodic_memory)
            new_hstate, _, _ = train_state.apply_fn(
                train_state.params,
                (obs_batch, done_batch),
                hstate
            )

            # Flatten and stop gradient on hidden state
            h_c, h_h = new_hstate
            hstate_flat = jnp.concatenate([
                h_c.reshape(1, -1),
                h_h.reshape(1, -1)
            ], axis=-1)
            hstate_flat = jax.lax.stop_gradient(hstate_flat)

            # Apply probe if available
            if hasattr(train_state, 'probe_params') and train_state.probe_params is not None:
                # Get probe from agent
                if hasattr(agent, 'probe'):
                    probe = agent.probe
                elif hasattr(agent, 'curriculum_probe'):
                    probe = agent.curriculum_probe
                else:
                    # No probe found
                    random_baselines = compute_random_baselines()
                    return float(random_baselines['total_loss']), {'error': 'no_probe_found'}

                # Apply probe to hidden state
                predictions = probe.apply(
                    train_state.probe_params,
                    hstate_flat,
                    episode_return=jnp.zeros(1),
                    episode_solved=jnp.zeros(1),
                    episode_length=jnp.ones(1) * 50,
                )

                # Compute probe loss
                level_obj = create_level_object(level)
                loss, metrics = compute_probe_loss(predictions, level_obj)

                return float(loss), {k: float(v) for k, v in metrics.items()}
            else:
                # No probe available - return max loss
                random_baselines = compute_random_baselines()
                return float(random_baselines['total_loss']), {'error': 'no_probe_params'}

    except Exception as e:
        # Return random baseline on error
        random_baselines = compute_random_baselines()
        return float(random_baselines['total_loss']), {'error': str(e)}


def compute_random_baseline_loss() -> float:
    """
    Get the random baseline total loss for normalization.

    Returns:
        Total loss expected from a random (untrained) predictor
    """
    baselines = compute_random_baselines()
    return float(baselines['total_loss'])


def compute_information_gain(loss: float, random_baseline: Optional[float] = None) -> float:
    """
    Compute information gain relative to random baseline.

    Args:
        loss: Actual prediction/probe loss
        random_baseline: Random baseline loss (computed if None)

    Returns:
        Information gain (random_baseline - loss, clipped to >= 0)
    """
    if random_baseline is None:
        random_baseline = compute_random_baseline_loss()

    return max(0.0, random_baseline - loss)


def compute_normalized_loss(loss: float, random_baseline: Optional[float] = None) -> float:
    """
    Compute normalized loss (0 = random, 1 = perfect).

    Args:
        loss: Actual prediction/probe loss
        random_baseline: Random baseline loss (computed if None)

    Returns:
        Normalized loss in [0, 1] where 1 is perfect prediction
    """
    if random_baseline is None:
        random_baseline = compute_random_baseline_loss()

    if random_baseline <= 0:
        return 0.0

    normalized = 1.0 - (loss / random_baseline)
    return max(0.0, min(1.0, normalized))
