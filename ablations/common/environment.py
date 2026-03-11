"""
Environment setup utilities for curriculum awareness ablations.

Contains:
- setup_environment: Create Maze environment with wrappers
- setup_level_sampler: Create PLR/ACCEL level sampler
- Level generation and mutation functions
"""

from typing import Tuple, Callable
import jax
import jax.numpy as jnp
import chex

from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level, make_level_generator, make_level_mutator_minimax
from jaxued.level_sampler import LevelSampler
from jaxued.wrappers import AutoReplayWrapper

# Patch Level to support dict-style access (level['wall_map'], 'wall_map' in level, level.get(...))
# Many experiment files use dict-style access but Level is a flax struct.
if not hasattr(Level, '__getitem__'):
    Level.__getitem__ = lambda self, key: getattr(self, key)
    Level.__contains__ = lambda self, key: hasattr(self, key)
    Level.get = lambda self, key, default=None: getattr(self, key, default)


def setup_environment(
    max_height: int = 13,
    max_width: int = 13,
    agent_view_size: int = 5,
    normalize_obs: bool = True,
    n_walls: int = 25,
) -> Tuple[Maze, Maze, Callable, MazeRenderer, Callable]:
    """
    Setup Maze environment with standard configuration.

    Args:
        max_height: Maximum maze height
        max_width: Maximum maze width
        agent_view_size: Agent's view size
        normalize_obs: Whether to normalize observations
        n_walls: Number of walls for random level generation

    Returns:
        env: Training environment (with AutoReplayWrapper)
        eval_env: Evaluation environment (no wrapper)
        sample_random_level: Function to generate random levels
        env_renderer: Renderer for visualization
        mutate_level: Function to mutate levels
    """
    # Base environment
    base_env = Maze(
        max_height=max_height,
        max_width=max_width,
        agent_view_size=agent_view_size,
        normalize_obs=normalize_obs,
    )

    # Evaluation environment (same as base)
    eval_env = base_env

    # Level generator
    sample_random_level = make_level_generator(max_height, max_width, n_walls)

    # Renderer
    env_renderer = MazeRenderer(base_env, tile_size=8)

    # Training environment with auto-replay wrapper
    env = AutoReplayWrapper(base_env)

    # Level mutator (ACCEL)
    mutate_level = make_level_mutator_minimax(100)

    return env, eval_env, sample_random_level, env_renderer, mutate_level


def setup_level_sampler(
    capacity: int = 4000,
    replay_prob: float = 0.8,
    staleness_coeff: float = 0.3,
    minimum_fill_ratio: float = 0.5,
    prioritization: str = "rank",
    temperature: float = 0.3,
    top_k: int = 4,
    duplicate_check: bool = True,
) -> LevelSampler:
    """
    Setup PLR level sampler with standard configuration.

    Args:
        capacity: Maximum number of levels in buffer
        replay_prob: Probability of replaying vs generating new level
        staleness_coeff: Coefficient for staleness penalty
        minimum_fill_ratio: Minimum buffer fill before replay
        prioritization: "rank" or "topk"
        temperature: Temperature for score-based sampling
        top_k: K for top-k prioritization
        duplicate_check: Whether to check for duplicate levels

    Returns:
        LevelSampler instance
    """
    return LevelSampler(
        capacity=capacity,
        replay_prob=replay_prob,
        staleness_coeff=staleness_coeff,
        minimum_fill_ratio=minimum_fill_ratio,
        prioritization=prioritization,
        prioritization_params={"temperature": temperature, "k": top_k},
        duplicate_check=duplicate_check,
    )


def compute_score_maxmc(
    dones: chex.Array,
    values: chex.Array,
    max_returns: chex.Array,
) -> chex.Array:
    """
    Compute MaxMC score for PLR.

    MaxMC = max over episode of (max_return - value_estimate)
    """
    from jaxued.utils import max_mc
    return max_mc(dones, values, max_returns)


def compute_score_pvl(
    dones: chex.Array,
    advantages: chex.Array,
) -> chex.Array:
    """
    Compute Positive Value Loss score for PLR.

    PVL = mean of positive advantages (learning potential)
    """
    from jaxued.utils import positive_value_loss
    return positive_value_loss(dones, advantages)


def compute_score(
    config: dict,
    dones: chex.Array,
    values: chex.Array,
    max_returns: chex.Array,
    advantages: chex.Array,
) -> chex.Array:
    """
    Compute level score based on config.

    Args:
        config: Config dict with 'score_function' key
        dones: Done flags
        values: Value estimates
        max_returns: Maximum returns achieved
        advantages: GAE advantages

    Returns:
        Score for each environment
    """
    if config["score_function"] == "MaxMC":
        return compute_score_maxmc(dones, values, max_returns)
    elif config["score_function"] == "pvl":
        return compute_score_pvl(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")


def get_eval_levels(level_names: list) -> Level:
    """
    Load predefined evaluation levels by name.

    Args:
        level_names: List of level names (e.g., ["SixteenRooms", "Labyrinth"])

    Returns:
        Batched Level struct
    """
    return Level.load_prefabs(level_names)


# =============================================================================
# BRANCH TRANSITION UTILITIES
# =============================================================================

def get_branch_name(branch_id: int) -> str:
    """Convert branch ID to human-readable name."""
    names = {0: "DR", 1: "Replay", 2: "Mutate"}
    return names.get(branch_id, f"Unknown({branch_id})")


def get_transition_name(prev_branch: int, curr_branch: int) -> str:
    """Get human-readable transition name."""
    return f"{get_branch_name(prev_branch)}->{get_branch_name(curr_branch)}"


def is_replay_to_mutate(prev_branch: int, curr_branch: int) -> bool:
    """Check if transition is Replay->Mutate (1-to-1 correspondence)."""
    return prev_branch == 1 and curr_branch == 2
