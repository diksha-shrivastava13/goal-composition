"""
Curriculum state management for cross-episode memory.

Contains:
- CurriculumState: Tracks curriculum history across episodes
- NoveltyLearnabilityState: Tracks prediction errors for computing metrics
- Functions for creating, updating, and extracting features from these states
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import chex
from flax import struct

from .types import DEFAULT_ENV_HEIGHT, DEFAULT_ENV_WIDTH


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_CURRICULUM_HISTORY_LENGTH = 64
DEFAULT_CURRICULUM_HIDDEN_SIZE = 128


# =============================================================================
# CURRICULUM STATE FOR CROSS-EPISODE MEMORY
# =============================================================================

@struct.dataclass
class CurriculumState:
    """
    Tracks curriculum history across episodes for next-level prediction.

    This enables the prediction head to learn patterns in the curriculum
    that span multiple episodes, which is essential for predicting what
    levels the regret-based curriculum will prioritize.
    """
    # Rolling buffer of recent level features (circular buffer)
    recent_wall_densities: chex.Array       # (history_length,) - fraction of walls
    recent_goal_positions: chex.Array       # (history_length, 2) - [x, y] coords
    recent_agent_positions: chex.Array      # (history_length, 2) - [x, y] coords
    recent_agent_dirs: chex.Array           # (history_length,) - direction 0-3
    recent_scores: chex.Array               # (history_length,) - regret scores
    recent_branches: chex.Array             # (history_length,) - 0=DR, 1=replay, 2=mutate

    # Buffer summary statistics (updated each step)
    buffer_size: int
    buffer_mean_score: float
    buffer_score_std: float
    buffer_max_score: float

    # Training progress
    training_step: int
    total_replay_steps: int      # Count of replay/mutate steps (curriculum steps)
    total_random_steps: int      # Count of random generation steps

    # Tier 1 ground truth tracking
    recent_path_lengths: chex.Array      # (history_length,) — JIT BFS shortest path (normalized)
    recent_goal_distances: chex.Array    # (history_length,) — Manhattan distance (normalized)

    # Tier 2 ground truth tracking
    recent_returns: chex.Array           # (history_length,) — agent episode returns

    # Circular buffer pointer
    head_pointer: int
    history_filled: bool         # Whether buffer has been filled at least once


def create_curriculum_state(
    history_length: int = DEFAULT_CURRICULUM_HISTORY_LENGTH,
) -> CurriculumState:
    """Initialize an empty CurriculumState."""
    return CurriculumState(
        recent_wall_densities=jnp.zeros(history_length, dtype=jnp.float32),
        recent_goal_positions=jnp.zeros((history_length, 2), dtype=jnp.int32),
        recent_agent_positions=jnp.zeros((history_length, 2), dtype=jnp.int32),
        recent_agent_dirs=jnp.zeros(history_length, dtype=jnp.int32),
        recent_scores=jnp.zeros(history_length, dtype=jnp.float32),
        recent_branches=jnp.zeros(history_length, dtype=jnp.int32),
        recent_path_lengths=jnp.zeros(history_length, dtype=jnp.float32),
        recent_goal_distances=jnp.zeros(history_length, dtype=jnp.float32),
        recent_returns=jnp.zeros(history_length, dtype=jnp.float32),
        buffer_size=0,
        buffer_mean_score=0.0,
        buffer_score_std=0.0,
        buffer_max_score=-jnp.inf,
        training_step=0,
        total_replay_steps=0,
        total_random_steps=0,
        head_pointer=0,
        history_filled=False,
    )


def update_curriculum_state(
    curriculum_state: CurriculumState,
    level,  # Level dataclass from jaxued
    branch: int,
    score: float,
    sampler_stats: dict,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
    episode_return: float = 0.0,
    path_length: float = 0.0,
    goal_distance: float = 0.0,
) -> CurriculumState:
    """
    Update curriculum state with a new level observation.

    Args:
        curriculum_state: Current curriculum state
        level: The level that was just used (Level dataclass)
        branch: Which branch generated this level (0=DR, 1=replay, 2=mutate)
        score: The regret score for this level
        sampler_stats: Statistics from the level sampler
        env_height: Environment height for wall density calculation
        env_width: Environment width for wall density calculation
        episode_return: Agent's actual episode return
        path_length: Normalized shortest path length (from BFS)
        goal_distance: Normalized Manhattan distance from agent to goal

    Returns:
        Updated CurriculumState
    """
    history_length = curriculum_state.recent_wall_densities.shape[0]
    ptr = curriculum_state.head_pointer

    # Compute wall density
    wall_density = level.wall_map.sum() / (env_height * env_width)

    # Update circular buffers
    new_wall_densities = curriculum_state.recent_wall_densities.at[ptr].set(wall_density)
    new_goal_positions = curriculum_state.recent_goal_positions.at[ptr].set(level.goal_pos)
    new_agent_positions = curriculum_state.recent_agent_positions.at[ptr].set(level.agent_pos)
    new_agent_dirs = curriculum_state.recent_agent_dirs.at[ptr].set(level.agent_dir)
    new_scores = curriculum_state.recent_scores.at[ptr].set(score)
    new_branches = curriculum_state.recent_branches.at[ptr].set(branch)
    new_path_lengths = curriculum_state.recent_path_lengths.at[ptr].set(path_length)
    new_goal_distances = curriculum_state.recent_goal_distances.at[ptr].set(goal_distance)
    new_returns = curriculum_state.recent_returns.at[ptr].set(episode_return)

    # Update pointer (circular)
    new_ptr = (ptr + 1) % history_length
    new_history_filled = curriculum_state.history_filled | (new_ptr == 0)

    # Update step counts
    # Convert branch to array if needed to ensure JAX-compatible operations
    branch_arr = jnp.asarray(branch)
    is_curriculum_step = (branch_arr == 1) | (branch_arr == 2)  # replay or mutate
    new_total_replay = curriculum_state.total_replay_steps + is_curriculum_step.astype(jnp.int32)
    new_total_random = curriculum_state.total_random_steps + (branch_arr == 0).astype(jnp.int32)

    return CurriculumState(
        recent_wall_densities=new_wall_densities,
        recent_goal_positions=new_goal_positions,
        recent_agent_positions=new_agent_positions,
        recent_agent_dirs=new_agent_dirs,
        recent_scores=new_scores,
        recent_branches=new_branches,
        recent_path_lengths=new_path_lengths,
        recent_goal_distances=new_goal_distances,
        recent_returns=new_returns,
        buffer_size=sampler_stats.get('size', 0),
        buffer_mean_score=sampler_stats.get('mean_score', 0.0),
        buffer_score_std=sampler_stats.get('score_std', 0.0),
        buffer_max_score=sampler_stats.get('max_score', -jnp.inf),
        training_step=curriculum_state.training_step + 1,
        total_replay_steps=new_total_replay,
        total_random_steps=new_total_random,
        head_pointer=new_ptr,
        history_filled=new_history_filled,
    )


def get_curriculum_features(
    curriculum_state: CurriculumState,
    max_training_steps: int = 30000,
    max_buffer_capacity: int = 4000,
) -> chex.Array:
    """
    Extract feature vector from curriculum state for the encoder.

    Returns a flattened array of curriculum features suitable for
    input to a neural network.

    Args:
        curriculum_state: Current curriculum state
        max_training_steps: Maximum training steps for normalization
        max_buffer_capacity: Maximum buffer capacity for normalization

    Returns:
        Feature vector of curriculum history and statistics
    """
    history_length = curriculum_state.recent_wall_densities.shape[0]

    # Normalize training progress
    normalized_step = curriculum_state.training_step / max_training_steps

    safe_training_step = jnp.maximum(curriculum_state.training_step, 1)
    replay_fraction = curriculum_state.total_replay_steps / safe_training_step

    # Buffer statistics (normalized)
    buffer_fill = curriculum_state.buffer_size / max_buffer_capacity
    normalized_mean_score = jnp.clip(curriculum_state.buffer_mean_score, -10, 10) / 10.0
    normalized_max_score = jnp.clip(curriculum_state.buffer_max_score, -10, 10) / 10.0

    # Recent level statistics
    mean_wall_density = curriculum_state.recent_wall_densities.mean()
    std_wall_density = curriculum_state.recent_wall_densities.std()
    mean_score = curriculum_state.recent_scores.mean()

    # Branch distribution in recent history
    branch_counts = jnp.array([
        (curriculum_state.recent_branches == 0).sum(),
        (curriculum_state.recent_branches == 1).sum(),
        (curriculum_state.recent_branches == 2).sum(),
    ], dtype=jnp.float32) / history_length

    # Combine into feature vector (including new tier history)
    features = jnp.concatenate([
        jnp.array([normalized_step, replay_fraction, buffer_fill]),
        jnp.array([normalized_mean_score, normalized_max_score]),
        jnp.array([mean_wall_density, std_wall_density, mean_score]),
        branch_counts,
        curriculum_state.recent_wall_densities,  # Full history (64,)
        curriculum_state.recent_scores,          # Full history (64,)
        curriculum_state.recent_returns,         # Full history (64,)
        curriculum_state.recent_goal_distances,  # Full history (64,)
        curriculum_state.recent_path_lengths,    # Full history (64,)
    ])

    return features


# =============================================================================
# NOVELTY AND LEARNABILITY METRICS
# Based on: https://arxiv.org/html/2406.04268
#
# Novelty: For fixed history length, prediction error INCREASES looking further
#          into the future. High novelty = harder to predict future environments.
#
# Learnability: For fixed prediction target, more history REDUCES prediction error.
#               High learnability = history helps predict next environment.
#
# Open-ended curriculum: Novelty and Learnability are both high and balanced.
# =============================================================================

@struct.dataclass
class NoveltyLearnabilityState:
    """Tracks prediction errors at different history lengths for computing metrics."""
    # Rolling buffer of prediction losses at different history lengths
    # Shape: (num_history_lengths, buffer_size)
    loss_by_history_length: chex.Array  # e.g., history lengths [8, 16, 32, 64]
    # Rolling buffer of prediction losses over time (for novelty)
    loss_over_time: chex.Array  # Recent losses to measure trend
    # Counters
    buffer_ptr: int
    total_samples: int
    # Configuration
    history_lengths: chex.Array  # e.g., [8, 16, 32, 64]


def create_novelty_learnability_state(
    history_lengths: list = None,
    buffer_size: int = 100,
) -> NoveltyLearnabilityState:
    """Create state for tracking novelty and learnability metrics."""
    if history_lengths is None:
        history_lengths = [8, 16, 32, 64]
    num_lengths = len(history_lengths)
    return NoveltyLearnabilityState(
        loss_by_history_length=jnp.zeros((num_lengths, buffer_size)),
        loss_over_time=jnp.zeros(buffer_size),
        buffer_ptr=0,
        total_samples=0,
        history_lengths=jnp.array(history_lengths),
    )


def compute_learnability(
    nl_state: NoveltyLearnabilityState,
) -> Tuple[float, dict]:
    # NOTE: This is the NL-state version (takes NoveltyLearnabilityState).
    # metrics.py has a separate compute_learnability(loss_history, step_history, total_samples, current_step)
    # for ProbeTrackingState fields. Callers alias via: from .curriculum_state import compute_learnability as compute_learnability_from_nl_state
    """
    Compute learnability: does more history reduce prediction error?

    Learnability = negative slope of (history_length vs prediction_loss).
    High learnability means longer history helps predict better.

    Returns:
        learnability: Scalar measure (higher = more learnable)
        details: Dict with per-history-length losses
    """
    # Get mean loss for each history length
    buffer_size = nl_state.loss_by_history_length.shape[1]
    valid_samples = jnp.minimum(nl_state.total_samples, buffer_size)

    # Mask for valid entries
    mask = jnp.arange(buffer_size) < valid_samples

    # Mean loss per history length
    mean_losses = (nl_state.loss_by_history_length * mask).sum(axis=1) / jnp.maximum(valid_samples, 1)

    # Learnability = negative correlation between history length and loss
    # If loss decreases with more history, learnability is positive
    history_lengths = nl_state.history_lengths.astype(jnp.float32)

    # Compute correlation (simplified: just the slope of linear regression)
    x_mean = history_lengths.mean()
    y_mean = mean_losses.mean()
    numerator = ((history_lengths - x_mean) * (mean_losses - y_mean)).sum()
    denominator = ((history_lengths - x_mean) ** 2).sum() + 1e-8
    slope = numerator / denominator

    # Learnability is negative slope (positive when loss decreases with more history)
    learnability = -slope

    details = {
        f"loss_at_history_{int(h)}": float(l)
        for h, l in zip(nl_state.history_lengths, mean_losses)
    }
    details["learnability_slope"] = float(-slope)

    return float(learnability), details


def compute_novelty(
    # NOTE: NL-state version. metrics.py has a separate compute_novelty with ProbeTrackingState args.
    nl_state: NoveltyLearnabilityState,
    window_size: int = 50,
) -> Tuple[float, dict]:
    """
    Compute novelty: does prediction error increase over time?

    Novelty = positive slope of (time vs prediction_loss).
    High novelty means curriculum is generating increasingly surprising environments.

    Returns:
        novelty: Scalar measure (higher = more novel)
        details: Dict with trend information
    """
    buffer_size = nl_state.loss_over_time.shape[0]
    valid_samples = jnp.minimum(nl_state.total_samples, buffer_size)

    # Get recent losses (last window_size samples)
    recent_start = jnp.maximum(0, valid_samples - window_size)
    recent_losses = jnp.where(
        jnp.arange(buffer_size) >= recent_start,
        nl_state.loss_over_time,
        0.0
    )
    recent_count = jnp.minimum(valid_samples, window_size)

    # Compute trend (slope of loss over time)
    time_indices = jnp.arange(buffer_size).astype(jnp.float32)
    mask = (jnp.arange(buffer_size) >= recent_start) & (jnp.arange(buffer_size) < valid_samples)

    x_mean = (time_indices * mask).sum() / jnp.maximum(mask.sum(), 1)
    y_mean = (nl_state.loss_over_time * mask).sum() / jnp.maximum(mask.sum(), 1)

    numerator = ((time_indices - x_mean) * (nl_state.loss_over_time - y_mean) * mask).sum()
    denominator = ((time_indices - x_mean) ** 2 * mask).sum() + 1e-8
    slope = numerator / denominator

    # Novelty is positive slope (positive when loss increases over time)
    novelty = slope

    # Also compute instantaneous novelty (current loss vs moving average)
    moving_avg = (nl_state.loss_over_time * mask).sum() / jnp.maximum(mask.sum(), 1)
    current_loss = nl_state.loss_over_time[jnp.minimum(valid_samples - 1, buffer_size - 1)]
    instantaneous_novelty = current_loss - moving_avg

    details = {
        "novelty_slope": float(slope),
        "instantaneous_novelty": float(instantaneous_novelty),
        "recent_mean_loss": float(moving_avg),
        "current_loss": float(current_loss),
    }

    return float(novelty), details


def compute_openendedness_score(
    novelty: float,
    learnability: float,
) -> Tuple[float, str]:
    """
    Compute open-endedness score based on balance of novelty and learnability.

    Open-ended curriculum: Both novelty and learnability are high and balanced.
    - High novelty + High learnability = Open-ended (ideal)
    - High novelty + Low learnability = Random/chaotic
    - Low novelty + High learnability = Stagnant/converged
    - Low novelty + Low learnability = Degenerate

    Returns:
        score: Combined open-endedness score
        regime: String describing the current regime
    """
    # Normalize to [0, 1] using sigmoid-like transform
    novelty_norm = 2.0 / (1.0 + jnp.exp(-novelty)) - 1.0  # Maps to [-1, 1]
    learnability_norm = 2.0 / (1.0 + jnp.exp(-learnability)) - 1.0

    # Open-endedness = geometric mean when both positive, else 0
    both_positive = (novelty_norm > 0) & (learnability_norm > 0)
    score = jnp.where(
        both_positive,
        jnp.sqrt(novelty_norm * learnability_norm),
        0.0
    )

    # Determine regime
    if novelty > 0.1 and learnability > 0.1:
        regime = "open-ended"
    elif novelty > 0.1 and learnability <= 0.1:
        regime = "chaotic"
    elif novelty <= 0.1 and learnability > 0.1:
        regime = "converging"
    else:
        regime = "stagnant"

    return float(score), regime


def update_novelty_learnability_state(
    nl_state: NoveltyLearnabilityState,
    prediction_loss: float,
    history_length_idx: int = -1,  # -1 means use the full history (last index)
) -> NoveltyLearnabilityState:
    """Update novelty/learnability tracking with new prediction loss."""
    buffer_size = nl_state.loss_over_time.shape[0]
    ptr = nl_state.buffer_ptr % buffer_size

    # Update loss over time
    new_loss_over_time = nl_state.loss_over_time.at[ptr].set(prediction_loss)

    # Update loss by history length (default: full history = last index)
    if history_length_idx < 0:
        history_length_idx = len(nl_state.history_lengths) - 1
    new_loss_by_history = nl_state.loss_by_history_length.at[history_length_idx, ptr].set(prediction_loss)

    return nl_state.replace(
        loss_over_time=new_loss_over_time,
        loss_by_history_length=new_loss_by_history,
        buffer_ptr=(ptr + 1) % buffer_size,
        total_samples=nl_state.total_samples + 1,
    )


# =============================================================================
# TIER TARGET COMPUTATION (JIT-compatible)
# =============================================================================

def compute_shortest_path_length(
    wall_map: chex.Array,
    agent_pos: chex.Array,
    goal_pos: chex.Array,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
    max_iters: int = 169,
) -> float:
    """JIT-compatible BFS on grid using jax.lax.fori_loop.

    Returns normalized shortest path length (distance / max_possible).
    Returns 1.0 if goal is unreachable.
    """
    max_dist = jnp.float32(env_height * env_width)
    # Initialize distance array: inf everywhere, 0 at agent position
    dist = jnp.full((env_height, env_width), max_dist, dtype=jnp.float32)
    agent_y, agent_x = agent_pos[1], agent_pos[0]
    dist = dist.at[agent_y, agent_x].set(0.0)

    # BFS via iterative relaxation (Bellman-Ford style, converges in grid diameter iterations)
    def relax_step(_, dist):
        # For each cell, check if any neighbor offers a shorter path
        # Shift in 4 directions and take min
        up = jnp.roll(dist, -1, axis=0).at[-1, :].set(max_dist)
        down = jnp.roll(dist, 1, axis=0).at[0, :].set(max_dist)
        left = jnp.roll(dist, -1, axis=1).at[:, -1].set(max_dist)
        right = jnp.roll(dist, 1, axis=1).at[:, 0].set(max_dist)

        neighbor_min = jnp.minimum(jnp.minimum(up, down), jnp.minimum(left, right))
        # New distance: min of current or (neighbor + 1), but only for passable cells
        passable = ~wall_map.astype(jnp.bool_)
        new_dist = jnp.where(passable, jnp.minimum(dist, neighbor_min + 1.0), max_dist)
        # Agent position always has distance 0
        new_dist = new_dist.at[agent_y, agent_x].set(0.0)
        return new_dist

    dist = jax.lax.fori_loop(0, max_iters, relax_step, dist)
    goal_y, goal_x = goal_pos[1], goal_pos[0]
    path_len = dist[goal_y, goal_x]
    # Normalize by max possible distance
    max_possible = jnp.float32(env_height + env_width)
    return jnp.clip(path_len / max_possible, 0.0, 1.0)


def compute_tier_novelty(
    curriculum_state: CurriculumState,
    current_difficulty: chex.Array,
) -> float:
    """Min Euclidean distance in (density, path_len, goal_dist) space to recent levels.

    Args:
        curriculum_state: Current curriculum state with history buffers
        current_difficulty: (3,) array of [wall_density, path_length, goal_distance]

    Returns:
        Novelty score (min distance to recent levels)
    """
    history_length = curriculum_state.recent_wall_densities.shape[0]
    recent = jnp.stack([
        curriculum_state.recent_wall_densities,
        curriculum_state.recent_path_lengths,
        curriculum_state.recent_goal_distances,
    ], axis=-1)  # (history_length, 3)
    dists = jnp.sqrt(((current_difficulty[None, :] - recent) ** 2).sum(axis=-1))
    # Mask invalid entries (unfilled buffer positions)
    valid_count = jnp.where(
        curriculum_state.history_filled,
        history_length,
        curriculum_state.head_pointer,
    )
    valid_mask = jnp.arange(history_length) < valid_count
    return jnp.where(valid_mask, dists, jnp.float32(jnp.inf)).min()


def compute_tier_unusualness(
    curriculum_state: CurriculumState,
    current_difficulty: chex.Array,
) -> float:
    """Mean z-score across difficulty features.

    Args:
        curriculum_state: Current curriculum state
        current_difficulty: (3,) array of [wall_density, path_length, goal_distance]

    Returns:
        Unusualness score (mean absolute z-score)
    """
    means = jnp.array([
        curriculum_state.recent_wall_densities.mean(),
        curriculum_state.recent_path_lengths.mean(),
        curriculum_state.recent_goal_distances.mean(),
    ])
    stds = jnp.array([
        curriculum_state.recent_wall_densities.std(),
        curriculum_state.recent_path_lengths.std(),
        curriculum_state.recent_goal_distances.std(),
    ])
    z_scores = jnp.abs(current_difficulty - means) / (stds + 1e-8)
    return z_scores.mean()


def compute_drift_direction(
    curriculum_state: CurriculumState,
    k: int = 16,
) -> chex.Array:
    """Diff of rolling means of difficulty stats.

    Returns (3,) vector: mean(last_k) - mean(prev_k) for
    [wall_density, path_length, goal_distance].
    """
    history_length = curriculum_state.recent_wall_densities.shape[0]
    ptr = curriculum_state.head_pointer

    # Build masks using vectorized index comparison
    indices = jnp.arange(history_length)

    # Last K entries: positions ptr-1, ptr-2, ..., ptr-K (circular)
    last_k_positions = (ptr - 1 - jnp.arange(k)) % history_length
    last_k_mask = jnp.isin(indices, last_k_positions)

    # Prev K entries: positions ptr-K-1, ptr-K-2, ..., ptr-2K (circular)
    prev_k_positions = (ptr - 1 - k - jnp.arange(k)) % history_length
    prev_k_mask = jnp.isin(indices, prev_k_positions)

    # Compute means for each difficulty feature
    def masked_mean(arr, mask):
        safe_count = jnp.maximum(mask.astype(jnp.float32).sum(), 1.0)
        return (arr * mask.astype(jnp.float32)).sum() / safe_count

    last_density = masked_mean(curriculum_state.recent_wall_densities, last_k_mask)
    prev_density = masked_mean(curriculum_state.recent_wall_densities, prev_k_mask)
    last_path = masked_mean(curriculum_state.recent_path_lengths, last_k_mask)
    prev_path = masked_mean(curriculum_state.recent_path_lengths, prev_k_mask)
    last_goal = masked_mean(curriculum_state.recent_goal_distances, last_k_mask)
    prev_goal = masked_mean(curriculum_state.recent_goal_distances, prev_k_mask)

    return jnp.array([
        last_density - prev_density,
        last_path - prev_path,
        last_goal - prev_goal,
    ])


def compute_tier_targets(
    curriculum_state: CurriculumState,
    level,
    episode_return: float,
    score: float,
    branch: int,
    is_paired: bool = False,
    ant_return: Optional[float] = None,
    pro_return: Optional[float] = None,
    adversary_entropy: Optional[float] = None,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> dict:
    """Assemble ground truth dict for all tier targets. Fully JIT-compatible.

    Args:
        curriculum_state: Current curriculum state
        level: The level dataclass
        episode_return: Agent's actual episode return
        score: Level sampler score
        branch: Branch index (0=DR, 1=replay, 2=mutate)
        is_paired: Whether using PAIRED training
        ant_return: Antagonist return (PAIRED only)
        pro_return: Protagonist return (PAIRED only)
        adversary_entropy: Adversary policy entropy (PAIRED only)
        env_height: Environment height
        env_width: Environment width

    Returns:
        Dict with all tier target values
    """
    # Tier 1: Curriculum Dynamics
    wall_density = level.wall_map.sum() / (env_height * env_width)
    goal_distance = jnp.abs(level.goal_pos.astype(jnp.float32) - level.agent_pos.astype(jnp.float32)).sum() / (env_height + env_width)
    path_length = compute_shortest_path_length(
        level.wall_map, level.agent_pos, level.goal_pos,
        env_height=env_height, env_width=env_width,
    )
    difficulty = jnp.array([wall_density, path_length, goal_distance])

    # Regret
    ant_return_safe = ant_return if ant_return is not None else jnp.float32(0.0)
    pro_return_safe = pro_return if pro_return is not None else jnp.float32(0.0)
    regret = jnp.where(
        jnp.bool_(is_paired),
        ant_return_safe - pro_return_safe,
        -episode_return,  # Approximate: negative return as proxy for non-PAIRED
    )

    # Tier 2: Agent-Curriculum Interaction
    novelty = compute_tier_novelty(curriculum_state, difficulty)
    unusualness = compute_tier_unusualness(curriculum_state, difficulty)

    # Regret source (PAIRED only): pro_weakness / (pro_weakness + ant_strength + eps)
    # pro_weakness = max(0, -pro_return), ant_strength = max(0, ant_return)
    pro_weakness = jnp.maximum(-pro_return_safe, 0.0)
    ant_strength = jnp.maximum(ant_return_safe, 0.0)
    regret_source = jnp.where(
        jnp.bool_(is_paired),
        pro_weakness / (pro_weakness + ant_strength + 1e-8),
        jnp.float32(0.0),
    )

    # Tier 3: Meta-Curriculum
    drift = compute_drift_direction(curriculum_state)
    adv_entropy = jnp.where(
        jnp.bool_(is_paired),
        adversary_entropy if adversary_entropy is not None else jnp.float32(0.0),
        jnp.float32(0.0),
    )

    return {
        't1_regret': regret,
        't1_difficulty': difficulty,
        't1_branch': jnp.int32(branch),
        't1_score': jnp.float32(score),
        't2_return': jnp.float32(episode_return),
        't2_regret_source': regret_source,
        't2_novelty': novelty,
        't2_unusualness': unusualness,
        't3_drift': drift,
        't3_adv_entropy': adv_entropy,
        # Also return computed values for curriculum state update
        '_path_length': path_length,
        '_goal_distance': goal_distance,
    }


def compute_distribution_divergence(
    predictions: dict,
    actual_levels_batch,  # Batch of Level dataclasses
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> dict:
    """
    Compute KL and JS divergence between predicted and actual distributions.

    This requires a batch of actual levels to estimate the empirical distribution.

    Args:
        predictions: Dict from CurriculumPredictionHead
        actual_levels_batch: Batch of Levels to estimate actual distribution
        env_height: Environment height
        env_width: Environment width

    Returns:
        Dict with divergence metrics
    """
    batch_size = actual_levels_batch.wall_map.shape[0]
    grid_size = env_height * env_width

    # Compute empirical goal distribution from batch
    goal_indices = actual_levels_batch.goal_pos[:, 1] * env_width + actual_levels_batch.goal_pos[:, 0]
    empirical_goal_counts = jnp.zeros(grid_size).at[goal_indices].add(1.0)
    empirical_goal_dist = empirical_goal_counts / batch_size + 1e-10  # Add small epsilon

    # Predicted goal distribution
    predicted_goal_dist = jax.nn.softmax(predictions['goal_logits']) + 1e-10

    # KL divergence: KL(empirical || predicted)
    goal_kl = jnp.sum(empirical_goal_dist * jnp.log(empirical_goal_dist / predicted_goal_dist))

    # JS divergence: 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
    m_dist = 0.5 * (empirical_goal_dist + predicted_goal_dist)
    goal_js = 0.5 * jnp.sum(empirical_goal_dist * jnp.log(empirical_goal_dist / m_dist))
    goal_js += 0.5 * jnp.sum(predicted_goal_dist * jnp.log(predicted_goal_dist / m_dist))

    # Compute empirical wall density
    empirical_wall_density = actual_levels_batch.wall_map.mean()
    predicted_wall_density = jax.nn.sigmoid(predictions['wall_logits']).mean()

    return {
        'goal_kl': goal_kl,
        'goal_js': goal_js,
        'empirical_wall_density': empirical_wall_density,
        'predicted_wall_density': predicted_wall_density,
        'wall_density_error': jnp.abs(empirical_wall_density - predicted_wall_density),
    }
