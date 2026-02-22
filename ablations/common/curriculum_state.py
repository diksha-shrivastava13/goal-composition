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

    # Combine into feature vector
    features = jnp.concatenate([
        jnp.array([normalized_step, replay_fraction, buffer_fill]),
        jnp.array([normalized_mean_score, normalized_max_score]),
        jnp.array([mean_wall_density, std_wall_density, mean_score]),
        branch_counts,
        curriculum_state.recent_wall_densities,  # Full history
        curriculum_state.recent_scores,          # Full history
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
