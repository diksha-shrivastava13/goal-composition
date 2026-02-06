import os
import json
import time
from typing import Sequence, Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import core, struct
from flax.training.train_state import TrainState as BaseTrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import orbax.checkpoint as ocp

import wandb
import chex
from enum import IntEnum
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from jaxued.linen import ResetRNN
from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level, make_level_generator, make_level_mutator_minimax   #TODO: the minimax mutator will still cause some more predictability in pattern, as it only flips certain things
from jaxued.level_sampler import LevelSampler
from jaxued.utils import max_mc, positive_value_loss, compute_max_returns
from jaxued.wrappers import AutoReplayWrapper


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_ENV_HEIGHT = 13
DEFAULT_ENV_WIDTH = 13
DEFAULT_CALIBRATION_BINS = 10


class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1


# =============================================================================
# PROBE-SPECIFIC STATE TRACKING
# =============================================================================

# Default hidden state size (LSTM c + h, each 256)
DEFAULT_HSTATE_DIM = 512


@struct.dataclass
class ProbeTrackingState:
    """
    Tracks probe metrics over time for computing novelty, learnability,
    calibration, and correlation with agent performance.

    This is specific to the probe approach: testing what the agent's
    hidden state has implicitly learned about curriculum dynamics.

    Bug Fix #2: Now tracks actual training steps for proper learnability.
    Bug Fix #3: Hidden state dimension is configurable via hstate_dim field.
    """
    # Rolling buffer of probe losses with associated training steps
    loss_history: chex.Array              # (buffer_size,)
    training_step_history: chex.Array     # (buffer_size,) - actual training step for each entry
    # Per-branch loss history for comparison
    branch_loss_history: chex.Array       # (3, buffer_size) - [random, replay, mutate]
    branch_step_history: chex.Array       # (3, buffer_size) - training step for each branch entry
    # Agent performance correlation tracking
    agent_returns_history: chex.Array     # (buffer_size,) - returns when probe made prediction
    probe_accuracy_history: chex.Array    # (buffer_size,) - probe accuracy at same time
    # Hidden state statistics per branch (dimension configurable)
    hstate_mean_by_branch: chex.Array     # (3, hstate_dim) - mean hidden state per branch
    hstate_var_by_branch: chex.Array      # (3, hstate_dim) - variance of hidden state per branch
    hstate_count_by_branch: chex.Array    # (3,) - count per branch for running stats
    # Sample counts per branch (for reporting)
    branch_sample_counts: chex.Array      # (3,) - total samples per branch
    # Buffer management
    buffer_ptr: int
    total_samples: int
    current_training_step: int            # Actual training step counter
    # Per-branch pointers
    branch_ptrs: chex.Array               # (3,)
    # Configuration
    hstate_dim: int                       # Hidden state dimension (Bug Fix #3)


def create_probe_tracking_state(
    buffer_size: int = 500,
    hstate_dim: int = DEFAULT_HSTATE_DIM,
) -> ProbeTrackingState:
    """Initialize probe tracking state with configurable hidden state dimension."""
    return ProbeTrackingState(
        loss_history=jnp.zeros(buffer_size),
        training_step_history=jnp.zeros(buffer_size, dtype=jnp.int32),
        branch_loss_history=jnp.zeros((3, buffer_size)),
        branch_step_history=jnp.zeros((3, buffer_size), dtype=jnp.int32),
        agent_returns_history=jnp.zeros(buffer_size),
        probe_accuracy_history=jnp.zeros(buffer_size),
        hstate_mean_by_branch=jnp.zeros((3, hstate_dim)),
        hstate_var_by_branch=jnp.zeros((3, hstate_dim)),
        hstate_count_by_branch=jnp.zeros(3),
        branch_sample_counts=jnp.zeros(3, dtype=jnp.int32),
        buffer_ptr=0,
        total_samples=0,
        current_training_step=0,
        branch_ptrs=jnp.zeros(3, dtype=jnp.int32),
        hstate_dim=hstate_dim,
    )


def update_probe_tracking_state(
    state: ProbeTrackingState,
    probe_loss: jnp.ndarray,  # Bug Fix #4: Use jnp types, not Python float
    branch: jnp.ndarray,
    agent_return: jnp.ndarray,
    probe_accuracy: jnp.ndarray,
    hstate_flat: chex.Array,  # (hstate_dim,)
    training_step: jnp.ndarray,  # Actual training step
) -> ProbeTrackingState:
    """Update probe tracking with new observation."""
    buffer_size = state.loss_history.shape[0]
    ptr = state.buffer_ptr % buffer_size

    # Update general loss history with training step
    new_loss_history = state.loss_history.at[ptr].set(probe_loss)
    new_step_history = state.training_step_history.at[ptr].set(training_step)
    new_agent_returns = state.agent_returns_history.at[ptr].set(agent_return)
    new_probe_accuracy = state.probe_accuracy_history.at[ptr].set(probe_accuracy)

    # Update per-branch loss history with step
    branch_ptr = state.branch_ptrs[branch] % buffer_size
    new_branch_loss = state.branch_loss_history.at[branch, branch_ptr].set(probe_loss)
    new_branch_step = state.branch_step_history.at[branch, branch_ptr].set(training_step)
    new_branch_ptrs = state.branch_ptrs.at[branch].set(branch_ptr + 1)

    # Update sample counts per branch
    new_branch_counts = state.branch_sample_counts.at[branch].add(1)

    # Update running hidden state statistics (Welford's algorithm)
    count = state.hstate_count_by_branch[branch]
    new_count = count + 1
    delta = hstate_flat - state.hstate_mean_by_branch[branch]
    new_mean = state.hstate_mean_by_branch.at[branch].set(
        state.hstate_mean_by_branch[branch] + delta / new_count
    )
    delta2 = hstate_flat - new_mean[branch]
    new_var = state.hstate_var_by_branch.at[branch].set(
        state.hstate_var_by_branch[branch] + delta * delta2
    )
    new_hstate_count = state.hstate_count_by_branch.at[branch].set(new_count)

    return state.replace(
        loss_history=new_loss_history,
        training_step_history=new_step_history,
        branch_loss_history=new_branch_loss,
        branch_step_history=new_branch_step,
        agent_returns_history=new_agent_returns,
        probe_accuracy_history=new_probe_accuracy,
        hstate_mean_by_branch=new_mean,
        hstate_var_by_branch=new_var,
        hstate_count_by_branch=new_hstate_count,
        branch_sample_counts=new_branch_counts,
        buffer_ptr=(ptr + 1) % buffer_size,
        total_samples=state.total_samples + 1,
        current_training_step=training_step + 1,
        branch_ptrs=new_branch_ptrs,
    )


# =============================================================================
# NOVELTY AND LEARNABILITY FOR PROBE (Fixed to use actual training steps)
# =============================================================================

@struct.dataclass
class ParetoHistoryState:
    """
    Tracks novelty/learnability history at eval checkpoints for Pareto visualization.

    Fix: Uses actual training steps, not buffer indices.
    """
    novelty_history: chex.Array           # (max_checkpoints,)
    learnability_history: chex.Array      # (max_checkpoints,)
    training_steps: chex.Array            # (max_checkpoints,) - actual step numbers
    checkpoint_ptr: int
    num_checkpoints: int


def create_pareto_history_state(max_checkpoints: int = 200) -> ParetoHistoryState:
    """Create state for tracking Pareto history at eval checkpoints."""
    return ParetoHistoryState(
        novelty_history=jnp.zeros(max_checkpoints),
        learnability_history=jnp.zeros(max_checkpoints),
        training_steps=jnp.zeros(max_checkpoints, dtype=jnp.int32),
        checkpoint_ptr=0,
        num_checkpoints=0,
    )


def update_pareto_history(
    state: ParetoHistoryState,
    novelty: float,
    learnability: float,
    training_step: int,
) -> ParetoHistoryState:
    """Update Pareto history at eval checkpoint."""
    max_checkpoints = state.novelty_history.shape[0]
    ptr = state.checkpoint_ptr % max_checkpoints

    return state.replace(
        novelty_history=state.novelty_history.at[ptr].set(novelty),
        learnability_history=state.learnability_history.at[ptr].set(learnability),
        training_steps=state.training_steps.at[ptr].set(training_step),
        checkpoint_ptr=ptr + 1,
        num_checkpoints=jnp.minimum(state.num_checkpoints + 1, max_checkpoints),
    )


def _compute_learnability_jit(
    probe_tracking: ProbeTrackingState,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JIT-compatible learnability computation returning JAX arrays.

    Returns: (learnability, early_loss, late_loss, early_count, late_count, median_step)
    """
    buffer_size = probe_tracking.loss_history.shape[0]
    valid_samples = jnp.minimum(probe_tracking.total_samples, buffer_size)

    steps = probe_tracking.training_step_history
    losses = probe_tracking.loss_history

    valid_mask = jnp.arange(buffer_size) < valid_samples
    max_step = probe_tracking.current_training_step
    median_step = max_step // 2

    early_mask = valid_mask & (steps < median_step) & (steps > 0)
    late_mask = valid_mask & (steps >= median_step)

    early_count = early_mask.sum()
    late_count = late_mask.sum()

    early_loss = jnp.where(
        early_count > 0,
        (losses * early_mask).sum() / jnp.maximum(early_count, 1),
        0.0
    )
    late_loss = jnp.where(
        late_count > 0,
        (losses * late_mask).sum() / jnp.maximum(late_count, 1),
        0.0
    )

    learnability = early_loss - late_loss
    return learnability, early_loss, late_loss, early_count, late_count, median_step


def compute_learnability(
    probe_tracking: ProbeTrackingState,
) -> Tuple[float, dict]:
    """
    Compute learnability: does the probe learn to predict better over time?

    For probe: High learnability = agent's hidden state encodes more curriculum
    information as training progresses.

    Bug Fix #2: Uses actual training steps, not buffer indices.
    Bug Fix #4: JIT-compatible - returns JAX scalars, details dict built outside JIT.

    Returns:
        learnability: Scalar (positive = improving)
        details: Dict with trend info
    """
    learnability, early_loss, late_loss, early_count, late_count, median_step = \
        _compute_learnability_jit(probe_tracking)

    details = {
        "early_loss": float(early_loss),
        "late_loss": float(late_loss),
        "improvement": float(learnability),
        "early_samples": int(early_count),
        "late_samples": int(late_count),
        "median_step": int(median_step),
    }

    return float(learnability), details


def _compute_novelty_jit(
    probe_tracking: ProbeTrackingState,
    window_steps: int = 1000,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JIT-compatible novelty computation returning JAX arrays.

    Returns: (novelty, instantaneous_novelty, y_mean, current_loss, count, window_start)
    """
    buffer_size = probe_tracking.loss_history.shape[0]
    valid_samples = jnp.minimum(probe_tracking.total_samples, buffer_size)

    steps = probe_tracking.training_step_history
    losses = probe_tracking.loss_history
    max_step = probe_tracking.current_training_step

    window_start = jnp.maximum(0, max_step - window_steps)
    valid_mask = jnp.arange(buffer_size) < valid_samples
    mask = valid_mask & (steps >= window_start) & (steps > 0)

    x = steps.astype(jnp.float32)
    y = losses

    x_masked = jnp.where(mask, x, 0.0)
    y_masked = jnp.where(mask, y, 0.0)
    count = mask.sum()

    x_mean = x_masked.sum() / jnp.maximum(count, 1)
    y_mean = y_masked.sum() / jnp.maximum(count, 1)

    numerator = jnp.sum((x - x_mean) * (y - y_mean) * mask)
    denominator = jnp.sum((x - x_mean) ** 2 * mask) + 1e-8
    slope = numerator / denominator

    novelty = slope

    most_recent_idx = jnp.argmax(steps * valid_mask.astype(jnp.float32))
    current_loss = losses[most_recent_idx]
    instantaneous_novelty = current_loss - y_mean

    return novelty, instantaneous_novelty, y_mean, current_loss, count, window_start


def compute_novelty(
    probe_tracking: ProbeTrackingState,
    window_steps: int = 1000,
) -> Tuple[float, dict]:
    """
    Compute novelty: is prediction error increasing (curriculum getting harder)?

    For probe: High novelty = curriculum is presenting increasingly novel
    levels that even the agent's learned representations struggle with.

    Bug Fix #2: Uses actual training steps for proper trend computation.
    Bug Fix #4: JIT-compatible - core computation uses JAX arrays.

    Returns:
        novelty: Scalar (positive = increasing difficulty)
        details: Dict with trend info
    """
    novelty, instantaneous_novelty, y_mean, current_loss, count, window_start = \
        _compute_novelty_jit(probe_tracking, window_steps)

    details = {
        "novelty_slope": float(novelty),
        "instantaneous_novelty": float(instantaneous_novelty),
        "recent_mean_loss": float(y_mean),
        "current_loss": float(current_loss),
        "window_samples": int(count),
        "window_start_step": int(window_start),
    }

    return float(novelty), details


def compute_openendedness_score(novelty: float, learnability: float) -> Tuple[float, str]:
    """
    Compute open-endedness score from novelty and learnability.

    Open-ended = both novelty and learnability are positive and balanced.
    """
    novelty_norm = 2.0 / (1.0 + np.exp(-novelty)) - 1.0
    learnability_norm = 2.0 / (1.0 + np.exp(-learnability)) - 1.0

    both_positive = (novelty_norm > 0) and (learnability_norm > 0)
    score = np.sqrt(novelty_norm * learnability_norm) if both_positive else 0.0

    if novelty > 0.1 and learnability > 0.1:
        regime = "open-ended"
    elif novelty > 0.1 and learnability <= 0.1:
        regime = "chaotic"
    elif novelty <= 0.1 and learnability > 0.1:
        regime = "converging"
    else:
        regime = "stagnant"

    return float(score), regime


# =============================================================================
# RANDOM BASELINES
# =============================================================================

def compute_random_baselines(
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> dict:
    """
    Compute expected accuracy/loss of random (untrained) predictions.

    This provides the baseline against which probe performance should be measured.
    Information gain = probe_metric - random_baseline
    """
    grid_size = env_height * env_width

    return {
        # Accuracy baselines
        'wall_accuracy': 0.5,  # Binary prediction, 50% by chance
        'goal_top1': 1.0 / grid_size,  # ~0.59% for 13x13
        'agent_pos_top1': 1.0 / grid_size,  # ~0.59% for 13x13
        'agent_dir': 0.25,  # 4 directions, 25% by chance
        # Loss baselines (cross-entropy of uniform prediction)
        'wall_loss': 0.693,  # -log(0.5) = ln(2)
        'goal_loss': np.log(grid_size),  # ~5.13 for 13x13
        'agent_pos_loss': np.log(grid_size),  # ~5.13 for 13x13
        'agent_dir_loss': np.log(4),  # ~1.39
        'total_loss': 0.693 + 2 * np.log(grid_size) + np.log(4),  # Sum of above
    }


# =============================================================================
# CALIBRATION METRICS (JIT-compatible)
# =============================================================================

def compute_calibration_metrics(
    predictions: dict,
    actual_level,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> dict:
    """
    Compute calibration metrics (Expected Calibration Error) for probe predictions.

    ECE measures how well-calibrated the predicted probabilities are.
    This version is JIT-compatible using vectorized operations.
    """
    # Wall calibration: ECE for binary wall predictions
    wall_probs = jax.nn.sigmoid(predictions['wall_logits']).flatten()
    wall_targets = actual_level.wall_map.astype(jnp.float32).flatten()
    total_samples = wall_probs.shape[0]

    # Vectorized ECE computation (Bug Fix #1: no Python for loop)
    bin_boundaries = jnp.linspace(0, 1, n_bins + 1)
    bin_indices = jnp.digitize(wall_probs, bin_boundaries) - 1
    bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)

    def compute_bin_ece(bin_idx):
        mask = bin_indices == bin_idx
        bin_size = mask.sum()
        # Use jnp.where for safe mean computation
        bin_confidence = jnp.where(
            bin_size > 0,
            jnp.sum(wall_probs * mask) / jnp.maximum(bin_size, 1),
            0.0
        )
        bin_accuracy = jnp.where(
            bin_size > 0,
            jnp.sum(wall_targets * mask) / jnp.maximum(bin_size, 1),
            0.0
        )
        return jnp.where(
            bin_size > 0,
            bin_size * jnp.abs(bin_confidence - bin_accuracy),
            0.0
        )

    # Vectorize over bins
    bin_eces = jax.vmap(compute_bin_ece)(jnp.arange(n_bins))
    wall_ece = bin_eces.sum() / total_samples

    # Position predictions: probability assigned to actual position
    goal_probs = jax.nn.softmax(predictions['goal_logits'])
    goal_idx = actual_level.goal_pos[1] * env_width + actual_level.goal_pos[0]
    goal_prob_at_actual = goal_probs[goal_idx]

    agent_pos_probs = jax.nn.softmax(predictions['agent_pos_logits'])
    agent_pos_idx = actual_level.agent_pos[1] * env_width + actual_level.agent_pos[0]
    agent_pos_prob_at_actual = agent_pos_probs[agent_pos_idx]

    agent_dir_probs = jax.nn.softmax(predictions['agent_dir_logits'])
    agent_dir_prob_at_actual = agent_dir_probs[actual_level.agent_dir]

    # Compute accuracy metrics
    wall_accuracy = ((wall_probs > 0.5) == wall_targets).mean()
    goal_top1_correct = (goal_probs.argmax() == goal_idx).astype(jnp.float32)
    agent_pos_top1_correct = (agent_pos_probs.argmax() == agent_pos_idx).astype(jnp.float32)
    agent_dir_correct = (agent_dir_probs.argmax() == actual_level.agent_dir).astype(jnp.float32)

    return {
        'wall_ece': wall_ece,
        'wall_accuracy': wall_accuracy,
        'goal_prob_at_actual': goal_prob_at_actual,
        'goal_top1_correct': goal_top1_correct,
        'agent_pos_prob_at_actual': agent_pos_prob_at_actual,
        'agent_pos_top1_correct': agent_pos_top1_correct,
        'agent_dir_prob_at_actual': agent_dir_prob_at_actual,
        'agent_dir_correct': agent_dir_correct,
    }


def compute_distributional_calibration_metrics(
    predictions_batch: dict,
    actual_levels_batch,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> dict:
    """
    Compute distributional calibration metrics comparing batch-averaged predictions
    to empirical level distributions.

    This is appropriate for all branches because it doesn't assume 1-to-1 correspondence
    between predictions and actuals. Instead, it asks: "Does the mean predicted
    distribution match the empirical distribution of levels in this batch?"

    Args:
        predictions_batch: Batched predictions from probe (batch_size, ...)
        actual_levels_batch: Batch of Level dataclasses
        env_height: Environment height
        env_width: Environment width

    Returns:
        Dictionary with distributional calibration metrics
    """
    batch_size = actual_levels_batch.wall_map.shape[0]
    grid_size = env_height * env_width

    # ===== WALL DISTRIBUTION CALIBRATION =====
    # Mean predicted wall probability per cell
    mean_wall_probs = jax.nn.sigmoid(predictions_batch['wall_logits']).mean(axis=0)  # (H, W)
    # Empirical wall frequency per cell
    empirical_wall_freq = actual_levels_batch.wall_map.astype(jnp.float32).mean(axis=0)  # (H, W)
    # MSE between mean prediction and empirical frequency
    wall_dist_calibration = jnp.mean((mean_wall_probs - empirical_wall_freq) ** 2)
    # Also compute cell-wise accuracy: how often does mean prediction > 0.5 match majority
    mean_wall_binary = (mean_wall_probs > 0.5).astype(jnp.float32)
    empirical_wall_majority = (empirical_wall_freq > 0.5).astype(jnp.float32)
    wall_dist_accuracy = (mean_wall_binary == empirical_wall_majority).mean()

    # ===== GOAL POSITION DISTRIBUTION CALIBRATION =====
    # Mean predicted goal distribution
    mean_goal_probs = jax.nn.softmax(predictions_batch['goal_logits']).mean(axis=0)  # (H*W,)
    # Empirical goal distribution
    goal_indices = actual_levels_batch.goal_pos[:, 1] * env_width + actual_levels_batch.goal_pos[:, 0]
    empirical_goal_counts = jnp.zeros(grid_size).at[goal_indices].add(1.0)
    empirical_goal_dist = empirical_goal_counts / batch_size
    # Cross-entropy: -sum(empirical * log(predicted))
    goal_dist_calibration = -jnp.sum(empirical_goal_dist * jnp.log(mean_goal_probs + 1e-10))
    # Top-1 match: does mode of prediction match mode of empirical?
    goal_dist_mode_match = (mean_goal_probs.argmax() == empirical_goal_dist.argmax()).astype(jnp.float32)

    # ===== AGENT POSITION DISTRIBUTION CALIBRATION =====
    mean_agent_pos_probs = jax.nn.softmax(predictions_batch['agent_pos_logits']).mean(axis=0)
    agent_pos_indices = actual_levels_batch.agent_pos[:, 1] * env_width + actual_levels_batch.agent_pos[:, 0]
    empirical_agent_pos_counts = jnp.zeros(grid_size).at[agent_pos_indices].add(1.0)
    empirical_agent_pos_dist = empirical_agent_pos_counts / batch_size
    agent_pos_dist_calibration = -jnp.sum(empirical_agent_pos_dist * jnp.log(mean_agent_pos_probs + 1e-10))
    agent_pos_dist_mode_match = (mean_agent_pos_probs.argmax() == empirical_agent_pos_dist.argmax()).astype(jnp.float32)

    # ===== AGENT DIRECTION DISTRIBUTION CALIBRATION =====
    mean_agent_dir_probs = jax.nn.softmax(predictions_batch['agent_dir_logits']).mean(axis=0)  # (4,)
    # Empirical direction distribution
    empirical_dir_counts = jnp.zeros(4).at[actual_levels_batch.agent_dir].add(1.0)
    empirical_dir_dist = empirical_dir_counts / batch_size
    agent_dir_dist_calibration = -jnp.sum(empirical_dir_dist * jnp.log(mean_agent_dir_probs + 1e-10))
    agent_dir_dist_mode_match = (mean_agent_dir_probs.argmax() == empirical_dir_dist.argmax()).astype(jnp.float32)

    return {
        # Calibration errors (lower is better)
        'wall_dist_calibration': wall_dist_calibration,
        'goal_dist_calibration': goal_dist_calibration,
        'agent_pos_dist_calibration': agent_pos_dist_calibration,
        'agent_dir_dist_calibration': agent_dir_dist_calibration,
        # Accuracy metrics (higher is better)
        'wall_dist_accuracy': wall_dist_accuracy,
        'goal_dist_mode_match': goal_dist_mode_match,
        'agent_pos_dist_mode_match': agent_pos_dist_mode_match,
        'agent_dir_dist_mode_match': agent_dir_dist_mode_match,
    }


def compute_per_instance_calibration_batch(
    predictions_batch: dict,
    actual_levels_batch,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> dict:
    """
    Compute per-instance calibration metrics with direct index pairing.

    This is ONLY meaningful when there's 1-to-1 correspondence between
    predictions[i] and actuals[i], which occurs in Replay->Mutate transition.

    For other transitions (DR->R, M->DR), these metrics are computed but
    represent noise since the pairing is arbitrary.

    Args:
        predictions_batch: Batched predictions from probe (batch_size, ...)
        actual_levels_batch: Batch of Level dataclasses
        env_height: Environment height
        env_width: Environment width

    Returns:
        Dictionary with per-instance calibration metrics averaged over batch
    """
    batch_size = actual_levels_batch.wall_map.shape[0]
    grid_size = env_height * env_width

    # ===== WALL ACCURACY (per instance, averaged) =====
    wall_probs = jax.nn.sigmoid(predictions_batch['wall_logits'])  # (B, H, W)
    wall_targets = actual_levels_batch.wall_map.astype(jnp.float32)  # (B, H, W)
    # Per-instance accuracy
    wall_correct = ((wall_probs > 0.5) == wall_targets).astype(jnp.float32)
    wall_accuracy_per_instance = wall_correct.mean(axis=(1, 2))  # (B,)
    wall_accuracy = wall_accuracy_per_instance.mean()

    # ===== GOAL POSITION ACCURACY (per instance, averaged) =====
    goal_probs = jax.nn.softmax(predictions_batch['goal_logits'])  # (B, H*W)
    goal_indices = actual_levels_batch.goal_pos[:, 1] * env_width + actual_levels_batch.goal_pos[:, 0]
    goal_predicted = goal_probs.argmax(axis=1)  # (B,)
    goal_correct = (goal_predicted == goal_indices).astype(jnp.float32)
    goal_accuracy = goal_correct.mean()
    # Probability assigned to correct position (confidence at actual)
    goal_prob_at_actual = jax.vmap(lambda p, i: p[i])(goal_probs, goal_indices).mean()

    # ===== AGENT POSITION ACCURACY (per instance, averaged) =====
    agent_pos_probs = jax.nn.softmax(predictions_batch['agent_pos_logits'])  # (B, H*W)
    agent_pos_indices = actual_levels_batch.agent_pos[:, 1] * env_width + actual_levels_batch.agent_pos[:, 0]
    agent_pos_predicted = agent_pos_probs.argmax(axis=1)  # (B,)
    agent_pos_correct = (agent_pos_predicted == agent_pos_indices).astype(jnp.float32)
    agent_pos_accuracy = agent_pos_correct.mean()
    agent_pos_prob_at_actual = jax.vmap(lambda p, i: p[i])(agent_pos_probs, agent_pos_indices).mean()

    # ===== AGENT DIRECTION ACCURACY (per instance, averaged) =====
    agent_dir_probs = jax.nn.softmax(predictions_batch['agent_dir_logits'])  # (B, 4)
    agent_dir_predicted = agent_dir_probs.argmax(axis=1)  # (B,)
    agent_dir_correct = (agent_dir_predicted == actual_levels_batch.agent_dir).astype(jnp.float32)
    agent_dir_accuracy = agent_dir_correct.mean()
    agent_dir_prob_at_actual = jax.vmap(lambda p, i: p[i])(agent_dir_probs, actual_levels_batch.agent_dir).mean()

    # Combined accuracy (weighted average matching original compute_calibration_metrics)
    combined_accuracy = (
        wall_accuracy * 0.4 +
        goal_correct.mean() * 0.2 +
        agent_pos_correct.mean() * 0.2 +
        agent_dir_correct.mean() * 0.2
    )

    return {
        # Accuracy metrics (per-instance, batch-averaged)
        'wall_accuracy': wall_accuracy,
        'goal_accuracy': goal_accuracy,
        'agent_pos_accuracy': agent_pos_accuracy,
        'agent_dir_accuracy': agent_dir_accuracy,
        'combined_accuracy': combined_accuracy,
        # Confidence at actual (how much probability mass on correct answer)
        'goal_prob_at_actual': goal_prob_at_actual,
        'agent_pos_prob_at_actual': agent_pos_prob_at_actual,
        'agent_dir_prob_at_actual': agent_dir_prob_at_actual,
    }


# =============================================================================
# DISTRIBUTION DIVERGENCE METRICS
# =============================================================================

def compute_distribution_divergence(
    predictions: dict,
    actual_levels_batch,  # Batch of Level dataclasses
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> dict:
    """
    Compute KL and JS divergence between predicted and empirical distributions.

    This requires a batch of actual levels to estimate the empirical distribution.
    """
    batch_size = actual_levels_batch.wall_map.shape[0]
    grid_size = env_height * env_width

    # Compute empirical goal distribution from batch
    goal_indices = actual_levels_batch.goal_pos[:, 1] * env_width + actual_levels_batch.goal_pos[:, 0]
    goal_counts = jnp.zeros(grid_size).at[goal_indices].add(1)
    empirical_goal_dist = goal_counts / batch_size + 1e-10

    # Predicted goal distribution
    predicted_goal_dist = jax.nn.softmax(predictions['goal_logits']) + 1e-10

    # KL divergence: KL(empirical || predicted)
    goal_kl = jnp.sum(empirical_goal_dist * jnp.log(empirical_goal_dist / predicted_goal_dist))

    # JS divergence: 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
    m_dist = 0.5 * (empirical_goal_dist + predicted_goal_dist)
    goal_js = 0.5 * jnp.sum(empirical_goal_dist * jnp.log(empirical_goal_dist / m_dist))
    goal_js += 0.5 * jnp.sum(predicted_goal_dist * jnp.log(predicted_goal_dist / m_dist))

    # Compute empirical agent position distribution
    agent_pos_indices = actual_levels_batch.agent_pos[:, 1] * env_width + actual_levels_batch.agent_pos[:, 0]
    agent_pos_counts = jnp.zeros(grid_size).at[agent_pos_indices].add(1)
    empirical_agent_pos_dist = agent_pos_counts / batch_size + 1e-10

    predicted_agent_pos_dist = jax.nn.softmax(predictions['agent_pos_logits']) + 1e-10
    agent_pos_kl = jnp.sum(empirical_agent_pos_dist * jnp.log(empirical_agent_pos_dist / predicted_agent_pos_dist))

    # Compute empirical wall density
    empirical_wall_density = actual_levels_batch.wall_map.mean()
    predicted_wall_density = jax.nn.sigmoid(predictions['wall_logits']).mean()

    return {
        'goal_kl': goal_kl,
        'goal_js': goal_js,
        'agent_pos_kl': agent_pos_kl,
        'empirical_wall_density': empirical_wall_density,
        'predicted_wall_density': predicted_wall_density,
        'wall_density_error': jnp.abs(empirical_wall_density - predicted_wall_density),
    }


def compute_probe_correlation_with_performance(
    probe_tracking: ProbeTrackingState,
) -> dict:
    """
    Compute correlation between probe accuracy and agent performance.

    Key insight: If probe accuracy correlates with agent returns, it suggests
    the agent's representations that help prediction also help performance.
    """
    buffer_size = probe_tracking.loss_history.shape[0]
    valid_samples = jnp.minimum(probe_tracking.total_samples, buffer_size)

    mask = jnp.arange(buffer_size) < valid_samples

    # Compute Pearson correlation
    x = probe_tracking.probe_accuracy_history
    y = probe_tracking.agent_returns_history

    x_mean = (x * mask).sum() / jnp.maximum(mask.sum(), 1)
    y_mean = (y * mask).sum() / jnp.maximum(mask.sum(), 1)

    numerator = ((x - x_mean) * (y - y_mean) * mask).sum()
    x_std = jnp.sqrt(((x - x_mean) ** 2 * mask).sum() + 1e-8)
    y_std = jnp.sqrt(((y - y_mean) ** 2 * mask).sum() + 1e-8)

    correlation = numerator / (x_std * y_std + 1e-8)

    return {
        'probe_return_correlation': float(correlation),
        'mean_probe_accuracy': float(x_mean),
        'mean_agent_return': float(y_mean),
    }


# =============================================================================
# HIDDEN STATE ANALYSIS
# =============================================================================

def compute_hidden_state_statistics(
    probe_tracking: ProbeTrackingState,
) -> dict:
    """
    Compute statistics about hidden states per branch.

    This reveals whether the agent develops different representations
    for different curriculum branches.
    """
    branch_names = ['random', 'replay', 'mutate']
    stats = {}

    for i, name in enumerate(branch_names):
        count = probe_tracking.hstate_count_by_branch[i]
        mean = probe_tracking.hstate_mean_by_branch[i]
        # Finalize variance (Welford's algorithm)
        var = jnp.where(count > 1,
                       probe_tracking.hstate_var_by_branch[i] / (count - 1),
                       jnp.zeros_like(probe_tracking.hstate_var_by_branch[i]))

        stats[f'hstate_{name}_mean_norm'] = float(jnp.linalg.norm(mean))
        stats[f'hstate_{name}_mean_std'] = float(mean.std())
        stats[f'hstate_{name}_var_mean'] = float(var.mean())
        stats[f'hstate_{name}_activation_sparsity'] = float((jnp.abs(mean) < 0.1).mean())

    # Compute inter-branch distances
    for i, name_i in enumerate(branch_names):
        for j, name_j in enumerate(branch_names):
            if i < j:
                dist = jnp.linalg.norm(
                    probe_tracking.hstate_mean_by_branch[i] -
                    probe_tracking.hstate_mean_by_branch[j]
                )
                stats[f'hstate_dist_{name_i}_{name_j}'] = float(dist)

    return stats


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_wall_prediction_heatmap(
    predictions: dict,
    actual_level,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> np.ndarray:
    """
    Create visualization comparing predicted vs actual wall maps.

    NOTE: This shows a SINGLE sample. For batch-aware visualization,
    use create_batch_wall_prediction_heatmap().

    Returns RGB image as numpy array.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Predicted wall probabilities
    wall_probs = jax.nn.sigmoid(predictions['wall_logits'])
    im0 = axes[0].imshow(np.array(wall_probs), cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Predicted Wall Probs\n(single sample)')
    plt.colorbar(im0, ax=axes[0])

    # Actual wall map
    im1 = axes[1].imshow(np.array(actual_level.wall_map), cmap='binary', vmin=0, vmax=1)
    axes[1].set_title('Actual Wall Map\n(single sample)')
    plt.colorbar(im1, ax=axes[1])

    # Difference (error)
    error = np.abs(np.array(wall_probs) - np.array(actual_level.wall_map.astype(jnp.float32)))
    im2 = axes[2].imshow(error, cmap='Reds', vmin=0, vmax=1)
    axes[2].set_title('Prediction Error')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_batch_wall_prediction_summary(
    predictions_batch: dict,
    levels_batch,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
    n_samples: int = 4,
) -> np.ndarray:
    """
    Create visualization showing prediction VARIANCE across batch.

    This addresses the averaging problem by showing:
    1. Mean predicted wall density vs mean actual wall density
    2. Prediction variance (uncertainty) per cell
    3. Multiple samples from the batch

    Returns RGB image as numpy array.
    """
    batch_size = predictions_batch['wall_logits'].shape[0]
    n_samples = min(n_samples, batch_size)

    fig = plt.figure(figsize=(16, 8))

    # Top row: Aggregated statistics
    ax1 = fig.add_subplot(2, 4, 1)
    wall_probs_batch = jax.nn.sigmoid(predictions_batch['wall_logits'])  # (batch, H, W)
    mean_pred = np.array(wall_probs_batch.mean(axis=0))
    im1 = ax1.imshow(mean_pred, cmap='hot', vmin=0, vmax=1)
    ax1.set_title(f'Mean Predicted\n(n={batch_size})')
    plt.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(2, 4, 2)
    mean_actual = np.array(levels_batch.wall_map.astype(jnp.float32).mean(axis=0))
    im2 = ax2.imshow(mean_actual, cmap='hot', vmin=0, vmax=1)
    ax2.set_title(f'Mean Actual\n(n={batch_size})')
    plt.colorbar(im2, ax=ax2)

    ax3 = fig.add_subplot(2, 4, 3)
    pred_variance = np.array(wall_probs_batch.var(axis=0))
    im3 = ax3.imshow(pred_variance, cmap='Purples', vmin=0, vmax=0.25)
    ax3.set_title('Prediction Variance\n(uncertainty)')
    plt.colorbar(im3, ax=ax3)

    ax4 = fig.add_subplot(2, 4, 4)
    mean_error = np.abs(mean_pred - mean_actual)
    im4 = ax4.imshow(mean_error, cmap='Reds', vmin=0, vmax=1)
    ax4.set_title('Mean Absolute Error')
    plt.colorbar(im4, ax=ax4)

    # Bottom row: Individual samples
    indices = np.linspace(0, batch_size - 1, n_samples, dtype=int)
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(2, 4, 5 + i)
        pred = np.array(jax.nn.sigmoid(predictions_batch['wall_logits'][idx]))
        actual = np.array(levels_batch.wall_map[idx])

        # Show prediction with actual walls outlined
        ax.imshow(pred, cmap='hot', vmin=0, vmax=1)
        # Overlay actual walls as contours
        ax.contour(actual, levels=[0.5], colors='cyan', linewidths=1)
        acc = ((pred > 0.5) == actual).mean()
        ax.set_title(f'Sample {idx}\nAcc: {acc:.1%}')
        ax.axis('off')

    plt.suptitle('Batch Wall Prediction Summary (cyan contours = actual walls)', y=1.02)
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_batch_position_prediction_summary(
    predictions_batch: dict,
    levels_batch,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> np.ndarray:
    """
    Create visualization showing position prediction across batch.

    Shows:
    1. Mean predicted goal distribution with all actual goals marked
    2. Prediction entropy (how uncertain is the model?)
    3. Mean predicted agent position distribution with all actual positions marked

    Returns RGB image as numpy array.
    """
    batch_size = predictions_batch['goal_logits'].shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Goal predictions
    goal_probs_batch = jax.nn.softmax(predictions_batch['goal_logits'])  # (batch, H*W)
    goal_probs_batch = goal_probs_batch.reshape(batch_size, env_height, env_width)

    # Mean goal distribution
    mean_goal_probs = np.array(goal_probs_batch.mean(axis=0))
    im0 = axes[0, 0].imshow(mean_goal_probs, cmap='hot')
    # Mark all actual goal positions
    for i in range(batch_size):
        gx, gy = levels_batch.goal_pos[i]
        axes[0, 0].scatter(gx, gy, c='lime', s=30, marker='o', alpha=0.5, edgecolors='white', linewidths=0.5)
    axes[0, 0].set_title(f'Mean Goal Prediction\n(dots = {batch_size} actual goals)')
    plt.colorbar(im0, ax=axes[0, 0])

    # Goal entropy (uncertainty)
    goal_entropy = -np.array((goal_probs_batch * jnp.log(goal_probs_batch + 1e-10)).sum(axis=(1, 2)))
    max_entropy = np.log(env_height * env_width)
    axes[0, 1].hist(goal_entropy, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(max_entropy, color='red', linestyle='--', label=f'Max entropy ({max_entropy:.1f})')
    axes[0, 1].set_xlabel('Prediction Entropy')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Goal Prediction Entropy\n(lower = more confident)')
    axes[0, 1].legend()

    # Goal prediction accuracy distribution
    goal_indices = levels_batch.goal_pos[:, 1] * env_width + levels_batch.goal_pos[:, 0]
    goal_probs_flat = jax.nn.softmax(predictions_batch['goal_logits'])
    prob_at_actual = np.array([goal_probs_flat[i, goal_indices[i]] for i in range(batch_size)])
    axes[0, 2].hist(prob_at_actual, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(1.0 / (env_height * env_width), color='red', linestyle='--', label=f'Random ({1/(env_height*env_width):.3f})')
    axes[0, 2].set_xlabel('P(actual goal)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Probability at Actual Goal\n(higher = better)')
    axes[0, 2].legend()

    # Agent position predictions
    agent_probs_batch = jax.nn.softmax(predictions_batch['agent_pos_logits'])
    agent_probs_batch = agent_probs_batch.reshape(batch_size, env_height, env_width)

    mean_agent_probs = np.array(agent_probs_batch.mean(axis=0))
    im1 = axes[1, 0].imshow(mean_agent_probs, cmap='hot')
    for i in range(batch_size):
        ax, ay = levels_batch.agent_pos[i]
        axes[1, 0].scatter(ax, ay, c='cyan', s=30, marker='^', alpha=0.5, edgecolors='white', linewidths=0.5)
    axes[1, 0].set_title(f'Mean Agent Pos Prediction\n(triangles = {batch_size} actual positions)')
    plt.colorbar(im1, ax=axes[1, 0])

    # Agent entropy
    agent_entropy = -np.array((agent_probs_batch * jnp.log(agent_probs_batch + 1e-10)).sum(axis=(1, 2)))
    axes[1, 1].hist(agent_entropy, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(max_entropy, color='red', linestyle='--', label=f'Max entropy ({max_entropy:.1f})')
    axes[1, 1].set_xlabel('Prediction Entropy')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Agent Pos Prediction Entropy')
    axes[1, 1].legend()

    # Direction accuracy
    dir_probs = jax.nn.softmax(predictions_batch['agent_dir_logits'])
    dir_correct = np.array((dir_probs.argmax(axis=1) == levels_batch.agent_dir).astype(float))
    axes[1, 2].bar(['Correct', 'Incorrect'], [dir_correct.mean(), 1 - dir_correct.mean()],
                   color=['green', 'red'], alpha=0.7)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axhline(0.25, color='gray', linestyle='--', label='Random (25%)')
    axes[1, 2].set_title(f'Direction Accuracy: {dir_correct.mean():.1%}')
    axes[1, 2].legend()

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_position_prediction_heatmap(
    predictions: dict,
    actual_level,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> np.ndarray:
    """
    Create visualization comparing predicted vs actual positions.

    Returns RGB image as numpy array.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Goal position distribution
    goal_probs = jax.nn.softmax(predictions['goal_logits']).reshape(env_height, env_width)
    im0 = axes[0].imshow(np.array(goal_probs), cmap='hot')
    actual_goal = actual_level.goal_pos
    axes[0].scatter(actual_goal[0], actual_goal[1], c='green', s=100, marker='*', label='Actual')
    axes[0].set_title('Goal Position (from hidden state)')
    axes[0].legend()
    plt.colorbar(im0, ax=axes[0])

    # Agent position distribution
    agent_probs = jax.nn.softmax(predictions['agent_pos_logits']).reshape(env_height, env_width)
    im1 = axes[1].imshow(np.array(agent_probs), cmap='hot')
    actual_agent = actual_level.agent_pos
    axes[1].scatter(actual_agent[0], actual_agent[1], c='blue', s=100, marker='^', label='Actual')
    axes[1].set_title('Agent Position (from hidden state)')
    axes[1].legend()
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_matched_pairs_visualization(
    predictions_batch: dict,
    levels_batch,
    matched_indices: jnp.ndarray,
    n_pairs: int = 4,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> np.ndarray:
    """
    Create visualization showing greedy-matched prediction/actual pairs.

    Shows n_pairs of (predicted wall map, matched actual wall map) side by side.
    This demonstrates how well predictions match their best-matching actual levels.
    """
    batch_size = min(predictions_batch['wall_logits'].shape[0], levels_batch.wall_map.shape[0])
    n_pairs = min(n_pairs, batch_size)

    fig, axes = plt.subplots(n_pairs, 3, figsize=(10, 3 * n_pairs))
    if n_pairs == 1:
        axes = axes[None, :]

    fig.suptitle('Greedy-Matched Prediction/Actual Pairs', fontsize=12, y=1.02)

    # Get matched levels
    matched_levels = jax.tree_util.tree_map(
        lambda x: x[matched_indices], levels_batch
    )

    for i in range(n_pairs):
        # Predicted walls
        pred_probs = np.array(jax.nn.sigmoid(predictions_batch['wall_logits'][i]))
        axes[i, 0].imshow(pred_probs, cmap='hot', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Pred[{i}]')
        axes[i, 0].axis('off')

        # Matched actual walls
        actual_walls = np.array(matched_levels.wall_map[i])
        axes[i, 1].imshow(actual_walls, cmap='binary', vmin=0, vmax=1)
        matched_idx = int(matched_indices[i])
        axes[i, 1].set_title(f'Matched Actual[{matched_idx}]')
        axes[i, 1].axis('off')

        # Error map
        pred_binary = (pred_probs > 0.5).astype(float)
        error = np.abs(pred_binary - actual_walls)
        accuracy = 1.0 - error.mean()
        axes[i, 2].imshow(error, cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Error (Acc: {accuracy:.1%})')
        axes[i, 2].axis('off')

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_replay_to_mutate_heatmap(
    predictions_batch: dict,
    mutation_levels_batch,
    n_samples: int = 4,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> np.ndarray:
    """
    Create visualization for Replay->Mutate transition (true 1-to-1 correspondence).

    This is the only transition where pred[i] should match actual[i] by index,
    because mutation_level[i] = mutate(replay_level[i]), and pred[i] comes from
    hidden state after experiencing replay_level[i].
    """
    batch_size = min(predictions_batch['wall_logits'].shape[0], mutation_levels_batch.wall_map.shape[0])
    n_samples = min(n_samples, batch_size)

    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    fig.suptitle('Replay→Mutate: Per-Instance Correspondence', fontsize=12, y=1.02)

    total_accuracy = 0.0
    for i in range(n_samples):
        # Predicted walls
        pred_probs = np.array(jax.nn.sigmoid(predictions_batch['wall_logits'][i]))
        axes[i, 0].imshow(pred_probs, cmap='hot', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Pred from H[{i}]')
        axes[i, 0].axis('off')

        # Actual mutation level (direct index correspondence)
        actual_walls = np.array(mutation_levels_batch.wall_map[i])
        axes[i, 1].imshow(actual_walls, cmap='binary', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Mutate Level[{i}]')
        axes[i, 1].axis('off')

        # Error map
        pred_binary = (pred_probs > 0.5).astype(float)
        error = np.abs(pred_binary - actual_walls)
        accuracy = 1.0 - error.mean()
        total_accuracy += accuracy
        axes[i, 2].imshow(error, cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Error (Acc: {accuracy:.1%})')
        axes[i, 2].axis('off')

    avg_accuracy = total_accuracy / n_samples
    fig.text(0.5, 0.01, f'Mean Wall Accuracy: {avg_accuracy:.1%}', ha='center', fontsize=10)

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_probe_loss_by_branch_plot(
    probe_tracking: ProbeTrackingState,
) -> np.ndarray:
    """
    Create bar chart comparing probe losses across branches.

    Key insight: If replay/mutate branches have lower loss than random,
    it suggests the agent has learned curriculum-specific representations.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    branch_names = ['Random (DR)', 'Replay', 'Mutate']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    buffer_size = probe_tracking.branch_loss_history.shape[1]
    losses = []
    stds = []

    for i in range(3):
        valid = jnp.minimum(probe_tracking.branch_ptrs[i], buffer_size)
        mask = jnp.arange(buffer_size) < valid
        mean_loss = jnp.where(mask.sum() > 0,
                             (probe_tracking.branch_loss_history[i] * mask).sum() / mask.sum(),
                             0.0)
        std_loss = jnp.where(mask.sum() > 1,
                            jnp.sqrt(((probe_tracking.branch_loss_history[i] - mean_loss) ** 2 * mask).sum() / (mask.sum() - 1)),
                            0.0)
        losses.append(float(mean_loss))
        stds.append(float(std_loss))

    bars = ax.bar(branch_names, losses, yerr=stds, color=colors, capsize=5, alpha=0.8)
    ax.set_ylabel('Probe Loss (lower = more predictable)')
    ax.set_title('Probe Loss by Curriculum Branch\n(What has the agent learned about each branch?)')
    ax.set_ylim(bottom=0)

    # Add value labels on bars
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{loss:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_novelty_learnability_plot(
    novelty: float,
    learnability: float,
    score: float,
    regime: str,
) -> np.ndarray:
    """
    Create Pareto frontier visualization of novelty vs learnability.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot quadrants
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Add quadrant labels
    ax.text(0.7, 0.7, 'Open-ended\n(ideal)', ha='center', va='center',
            fontsize=10, alpha=0.5, transform=ax.transAxes)
    ax.text(0.3, 0.7, 'Chaotic', ha='center', va='center',
            fontsize=10, alpha=0.5, transform=ax.transAxes)
    ax.text(0.7, 0.3, 'Converging', ha='center', va='center',
            fontsize=10, alpha=0.5, transform=ax.transAxes)
    ax.text(0.3, 0.3, 'Stagnant', ha='center', va='center',
            fontsize=10, alpha=0.5, transform=ax.transAxes)

    # Plot current point
    color = {'open-ended': 'green', 'chaotic': 'red', 'converging': 'blue', 'stagnant': 'gray'}[regime]
    ax.scatter([novelty], [learnability], s=200, c=color, marker='o', zorder=5)
    ax.annotate(f'Current\n({regime})\nScore: {score:.3f}',
                (novelty, learnability), textcoords="offset points",
                xytext=(10, 10), fontsize=9)

    ax.set_xlabel('Novelty (curriculum surprise)')
    ax.set_ylabel('Learnability (probe improvement)')
    ax.set_title('Novelty-Learnability Space\n(Probe perspective on curriculum dynamics)')

    # Set axis limits symmetrically
    max_val = max(abs(novelty), abs(learnability), 0.5) * 1.5
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_hidden_state_tsne_plot(
    hstate_samples: chex.Array,    # (n_samples, 512)
    branch_labels: chex.Array,     # (n_samples,)
    max_samples: int = 500,
) -> np.ndarray:
    """
    Create t-SNE visualization of hidden states colored by branch.

    Note: This requires sklearn for t-SNE. Falls back to PCA if not available.
    """
    try:
        from sklearn.manifold import TSNE
        use_tsne = True
    except ImportError:
        use_tsne = False

    fig, ax = plt.subplots(figsize=(8, 6))

    n_samples = min(len(hstate_samples), max_samples)
    if n_samples < 10:
        ax.text(0.5, 0.5, 'Not enough samples yet', ha='center', va='center', transform=ax.transAxes)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return img

    # Subsample if needed
    indices = np.random.choice(len(hstate_samples), n_samples, replace=False)
    X = np.array(hstate_samples[indices])
    labels = np.array(branch_labels[indices])

    if use_tsne:
        tsne = TSNE(n_components=2, perplexity=min(30, n_samples - 1), random_state=42)
        X_2d = tsne.fit_transform(X)
        method = 't-SNE'
    else:
        # Fallback to PCA
        X_centered = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        X_2d = X_centered @ Vt[:2].T
        method = 'PCA'

    # Plot by branch
    branch_names = ['Random', 'Replay', 'Mutate']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for i, (name, color) in enumerate(zip(branch_names, colors)):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=name, alpha=0.6, s=20)

    ax.legend()
    ax.set_xlabel(f'{method} Component 1')
    ax.set_ylabel(f'{method} Component 2')
    ax.set_title(f'Hidden State {method} by Branch\n(Do different branches produce different representations?)')

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_correlation_scatter_plot(
    probe_tracking: ProbeTrackingState,
) -> np.ndarray:
    """
    Create scatter plot of probe accuracy vs agent return.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    buffer_size = probe_tracking.loss_history.shape[0]
    valid_samples = min(int(probe_tracking.total_samples), buffer_size)

    if valid_samples < 10:
        ax.text(0.5, 0.5, 'Not enough samples yet', ha='center', va='center', transform=ax.transAxes)
    else:
        x = np.array(probe_tracking.probe_accuracy_history[:valid_samples])
        y = np.array(probe_tracking.agent_returns_history[:valid_samples])

        ax.scatter(x, y, alpha=0.5, s=10)

        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label=f'Trend (slope={z[0]:.3f})')

        # Compute correlation
        corr_stats = compute_probe_correlation_with_performance(probe_tracking)
        ax.set_title(f'Probe Accuracy vs Agent Return\n(r = {corr_stats["probe_return_correlation"]:.3f})')
        ax.legend()

    ax.set_xlabel('Probe Accuracy')
    ax.set_ylabel('Agent Return')

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


# =============================================================================
# INFORMATION CONTENT DASHBOARD
# =============================================================================

def create_information_content_dashboard(
    probe_metrics: dict,
    probe_tracking: ProbeTrackingState,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> np.ndarray:
    """
    Create an Information Content Dashboard showing probe performance vs baselines.

    This visualization answers: "How much curriculum info does the agent encode?"
    """
    fig = plt.figure(figsize=(14, 10))

    # Get random baselines
    baselines = compute_random_baselines(env_height, env_width)

    # --- Panel 1: Information Gain Table (top left) ---
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.axis('off')

    # Extract metrics (using distributional calibration for fair comparison)
    metrics_table = [
        ['Component', 'Dist Acc', 'Random Base', 'Info Gain'],
        ['Walls', f'{probe_metrics.get("wall_dist_accuracy", 0):.1%}',
         f'{baselines["wall_accuracy"]:.1%}',
         f'+{(probe_metrics.get("wall_dist_accuracy", 0) - baselines["wall_accuracy"]):.1%}'],
        ['Goal Pos', f'{probe_metrics.get("goal_dist_mode_match", 0):.1%}',
         f'{baselines["goal_top1"]:.1%}',
         f'+{(probe_metrics.get("goal_dist_mode_match", 0) - baselines["goal_top1"]):.1%}'],
        ['Agent Pos', f'{probe_metrics.get("agent_pos_dist_mode_match", 0):.1%}',
         f'{baselines["agent_pos_top1"]:.1%}',
         f'+{(probe_metrics.get("agent_pos_dist_mode_match", 0) - baselines["agent_pos_top1"]):.1%}'],
        ['Direction', f'{probe_metrics.get("agent_dir_dist_mode_match", 0):.1%}',
         f'{baselines["agent_dir"]:.1%}',
         f'+{(probe_metrics.get("agent_dir_dist_mode_match", 0) - baselines["agent_dir"]):.1%}'],
    ]

    table = ax1.table(cellText=metrics_table, loc='center', cellLoc='center',
                      colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color the header and info gain column
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    for i in range(1, 5):
        gain = probe_metrics.get(['wall_dist_accuracy', 'goal_dist_mode_match',
                                   'agent_pos_dist_mode_match', 'agent_dir_dist_mode_match'][i-1], 0)
        baseline = [baselines['wall_accuracy'], baselines['goal_top1'],
                   baselines['agent_pos_top1'], baselines['agent_dir']][i-1]
        if gain > baseline * 1.5:
            table[(i, 3)].set_facecolor('#C8E6C9')  # Green for good gain
        elif gain > baseline:
            table[(i, 3)].set_facecolor('#FFF9C4')  # Yellow for some gain

    ax1.set_title('Information Content Summary', fontsize=12, fontweight='bold')

    # --- Panel 2: Info Gain by Branch (top right) ---
    ax2 = fig.add_subplot(2, 2, 2)

    branch_names = ['Random', 'Replay', 'Mutate']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    x = np.arange(len(branch_names))
    width = 0.2

    # Compute info gain per branch
    buffer_size = probe_tracking.branch_loss_history.shape[1]
    branch_losses = []
    branch_counts = []
    for i in range(3):
        valid = int(jnp.minimum(probe_tracking.branch_ptrs[i], buffer_size))
        if valid > 0:
            mask = jnp.arange(buffer_size) < valid
            mean_loss = float((probe_tracking.branch_loss_history[i] * mask).sum() / mask.sum())
        else:
            mean_loss = baselines['total_loss']
        branch_losses.append(mean_loss)
        branch_counts.append(int(probe_tracking.branch_sample_counts[i]))

    # Info gain = baseline_loss - probe_loss (higher is better)
    info_gains = [baselines['total_loss'] - loss for loss in branch_losses]

    bars = ax2.bar(branch_names, info_gains, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Information Gain (baseline - probe loss)')
    ax2.set_title('Information Gain by Branch')

    # Add sample counts as labels
    for bar, count in zip(bars, branch_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'n={count}', ha='center', va='bottom', fontsize=9)

    # --- Panel 3: Loss Over Training (bottom left) ---
    ax3 = fig.add_subplot(2, 2, 3)

    valid_samples = min(int(probe_tracking.total_samples), buffer_size)
    if valid_samples > 10:
        steps = np.array(probe_tracking.training_step_history[:valid_samples])
        losses = np.array(probe_tracking.loss_history[:valid_samples])

        # Sort by training step
        sort_idx = np.argsort(steps)
        steps_sorted = steps[sort_idx]
        losses_sorted = losses[sort_idx]

        ax3.plot(steps_sorted, losses_sorted, 'b-', alpha=0.3, linewidth=0.5)

        # Add smoothed line
        window = min(50, valid_samples // 5)
        if window > 1:
            smoothed = np.convolve(losses_sorted, np.ones(window)/window, mode='valid')
            ax3.plot(steps_sorted[window-1:], smoothed, 'b-', linewidth=2, label='Smoothed')

        ax3.axhline(y=baselines['total_loss'], color='red', linestyle='--',
                   label=f'Random Baseline ({baselines["total_loss"]:.2f})')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Probe Loss')
        ax3.set_title('Probe Loss Over Training')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Not enough data yet', ha='center', va='center',
                transform=ax3.transAxes)

    # --- Panel 4: Component Breakdown (bottom right) ---
    ax4 = fig.add_subplot(2, 2, 4)

    components = ['Wall', 'Goal', 'Agent Pos', 'Direction']
    probe_losses = [
        probe_metrics.get('wall_loss', baselines['wall_loss']),
        probe_metrics.get('goal_loss', baselines['goal_loss']),
        probe_metrics.get('agent_pos_loss', baselines['agent_pos_loss']),
        probe_metrics.get('agent_dir_loss', baselines['agent_dir_loss']),
    ]
    baseline_losses = [
        baselines['wall_loss'],
        baselines['goal_loss'],
        baselines['agent_pos_loss'],
        baselines['agent_dir_loss'],
    ]

    x = np.arange(len(components))
    width = 0.35

    bars1 = ax4.bar(x - width/2, probe_losses, width, label='Probe', color='#3498db')
    bars2 = ax4.bar(x + width/2, baseline_losses, width, label='Random', color='#95a5a6')

    ax4.set_ylabel('Loss (lower is better)')
    ax4.set_title('Component-wise Loss Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(components)
    ax4.legend()

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_pareto_trajectory_plot(
    pareto_history: ParetoHistoryState,
) -> np.ndarray:
    """
    Create Pareto frontier visualization with training trajectory.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    num_points = int(pareto_history.num_checkpoints)
    if num_points < 2:
        for ax in axes:
            ax.text(0.5, 0.5, 'Not enough checkpoints yet',
                   ha='center', va='center', transform=ax.transAxes)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return img

    novelty = np.array(pareto_history.novelty_history[:num_points])
    learnability = np.array(pareto_history.learnability_history[:num_points])
    steps = np.array(pareto_history.training_steps[:num_points])

    # Left plot: Novelty vs Learnability scatter with trajectory
    ax = axes[0]
    colors = np.linspace(0, 1, num_points)
    scatter = ax.scatter(learnability, novelty, c=colors, cmap='viridis', s=50, alpha=0.7)
    ax.plot(learnability, novelty, 'k-', alpha=0.3, linewidth=1)

    ax.scatter([learnability[0]], [novelty[0]], c='green', s=200, marker='o',
              label='Start', zorder=5)
    ax.scatter([learnability[-1]], [novelty[-1]], c='red', s=200, marker='s',
              label='End', zorder=5)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    ax.text(0.95, 0.95, 'Open-ended', transform=ax.transAxes,
           ha='right', va='top', fontsize=10, color='green')
    ax.text(0.05, 0.95, 'Chaotic', transform=ax.transAxes,
           ha='left', va='top', fontsize=10, color='orange')
    ax.text(0.95, 0.05, 'Converging', transform=ax.transAxes,
           ha='right', va='bottom', fontsize=10, color='blue')
    ax.text(0.05, 0.05, 'Stagnant', transform=ax.transAxes,
           ha='left', va='bottom', fontsize=10, color='gray')

    ax.set_xlabel('Learnability')
    ax.set_ylabel('Novelty')
    ax.set_title('Pareto Frontier: Novelty vs Learnability')
    ax.legend(loc='upper left')
    plt.colorbar(scatter, ax=ax, label='Training Progress')

    # Right plot: Time series
    ax = axes[1]
    ax.plot(steps, novelty, 'b-', label='Novelty', linewidth=2)
    ax.plot(steps, learnability, 'g-', label='Learnability', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Metric Value')
    ax.set_title('Novelty and Learnability Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


# =============================================================================
# FRESH PROBE EVALUATION (Clean measurement at eval time)
# =============================================================================

def evaluate_fresh_probe(
    agent_params,
    agent_apply_fn,
    env,
    env_params,
    sample_random_level,
    level_sampler,
    sampler_state,
    rng: chex.PRNGKey,
    num_episodes: int = 50,
    probe_train_steps: int = 100,
    probe_lr: float = 1e-3,
    num_train_envs: int = 32,
    num_steps: int = 256,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
    hstate_dim: int = DEFAULT_HSTATE_DIM,
) -> dict:
    """
    Evaluate probe information content with a freshly initialized probe.

    This provides a clean measurement of "how much information is extractable
    from current representations" without the confound of probe training history.

    Args:
        agent_params: Current agent parameters
        agent_apply_fn: Agent network apply function
        env: Environment
        env_params: Environment parameters
        sample_random_level: Random level generator
        level_sampler: Level sampler
        sampler_state: Current sampler state
        rng: Random key
        num_episodes: Number of episodes to collect hidden states
        probe_train_steps: Steps to train fresh probe
        probe_lr: Learning rate for fresh probe
        num_train_envs: Number of parallel environments
        num_steps: Steps per episode
        env_height: Environment height
        env_width: Environment width
        hstate_dim: Hidden state dimension

    Returns:
        Dict with fresh probe metrics and comparison to random baseline
    """
    # Initialize fresh probe (without episode context for evaluation simplicity)
    rng, rng_probe = jax.random.split(rng)
    probe_network = CurriculumProbe(
        env_height=env_height, env_width=env_width,
        use_episode_context=False,  # Evaluation uses hidden state only
    )
    dummy_input = jnp.zeros((1, hstate_dim))
    fresh_probe_params = probe_network.init(rng_probe, dummy_input)
    probe_opt = optax.adam(learning_rate=probe_lr)
    probe_opt_state = probe_opt.init(fresh_probe_params)

    # Collect hidden states and levels
    collected_hstates = []
    collected_levels = []

    for _ in range(num_episodes):
        rng, rng_level, rng_reset, rng_rollout = jax.random.split(rng, 4)

        # Generate level
        level = sample_random_level(rng_level)

        # Run agent to get hidden state
        init_obs, init_env_state = env.reset_to_level(rng_reset, level, env_params)
        init_obs_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)
        init_env_state_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_env_state)
        init_hstate = nn.OptimizedLSTMCell(features=256).initialize_carry(
            jax.random.PRNGKey(0), (1, 256)
        )

        # Simple rollout to get final hidden state
        hstate = init_hstate
        obs = init_obs_batch
        env_state = init_env_state_batch
        done = jnp.zeros(1, dtype=bool)

        for _ in range(num_steps):
            rng_rollout, rng_action = jax.random.split(rng_rollout)
            x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
            hstate, pi, _ = agent_apply_fn(agent_params, x, hstate)
            action = pi.sample(seed=rng_action).squeeze(0)
            obs, env_state, _, done, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(jax.random.split(rng_action, 1), env_state, action, env_params)

        # Flatten hidden state
        h_c, h_h = hstate
        hstate_flat = jnp.concatenate([h_c[0], h_h[0]], axis=-1)

        collected_hstates.append(hstate_flat)
        collected_levels.append(level)

    # Stack collected data
    hstates = jnp.stack(collected_hstates)  # (num_episodes, hstate_dim)

    # Train fresh probe
    for step in range(probe_train_steps):
        rng, rng_batch = jax.random.split(rng)
        batch_idx = jax.random.randint(rng_batch, (min(32, num_episodes),), 0, num_episodes)

        def probe_loss_fn(params):
            batch_hstates = hstates[batch_idx]
            predictions = probe_network.apply(params, batch_hstates)

            total_loss = 0.0
            for i, idx in enumerate(batch_idx):
                level = jax.tree_util.tree_map(lambda x: x, collected_levels[int(idx)])
                pred_i = jax.tree_util.tree_map(lambda x: x[i], predictions)
                loss_i, _ = compute_probe_loss(pred_i, level, env_height, env_width)
                total_loss += loss_i

            return total_loss / len(batch_idx)

        loss, grads = jax.value_and_grad(probe_loss_fn)(fresh_probe_params)
        updates, probe_opt_state = probe_opt.update(grads, probe_opt_state, fresh_probe_params)
        fresh_probe_params = optax.apply_updates(fresh_probe_params, updates)

    # Evaluate fresh probe
    total_metrics = {
        'wall_loss': 0.0, 'goal_loss': 0.0, 'agent_pos_loss': 0.0,
        'agent_dir_loss': 0.0, 'total_loss': 0.0,
        'wall_accuracy': 0.0, 'goal_top1_correct': 0.0,
        'agent_pos_top1_correct': 0.0, 'agent_dir_correct': 0.0,
    }

    for i in range(len(collected_hstates)):
        hstate_flat = collected_hstates[i][None, ...]
        predictions = probe_network.apply(fresh_probe_params, hstate_flat)
        predictions = jax.tree_util.tree_map(lambda x: x[0], predictions)

        _, loss_metrics = compute_probe_loss(predictions, collected_levels[i], env_height, env_width)
        calib_metrics = compute_calibration_metrics(predictions, collected_levels[i],
                                                    env_height=env_height, env_width=env_width)

        for key in ['wall_loss', 'goal_loss', 'agent_pos_loss', 'agent_dir_loss', 'total_loss']:
            total_metrics[key] += float(loss_metrics[key])
        for key in ['wall_accuracy', 'goal_top1_correct', 'agent_pos_top1_correct', 'agent_dir_correct']:
            total_metrics[key] += float(calib_metrics[key])

    # Average metrics
    for key in total_metrics:
        total_metrics[key] /= len(collected_hstates)

    # Compare to random baseline
    baselines = compute_random_baselines(env_height, env_width)
    total_metrics['info_gain_wall'] = total_metrics['wall_accuracy'] - baselines['wall_accuracy']
    total_metrics['info_gain_goal'] = total_metrics['goal_top1_correct'] - baselines['goal_top1']
    total_metrics['info_gain_agent_pos'] = total_metrics['agent_pos_top1_correct'] - baselines['agent_pos_top1']
    total_metrics['info_gain_agent_dir'] = total_metrics['agent_dir_correct'] - baselines['agent_dir']
    total_metrics['info_gain_total'] = (
        total_metrics['info_gain_wall'] + total_metrics['info_gain_goal'] +
        total_metrics['info_gain_agent_pos'] + total_metrics['info_gain_agent_dir']
    ) / 4

    return total_metrics


# =============================================================================
# EMPIRICAL BASELINE (Probe untrained agent)
# =============================================================================

def compute_empirical_baseline(
    env,
    env_params,
    sample_random_level,
    rng: chex.PRNGKey,
    num_episodes: int = 50,
    probe_train_steps: int = 100,
    probe_lr: float = 1e-3,
    num_steps: int = 256,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
    hstate_dim: int = DEFAULT_HSTATE_DIM,
) -> dict:
    """
    Compute empirical baseline by probing an UNTRAINED agent's hidden states.

    This tells us what information is available from the LSTM architecture
    itself vs what the agent has learned from training.

    The difference between trained probe and this baseline shows true learning.

    Returns:
        Dict with baseline metrics from untrained agent
    """
    # Initialize random (untrained) agent
    rng, rng_agent, rng_probe = jax.random.split(rng, 3)

    # Create untrained agent network
    from functools import partial

    # Simple untrained LSTM - just the architecture with random weights
    class UntrainedAgent(nn.Module):
        action_dim: int = 7

        @nn.compact
        def __call__(self, inputs, hidden):
            obs, dones = inputs
            img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
            img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
            img_embed = nn.relu(img_embed)
            dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
            dir_embed = nn.Dense(5)(dir_embed)
            embedding = jnp.append(img_embed, dir_embed, axis=-1)
            hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))(
                (embedding, dones), initial_carry=hidden
            )
            actor_logits = nn.Dense(self.action_dim)(embedding)
            pi = distrax.Categorical(logits=actor_logits)
            critic = nn.Dense(1)(embedding)
            return hidden, pi, jnp.squeeze(critic, axis=-1)

    untrained_agent = UntrainedAgent()

    # Initialize with random weights
    dummy_level = sample_random_level(rng)
    dummy_obs, _ = env.reset_to_level(rng, dummy_level, env_params)
    dummy_obs = jax.tree_util.tree_map(lambda x: x[None, None, ...], dummy_obs)
    dummy_dones = jnp.zeros((1, 1), dtype=bool)
    dummy_hstate = nn.OptimizedLSTMCell(features=256).initialize_carry(
        jax.random.PRNGKey(0), (1, 256)
    )
    untrained_params = untrained_agent.init(rng_agent, (dummy_obs, dummy_dones), dummy_hstate)

    # Collect hidden states from untrained agent
    collected_hstates = []
    collected_levels = []

    for _ in range(num_episodes):
        rng, rng_level, rng_reset, rng_rollout = jax.random.split(rng, 4)
        level = sample_random_level(rng_level)

        init_obs, init_env_state = env.reset_to_level(rng_reset, level, env_params)
        init_obs_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)
        init_env_state_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_env_state)
        hstate = nn.OptimizedLSTMCell(features=256).initialize_carry(
            jax.random.PRNGKey(0), (1, 256)
        )

        obs = init_obs_batch
        env_state = init_env_state_batch
        done = jnp.zeros(1, dtype=bool)

        for _ in range(num_steps):
            rng_rollout, rng_action = jax.random.split(rng_rollout)
            x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
            hstate, pi, _ = untrained_agent.apply(untrained_params, x, hstate)
            action = pi.sample(seed=rng_action).squeeze(0)
            obs, env_state, _, done, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(jax.random.split(rng_action, 1), env_state, action, env_params)

        h_c, h_h = hstate
        hstate_flat = jnp.concatenate([h_c[0], h_h[0]], axis=-1)
        collected_hstates.append(hstate_flat)
        collected_levels.append(level)

    # Train probe on untrained agent's hidden states (without episode context)
    probe_network = CurriculumProbe(
        env_height=env_height, env_width=env_width,
        use_episode_context=False,
    )
    dummy_input = jnp.zeros((1, hstate_dim))
    probe_params = probe_network.init(rng_probe, dummy_input)
    probe_opt = optax.adam(learning_rate=probe_lr)
    probe_opt_state = probe_opt.init(probe_params)

    hstates = jnp.stack(collected_hstates)

    for step in range(probe_train_steps):
        rng, rng_batch = jax.random.split(rng)
        batch_idx = jax.random.randint(rng_batch, (min(32, num_episodes),), 0, num_episodes)

        def probe_loss_fn(params):
            batch_hstates = hstates[batch_idx]
            predictions = probe_network.apply(params, batch_hstates)
            total_loss = 0.0
            for i, idx in enumerate(batch_idx):
                level = collected_levels[int(idx)]
                pred_i = jax.tree_util.tree_map(lambda x: x[i], predictions)
                loss_i, _ = compute_probe_loss(pred_i, level, env_height, env_width)
                total_loss += loss_i
            return total_loss / len(batch_idx)

        loss, grads = jax.value_and_grad(probe_loss_fn)(probe_params)
        updates, probe_opt_state = probe_opt.update(grads, probe_opt_state, probe_params)
        probe_params = optax.apply_updates(probe_params, updates)

    # Evaluate
    total_metrics = {
        'wall_accuracy': 0.0, 'goal_top1_correct': 0.0,
        'agent_pos_top1_correct': 0.0, 'agent_dir_correct': 0.0,
        'total_loss': 0.0,
    }

    for i in range(len(collected_hstates)):
        hstate_flat = collected_hstates[i][None, ...]
        predictions = probe_network.apply(probe_params, hstate_flat)
        predictions = jax.tree_util.tree_map(lambda x: x[0], predictions)
        _, loss_metrics = compute_probe_loss(predictions, collected_levels[i], env_height, env_width)
        calib_metrics = compute_calibration_metrics(predictions, collected_levels[i],
                                                    env_height=env_height, env_width=env_width)
        total_metrics['total_loss'] += float(loss_metrics['total_loss'])
        for key in ['wall_accuracy', 'goal_top1_correct', 'agent_pos_top1_correct', 'agent_dir_correct']:
            total_metrics[key] += float(calib_metrics[key])

    for key in total_metrics:
        total_metrics[key] /= len(collected_hstates)

    return total_metrics


# =============================================================================
# N-STEP PROBE EVALUATION
# =============================================================================

def evaluate_nstep_prediction(
    probe_params,
    agent_params,
    agent_apply_fn,
    env,
    env_params,
    sample_random_level,
    level_sampler,
    sampler_state,
    rng: chex.PRNGKey,
    n_steps: int = 5,
    num_episodes: int = 50,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
    hstate_dim: int = DEFAULT_HSTATE_DIM,
) -> dict:
    """
    Evaluate N-step prediction: can the probe predict the level N steps ahead?

    This tests temporal awareness - does the agent's hidden state encode
    information about upcoming curriculum dynamics, not just the current level?

    Args:
        probe_params: Trained probe parameters
        agent_params: Agent parameters
        agent_apply_fn: Agent network apply function
        env: Environment
        env_params: Environment parameters
        sample_random_level: Random level generator
        level_sampler: Level sampler
        sampler_state: Current sampler state
        rng: Random key
        n_steps: How many steps ahead to predict (1 = next level, 5 = level 5 steps later)
        num_episodes: Number of evaluation episodes
        env_height: Environment height
        env_width: Environment width
        hstate_dim: Hidden state dimension

    Returns:
        Dict with N-step prediction accuracy at each step ahead
    """
    probe_network = CurriculumProbe(
        env_height=env_height, env_width=env_width,
        use_episode_context=False,  # N-step uses hidden state only
    )
    results = {f'step_{k}': {'loss': 0.0, 'accuracy': 0.0} for k in range(1, n_steps + 1)}

    for episode in range(num_episodes):
        rng, rng_episode = jax.random.split(rng)

        # Collect hidden states and upcoming levels for N steps
        hstates = []
        levels = []

        # Generate a sequence of levels
        level_sequence = []
        for _ in range(n_steps + 1):
            rng_episode, rng_level = jax.random.split(rng_episode)
            level = sample_random_level(rng_level)
            level_sequence.append(level)

        # Run agent through each level, collecting hidden states
        hstate = nn.OptimizedLSTMCell(features=256).initialize_carry(
            jax.random.PRNGKey(0), (1, 256)
        )

        for step_idx, level in enumerate(level_sequence[:-1]):  # Exclude last (no future level)
            rng_episode, rng_reset, rng_rollout = jax.random.split(rng_episode, 3)

            # Run agent on level
            init_obs, init_env_state = env.reset_to_level(rng_reset, level, env_params)
            init_obs_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)
            init_env_state_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_env_state)

            obs = init_obs_batch
            env_state = init_env_state_batch
            done = jnp.zeros(1, dtype=bool)

            # Rollout to update hidden state
            for _ in range(256):  # num_steps
                rng_rollout, rng_action = jax.random.split(rng_rollout)
                x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
                hstate, pi, _ = agent_apply_fn(agent_params, x, hstate)
                action = pi.sample(seed=rng_action).squeeze(0)
                obs, env_state, _, done, _ = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(jax.random.split(rng_action, 1), env_state, action, env_params)

            # Store hidden state after this level
            h_c, h_h = hstate
            hstate_flat = jnp.concatenate([h_c[0], h_h[0]], axis=-1)
            hstates.append(hstate_flat)
            levels.append(level)

        # Now evaluate: for each collected hidden state, predict N steps ahead
        for k in range(1, n_steps + 1):  # k steps ahead
            valid_predictions = 0
            total_loss = 0.0
            total_accuracy = 0.0

            for idx in range(len(hstates)):
                target_idx = idx + k
                if target_idx >= len(level_sequence):
                    continue

                hstate_flat = hstates[idx]
                target_level = level_sequence[target_idx]

                # Probe prediction
                probe_input = jax.lax.stop_gradient(hstate_flat[None, ...])
                predictions = probe_network.apply(probe_params, probe_input)
                predictions = jax.tree_util.tree_map(lambda x: x[0], predictions)

                loss, _ = compute_probe_loss(predictions, target_level, env_height, env_width)
                calibration = compute_calibration_metrics(predictions, target_level,
                                                          env_height=env_height, env_width=env_width)

                total_loss += float(loss)
                total_accuracy += float(calibration['wall_accuracy'])
                valid_predictions += 1

            if valid_predictions > 0:
                results[f'step_{k}']['loss'] += total_loss / valid_predictions
                results[f'step_{k}']['accuracy'] += total_accuracy / valid_predictions

    # Average over episodes
    for k in range(1, n_steps + 1):
        results[f'step_{k}']['loss'] /= num_episodes
        results[f'step_{k}']['accuracy'] /= num_episodes

    return results


# =============================================================================
# CURRICULUM PROBE NETWORK
# =============================================================================

class CurriculumProbe(nn.Module):
    """
    Probes agent's LSTM hidden state to predict next level features.

    This is a completely separate network from the agent. It learns to decode
    information from the agent's hidden state, but gradients do NOT flow back
    to the agent (stop_gradient is applied by the caller).

    The probe tests: "Has the agent's internal representation learned to encode
    information about the curriculum dynamics, purely from RL experience?"

    Enhanced with episode context (return, solved, length) to give the probe
    access to the same information the agent experiences.
    """
    env_height: int = DEFAULT_ENV_HEIGHT
    env_width: int = DEFAULT_ENV_WIDTH
    use_episode_context: bool = True  # Whether to use return/solved/length

    @nn.compact
    def __call__(
        self,
        hidden_state: chex.Array,
        episode_return: Optional[chex.Array] = None,
        episode_solved: Optional[chex.Array] = None,
        episode_length: Optional[chex.Array] = None,
    ) -> dict:
        """
        Args:
            hidden_state: Flattened LSTM state, shape (batch, hstate_dim)
                          Caller should apply stop_gradient before passing.
            episode_return: Return from last episode, shape (batch,) or (batch, 1)
            episode_solved: Whether agent solved (reached goal), shape (batch,) or (batch, 1)
            episode_length: Number of steps in episode, shape (batch,) or (batch, 1)

        Returns:
            Dict with prediction logits for each level component
        """
        # Build input features
        features = [hidden_state]

        if self.use_episode_context:
            # Add episode context if provided
            if episode_return is not None:
                # Ensure shape is (batch, 1)
                ret = episode_return.reshape(-1, 1) if episode_return.ndim == 1 else episode_return
                features.append(ret)
            if episode_solved is not None:
                solved = episode_solved.astype(jnp.float32).reshape(-1, 1) if episode_solved.ndim == 1 else episode_solved.astype(jnp.float32)
                features.append(solved)
            if episode_length is not None:
                # Normalize length by max episode length (assume 256)
                length_norm = (episode_length.astype(jnp.float32) / 256.0).reshape(-1, 1) if episode_length.ndim == 1 else episode_length.astype(jnp.float32) / 256.0
                features.append(length_norm)

        # Concatenate all features
        if len(features) > 1:
            x = jnp.concatenate(features, axis=-1)
        else:
            x = hidden_state

        # Probe encoder - learns to extract curriculum-relevant features
        x = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="probe_encoder_0"
        )(x)
        x = nn.relu(x)

        x = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="probe_encoder_1"
        )(x)
        x = nn.relu(x)

        # Prediction heads

        # Wall prediction: per-cell probability (BCE loss)
        wall_logits = nn.Dense(
            self.env_height * self.env_width,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="probe_wall"
        )(x)
        wall_logits = wall_logits.reshape(-1, self.env_height, self.env_width)

        # Goal position: softmax over all cells
        goal_logits = nn.Dense(
            self.env_height * self.env_width,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="probe_goal"
        )(x)

        # Agent spawn position: softmax over all cells
        agent_pos_logits = nn.Dense(
            self.env_height * self.env_width,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="probe_agent_pos"
        )(x)

        # Agent direction: softmax over 4 directions
        agent_dir_logits = nn.Dense(
            4,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="probe_agent_dir"
        )(x)

        return {
            'wall_logits': wall_logits,           # (batch, H, W)
            'goal_logits': goal_logits,           # (batch, H * W)
            'agent_pos_logits': agent_pos_logits, # (batch, H * W)
            'agent_dir_logits': agent_dir_logits, # (batch, 4)
        }


def compute_probe_loss(
    predictions: dict,
    actual_level,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute loss between probe predictions and actual level.

    Args:
        predictions: Dict from CurriculumProbe (for single level, not batched)
        actual_level: The actual Level that was generated
        env_height: Environment height
        env_width: Environment width

    Returns:
        total_loss: Scalar loss
        metrics: Dict with individual loss components
    """
    # Wall prediction loss: Binary cross-entropy per cell
    wall_targets = actual_level.wall_map.astype(jnp.float32)
    wall_bce = optax.sigmoid_binary_cross_entropy(
        predictions['wall_logits'],
        wall_targets
    )
    wall_loss = wall_bce.mean()

    # Goal position loss: Cross-entropy over flattened grid
    goal_idx = actual_level.goal_pos[1] * env_width + actual_level.goal_pos[0]
    goal_loss = optax.softmax_cross_entropy_with_integer_labels(
        predictions['goal_logits'],
        goal_idx
    )

    # Agent position loss: Cross-entropy over flattened grid
    agent_pos_idx = actual_level.agent_pos[1] * env_width + actual_level.agent_pos[0]
    agent_pos_loss = optax.softmax_cross_entropy_with_integer_labels(
        predictions['agent_pos_logits'],
        agent_pos_idx
    )

    # Agent direction loss: Cross-entropy over 4 directions
    agent_dir_loss = optax.softmax_cross_entropy_with_integer_labels(
        predictions['agent_dir_logits'],
        actual_level.agent_dir
    )

    total_loss = wall_loss + goal_loss + agent_pos_loss + agent_dir_loss

    metrics = {
        'wall_loss': wall_loss,
        'goal_loss': goal_loss,
        'agent_pos_loss': agent_pos_loss,
        'agent_dir_loss': agent_dir_loss,
        'total_loss': total_loss,
    }

    return total_loss, metrics


def compute_distributional_probe_loss(
    predictions_batch: dict,
    actual_levels_batch,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute distributional loss between probe predictions and actual levels.

    Unlike per-instance loss which requires 1-to-1 correspondence between
    prediction[i] and level[i], this compares predicted vs empirical batch
    distributions. Appropriate when there's no meaningful index correspondence
    (e.g., DR branch where levels are randomly sampled).

    Args:
        predictions_batch: Dict from CurriculumProbe with batch dimension
            - wall_logits: (batch, H, W)
            - goal_logits: (batch, H * W)
            - agent_pos_logits: (batch, H * W)
            - agent_dir_logits: (batch, 4)
        actual_levels_batch: Batch of Level dataclasses
        env_height: Environment height
        env_width: Environment width

    Returns:
        total_loss: Scalar distributional loss
        metrics: Dict with individual loss components
    """
    batch_size = actual_levels_batch.wall_map.shape[0]
    grid_size = env_height * env_width

    # ===== WALL DISTRIBUTION LOSS =====
    # Mean predicted wall probability per cell
    mean_pred_wall_probs = jax.nn.sigmoid(predictions_batch['wall_logits']).mean(axis=0)  # (H, W)
    # Mean actual wall presence per cell
    mean_actual_walls = actual_levels_batch.wall_map.astype(jnp.float32).mean(axis=0)  # (H, W)
    # MSE between mean distributions
    wall_dist_loss = ((mean_pred_wall_probs - mean_actual_walls) ** 2).mean()

    # ===== GOAL POSITION DISTRIBUTION LOSS =====
    # Empirical goal distribution from batch
    goal_indices = actual_levels_batch.goal_pos[:, 1] * env_width + actual_levels_batch.goal_pos[:, 0]
    empirical_goal_dist = jnp.zeros(grid_size).at[goal_indices].add(1.0) / batch_size + 1e-10
    # Mean predicted goal distribution
    mean_pred_goal_dist = jax.nn.softmax(predictions_batch['goal_logits']).mean(axis=0) + 1e-10
    # Cross-entropy: -sum(p_empirical * log(p_pred))
    goal_dist_loss = -jnp.sum(empirical_goal_dist * jnp.log(mean_pred_goal_dist))

    # ===== AGENT POSITION DISTRIBUTION LOSS =====
    agent_pos_indices = actual_levels_batch.agent_pos[:, 1] * env_width + actual_levels_batch.agent_pos[:, 0]
    empirical_agent_pos_dist = jnp.zeros(grid_size).at[agent_pos_indices].add(1.0) / batch_size + 1e-10
    mean_pred_agent_pos_dist = jax.nn.softmax(predictions_batch['agent_pos_logits']).mean(axis=0) + 1e-10
    agent_pos_dist_loss = -jnp.sum(empirical_agent_pos_dist * jnp.log(mean_pred_agent_pos_dist))

    # ===== AGENT DIRECTION DISTRIBUTION LOSS =====
    dir_counts = jnp.zeros(4).at[actual_levels_batch.agent_dir].add(1.0)
    empirical_dir_dist = dir_counts / batch_size + 1e-10
    mean_pred_dir_dist = jax.nn.softmax(predictions_batch['agent_dir_logits']).mean(axis=0) + 1e-10
    agent_dir_dist_loss = -jnp.sum(empirical_dir_dist * jnp.log(mean_pred_dir_dist))

    # Total loss
    total_loss = wall_dist_loss + goal_dist_loss + agent_pos_dist_loss + agent_dir_dist_loss

    metrics = {
        'wall_dist_loss': wall_dist_loss,
        'goal_dist_loss': goal_dist_loss,
        'agent_pos_dist_loss': agent_pos_dist_loss,
        'agent_dir_dist_loss': agent_dir_dist_loss,
        'total_dist_loss': total_loss,
    }

    return total_loss, metrics


def compute_greedy_matching(
    predictions_batch: dict,
    actual_levels_batch,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute greedy matching between predictions and actual levels for visualization.

    For each prediction, finds the actual level with minimum loss.
    O(n^2) but only used for visualization, not training.

    Args:
        predictions_batch: Dict from CurriculumProbe with batch dimension
        actual_levels_batch: Batch of Level dataclasses

    Returns:
        matched_indices: (batch_size,) - for each prediction, index of best-matched level
        match_losses: (batch_size,) - loss for each matched pair
    """
    batch_size = actual_levels_batch.wall_map.shape[0]

    def compute_pair_loss(pred_idx, level_idx):
        """Compute loss between prediction pred_idx and level level_idx."""
        pred = jax.tree_util.tree_map(lambda x: x[pred_idx], predictions_batch)
        level = jax.tree_util.tree_map(lambda x: x[level_idx], actual_levels_batch)
        loss, _ = compute_probe_loss(pred, level, env_height, env_width)
        return loss

    # Compute full cost matrix (batch_size x batch_size)
    def compute_row_losses(pred_idx):
        return jax.vmap(lambda level_idx: compute_pair_loss(pred_idx, level_idx))(
            jnp.arange(batch_size)
        )

    cost_matrix = jax.vmap(compute_row_losses)(jnp.arange(batch_size))  # (batch, batch)

    # Greedy matching: for each prediction, pick minimum cost level
    matched_indices = cost_matrix.argmin(axis=1)  # (batch_size,)
    match_losses = jnp.take_along_axis(cost_matrix, matched_indices[:, None], axis=1).squeeze(-1)

    return matched_indices, match_losses


def compute_per_instance_loss_batch(
    predictions_batch: dict,
    actual_levels_batch,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute per-instance loss with direct index pairing (pred[i] vs level[i]).

    Use ONLY when there's meaningful 1-to-1 correspondence (Replay->Mutate branch).

    Args:
        predictions_batch: Dict from CurriculumProbe with batch dimension
        actual_levels_batch: Batch of Level dataclasses

    Returns:
        mean_loss: Scalar mean loss over batch
        metrics: Dict with individual loss components (averaged over batch)
    """
    batch_size = actual_levels_batch.wall_map.shape[0]

    def single_instance_loss(idx):
        pred = jax.tree_util.tree_map(lambda x: x[idx], predictions_batch)
        level = jax.tree_util.tree_map(lambda x: x[idx], actual_levels_batch)
        return compute_probe_loss(pred, level, env_height, env_width)

    # Vectorize over batch
    losses, metrics_batch = jax.vmap(single_instance_loss)(jnp.arange(batch_size))
    mean_loss = losses.mean()
    avg_metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics_batch)

    return mean_loss, avg_metrics


def compute_matched_accuracy_metrics(
    predictions_batch: dict,
    actual_levels_batch,
    matched_indices: jnp.ndarray,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> dict:
    """
    Compute accuracy metrics using greedy-matched pairs for visualization.

    Args:
        predictions_batch: Dict from CurriculumProbe with batch dimension
        actual_levels_batch: Batch of Level dataclasses
        matched_indices: (batch_size,) indices from greedy matching

    Returns:
        Dict with accuracy metrics based on matched pairs
    """
    # Reorder actual levels according to matching
    matched_levels = jax.tree_util.tree_map(
        lambda x: x[matched_indices], actual_levels_batch
    )

    # Wall accuracy
    pred_walls = (jax.nn.sigmoid(predictions_batch['wall_logits']) > 0.5).astype(jnp.float32)
    actual_walls = matched_levels.wall_map.astype(jnp.float32)
    wall_accuracies = (pred_walls == actual_walls).mean(axis=(1, 2))  # (batch,)

    # Goal accuracy (top-1)
    pred_goal_indices = predictions_batch['goal_logits'].argmax(axis=1)
    actual_goal_indices = matched_levels.goal_pos[:, 1] * env_width + matched_levels.goal_pos[:, 0]
    goal_correct = (pred_goal_indices == actual_goal_indices).astype(jnp.float32)

    # Agent position accuracy (top-1)
    pred_agent_pos_indices = predictions_batch['agent_pos_logits'].argmax(axis=1)
    actual_agent_pos_indices = matched_levels.agent_pos[:, 1] * env_width + matched_levels.agent_pos[:, 0]
    agent_pos_correct = (pred_agent_pos_indices == actual_agent_pos_indices).astype(jnp.float32)

    # Direction accuracy
    pred_dirs = predictions_batch['agent_dir_logits'].argmax(axis=1)
    dir_correct = (pred_dirs == matched_levels.agent_dir).astype(jnp.float32)

    return {
        'matched_wall_accuracy': wall_accuracies.mean(),
        'matched_goal_accuracy': goal_correct.mean(),
        'matched_agent_pos_accuracy': agent_pos_correct.mean(),
        'matched_dir_accuracy': dir_correct.mean(),
        'wall_accuracy_std': wall_accuracies.std(),
    }


# =============================================================================
# TRAIN STATE
# =============================================================================

class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    # used for logging
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)

    # Probe fields (only used if use_probe=True)
    probe_params: Optional[chex.ArrayTree] = struct.field(pytree_node=True, default=None)
    probe_opt_state: Optional[chex.ArrayTree] = struct.field(pytree_node=True, default=None)

    # Hidden state tracking for probe
    # current_hstate: hstate from the rollout that just completed (set by branch functions)
    # prev_hstate: hstate from the PREVIOUS step (used to predict current level)
    current_hstate: Optional[chex.ArrayTree] = struct.field(pytree_node=True, default=None)
    prev_hstate: Optional[chex.ArrayTree] = struct.field(pytree_node=True, default=None)
    current_branch: int = struct.field(pytree_node=True, default=0)
    prev_branch: int = struct.field(pytree_node=True, default=0)
    # Flag to track if we have valid prev_hstate (False on first step)
    has_valid_prev_hstate: bool = struct.field(pytree_node=True, default=False)

    # Probe tracking state for metrics (novelty, learnability, correlation, etc.)
    probe_tracking: Optional[ProbeTrackingState] = struct.field(pytree_node=True, default=None)

    # Pareto history for tracking novelty/learnability over training
    pareto_history: Optional[ParetoHistoryState] = struct.field(pytree_node=True, default=None)

    # Hidden state samples for t-SNE visualization (collected periodically)
    hstate_samples: Optional[chex.Array] = struct.field(pytree_node=True, default=None)  # (max_samples, hstate_dim)
    hstate_sample_branches: Optional[chex.Array] = struct.field(pytree_node=True, default=None)  # (max_samples,)
    hstate_sample_ptr: int = struct.field(pytree_node=True, default=0)

    # Track last agent return for correlation
    last_agent_return: jnp.ndarray = struct.field(pytree_node=True, default=0.0)

    # Actual training step counter for proper time tracking
    training_step: int = struct.field(pytree_node=True, default=0)

    # Episode context for probe (what the agent experiences)
    # These are batched: (num_envs,)
    last_episode_return: Optional[chex.Array] = struct.field(pytree_node=True, default=None)
    last_episode_solved: Optional[chex.Array] = struct.field(pytree_node=True, default=None)
    last_episode_length: Optional[chex.Array] = struct.field(pytree_node=True, default=None)

    # Track previous branch for Replay->Mutate detection
    # 0=DR, 1=Replay, 2=Mutate
    # When prev_branch=1 and current_branch=2, we have Replay->Mutate with 1-to-1 correspondence


def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets"""

    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values


def sample_trajectories_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
) -> Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]:
    """This samples trajectories from the environment using the agent specified by the train_state"""

    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, info)

    (rng, train_state, hstate, last_obs, last_env_state, last_done), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_envs, dtype=bool),
        ),
        None,
        length=max_episode_length,
    )

    x = jax.tree_util.tree_map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)
    return (rng, train_state, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj

def evaluate_rnn(
        rng: chex.PRNGKey,
        env: UnderspecifiedEnv,
        env_params: EnvParams,
        train_state: TrainState,
        init_hstate: chex.ArrayTree,
        init_obs: Observation,
        init_env_state: EnvState,
        max_episode_length: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:

    """This runs the RNN on the environment, given an initial state and observation, and returns (states, rewards, episode lengths"""
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]

    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)

        next_mask = mask & ~done
        episode_length += mask

        return (rng, hstate, obs, next_state, done, next_mask, episode_length), (state, reward)

    (_, _, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (
            rng,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_levels, dtype=bool),
            jnp.ones(num_levels, dtype=bool),
            jnp.zeros(num_levels, dtype=jnp.int32),
        ),
        None,
        length=max_episode_length,
    )

    return states, rewards, episode_lengths


def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool=True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state"""

    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages

    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate, obs, actions, last_dones, log_probs, values, targets, advantages = minibatch

            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(params, (obs, last_dones), init_hstate)
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_probs_pred - log_probs)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)).mean()

                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()

                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy
                return loss, (l_vf, l_clip, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0)
                .reshape(n_minibatch, -1, *x.shape[1:]),
                init_hstate,
            ),
            *jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)


class ActorCritic(nn.Module):
    """This is an actor critic class that uses an LSTM"""
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs

        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)

        embedding = jnp.append(img_embed, dir_embed, axis=-1)
        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)

        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))


def setup_checkpointing(config: dict, train_state: TrainState, env: UnderspecifiedEnv, env_params: EnvParams) -> ocp.CheckpointManager:
    """This takes in the train state and config, and returns an orbax checkpoint manager."""
    overall_save_dir = os.path.join(os.getcwd(), "checkpoints", f"{config['run_name']}", str(config['seed']))
    os.makedirs(overall_save_dir, exist_ok=True)

    with open(os.path.join(overall_save_dir, "config.json"), "w+") as f:
        f.write(json.dumps(config.as_dict(), indent=True))

    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, "models"),
        checkpointers=ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config["checkpoint_save_interval"],
            max_to_keep=config["max_number_of_checkpoints"],
        )
    )
    return checkpoint_manager


def train_state_to_log_dict(train_state: TrainState, level_sampler: LevelSampler) -> dict:
    """To prevent the entire large train_state to be copied to the cpu when doing logging, this function returns all the
    important information in a dict format"""

    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)

    return {
        "log":{
            "level_sampler/size": sampler["size"],
            'level_sampler/episode_count': sampler["episode_count"],
            "level_sampler/max_score": sampler["scores"].max(),
            "level_sampler/weighted_score": (sampler["scores"] * level_sampler.level_weights(sampler)).sum(),
            "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
        },
        "info":{
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        },
        # Include train_state for probe tracking access
        "train_state": train_state,
    }


def compute_score(config, dones, values, max_returns, advantages):
    if config["score_function"] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config["score_function"] == "pvl":
        return positive_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")


def main(config=None, project="JaxUED-minigrid-maze"):
    tags = []
    if not config["exploratory_grad_updates"]:
        tags.append("robust")
    if config["use_accel"]:
        tags.append("ACCEL")
    else:
        tags.append("PLR")
    if config.get("use_probe", False):
        tags.append("probe")

    run = wandb.init(config=config, project=project, group=config["run_name"], tags=tags)
    config = wandb.config

    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_lengths/*", step_metric="num_updates")

    # Probe metrics - comprehensive tracking (if enabled)
    if config.get("use_probe", False):
        wandb.define_metric("probe/*", step_metric="num_updates")
        wandb.define_metric("probe/all/*", step_metric="num_updates")
        wandb.define_metric("probe/random/*", step_metric="num_updates")
        wandb.define_metric("probe/replay/*", step_metric="num_updates")
        wandb.define_metric("probe/mutate/*", step_metric="num_updates")
        wandb.define_metric("probe/calibration/*", step_metric="num_updates")
        wandb.define_metric("probe/accuracy/*", step_metric="num_updates")
        wandb.define_metric("probe/dynamics/*", step_metric="num_updates")
        wandb.define_metric("probe/correlation/*", step_metric="num_updates")
        wandb.define_metric("probe/hstate/*", step_metric="num_updates")
        wandb.define_metric("probe/divergence/*", step_metric="num_updates")
        wandb.define_metric("probe/info_gain/*", step_metric="num_updates")
        wandb.define_metric("probe/samples/*", step_metric="num_updates")
        wandb.define_metric("probe/images/*", step_metric="num_updates")
        wandb.define_metric("probe/final/*", step_metric="num_updates")
        wandb.define_metric("baselines/*", step_metric="num_updates")

    def log_eval(stats, train_state_info):
        print(f"Logging update: {stats['update_count']}")

        # generic stats
        env_steps = stats["update_count"] * config["num_train_envs"] * config["num_steps"]
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / stats["time_delta"],
        }

        # evaluation performance
        solve_rates = stats["eval_solve_rates"]
        returns = stats["eval_returns"]
        log_dict.update({f"solve_rate/{name}": solve_rate for name, solve_rate in zip(config["eval_levels"], solve_rates)})
        log_dict.update({f"solve_rate/mean": solve_rates.mean()})
        log_dict.update({f"return/{name}": ret for name, ret in zip(config["eval_levels"], returns)})
        log_dict.update({f"return/mean": returns.mean()})
        log_dict.update({f"eval_ep_lengths/mean": stats["eval_ep_lengths"].mean()})

        # level sampler
        log_dict.update(train_state_info["log"])

        # images
        log_dict.update({"images/highest_scoring_level": wandb.Image(np.array(stats["highest_scoring_level"]), caption="Highest Scoring Level")})
        log_dict.update({"images/highest_weighted_level": wandb.Image(np.array(stats["highest_weighted_level"]), caption="highest Weighted Level")})

        for s in ["dr", "replay", "mutation"]:
            if train_state_info["info"][f"num_{s}_updates"] > 0:
                log_dict.update({f"images/{s}_levels": [wandb.Image(np.array(image)) for image in stats[f"{s}_levels"]]})

        # animations
        for i, level_name in enumerate(config["eval_levels"]):
            frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
            frames = np.array(frames[:episode_length])
            log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4, format="gif")})

        # Probe metrics (if enabled)
        if config.get("use_probe", False) and "probe" in stats:
            probe_stats = stats["probe"]
            # Metrics are stacked from jax.lax.scan, take mean

            # Distributional loss metrics (main training signal)
            log_dict.update({
                "probe/dist_loss/wall": float(probe_stats["wall_dist_loss"].mean()),
                "probe/dist_loss/goal": float(probe_stats["goal_dist_loss"].mean()),
                "probe/dist_loss/agent_pos": float(probe_stats["agent_pos_dist_loss"].mean()),
                "probe/dist_loss/agent_dir": float(probe_stats["agent_dir_dist_loss"].mean()),
                "probe/dist_loss/total": float(probe_stats["total_dist_loss"].mean()),
            })

            # Legacy per-instance loss metrics (for backward compatibility)
            log_dict.update({
                "probe/all/wall_loss": float(probe_stats["wall_loss"].mean()),
                "probe/all/goal_loss": float(probe_stats["goal_loss"].mean()),
                "probe/all/agent_pos_loss": float(probe_stats["agent_pos_loss"].mean()),
                "probe/all/agent_dir_loss": float(probe_stats["agent_dir_loss"].mean()),
                "probe/all/total_loss": float(probe_stats["total_loss"].mean()),
            })

            # Replay->Mutate specific per-instance metrics (meaningful 1-to-1 correspondence)
            if "is_replay_to_mutate" in probe_stats:
                r2m_mask = probe_stats["is_replay_to_mutate"] > 0.5
                if r2m_mask.any():
                    log_dict.update({
                        "probe/replay_to_mutate/per_instance_loss": float(probe_stats["per_instance_loss"][r2m_mask].mean()),
                        "probe/replay_to_mutate/per_instance_wall_loss": float(probe_stats["per_instance_wall_loss"][r2m_mask].mean()),
                        "probe/replay_to_mutate/count": int(r2m_mask.sum()),
                    })

            # === DISTRIBUTIONAL CALIBRATION (all branches, fair comparison) ===
            log_dict.update({
                "probe/dist_calibration/wall": float(probe_stats["wall_dist_calibration"].mean()),
                "probe/dist_calibration/goal": float(probe_stats["goal_dist_calibration"].mean()),
                "probe/dist_calibration/agent_pos": float(probe_stats["agent_pos_dist_calibration"].mean()),
                "probe/dist_calibration/agent_dir": float(probe_stats["agent_dir_dist_calibration"].mean()),
                "probe/dist_accuracy/wall": float(probe_stats["wall_dist_accuracy"].mean()),
                "probe/dist_accuracy/goal_mode_match": float(probe_stats["goal_dist_mode_match"].mean()),
                "probe/dist_accuracy/agent_pos_mode_match": float(probe_stats["agent_pos_dist_mode_match"].mean()),
                "probe/dist_accuracy/agent_dir_mode_match": float(probe_stats["agent_dir_dist_mode_match"].mean()),
                "probe/dist_accuracy/combined": float(probe_stats["combined_accuracy"].mean()),
            })

            # === PER-INSTANCE CALIBRATION (R->M only, when 1-to-1 correspondence exists) ===
            if "is_replay_to_mutate" in probe_stats:
                r2m_mask = probe_stats["is_replay_to_mutate"] > 0.5
                if r2m_mask.any():
                    log_dict.update({
                        "probe/replay_to_mutate/wall_accuracy": float(probe_stats["r2m_wall_accuracy"][r2m_mask].mean()),
                        "probe/replay_to_mutate/goal_accuracy": float(probe_stats["r2m_goal_accuracy"][r2m_mask].mean()),
                        "probe/replay_to_mutate/agent_pos_accuracy": float(probe_stats["r2m_agent_pos_accuracy"][r2m_mask].mean()),
                        "probe/replay_to_mutate/agent_dir_accuracy": float(probe_stats["r2m_agent_dir_accuracy"][r2m_mask].mean()),
                        "probe/replay_to_mutate/combined_accuracy": float(probe_stats["r2m_combined_accuracy"][r2m_mask].mean()),
                        "probe/replay_to_mutate/goal_prob_at_actual": float(probe_stats["r2m_goal_prob_at_actual"][r2m_mask].mean()),
                        "probe/replay_to_mutate/agent_pos_prob_at_actual": float(probe_stats["r2m_agent_pos_prob_at_actual"][r2m_mask].mean()),
                        "probe/replay_to_mutate/agent_dir_prob_at_actual": float(probe_stats["r2m_agent_dir_prob_at_actual"][r2m_mask].mean()),
                    })

            # Per-branch metrics (if we have branch info)
            if "branch" in probe_stats:
                branch_names = {0: "random", 1: "replay", 2: "mutate"}
                branches = probe_stats["branch"]
                for branch_id, branch_name in branch_names.items():
                    mask = branches == branch_id
                    if mask.any():
                        log_dict.update({
                            # Distributional losses
                            f"probe/{branch_name}/dist_loss/wall": float(probe_stats["wall_dist_loss"][mask].mean()),
                            f"probe/{branch_name}/dist_loss/total": float(probe_stats["total_dist_loss"][mask].mean()),
                            # Legacy losses
                            f"probe/{branch_name}/wall_loss": float(probe_stats["wall_loss"][mask].mean()),
                            f"probe/{branch_name}/goal_loss": float(probe_stats["goal_loss"][mask].mean()),
                            f"probe/{branch_name}/agent_pos_loss": float(probe_stats["agent_pos_loss"][mask].mean()),
                            f"probe/{branch_name}/agent_dir_loss": float(probe_stats["agent_dir_loss"][mask].mean()),
                            f"probe/{branch_name}/total_loss": float(probe_stats["total_loss"][mask].mean()),
                            # Distributional calibration (all branches, fair comparison)
                            f"probe/{branch_name}/dist_calibration/wall": float(probe_stats["wall_dist_calibration"][mask].mean()),
                            f"probe/{branch_name}/dist_calibration/goal": float(probe_stats["goal_dist_calibration"][mask].mean()),
                            f"probe/{branch_name}/dist_accuracy/wall": float(probe_stats["wall_dist_accuracy"][mask].mean()),
                            f"probe/{branch_name}/dist_accuracy/goal_mode_match": float(probe_stats["goal_dist_mode_match"][mask].mean()),
                            f"probe/{branch_name}/dist_accuracy/agent_pos_mode_match": float(probe_stats["agent_pos_dist_mode_match"][mask].mean()),
                            f"probe/{branch_name}/dist_accuracy/combined": float(probe_stats["combined_accuracy"][mask].mean()),
                        })

            # ===== PROBE-SPECIFIC INSIGHTS FROM TRACKING STATE =====
            train_state = train_state_info.get("train_state")
            if train_state is not None and train_state.probe_tracking is not None:
                probe_tracking = train_state.probe_tracking

                # Novelty and Learnability
                novelty, novelty_details = compute_novelty(probe_tracking)
                learnability, learnability_details = compute_learnability(probe_tracking)
                openendedness_score, regime = compute_openendedness_score(novelty, learnability)

                log_dict.update({
                    "probe/dynamics/novelty": novelty,
                    "probe/dynamics/learnability": learnability,
                    "probe/dynamics/openendedness_score": openendedness_score,
                    "probe/dynamics/regime": regime,
                    "probe/dynamics/novelty_slope": novelty_details["novelty_slope"],
                    "probe/dynamics/instantaneous_novelty": novelty_details["instantaneous_novelty"],
                    "probe/dynamics/early_loss": learnability_details["early_loss"],
                    "probe/dynamics/late_loss": learnability_details["late_loss"],
                })

                # Update Pareto history for trajectory visualization
                if train_state.pareto_history is not None:
                    training_step = int(train_state.training_step)
                    # Note: We can't modify train_state here (outside JIT), so this is logged
                    # The actual pareto_history update happens inside train_and_eval_step
                    log_dict["probe/dynamics/training_step"] = training_step

                # ===== DISTRIBUTION DIVERGENCE =====
                # Compute divergence between predictions and actual level distribution
                try:
                    last_predictions = jax.tree_util.tree_map(lambda x: x[-1], probe_stats["_predictions"])
                    # Get the level batch for divergence computation
                    level_batch = jax.tree_util.tree_map(lambda x: x, probe_stats["_current_level"])
                    divergence_metrics = compute_distribution_divergence(
                        last_predictions, level_batch,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict.update({
                        "probe/divergence/goal_kl": float(divergence_metrics['goal_kl']),
                        "probe/divergence/goal_js": float(divergence_metrics['goal_js']),
                        "probe/divergence/agent_pos_kl": float(divergence_metrics['agent_pos_kl']),
                        "probe/divergence/wall_density_error": float(divergence_metrics['wall_density_error']),
                        "probe/divergence/empirical_wall_density": float(divergence_metrics['empirical_wall_density']),
                        "probe/divergence/predicted_wall_density": float(divergence_metrics['predicted_wall_density']),
                    })
                except Exception as e:
                    print(f"Warning: Could not compute distribution divergence: {e}")

                # ===== INFORMATION GAIN VS BASELINES =====
                random_baselines = compute_random_baselines(DEFAULT_ENV_HEIGHT, DEFAULT_ENV_WIDTH)
                wall_dist_acc = float(probe_stats["wall_dist_accuracy"].mean())
                goal_dist_mode = float(probe_stats["goal_dist_mode_match"].mean())
                log_dict.update({
                    "probe/info_gain/wall_vs_random": wall_dist_acc - random_baselines['wall_accuracy'],
                    "probe/info_gain/goal_vs_random": goal_dist_mode - random_baselines['goal_top1'],
                    "probe/info_gain/total_loss_improvement": random_baselines['total_loss'] - float(probe_stats["total_loss"].mean()),
                })

                # ===== BRANCH SAMPLE COUNTS =====
                log_dict.update({
                    "probe/samples/random_count": int(probe_tracking.branch_sample_counts[0]),
                    "probe/samples/replay_count": int(probe_tracking.branch_sample_counts[1]),
                    "probe/samples/mutate_count": int(probe_tracking.branch_sample_counts[2]),
                    "probe/samples/total": int(probe_tracking.total_samples),
                })

                # Correlation with agent performance
                corr_stats = compute_probe_correlation_with_performance(probe_tracking)
                log_dict.update({
                    "probe/correlation/probe_return_correlation": corr_stats["probe_return_correlation"],
                    "probe/correlation/mean_probe_accuracy": corr_stats["mean_probe_accuracy"],
                    "probe/correlation/mean_agent_return": corr_stats["mean_agent_return"],
                })

                # Hidden state statistics per branch
                hstate_stats = compute_hidden_state_statistics(probe_tracking)
                for key, value in hstate_stats.items():
                    log_dict[f"probe/hstate/{key}"] = value

                # ===== VISUALIZATIONS =====
                # Get last predictions and level for visualization
                last_predictions = jax.tree_util.tree_map(lambda x: x[-1], probe_stats["_predictions"])
                last_level = jax.tree_util.tree_map(lambda x: x[-1], probe_stats["_current_level"])

                # Wall prediction heatmap
                try:
                    wall_heatmap = create_wall_prediction_heatmap(
                        last_predictions, last_level,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["probe/images/wall_prediction_heatmap"] = wandb.Image(
                        wall_heatmap, caption="Wall Prediction from Hidden State"
                    )
                except Exception as e:
                    print(f"Warning: Could not create wall heatmap: {e}")

                # Position prediction heatmap
                try:
                    pos_heatmap = create_position_prediction_heatmap(
                        last_predictions, last_level,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["probe/images/position_prediction_heatmap"] = wandb.Image(
                        pos_heatmap, caption="Position Prediction from Hidden State"
                    )
                except Exception as e:
                    print(f"Warning: Could not create position heatmap: {e}")

                # ===== BATCH-AWARE VISUALIZATIONS (address averaging problem) =====
                # These show predictions across the full batch, not just one sample
                try:
                    # Get full batch of predictions and levels (last step of scan)
                    # Shape after scan: predictions_batch[key] = (scan_steps, batch_size, ...)
                    # We want the last scan step: (batch_size, ...)
                    def extract_last_step(x):
                        """Extract last scan step, handling various array dimensions."""
                        if not hasattr(x, 'ndim'):
                            return x  # Scalar or non-array
                        if x.ndim >= 2:
                            return x[-1]  # Take last along first (scan) dimension
                        return x  # 1D or 0D, return as-is

                    predictions_batch = jax.tree_util.tree_map(
                        extract_last_step, probe_stats["_predictions"]
                    )
                    levels_batch = jax.tree_util.tree_map(
                        extract_last_step, probe_stats["_current_level"]
                    )

                    # Compute greedy matching for visualization
                    matched_indices, match_losses = compute_greedy_matching(
                        predictions_batch, levels_batch,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )

                    # Compute matched accuracy metrics
                    matched_metrics = compute_matched_accuracy_metrics(
                        predictions_batch, levels_batch, matched_indices,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )

                    # Log matched accuracy (these are the "best case" accuracies with greedy matching)
                    log_dict.update({
                        "probe/matched/wall_accuracy": float(matched_metrics['matched_wall_accuracy']),
                        "probe/matched/goal_accuracy": float(matched_metrics['matched_goal_accuracy']),
                        "probe/matched/agent_pos_accuracy": float(matched_metrics['matched_agent_pos_accuracy']),
                        "probe/matched/dir_accuracy": float(matched_metrics['matched_dir_accuracy']),
                        "probe/matched/mean_match_loss": float(match_losses.mean()),
                    })

                    # Batch wall prediction summary (shows variance, multiple samples)
                    batch_wall_summary = create_batch_wall_prediction_summary(
                        predictions_batch, levels_batch,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH,
                        n_samples=4
                    )
                    log_dict["probe/images/batch_wall_summary"] = wandb.Image(
                        batch_wall_summary, caption="Batch Wall Prediction (shows variance & samples)"
                    )

                    # Batch position prediction summary (shows all actual positions)
                    batch_pos_summary = create_batch_position_prediction_summary(
                        predictions_batch, levels_batch,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["probe/images/batch_position_summary"] = wandb.Image(
                        batch_pos_summary, caption="Batch Position Prediction (shows all actuals)"
                    )

                    # Matched pairs visualization (greedy matching)
                    matched_pairs_viz = create_matched_pairs_visualization(
                        predictions_batch, levels_batch, matched_indices,
                        n_pairs=4,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["probe/images/matched_pairs"] = wandb.Image(
                        matched_pairs_viz, caption="Greedy-Matched Prediction/Actual Pairs"
                    )

                    # Replay→Mutate specific heatmap (true 1-to-1 correspondence)
                    # Check if the last step in this eval batch was a Replay→Mutate transition
                    if "is_replay_to_mutate" in probe_stats:
                        last_r2m = probe_stats["is_replay_to_mutate"][-1]
                        if float(last_r2m) > 0.5:
                            r2m_heatmap = create_replay_to_mutate_heatmap(
                                predictions_batch, levels_batch,
                                n_samples=4,
                                env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                            )
                            log_dict["probe/images/replay_to_mutate_heatmap"] = wandb.Image(
                                r2m_heatmap, caption="Replay→Mutate: Per-Instance Correspondence (pred[i] vs mutate(replay[i]))"
                            )
                except Exception as e:
                    print(f"Warning: Could not create batch visualizations: {e}")

                # Loss by branch comparison
                try:
                    branch_plot = create_probe_loss_by_branch_plot(probe_tracking)
                    log_dict["probe/images/loss_by_branch"] = wandb.Image(
                        branch_plot, caption="Probe Loss by Curriculum Branch"
                    )
                except Exception as e:
                    print(f"Warning: Could not create branch plot: {e}")

                # Novelty-Learnability Pareto plot
                try:
                    nl_plot = create_novelty_learnability_plot(
                        novelty, learnability, openendedness_score, regime
                    )
                    log_dict["probe/images/novelty_learnability"] = wandb.Image(
                        nl_plot, caption="Novelty-Learnability Space"
                    )
                except Exception as e:
                    print(f"Warning: Could not create N-L plot: {e}")

                # Correlation scatter plot
                try:
                    corr_plot = create_correlation_scatter_plot(probe_tracking)
                    log_dict["probe/images/accuracy_return_correlation"] = wandb.Image(
                        corr_plot, caption="Probe Accuracy vs Agent Return"
                    )
                except Exception as e:
                    print(f"Warning: Could not create correlation plot: {e}")

                # Hidden state t-SNE (only periodically, as it's expensive)
                if stats["update_count"] % (config.get("tsne_freq", 10) * config["eval_freq"]) == 0:
                    try:
                        n_samples = min(int(train_state.hstate_sample_ptr), train_state.hstate_samples.shape[0])
                        if n_samples >= 50:
                            tsne_plot = create_hidden_state_tsne_plot(
                                train_state.hstate_samples[:n_samples],
                                train_state.hstate_sample_branches[:n_samples],
                                max_samples=500
                            )
                            log_dict["probe/images/hidden_state_tsne"] = wandb.Image(
                                tsne_plot, caption="Hidden State t-SNE by Branch"
                            )
                    except Exception as e:
                        print(f"Warning: Could not create t-SNE plot: {e}")

                # Pareto trajectory plot (novelty vs learnability over training)
                if train_state.pareto_history is not None:
                    try:
                        pareto_plot = create_pareto_trajectory_plot(train_state.pareto_history)
                        log_dict["probe/images/pareto_trajectory"] = wandb.Image(
                            pareto_plot, caption="Novelty-Learnability Trajectory"
                        )
                    except Exception as e:
                        print(f"Warning: Could not create Pareto trajectory plot: {e}")

                # Information Content Dashboard
                try:
                    probe_metrics_for_dashboard = {
                        # Distributional calibration metrics (fair comparison across branches)
                        'wall_dist_accuracy': float(probe_stats["wall_dist_accuracy"].mean()),
                        'goal_dist_mode_match': float(probe_stats["goal_dist_mode_match"].mean()),
                        'agent_pos_dist_mode_match': float(probe_stats["agent_pos_dist_mode_match"].mean()),
                        'agent_dir_dist_mode_match': float(probe_stats["agent_dir_dist_mode_match"].mean()),
                        'wall_loss': float(probe_stats["wall_loss"].mean()),
                        'goal_loss': float(probe_stats["goal_loss"].mean()),
                        'agent_pos_loss': float(probe_stats["agent_pos_loss"].mean()),
                        'agent_dir_loss': float(probe_stats["agent_dir_loss"].mean()),
                    }
                    info_dashboard = create_information_content_dashboard(
                        probe_metrics_for_dashboard, probe_tracking,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["probe/images/information_dashboard"] = wandb.Image(
                        info_dashboard, caption="Information Content Dashboard"
                    )
                except Exception as e:
                    print(f"Warning: Could not create information dashboard: {e}")

        wandb.log(log_dict)

    # setup the environment
    env = Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    eval_env = env
    sample_random_level = make_level_generator(env.max_height, env.max_width, config["n_walls"])
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params
    mutate_level = make_level_mutator_minimax(100)

    # and the level sampler
    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config["top_k"]},
        duplicate_check=config["buffer_duplicate_check"],
    )


    @jax.jit
    def create_train_state(rng) -> TrainState:
        # creates the train state
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["num_minibatches"] * config["epoch_ppo"]))
                / config["num_updates"]
            )
            return config["lr"] * frac

        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
            obs,
        )

        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
        network = ActorCritic(env.action_space(env_params).n)
        network_params = network.init(rng, init_x, ActorCritic.initialize_carry((config["num_train_envs"],)))
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
        pholder_level = sample_random_level(jax.random.PRNGKey(0))
        sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
        pholder_level_batch = jax.tree_util.tree_map(lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0), pholder_level)

        # Initialize probe if enabled
        probe_params = None
        probe_opt_state = None
        probe_tracking = None
        hstate_samples = None
        hstate_sample_branches = None
        # Initialize hstates with proper shapes for JAX tracing
        init_hstate = ActorCritic.initialize_carry((config["num_train_envs"],))

        # Max samples to store for t-SNE visualization
        max_hstate_samples = config.get("max_hstate_samples", 1000)

        # Also initialize pareto history for novelty/learnability tracking
        pareto_history = None

        if config.get("use_probe", False):
            rng, rng_probe = jax.random.split(rng)
            use_episode_context = config.get("probe_use_episode_context", True)
            probe_network = CurriculumProbe(
                env_height=DEFAULT_ENV_HEIGHT,
                env_width=DEFAULT_ENV_WIDTH,
                use_episode_context=use_episode_context,
            )
            # Probe input is flattened LSTM state: (batch, 512)
            # LSTM cell has (c, h) each of size 256
            hstate_dim = DEFAULT_HSTATE_DIM
            num_envs = config["num_train_envs"]
            dummy_probe_input = jnp.zeros((num_envs, hstate_dim))
            # Initialize with episode context if enabled
            if use_episode_context:
                dummy_return = jnp.zeros((num_envs,))
                dummy_solved = jnp.zeros((num_envs,), dtype=bool)
                dummy_length = jnp.zeros((num_envs,))
                probe_params = probe_network.init(
                    rng_probe, dummy_probe_input,
                    episode_return=dummy_return,
                    episode_solved=dummy_solved,
                    episode_length=dummy_length,
                )
            else:
                probe_params = probe_network.init(rng_probe, dummy_probe_input)
            probe_tx = optax.adam(learning_rate=config.get("probe_lr", 1e-4))
            probe_opt_state = probe_tx.init(probe_params)

            # Initialize probe tracking state with configurable hstate_dim
            probe_tracking = create_probe_tracking_state(buffer_size=500, hstate_dim=hstate_dim)

            # Initialize hidden state sample storage for t-SNE
            hstate_samples = jnp.zeros((max_hstate_samples, hstate_dim))
            hstate_sample_branches = jnp.zeros(max_hstate_samples, dtype=jnp.int32)

            # Initialize pareto history for novelty/learnability over training
            pareto_history = create_pareto_history_state(max_checkpoints=config["num_updates"] // config["eval_freq"] + 10)

        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=0,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            mutation_last_level_batch=pholder_level_batch,
            probe_params=probe_params,
            probe_opt_state=probe_opt_state,
            current_hstate=init_hstate,
            prev_hstate=init_hstate,
            current_branch=0,
            prev_branch=0,
            has_valid_prev_hstate=False,
            probe_tracking=probe_tracking,
            pareto_history=pareto_history,
            hstate_samples=hstate_samples,
            hstate_sample_branches=hstate_sample_branches,
            hstate_sample_ptr=0,
            last_agent_return=jnp.array(0.0),
            training_step=0,
            # Initialize episode context for probe
            last_episode_return=jnp.zeros(config["num_train_envs"]),
            last_episode_solved=jnp.zeros(config["num_train_envs"], dtype=bool),
            last_episode_length=jnp.zeros(config["num_train_envs"]),
        )

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        """This is the main training loop. It basically calls either `on_new_levels`, `on_replay_levels`,
        or `on_mutate_levels` at every step."""

        def on_new_levels(rng: chex.PRNGKey, train_state: TrainState):
            """Samples new randomly-generated levels and evaluated the policy on these. It also then adds the levels to the
            level buffer if they have high-enough scores. The agent is updated on these trajectories iff `config["exploratory_grad_updates"]` is True"""

            sampler = train_state.sampler

            # reset
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, config["num_train_envs"]))
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), new_levels, env_params)

            # rollout
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )

            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            max_returns = compute_max_returns(dones, rewards)
            scores = compute_score(config, dones, values, max_returns, advantages)
            sampler, _ = level_sampler.insert_batch(sampler, new_levels, scores, {"max_return": max_returns})

            # update: train_state only modified if exploratory_grad_updates is on
            (rng, train_state), losses = update_actor_critic_rnn(
                rng,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
            )

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": new_levels.wall_map.sum() / config["num_train_envs"],
            }

            # Compute episode context for probe
            episode_returns = rewards.sum(axis=0)  # Sum over time, shape (num_envs,)
            mean_return = episode_returns.mean()
            # episode_solved: True if agent got positive reward (only goal gives reward > 0)
            episode_solved = (rewards > 0).any(axis=0)  # (num_envs,)
            # episode_length: first step where done=True, or num_steps if never done
            num_steps = dones.shape[0]
            has_done = dones.any(axis=0)  # (num_envs,)
            # argmax finds first True index (returns 0 if all False, but we handle that)
            first_done_idx = jnp.argmax(dones.astype(jnp.int32), axis=0)  # (num_envs,)
            episode_length = jnp.where(has_done, first_done_idx + 1, num_steps).astype(jnp.float32)

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=new_levels,
                current_hstate=hstate,  # Store for probe
                current_branch=0,       # DR branch = 0
                last_agent_return=mean_return,  # Track for correlation
                last_episode_return=episode_returns,
                last_episode_solved=episode_solved,
                last_episode_length=episode_length,
            )
            return (rng, train_state), metrics

        def on_replay_levels(rng: chex.PRNGKey, train_state: TrainState):
            """This samples levels from the level buffer, and updates the policy on them"""
            sampler = train_state.sampler

            # collect trajectories on replay levels
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, config["num_train_envs"])
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), levels, env_params)
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            max_returns = jnp.maximum(level_sampler.get_levels_extra(sampler, level_inds)["max_return"], compute_max_returns(dones, rewards))
            scores = compute_score(config, dones, values, max_returns, advantages)
            sampler = level_sampler.update_batch(sampler, level_inds, scores, {"max_return": max_returns})

            # update the policy using trajectories collected from levels
            (rng, train_state), losses = update_actor_critic_rnn(
                rng,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=True,
            )

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": levels.wall_map.sum() / config["num_train_envs"],
            }

            # Compute episode context for probe
            episode_returns = rewards.sum(axis=0)  # Sum over time, shape (num_envs,)
            mean_return = episode_returns.mean()
            # episode_solved: True if agent got positive reward (only goal gives reward > 0)
            episode_solved = (rewards > 0).any(axis=0)  # (num_envs,)
            # episode_length: first step where done=True, or num_steps if never done
            num_steps = dones.shape[0]
            has_done = dones.any(axis=0)  # (num_envs,)
            first_done_idx = jnp.argmax(dones.astype(jnp.int32), axis=0)  # (num_envs,)
            episode_length = jnp.where(has_done, first_done_idx + 1, num_steps).astype(jnp.float32)

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.REPLAY,
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
                current_hstate=hstate,  # Store for probe
                current_branch=1,       # Replay branch = 1
                last_agent_return=mean_return,  # Track for correlation
                last_episode_return=episode_returns,
                last_episode_solved=episode_solved,
                last_episode_length=episode_length,
            )
            return (rng, train_state), metrics

        def on_mutate_levels(rng: chex.PRNGKey, train_state: TrainState):
            """This mutates the previous batch of replay levels and potentially adds them to the level buffer.
            This also updates the policy iff `config["exploratory_grad_updates"]` is True."""

            sampler = train_state.sampler
            rng, rng_mutate, rng_reset = jax.random.split(rng, 3)

            # mutate
            parent_levels = train_state.replay_last_level_batch
            child_levels = jax.vmap(mutate_level, (0, 0, None))(jax.random.split(rng_mutate, config["num_train_envs"]), parent_levels, config["num_edits"])
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), child_levels, env_params)

            # rollout
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )

            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            max_returns = compute_max_returns(dones, rewards)
            scores = compute_score(config, dones, values, max_returns, advantages)
            sampler, _ = level_sampler.insert_batch(sampler, child_levels, scores, {"max_return": max_returns})

            # update: train_state only modified if exploratory_grad_updates is on
            (rng, train_state), losses = update_actor_critic_rnn(
                rng,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
            )

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": child_levels.wall_map.sum() / config["num_train_envs"],
            }

            # Compute episode context for probe
            episode_returns = rewards.sum(axis=0)  # Sum over time, shape (num_envs,)
            mean_return = episode_returns.mean()
            # episode_solved: True if agent got positive reward (only goal gives reward > 0)
            episode_solved = (rewards > 0).any(axis=0)  # (num_envs,)
            # episode_length: first step where done=True, or num_steps if never done
            num_steps = dones.shape[0]
            has_done = dones.any(axis=0)  # (num_envs,)
            first_done_idx = jnp.argmax(dones.astype(jnp.int32), axis=0)  # (num_envs,)
            episode_length = jnp.where(has_done, first_done_idx + 1, num_steps).astype(jnp.float32)

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_mutation_updates=train_state.num_mutation_updates + 1,
                mutation_last_level_batch=child_levels,
                current_hstate=hstate,  # Store for probe
                current_branch=2,       # Mutate branch = 2
                last_agent_return=mean_return,  # Track for correlation
                last_episode_return=episode_returns,
                last_episode_solved=episode_solved,
                last_episode_length=episode_length,
            )
            return (rng, train_state), metrics

        rng, train_state = carry
        rng, rng_replay = jax.random.split(rng)

        # Save prev_hstate before the branch executes (for probe prediction)
        saved_prev_hstate = train_state.prev_hstate
        saved_prev_branch = train_state.prev_branch

        # the train step makes a decision on which branch to take, either on_new, on_replay or on_mutate
        # on_mutate is only called if the replay branch has been taken before (as it uses train_state.update_state).

        if config["use_accel"]:
            s = train_state.update_state
            branch = (1 - s) * level_sampler.sample_replay_decision(train_state.sampler, rng_replay) + 2 * s
        else:
            branch = level_sampler.sample_replay_decision(train_state.sampler, rng_replay).astype(int)

        # Execute the branch (agent trains, stores current_hstate and current_branch)
        (rng, train_state), metrics = jax.lax.switch(
            branch,
            [
                on_new_levels,
                on_replay_levels,
                on_mutate_levels,
            ],
            rng, train_state,
        )

        # =====================================================================
        # PROBE: Predict current level from previous hidden state
        # =====================================================================
        if config.get("use_probe", False):
            # Get the level batch that was just used (based on which branch executed)
            current_level_batch = jax.lax.switch(
                train_state.current_branch,
                [
                    lambda: train_state.dr_last_level_batch,
                    lambda: train_state.replay_last_level_batch,
                    lambda: train_state.mutation_last_level_batch,
                ],
            )

            def compute_and_update_probe(operand):
                """
                Compute probe loss using DISTRIBUTIONAL loss (not per-instance index pairing).

                Uses:
                - Distributional loss for training (compares predicted vs empirical batch distributions)
                - Episode context (return, solved, length) as additional probe inputs
                - Greedy matching for visualization metrics
                - Per-instance loss for Replay->Mutate transition (where 1-to-1 correspondence exists)
                """
                train_state, saved_prev_hstate, current_level_batch, saved_prev_branch = operand
                num_envs = config["num_train_envs"]

                # Flatten LSTM hidden state: (c, h) each of shape (num_envs, 256) -> (num_envs, 512)
                h_c, h_h = saved_prev_hstate
                hstate_batch = jnp.concatenate([h_c, h_h], axis=-1)  # (num_envs, 512)
                # Apply stop_gradient - probe doesn't affect agent
                hstate_batch = jax.lax.stop_gradient(hstate_batch)

                # Get episode context (if available)
                episode_return = train_state.last_episode_return
                episode_solved = train_state.last_episode_solved
                episode_length = train_state.last_episode_length

                # Define loss function using DISTRIBUTIONAL loss
                def probe_loss_fn(probe_params):
                    probe_network = CurriculumProbe(
                        env_height=DEFAULT_ENV_HEIGHT,
                        env_width=DEFAULT_ENV_WIDTH,
                        use_episode_context=config.get("probe_use_episode_context", True),
                    )
                    # Run probe on full batch with episode context
                    predictions_batch = probe_network.apply(
                        probe_params,
                        hstate_batch,
                        episode_return=episode_return,
                        episode_solved=episode_solved,
                        episode_length=episode_length,
                    )

                    # Compute DISTRIBUTIONAL loss (not per-instance)
                    dist_loss, dist_metrics = compute_distributional_probe_loss(
                        predictions_batch,
                        current_level_batch,
                        env_height=DEFAULT_ENV_HEIGHT,
                        env_width=DEFAULT_ENV_WIDTH,
                    )

                    return dist_loss, (dist_metrics, predictions_batch)

                # Compute gradients and loss
                (loss, (dist_metrics, predictions_batch)), grads = jax.value_and_grad(
                    probe_loss_fn, has_aux=True
                )(train_state.probe_params)

                # Update probe parameters
                probe_tx = optax.adam(learning_rate=config.get("probe_lr", 1e-4))
                updates, new_opt_state = probe_tx.update(
                    grads, train_state.probe_opt_state, train_state.probe_params
                )
                new_probe_params = optax.apply_updates(train_state.probe_params, updates)

                # ===== COMPUTE DISTRIBUTIONAL CALIBRATION (all branches) =====
                # Compare batch-averaged predictions to empirical level distribution
                # This is meaningful for ALL branches - no 1-to-1 correspondence assumed
                dist_calibration = compute_distributional_calibration_metrics(
                    predictions_batch, current_level_batch,
                    env_height=DEFAULT_ENV_HEIGHT,
                    env_width=DEFAULT_ENV_WIDTH,
                )

                # ===== COMPUTE REPLAY->MUTATE SPECIFIC METRICS =====
                # When prev_branch=1 (Replay) and current_branch=2 (Mutate),
                # there IS 1-to-1 correspondence: mutation_level[i] = mutate(replay_level[i])
                # So pred[i] from hidden state after replay_level[i] should predict mutation_level[i]
                is_replay_to_mutate = (saved_prev_branch == 1) & (train_state.current_branch == 2)

                # Compute per-instance loss for Replay->Mutate (meaningful only when is_replay_to_mutate)
                per_instance_loss, per_instance_metrics = compute_per_instance_loss_batch(
                    predictions_batch, current_level_batch,
                    env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH,
                )

                # Compute per-instance CALIBRATION for R->M (meaningful only when is_replay_to_mutate)
                per_instance_calibration = compute_per_instance_calibration_batch(
                    predictions_batch, current_level_batch,
                    env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH,
                )

                # Use distributional accuracy for correlation tracking (fair for all branches)
                probe_accuracy = (
                    dist_calibration['wall_dist_accuracy'] * 0.4 +
                    dist_calibration['goal_dist_mode_match'] * 0.2 +
                    dist_calibration['agent_pos_dist_mode_match'] * 0.2 +
                    dist_calibration['agent_dir_dist_mode_match'] * 0.2
                )

                # Use first env's hidden state for tracking
                hstate_flat = jnp.concatenate([h_c[0], h_h[0]], axis=-1)

                # ===== UPDATE PROBE TRACKING STATE =====
                new_probe_tracking = update_probe_tracking_state(
                    train_state.probe_tracking,
                    probe_loss=loss,
                    branch=saved_prev_branch,
                    agent_return=train_state.last_agent_return,
                    probe_accuracy=probe_accuracy,
                    hstate_flat=hstate_flat,
                    training_step=train_state.training_step,
                )

                # ===== STORE HIDDEN STATE SAMPLE FOR T-SNE =====
                max_samples = train_state.hstate_samples.shape[0]
                sample_ptr = train_state.hstate_sample_ptr % max_samples
                new_hstate_samples = train_state.hstate_samples.at[sample_ptr].set(hstate_flat)
                new_hstate_sample_branches = train_state.hstate_sample_branches.at[sample_ptr].set(saved_prev_branch)

                new_train_state = train_state.replace(
                    probe_params=new_probe_params,
                    probe_opt_state=new_opt_state,
                    probe_tracking=new_probe_tracking,
                    hstate_samples=new_hstate_samples,
                    hstate_sample_branches=new_hstate_sample_branches,
                    hstate_sample_ptr=sample_ptr + 1,
                )

                # Build metrics dict
                probe_metrics = {
                    # Distributional loss (used for training)
                    'total_dist_loss': dist_metrics['total_dist_loss'],
                    'wall_dist_loss': dist_metrics['wall_dist_loss'],
                    'goal_dist_loss': dist_metrics['goal_dist_loss'],
                    'agent_pos_dist_loss': dist_metrics['agent_pos_dist_loss'],
                    'agent_dir_dist_loss': dist_metrics['agent_dir_dist_loss'],
                    # Per-instance loss (meaningful for Replay->Mutate)
                    'per_instance_loss': per_instance_loss,
                    'per_instance_wall_loss': per_instance_metrics['wall_loss'],
                    'is_replay_to_mutate': is_replay_to_mutate.astype(jnp.float32),
                    # Legacy: total_loss for backward compatibility
                    'total_loss': loss,
                    'wall_loss': dist_metrics['wall_dist_loss'],  # Map to dist loss
                    'goal_loss': dist_metrics['goal_dist_loss'],
                    'agent_pos_loss': dist_metrics['agent_pos_dist_loss'],
                    'agent_dir_loss': dist_metrics['agent_dir_dist_loss'],
                    # === DISTRIBUTIONAL CALIBRATION (all branches, fair comparison) ===
                    'branch': saved_prev_branch,
                    'wall_dist_calibration': dist_calibration['wall_dist_calibration'],
                    'goal_dist_calibration': dist_calibration['goal_dist_calibration'],
                    'agent_pos_dist_calibration': dist_calibration['agent_pos_dist_calibration'],
                    'agent_dir_dist_calibration': dist_calibration['agent_dir_dist_calibration'],
                    'wall_dist_accuracy': dist_calibration['wall_dist_accuracy'],
                    'goal_dist_mode_match': dist_calibration['goal_dist_mode_match'],
                    'agent_pos_dist_mode_match': dist_calibration['agent_pos_dist_mode_match'],
                    'agent_dir_dist_mode_match': dist_calibration['agent_dir_dist_mode_match'],
                    'combined_accuracy': probe_accuracy,  # Now uses distributional accuracy
                    # === PER-INSTANCE CALIBRATION (R->M only, use with is_replay_to_mutate) ===
                    'r2m_wall_accuracy': per_instance_calibration['wall_accuracy'],
                    'r2m_goal_accuracy': per_instance_calibration['goal_accuracy'],
                    'r2m_agent_pos_accuracy': per_instance_calibration['agent_pos_accuracy'],
                    'r2m_agent_dir_accuracy': per_instance_calibration['agent_dir_accuracy'],
                    'r2m_combined_accuracy': per_instance_calibration['combined_accuracy'],
                    'r2m_goal_prob_at_actual': per_instance_calibration['goal_prob_at_actual'],
                    'r2m_agent_pos_prob_at_actual': per_instance_calibration['agent_pos_prob_at_actual'],
                    'r2m_agent_dir_prob_at_actual': per_instance_calibration['agent_dir_prob_at_actual'],
                    # Store FULL BATCH for visualization (greedy matching done at log time)
                    '_predictions': predictions_batch,
                    '_current_level': current_level_batch,
                }

                return new_train_state, probe_metrics

            def skip_probe_first_step(operand):
                """First step only: no previous hidden state available yet, return placeholder."""
                train_state, _, current_level_batch, saved_prev_branch = operand
                num_envs = config["num_train_envs"]
                # Create batched dummy predictions to match the new structure
                dummy_predictions_batch = {
                    'wall_logits': jnp.zeros((num_envs, DEFAULT_ENV_HEIGHT, DEFAULT_ENV_WIDTH)),
                    'goal_logits': jnp.zeros((num_envs, DEFAULT_ENV_HEIGHT * DEFAULT_ENV_WIDTH)),
                    'agent_pos_logits': jnp.zeros((num_envs, DEFAULT_ENV_HEIGHT * DEFAULT_ENV_WIDTH)),
                    'agent_dir_logits': jnp.zeros((num_envs, 4)),
                }
                dummy_metrics = {
                    # Distributional loss metrics
                    'total_dist_loss': jnp.array(0.0),
                    'wall_dist_loss': jnp.array(0.0),
                    'goal_dist_loss': jnp.array(0.0),
                    'agent_pos_dist_loss': jnp.array(0.0),
                    'agent_dir_dist_loss': jnp.array(0.0),
                    # Per-instance loss metrics
                    'per_instance_loss': jnp.array(0.0),
                    'per_instance_wall_loss': jnp.array(0.0),
                    'is_replay_to_mutate': jnp.array(0.0),
                    # Legacy metrics (for backward compatibility)
                    'wall_loss': jnp.array(0.0),
                    'goal_loss': jnp.array(0.0),
                    'agent_pos_loss': jnp.array(0.0),
                    'agent_dir_loss': jnp.array(0.0),
                    'total_loss': jnp.array(0.0),
                    # === DISTRIBUTIONAL CALIBRATION (all branches) ===
                    'branch': saved_prev_branch,
                    'wall_dist_calibration': jnp.array(0.0),
                    'goal_dist_calibration': jnp.array(0.0),
                    'agent_pos_dist_calibration': jnp.array(0.0),
                    'agent_dir_dist_calibration': jnp.array(0.0),
                    'wall_dist_accuracy': jnp.array(0.0),
                    'goal_dist_mode_match': jnp.array(0.0),
                    'agent_pos_dist_mode_match': jnp.array(0.0),
                    'agent_dir_dist_mode_match': jnp.array(0.0),
                    'combined_accuracy': jnp.array(0.0),
                    # === PER-INSTANCE CALIBRATION (R->M only) ===
                    'r2m_wall_accuracy': jnp.array(0.0),
                    'r2m_goal_accuracy': jnp.array(0.0),
                    'r2m_agent_pos_accuracy': jnp.array(0.0),
                    'r2m_agent_dir_accuracy': jnp.array(0.0),
                    'r2m_combined_accuracy': jnp.array(0.0),
                    'r2m_goal_prob_at_actual': jnp.array(0.0),
                    'r2m_agent_pos_prob_at_actual': jnp.array(0.0),
                    'r2m_agent_dir_prob_at_actual': jnp.array(0.0),
                    # Full batch for visualization
                    '_predictions': dummy_predictions_batch,
                    '_current_level': current_level_batch,
                }
                return train_state, dummy_metrics

            # Use jax.lax.cond to conditionally compute probe (pass batch, not single level)
            train_state, probe_metrics = jax.lax.cond(
                train_state.has_valid_prev_hstate,
                compute_and_update_probe,
                skip_probe_first_step,
                (train_state, saved_prev_hstate, current_level_batch, saved_prev_branch),
            )
            metrics['probe'] = probe_metrics

            # Update prev_hstate for next iteration, set flag to True, and increment training step
            train_state = train_state.replace(
                prev_hstate=train_state.current_hstate,
                prev_branch=train_state.current_branch,
                has_valid_prev_hstate=True,
                training_step=train_state.training_step + 1,
            )
        else:
            # Still increment training_step even if probe is disabled
            train_state = train_state.replace(
                training_step=train_state.training_step + 1,
            )

        return (rng, train_state), metrics

    def eval(rng: chex.PRNGKey, train_state: TrainState):
        """This evaluates the current policy on the ste of evaluation levels specified by config["eval_levels"]."""

        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["eval_levels"])
        num_levels = len(config["eval_levels"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)
        states, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths

    @jax.jit
    def train_and_eval_step(runner_state, _):
        """This function runs the train_step for a certain number of iterations, and then evaluates teh policy."""

        # train
        (rng, train_state), metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])

        # eval
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, config["eval_num_attempts"]), train_state)

        # collect metrics
        eval_solve_rates = jnp.where(cum_rewards > 0, 1., 0.).mean(axis=0)
        eval_returns = cum_rewards.mean(axis=0)

        # just grab the first run
        states, episode_lengths = jax.tree_util.tree_map(lambda x: x[0], (states, episode_lengths))
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params)
        frames = images.transpose(0, 1, 4, 2, 3)

        metrics["update_count"] = train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"] = episode_lengths
        metrics["eval_animation"] = (frames, episode_lengths)
        metrics["dr_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.dr_last_level_batch, env_params)
        metrics["replay_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.replay_last_level_batch, env_params)
        metrics["mutation_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.mutation_last_level_batch, env_params)

        highest_scoring_level = level_sampler.get_levels(train_state.sampler, train_state.sampler["scores"].argmax())
        highest_weighted_level = level_sampler.get_levels(train_state.sampler, level_sampler.level_weights(train_state.sampler).argmax())

        metrics["highest_scoring_level"] = env_renderer.render_level(highest_scoring_level, env_params)
        metrics["highest_weighted_level"] = env_renderer.render_level(highest_weighted_level, env_params)

        # Update Pareto history at eval time (inside JIT for proper state management)
        if config.get("use_probe", False) and train_state.pareto_history is not None:
            # Use JIT-compatible versions (return JAX arrays, not Python floats)
            novelty, _, _, _, _, _ = _compute_novelty_jit(train_state.probe_tracking)
            learnability, _, _, _, _, _ = _compute_learnability_jit(train_state.probe_tracking)
            new_pareto_history = update_pareto_history(
                train_state.pareto_history,
                novelty=novelty,
                learnability=learnability,
                training_step=train_state.training_step,
            )
            train_state = train_state.replace(pareto_history=new_pareto_history)

        return (rng, train_state), metrics

    def eval_checkpoint(og_config):
        """This function is what is used to evaluate a saved checkpoint after training. It first loads the checkpoint and
        then runs evaluation."""
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))

        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, "config.json")) as f:
                config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_directory, "models"), item_handlers=ocp.StandardCheckpointHandler())

            train_state_og: TrainState = create_train_state(rng_init)
            step = checkpoint_manager.latest_step() if og_config["checkpoint_to_eval"] == -1 else og_config["checkpoint_to_eval"]

            loaded_checkpoint = checkpoint_manager.restore(step)
            params = loaded_checkpoint["params"]
            train_state = train_state_og.replace(params=params)
            return train_state, config

        train_state, config = load(rng_init, og_config["checkpoint_directory"])
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, og_config["eval_num_attempts"]), train_state)
        save_loc = og_config["checkpoint_directory"].replace("checkpoints", "results")
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(os.path.join(save_loc, "results.npz"), states=np.asarray(states), cum_rewards=np.asarray(cum_rewards), episode_lengths=np.asarray(episode_lengths), levels=config["eval_levels"])
        return states, cum_rewards, episode_lengths

    def post_training_probe_analysis(train_state: TrainState, num_samples: int = 500):
        """
        Comprehensive probe analysis after training.

        Runs the trained agent on many levels from each branch and analyzes
        what the agent has learned about curriculum dynamics.

        Fix: Uses actual replay buffer for replay branch analysis.
        """
        if not config.get("use_probe", False) or train_state.probe_params is None:
            print("Probe not enabled, skipping post-training analysis.")
            return {}

        print("\n" + "="*60)
        print("POST-TRAINING PROBE ANALYSIS")
        print("="*60)

        rng = jax.random.PRNGKey(42)
        results = {
            'branch_losses': {0: [], 1: [], 2: []},
            'branch_accuracies': {0: [], 1: [], 2: []},
            'hstates': [],
            'branches': [],
        }

        samples_per_branch = num_samples // 3

        # Collect samples from each branch
        for branch_id in range(3):
            branch_name = ['Random (DR)', 'Replay', 'Mutate'][branch_id]
            print(f"\nAnalyzing {branch_name} branch...")

            for i in range(samples_per_branch):
                rng, rng_level, rng_reset, rng_rollout = jax.random.split(rng, 4)

                # Generate level based on branch
                if branch_id == 0:  # Random
                    level = sample_random_level(rng_level)
                elif branch_id == 1:  # Replay - sample from actual replay buffer
                    # Use actual replay buffer via level_sampler (Fix: post-training analysis)
                    sampler_state, (level_inds, replay_levels) = level_sampler.sample_replay_levels(
                        train_state.sampler, rng_level, 1
                    )
                    level = jax.tree_util.tree_map(lambda x: x[0], replay_levels)
                else:  # Mutate - mutate from replay buffer levels
                    # Sample parent from replay buffer, then mutate
                    sampler_state, (level_inds, parent_levels) = level_sampler.sample_replay_levels(
                        train_state.sampler, rng_level, 1
                    )
                    parent = jax.tree_util.tree_map(lambda x: x[0], parent_levels)
                    level = mutate_level(rng_level, parent, config["num_edits"])

                # Run agent on level
                init_obs, init_env_state = env.reset_to_level(rng_reset, level, env_params)
                init_obs_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)
                init_env_state_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_env_state)

                # Get hidden state after rollout
                (_, _, hstate, _, _, _), _ = sample_trajectories_rnn(
                    rng_rollout, env, env_params, train_state,
                    ActorCritic.initialize_carry((1,)),
                    init_obs_batch, init_env_state_batch,
                    1, config["num_steps"],
                )

                # Flatten hidden state
                h_c, h_h = hstate
                hstate_flat = jnp.concatenate([h_c[0], h_h[0]], axis=-1)

                # Run probe (use same config as training)
                use_episode_context = config.get("probe_use_episode_context", True)
                probe_network = CurriculumProbe(
                    env_height=DEFAULT_ENV_HEIGHT,
                    env_width=DEFAULT_ENV_WIDTH,
                    use_episode_context=use_episode_context,
                )
                probe_input = jax.lax.stop_gradient(hstate_flat[None, ...])
                # Pass dummy episode context if probe was trained with it
                if use_episode_context:
                    dummy_return = jnp.zeros((1,))
                    dummy_solved = jnp.zeros((1,), dtype=bool)
                    dummy_length = jnp.zeros((1,))
                    predictions = probe_network.apply(
                        train_state.probe_params, probe_input,
                        episode_return=dummy_return,
                        episode_solved=dummy_solved,
                        episode_length=dummy_length,
                    )
                else:
                    predictions = probe_network.apply(train_state.probe_params, probe_input)
                predictions = jax.tree_util.tree_map(lambda x: x[0], predictions)

                # Compute loss and accuracy
                loss, _ = compute_probe_loss(predictions, level)
                calibration = compute_calibration_metrics(predictions, level)

                results['branch_losses'][branch_id].append(float(loss))
                results['branch_accuracies'][branch_id].append(float(calibration['wall_accuracy']))
                results['hstates'].append(np.array(hstate_flat))
                results['branches'].append(branch_id)

        # Print summary
        print("\n" + "-"*40)
        print("PROBE ANALYSIS SUMMARY")
        print("-"*40)

        for branch_id in range(3):
            branch_name = ['Random (DR)', 'Replay', 'Mutate'][branch_id]
            losses = results['branch_losses'][branch_id]
            accs = results['branch_accuracies'][branch_id]
            print(f"\n{branch_name}:")
            print(f"  Mean Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
            print(f"  Mean Wall Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

        # Key insight: compare branches
        print("\n" + "-"*40)
        print("KEY INSIGHTS")
        print("-"*40)

        random_loss = np.mean(results['branch_losses'][0])
        replay_loss = np.mean(results['branch_losses'][1])
        mutate_loss = np.mean(results['branch_losses'][2])

        if replay_loss < random_loss and mutate_loss < random_loss:
            print("✓ Agent has learned curriculum-specific patterns!")
            print(f"  Replay/Mutate branches are {((random_loss - replay_loss) / random_loss * 100):.1f}% more predictable than random.")
        elif replay_loss > random_loss:
            print("✗ Curriculum levels are LESS predictable than random.")
            print("  The curriculum may be generating more novel/surprising levels.")
        else:
            print("~ Mixed results across branches.")

        return results

    if config["mode"] == "eval":
        return eval_checkpoint(config)

    # set up the train states
    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)

    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    # =========================================================================
    # COMPUTE BASELINES (once at start of training)
    # =========================================================================
    empirical_baseline = None
    random_baselines = compute_random_baselines(DEFAULT_ENV_HEIGHT, DEFAULT_ENV_WIDTH)

    if config.get("use_probe", False):
        print("\n" + "="*60)
        print("COMPUTING BASELINES")
        print("="*60)

        # Theoretical random baseline (already computed above)
        print(f"Theoretical random baseline (wall acc): {random_baselines['wall_accuracy']:.2%}")
        print(f"Theoretical random baseline (goal top1): {random_baselines['goal_top1']:.2%}")

        # Empirical baseline: probe untrained agent
        print("\nComputing empirical baseline (probing untrained agent)...")
        rng_baseline, rng_train = jax.random.split(rng_train)
        try:
            empirical_baseline = compute_empirical_baseline(
                env=env,
                env_params=env_params,
                sample_random_level=sample_random_level,
                rng=rng_baseline,
                num_episodes=30,
                probe_train_steps=50,
                env_height=DEFAULT_ENV_HEIGHT,
                env_width=DEFAULT_ENV_WIDTH,
            )
            print(f"Empirical baseline (wall acc): {empirical_baseline['wall_accuracy']:.2%}")
            print(f"Empirical baseline (goal top1): {empirical_baseline['goal_top1_correct']:.2%}")

            # Log baselines to wandb
            wandb.log({
                "baselines/random_wall_accuracy": random_baselines['wall_accuracy'],
                "baselines/random_goal_top1": random_baselines['goal_top1'],
                "baselines/random_total_loss": random_baselines['total_loss'],
                "baselines/empirical_wall_accuracy": empirical_baseline['wall_accuracy'],
                "baselines/empirical_goal_top1": empirical_baseline['goal_top1_correct'],
                "baselines/empirical_total_loss": empirical_baseline['total_loss'],
            })
        except Exception as e:
            print(f"Warning: Failed to compute empirical baseline: {e}")
            empirical_baseline = None

        runner_state = (rng_train, train_state)

    # and run the train_eval_sep function for the specified number of updates
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics["time_delta"] = curr_time - start_time
        log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler))
        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(eval_step, runner_state[1])
            checkpoint_manager.wait_until_finished()

    # Run post-training probe analysis if enabled
    if config.get("use_probe", False):
        final_train_state = runner_state[1]

        # Standard post-training analysis
        probe_results = post_training_probe_analysis(final_train_state, num_samples=300)

        # Fresh probe evaluation (clean measurement)
        print("\nRunning fresh probe evaluation...")
        rng_fresh = jax.random.PRNGKey(999)
        try:
            fresh_probe_results = evaluate_fresh_probe(
                agent_params=final_train_state.params,
                agent_apply_fn=final_train_state.apply_fn,
                env=env,
                env_params=env_params,
                sample_random_level=sample_random_level,
                level_sampler=level_sampler,
                sampler_state=final_train_state.sampler,
                rng=rng_fresh,
                num_episodes=50,
                probe_train_steps=100,
                env_height=DEFAULT_ENV_HEIGHT,
                env_width=DEFAULT_ENV_WIDTH,
            )
            print(f"Fresh probe wall accuracy: {fresh_probe_results['wall_accuracy']:.2%}")
            print(f"Fresh probe info gain (wall): {fresh_probe_results['info_gain_wall']:.2%}")
        except Exception as e:
            print(f"Warning: Fresh probe evaluation failed: {e}")
            fresh_probe_results = None

        # N-step prediction evaluation
        print("\nRunning N-step prediction evaluation...")
        try:
            nstep_results = evaluate_nstep_prediction(
                probe_params=final_train_state.probe_params,
                agent_params=final_train_state.params,
                agent_apply_fn=final_train_state.apply_fn,
                env=env,
                env_params=env_params,
                sample_random_level=sample_random_level,
                level_sampler=level_sampler,
                sampler_state=final_train_state.sampler,
                rng=jax.random.PRNGKey(1000),
                n_steps=5,
                num_episodes=30,
                env_height=DEFAULT_ENV_HEIGHT,
                env_width=DEFAULT_ENV_WIDTH,
            )
            for k in range(1, 6):
                print(f"  Step {k} ahead - Loss: {nstep_results[f'step_{k}']['loss']:.4f}, Acc: {nstep_results[f'step_{k}']['accuracy']:.2%}")
        except Exception as e:
            print(f"Warning: N-step evaluation failed: {e}")
            nstep_results = None

        # Log final probe summary to wandb
        final_log = {}
        if probe_results:
            final_log.update({
                "probe/final/random_mean_loss": np.mean(probe_results['branch_losses'][0]),
                "probe/final/replay_mean_loss": np.mean(probe_results['branch_losses'][1]),
                "probe/final/mutate_mean_loss": np.mean(probe_results['branch_losses'][2]),
                "probe/final/random_mean_accuracy": np.mean(probe_results['branch_accuracies'][0]),
                "probe/final/replay_mean_accuracy": np.mean(probe_results['branch_accuracies'][1]),
                "probe/final/mutate_mean_accuracy": np.mean(probe_results['branch_accuracies'][2]),
            })

        if fresh_probe_results:
            final_log.update({
                "probe/final/fresh_wall_accuracy": fresh_probe_results['wall_accuracy'],
                "probe/final/fresh_goal_top1": fresh_probe_results['goal_top1_correct'],
                "probe/final/fresh_info_gain_total": fresh_probe_results['info_gain_total'],
            })

        if nstep_results:
            for k in range(1, 6):
                final_log[f"probe/final/nstep_{k}_loss"] = nstep_results[f'step_{k}']['loss']
                final_log[f"probe/final/nstep_{k}_accuracy"] = nstep_results[f'step_{k}']['accuracy']

        # Compare to baselines
        if empirical_baseline is not None and fresh_probe_results is not None:
            info_gain_vs_empirical = fresh_probe_results['wall_accuracy'] - empirical_baseline['wall_accuracy']
            final_log["probe/final/info_gain_vs_empirical"] = info_gain_vs_empirical
            print(f"\n*** Information gain vs untrained agent: {info_gain_vs_empirical:.2%} ***")

        if final_log:
            wandb.log(final_log)

    # Save final checkpoint as wandb artifact
    if config["checkpoint_save_interval"] > 0:
        try:
            artifact = wandb.Artifact(
                name=f"model-{config.get('run_name', 'probe')}-final",
                type="model",
                description=f"Final trained model with probe after {config['num_updates']} updates"
            )
            checkpoint_dir = os.path.join(
                os.getcwd(), "checkpoints",
                f"{config['run_name']}", str(config['seed']), "models"
            )
            if os.path.exists(checkpoint_dir):
                artifact.add_dir(checkpoint_dir)
                wandb.log_artifact(artifact)
                print(f"Saved final checkpoint to wandb as artifact")
        except Exception as e:
            print(f"Warning: Failed to save checkpoint artifact: {e}")

    return runner_state[1]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="JaxUED-minigrid-maze")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    # train vs eval
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--checkpoint_directory", type=str, default=None)
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1)

    # checkpointing
    parser.add_argument("--checkpoint_save_interval", type=int, default=2)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)

    # eval
    parser.add_argument("--eval_freq", type=int, default=250)
    parser.add_argument("--eval_num_attempts", type=int, default=10)
    parser.add_argument("--eval_levels", nargs="+", default=[
        "SixteenRooms",
        "SixteenRooms2",
        "Labyrinth",
        "LabyrinthFlipped",
        "Labyrinth2",
        "StandardMaze",
        "StandardMaze2",
        "StandardMaze3",
    ])
    group = parser.add_argument_group("Training params")

    # PPO
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--max_grad_norm", type=float, default=0.5)
    mut_group = group.add_mutually_exclusive_group()
    mut_group.add_argument("--num_updates", type=int, default=30000)
    mut_group.add_argument("--num_env_steps", type=int, default=None)
    group.add_argument("--num_steps", type=int, default=256)
    group.add_argument("--num_train_envs", type=int, default=32)
    group.add_argument("--num_minibatches", type=int, default=1)
    group.add_argument("--gamma", type=float, default=0.995)
    group.add_argument("--epoch_ppo", type=int, default=5)
    group.add_argument("--clip_eps", type=float, default=0.2)
    group.add_argument("--gae_lambda", type=float, default=0.98)
    group.add_argument("--entropy_coeff", type=float, default=1e-3)
    group.add_argument("--critic_coeff", type=float, default=0.5)

    # PLR
    group.add_argument("--score_function", type=str, default="MaxMC", choices=["MaxMC", "pvl"])
    group.add_argument("--exploratory_grad_updates", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--level_buffer_capacity", type=int, default=4000)
    group.add_argument("--replay_prob", type=float, default=0.8)
    group.add_argument("--staleness_coeff", type=float, default=0.3)
    group.add_argument("--temperature", type=float, default=0.3)
    group.add_argument("--top_k", type=int, default=4)
    group.add_argument("--minimum_fill_ratio", type=float, default=0.5)
    group.add_argument("--prioritization", type=str, default="rank", choices=["rank", "topk"])
    group.add_argument("--buffer_duplicate_check", action=argparse.BooleanOptionalAction, default=True)

    # ACCEL
    group.add_argument("--use_accel", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--num_edits", type=int, default=5)

    # Curriculum Probe (tests what agent has learned about curriculum from experience)
    group.add_argument("--use_probe", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--probe_lr", type=float, default=1e-4)
    group.add_argument("--probe_use_episode_context", action=argparse.BooleanOptionalAction, default=True,
                       help="Include episode return/solved/length as probe inputs")
    group.add_argument("--max_hstate_samples", type=int, default=1000,
                       help="Max hidden state samples to store for t-SNE visualization")
    group.add_argument("--tsne_freq", type=int, default=10,
                       help="Generate t-SNE plot every N eval steps")

    # env config
    group.add_argument("--agent_view_size", type=int, default=5)

    # dr
    group.add_argument("--n_walls", type=int, default=25)

    config = vars(parser.parse_args())
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"])
    config["group_name"] = "".join([str(config[key]) for key in sorted([a.dest for a in parser._action_groups[2]._group_actions])])

    if config["mode"] == "eval":
        os.environ["WANDB_MODE"] = "disabled"

    wandb.login()
    main(config, project=config["project"])
