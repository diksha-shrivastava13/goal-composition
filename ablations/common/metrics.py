"""
Metrics and evaluation utilities for curriculum awareness ablations.

Contains:
- Calibration metrics (distributional and per-instance)
- Random baselines
- Probe loss computation
- Curriculum prediction loss computation
- Distribution divergence metrics
"""

from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
import optax
import chex

from .types import DEFAULT_ENV_HEIGHT, DEFAULT_ENV_WIDTH, DEFAULT_CALIBRATION_BINS


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
        'agent_pos_top1': 1.0 / grid_size,
        'agent_dir': 0.25,  # 4 directions
        # Loss baselines (cross-entropy of uniform prediction)
        'wall_loss': 0.693,  # -log(0.5) = ln(2)
        'goal_loss': np.log(grid_size),  # ~5.13 for 13x13
        'agent_pos_loss': np.log(grid_size),
        'agent_dir_loss': np.log(4),  # ~1.39
        'total_loss': 0.693 + 2 * np.log(grid_size) + np.log(4),
    }


# =============================================================================
# PROBE LOSS COMPUTATION
# =============================================================================

def compute_probe_loss(
    predictions: dict,
    actual_level,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> tuple:
    """
    Compute probe loss for a single prediction/level pair.

    Args:
        predictions: Dict with wall_logits, goal_logits, agent_pos_logits, agent_dir_logits
        actual_level: Level dataclass with wall_map, goal_pos, agent_pos, agent_dir
        env_height: Environment height
        env_width: Environment width

    Returns:
        total_loss: Scalar loss
        loss_dict: Dict with per-component losses
    """
    grid_size = env_height * env_width

    # Wall loss: Binary cross-entropy
    wall_targets = actual_level.wall_map.astype(jnp.float32)
    wall_probs = jax.nn.sigmoid(predictions['wall_logits'])
    wall_loss = -jnp.mean(
        wall_targets * jnp.log(wall_probs + 1e-10) +
        (1 - wall_targets) * jnp.log(1 - wall_probs + 1e-10)
    )

    # Goal loss: Cross-entropy
    goal_idx = actual_level.goal_pos[1] * env_width + actual_level.goal_pos[0]
    goal_log_probs = jax.nn.log_softmax(predictions['goal_logits'])
    goal_loss = -goal_log_probs[goal_idx]

    # Agent position loss
    agent_pos_idx = actual_level.agent_pos[1] * env_width + actual_level.agent_pos[0]
    agent_pos_log_probs = jax.nn.log_softmax(predictions['agent_pos_logits'])
    agent_pos_loss = -agent_pos_log_probs[agent_pos_idx]

    # Agent direction loss
    agent_dir_log_probs = jax.nn.log_softmax(predictions['agent_dir_logits'])
    agent_dir_loss = -agent_dir_log_probs[actual_level.agent_dir]

    total_loss = wall_loss + goal_loss + agent_pos_loss + agent_dir_loss

    loss_dict = {
        'wall_loss': wall_loss,
        'goal_loss': goal_loss,
        'agent_pos_loss': agent_pos_loss,
        'agent_dir_loss': agent_dir_loss,
        'total_loss': total_loss,
    }

    return total_loss, loss_dict


def compute_probe_loss_batch(
    predictions_batch: dict,
    actual_levels_batch,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> tuple:
    """
    Compute probe loss for a batch of predictions/levels.

    Uses distributional loss - compares batch statistics rather than
    per-instance matching (which only makes sense for R->M transition).
    """
    batch_size = actual_levels_batch.wall_map.shape[0]
    grid_size = env_height * env_width

    # Wall loss: Mean BCE across batch
    wall_targets = actual_levels_batch.wall_map.astype(jnp.float32)
    wall_probs = jax.nn.sigmoid(predictions_batch['wall_logits'])
    wall_loss = -jnp.mean(
        wall_targets * jnp.log(wall_probs + 1e-10) +
        (1 - wall_targets) * jnp.log(1 - wall_probs + 1e-10)
    )

    # Goal loss: Cross-entropy (average over batch)
    goal_indices = actual_levels_batch.goal_pos[:, 1] * env_width + actual_levels_batch.goal_pos[:, 0]
    goal_log_probs = jax.nn.log_softmax(predictions_batch['goal_logits'])
    goal_loss = -jnp.mean(jax.vmap(lambda lp, idx: lp[idx])(goal_log_probs, goal_indices))

    # Agent position loss
    agent_pos_indices = actual_levels_batch.agent_pos[:, 1] * env_width + actual_levels_batch.agent_pos[:, 0]
    agent_pos_log_probs = jax.nn.log_softmax(predictions_batch['agent_pos_logits'])
    agent_pos_loss = -jnp.mean(jax.vmap(lambda lp, idx: lp[idx])(agent_pos_log_probs, agent_pos_indices))

    # Agent direction loss
    agent_dir_log_probs = jax.nn.log_softmax(predictions_batch['agent_dir_logits'])
    agent_dir_loss = -jnp.mean(jax.vmap(lambda lp, idx: lp[idx])(agent_dir_log_probs, actual_levels_batch.agent_dir))

    total_loss = wall_loss + goal_loss + agent_pos_loss + agent_dir_loss

    loss_dict = {
        'wall_loss': wall_loss,
        'goal_loss': goal_loss,
        'agent_pos_loss': agent_pos_loss,
        'agent_dir_loss': agent_dir_loss,
        'total_loss': total_loss,
    }

    return total_loss, loss_dict


# =============================================================================
# DISTRIBUTIONAL CALIBRATION METRICS
# =============================================================================

def compute_distributional_calibration_metrics(
    predictions_batch: dict,
    actual_levels_batch,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> dict:
    """
    Compute distributional calibration metrics comparing batch-averaged predictions
    to empirical level distributions.

    This is appropriate for all branches because it doesn't assume 1-to-1 correspondence.
    """
    batch_size = actual_levels_batch.wall_map.shape[0]
    grid_size = env_height * env_width

    # Wall distribution calibration
    mean_wall_probs = jax.nn.sigmoid(predictions_batch['wall_logits']).mean(axis=0)
    empirical_wall_freq = actual_levels_batch.wall_map.astype(jnp.float32).mean(axis=0)
    wall_dist_calibration = jnp.mean((mean_wall_probs - empirical_wall_freq) ** 2)
    mean_wall_binary = (mean_wall_probs > 0.5).astype(jnp.float32)
    empirical_wall_majority = (empirical_wall_freq > 0.5).astype(jnp.float32)
    wall_dist_accuracy = (mean_wall_binary == empirical_wall_majority).mean()

    # Goal position distribution calibration
    mean_goal_probs = jax.nn.softmax(predictions_batch['goal_logits']).mean(axis=0)
    goal_indices = actual_levels_batch.goal_pos[:, 1] * env_width + actual_levels_batch.goal_pos[:, 0]
    empirical_goal_counts = jnp.zeros(grid_size).at[goal_indices].add(1.0)
    empirical_goal_dist = empirical_goal_counts / batch_size
    goal_dist_calibration = -jnp.sum(empirical_goal_dist * jnp.log(mean_goal_probs + 1e-10))
    goal_dist_mode_match = (mean_goal_probs.argmax() == empirical_goal_dist.argmax()).astype(jnp.float32)

    # Agent position distribution calibration
    mean_agent_pos_probs = jax.nn.softmax(predictions_batch['agent_pos_logits']).mean(axis=0)
    agent_pos_indices = actual_levels_batch.agent_pos[:, 1] * env_width + actual_levels_batch.agent_pos[:, 0]
    empirical_agent_pos_counts = jnp.zeros(grid_size).at[agent_pos_indices].add(1.0)
    empirical_agent_pos_dist = empirical_agent_pos_counts / batch_size
    agent_pos_dist_calibration = -jnp.sum(empirical_agent_pos_dist * jnp.log(mean_agent_pos_probs + 1e-10))
    agent_pos_dist_mode_match = (mean_agent_pos_probs.argmax() == empirical_agent_pos_dist.argmax()).astype(jnp.float32)

    # Agent direction distribution calibration
    mean_agent_dir_probs = jax.nn.softmax(predictions_batch['agent_dir_logits']).mean(axis=0)
    empirical_dir_counts = jnp.zeros(4).at[actual_levels_batch.agent_dir].add(1.0)
    empirical_dir_dist = empirical_dir_counts / batch_size
    agent_dir_dist_calibration = -jnp.sum(empirical_dir_dist * jnp.log(mean_agent_dir_probs + 1e-10))
    agent_dir_dist_mode_match = (mean_agent_dir_probs.argmax() == empirical_dir_dist.argmax()).astype(jnp.float32)

    return {
        'wall_dist_calibration': wall_dist_calibration,
        'goal_dist_calibration': goal_dist_calibration,
        'agent_pos_dist_calibration': agent_pos_dist_calibration,
        'agent_dir_dist_calibration': agent_dir_dist_calibration,
        'wall_dist_accuracy': wall_dist_accuracy,
        'goal_dist_mode_match': goal_dist_mode_match,
        'agent_pos_dist_mode_match': agent_pos_dist_mode_match,
        'agent_dir_dist_mode_match': agent_dir_dist_mode_match,
    }


# =============================================================================
# PER-INSTANCE CALIBRATION METRICS
# =============================================================================

def compute_per_instance_calibration_batch(
    predictions_batch: dict,
    actual_levels_batch,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> dict:
    """
    Compute per-instance calibration metrics with direct index pairing.

    ONLY meaningful for Replay->Mutate transition where there's 1-to-1 correspondence.
    """
    batch_size = actual_levels_batch.wall_map.shape[0]
    grid_size = env_height * env_width

    # Wall accuracy (per instance, averaged)
    wall_probs = jax.nn.sigmoid(predictions_batch['wall_logits'])
    wall_targets = actual_levels_batch.wall_map.astype(jnp.float32)
    wall_correct = ((wall_probs > 0.5) == wall_targets).astype(jnp.float32)
    wall_accuracy_per_instance = wall_correct.mean(axis=(1, 2))
    wall_accuracy = wall_accuracy_per_instance.mean()

    # Goal position accuracy
    goal_probs = jax.nn.softmax(predictions_batch['goal_logits'])
    goal_indices = actual_levels_batch.goal_pos[:, 1] * env_width + actual_levels_batch.goal_pos[:, 0]
    goal_predicted = goal_probs.argmax(axis=1)
    goal_correct = (goal_predicted == goal_indices).astype(jnp.float32)
    goal_accuracy = goal_correct.mean()
    goal_prob_at_actual = jax.vmap(lambda p, i: p[i])(goal_probs, goal_indices).mean()

    # Agent position accuracy
    agent_pos_probs = jax.nn.softmax(predictions_batch['agent_pos_logits'])
    agent_pos_indices = actual_levels_batch.agent_pos[:, 1] * env_width + actual_levels_batch.agent_pos[:, 0]
    agent_pos_predicted = agent_pos_probs.argmax(axis=1)
    agent_pos_correct = (agent_pos_predicted == agent_pos_indices).astype(jnp.float32)
    agent_pos_accuracy = agent_pos_correct.mean()
    agent_pos_prob_at_actual = jax.vmap(lambda p, i: p[i])(agent_pos_probs, agent_pos_indices).mean()

    # Agent direction accuracy
    agent_dir_probs = jax.nn.softmax(predictions_batch['agent_dir_logits'])
    agent_dir_predicted = agent_dir_probs.argmax(axis=1)
    agent_dir_correct = (agent_dir_predicted == actual_levels_batch.agent_dir).astype(jnp.float32)
    agent_dir_accuracy = agent_dir_correct.mean()
    agent_dir_prob_at_actual = jax.vmap(lambda p, i: p[i])(agent_dir_probs, actual_levels_batch.agent_dir).mean()

    # Combined accuracy
    combined_accuracy = (
        wall_accuracy * 0.4 +
        goal_correct.mean() * 0.2 +
        agent_pos_correct.mean() * 0.2 +
        agent_dir_correct.mean() * 0.2
    )

    return {
        'wall_accuracy': wall_accuracy,
        'goal_accuracy': goal_accuracy,
        'agent_pos_accuracy': agent_pos_accuracy,
        'agent_dir_accuracy': agent_dir_accuracy,
        'combined_accuracy': combined_accuracy,
        'goal_prob_at_actual': goal_prob_at_actual,
        'agent_pos_prob_at_actual': agent_pos_prob_at_actual,
        'agent_dir_prob_at_actual': agent_dir_prob_at_actual,
    }


# =============================================================================
# NOVELTY AND LEARNABILITY
# =============================================================================

def compute_learnability(
    loss_history: chex.Array,
    step_history: chex.Array,
    total_samples: int,
    current_step: int,
) -> tuple:
    """
    Compute learnability: does the probe learn to predict better over time?

    High learnability = agent's hidden state encodes more curriculum
    information as training progresses.

    Returns:
        learnability: Scalar (positive = improving)
        details: Dict with trend info
    """
    buffer_size = loss_history.shape[0]
    valid_samples = jnp.minimum(total_samples, buffer_size)

    median_step = current_step // 2

    valid_mask = jnp.arange(buffer_size) < valid_samples
    early_mask = valid_mask & (step_history < median_step) & (step_history > 0)
    late_mask = valid_mask & (step_history >= median_step)

    early_count = early_mask.sum()
    late_count = late_mask.sum()

    early_loss = jnp.where(
        early_count > 0,
        (loss_history * early_mask).sum() / jnp.maximum(early_count, 1),
        0.0
    )
    late_loss = jnp.where(
        late_count > 0,
        (loss_history * late_mask).sum() / jnp.maximum(late_count, 1),
        0.0
    )

    learnability = early_loss - late_loss

    details = {
        "early_loss": float(early_loss),
        "late_loss": float(late_loss),
        "improvement": float(learnability),
        "early_samples": int(early_count),
        "late_samples": int(late_count),
        "median_step": int(median_step),
    }

    return float(learnability), details


def compute_novelty(
    loss_history: chex.Array,
    step_history: chex.Array,
    total_samples: int,
    current_step: int,
    window_steps: int = 1000,
) -> tuple:
    """
    Compute novelty: is prediction error increasing (curriculum getting harder)?

    High novelty = curriculum is presenting increasingly novel levels.

    Returns:
        novelty: Scalar (positive = increasing difficulty)
        details: Dict with trend info
    """
    buffer_size = loss_history.shape[0]
    valid_samples = jnp.minimum(total_samples, buffer_size)

    window_start = jnp.maximum(0, current_step - window_steps)
    valid_mask = jnp.arange(buffer_size) < valid_samples
    mask = valid_mask & (step_history >= window_start) & (step_history > 0)

    x = step_history.astype(jnp.float32)
    y = loss_history

    x_masked = jnp.where(mask, x, 0.0)
    y_masked = jnp.where(mask, y, 0.0)
    count = mask.sum()

    x_mean = x_masked.sum() / jnp.maximum(count, 1)
    y_mean = y_masked.sum() / jnp.maximum(count, 1)

    numerator = jnp.sum((x - x_mean) * (y - y_mean) * mask)
    denominator = jnp.sum((x - x_mean) ** 2 * mask) + 1e-8
    slope = numerator / denominator

    novelty = slope

    most_recent_idx = jnp.argmax(step_history * valid_mask.astype(jnp.float32))
    current_loss = loss_history[most_recent_idx]
    instantaneous_novelty = current_loss - y_mean

    details = {
        "novelty_slope": float(novelty),
        "instantaneous_novelty": float(instantaneous_novelty),
        "recent_mean_loss": float(y_mean),
        "current_loss": float(current_loss),
        "window_samples": int(count),
        "window_start_step": int(window_start),
    }

    return float(novelty), details


def compute_openendedness_score(novelty: float, learnability: float) -> tuple:
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
# CURRICULUM PREDICTION LOSS AND METRICS
# =============================================================================

def compute_curriculum_prediction_loss(
    predictions: dict,
    actual_level,  # Level dataclass
    wall_weight: float = 1.0,
    goal_weight: float = 1.0,
    agent_pos_weight: float = 1.0,
    agent_dir_weight: float = 1.0,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute loss between predicted distributions and actual level.

    Uses:
    - Binary cross-entropy for wall predictions (per-cell)
    - Cross-entropy for goal position (softmax over grid)
    - Cross-entropy for agent position (softmax over grid)
    - Cross-entropy for agent direction (softmax over 4 directions)

    Args:
        predictions: Dict from CurriculumPredictionHead
        actual_level: The actual Level that was generated
        wall_weight: Weight for wall prediction loss
        goal_weight: Weight for goal position loss
        agent_pos_weight: Weight for agent position loss
        agent_dir_weight: Weight for agent direction loss
        env_height: Environment height
        env_width: Environment width

    Returns:
        total_loss: Weighted sum of all losses
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

    # Weighted total loss
    total_loss = (
        wall_weight * wall_loss +
        goal_weight * goal_loss +
        agent_pos_weight * agent_pos_loss +
        agent_dir_weight * agent_dir_loss
    )

    metrics = {
        'wall_loss': wall_loss,
        'goal_loss': goal_loss,
        'agent_pos_loss': agent_pos_loss,
        'agent_dir_loss': agent_dir_loss,
        'total_loss': total_loss,
        # Also compute NLL (same as CE for these)
        'wall_nll': wall_loss,
        'goal_nll': goal_loss,
        'agent_pos_nll': agent_pos_loss,
        'agent_dir_nll': agent_dir_loss,
    }

    return total_loss, metrics


def compute_curriculum_prediction_loss_batch(
    predictions_batch: dict,
    actual_levels_batch,  # Batch of Level dataclasses
    wall_weight: float = 1.0,
    goal_weight: float = 1.0,
    agent_pos_weight: float = 1.0,
    agent_dir_weight: float = 1.0,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute curriculum prediction loss for a batch.

    Args:
        predictions_batch: Dict with batched predictions
        actual_levels_batch: Batch of Levels
        wall_weight: Weight for wall prediction loss
        goal_weight: Weight for goal position loss
        agent_pos_weight: Weight for agent position loss
        agent_dir_weight: Weight for agent direction loss
        env_height: Environment height
        env_width: Environment width

    Returns:
        total_loss: Mean loss across batch
        metrics: Dict with mean loss components
    """
    batch_size = actual_levels_batch.wall_map.shape[0]

    # Wall prediction loss: Binary cross-entropy per cell
    wall_targets = actual_levels_batch.wall_map.astype(jnp.float32)
    wall_bce = optax.sigmoid_binary_cross_entropy(
        predictions_batch['wall_logits'],
        wall_targets
    )
    wall_loss = wall_bce.mean()

    # Goal position loss
    goal_indices = actual_levels_batch.goal_pos[:, 1] * env_width + actual_levels_batch.goal_pos[:, 0]
    goal_losses = jax.vmap(optax.softmax_cross_entropy_with_integer_labels)(
        predictions_batch['goal_logits'], goal_indices
    )
    goal_loss = goal_losses.mean()

    # Agent position loss
    agent_pos_indices = actual_levels_batch.agent_pos[:, 1] * env_width + actual_levels_batch.agent_pos[:, 0]
    agent_pos_losses = jax.vmap(optax.softmax_cross_entropy_with_integer_labels)(
        predictions_batch['agent_pos_logits'], agent_pos_indices
    )
    agent_pos_loss = agent_pos_losses.mean()

    # Agent direction loss
    agent_dir_losses = jax.vmap(optax.softmax_cross_entropy_with_integer_labels)(
        predictions_batch['agent_dir_logits'], actual_levels_batch.agent_dir
    )
    agent_dir_loss = agent_dir_losses.mean()

    # Weighted total loss
    total_loss = (
        wall_weight * wall_loss +
        goal_weight * goal_loss +
        agent_pos_weight * agent_pos_loss +
        agent_dir_weight * agent_dir_loss
    )

    metrics = {
        'wall_loss': wall_loss,
        'goal_loss': goal_loss,
        'agent_pos_loss': agent_pos_loss,
        'agent_dir_loss': agent_dir_loss,
        'total_loss': total_loss,
    }

    return total_loss, metrics


# =============================================================================
# DISTRIBUTION DIVERGENCE METRICS
# =============================================================================

def compute_distribution_divergence(
    predictions: dict,
    actual_levels_batch,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> dict:
    """
    Compute KL and JS divergence between predicted and actual distributions.

    This requires a batch of actual levels to estimate the empirical distribution.

    Args:
        predictions: Dict with wall_logits, goal_logits, agent_pos_logits, agent_dir_logits
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
    empirical_goal_dist = empirical_goal_counts / batch_size + 1e-10

    # Predicted goal distribution (handle both single and batch predictions)
    if predictions['goal_logits'].ndim == 1:
        predicted_goal_dist = jax.nn.softmax(predictions['goal_logits']) + 1e-10
    else:
        predicted_goal_dist = jax.nn.softmax(predictions['goal_logits']).mean(axis=0) + 1e-10

    # KL divergence: KL(empirical || predicted)
    goal_kl = jnp.sum(empirical_goal_dist * jnp.log(empirical_goal_dist / predicted_goal_dist))

    # JS divergence
    m_dist = 0.5 * (empirical_goal_dist + predicted_goal_dist)
    goal_js = 0.5 * jnp.sum(empirical_goal_dist * jnp.log(empirical_goal_dist / m_dist))
    goal_js += 0.5 * jnp.sum(predicted_goal_dist * jnp.log(predicted_goal_dist / m_dist))

    # Agent position distribution
    agent_pos_indices = actual_levels_batch.agent_pos[:, 1] * env_width + actual_levels_batch.agent_pos[:, 0]
    empirical_agent_pos_counts = jnp.zeros(grid_size).at[agent_pos_indices].add(1.0)
    empirical_agent_pos_dist = empirical_agent_pos_counts / batch_size + 1e-10

    if predictions['agent_pos_logits'].ndim == 1:
        predicted_agent_pos_dist = jax.nn.softmax(predictions['agent_pos_logits']) + 1e-10
    else:
        predicted_agent_pos_dist = jax.nn.softmax(predictions['agent_pos_logits']).mean(axis=0) + 1e-10

    agent_pos_kl = jnp.sum(empirical_agent_pos_dist * jnp.log(empirical_agent_pos_dist / predicted_agent_pos_dist))

    # Compute wall density divergence
    empirical_wall_density = actual_levels_batch.wall_map.astype(jnp.float32).mean()
    if predictions['wall_logits'].ndim == 2:
        predicted_wall_density = jax.nn.sigmoid(predictions['wall_logits']).mean()
    else:
        predicted_wall_density = jax.nn.sigmoid(predictions['wall_logits'].mean(axis=0)).mean()

    return {
        'goal_kl': goal_kl,
        'goal_js': goal_js,
        'agent_pos_kl': agent_pos_kl,
        'empirical_wall_density': empirical_wall_density,
        'predicted_wall_density': predicted_wall_density,
        'wall_density_error': jnp.abs(empirical_wall_density - predicted_wall_density),
    }


def compute_per_cell_correlation(
    predictions_batch: dict,
    actual_levels_batch,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> dict:
    """
    Compute per-cell prediction correlation for walls.

    Measures how well the probe's predicted wall probability correlates
    with the actual wall presence across the batch for each cell.

    Args:
        predictions_batch: Dict with batched predictions
        actual_levels_batch: Batch of Levels
        env_height: Environment height
        env_width: Environment width

    Returns:
        Dict with per-cell correlation metrics
    """
    batch_size = actual_levels_batch.wall_map.shape[0]

    # Get wall predictions and actual
    wall_probs = jax.nn.sigmoid(predictions_batch['wall_logits'])  # (batch, H, W)
    wall_actual = actual_levels_batch.wall_map.astype(jnp.float32)  # (batch, H, W)

    # Compute correlation per cell (across batch dimension)
    # correlation(pred, actual) for each (i, j) cell
    def cell_correlation(i, j):
        pred = wall_probs[:, i, j]
        actual = wall_actual[:, i, j]
        pred_mean = pred.mean()
        actual_mean = actual.mean()
        pred_centered = pred - pred_mean
        actual_centered = actual - actual_mean
        numerator = (pred_centered * actual_centered).sum()
        denominator = jnp.sqrt((pred_centered ** 2).sum() * (actual_centered ** 2).sum()) + 1e-10
        return numerator / denominator

    # Compute for all cells
    correlations = jnp.zeros((env_height, env_width))
    for i in range(env_height):
        for j in range(env_width):
            correlations = correlations.at[i, j].set(cell_correlation(i, j))

    return {
        'per_cell_correlation': correlations,
        'mean_correlation': correlations.mean(),
        'min_correlation': correlations.min(),
        'max_correlation': correlations.max(),
        'std_correlation': correlations.std(),
    }


def aggregate_branch_metrics(
    metrics_list: list,
    branches: list,
) -> dict:
    """
    Aggregate metrics by branch type.

    Args:
        metrics_list: List of metric dicts from training steps
        branches: List of branch indices (0=DR, 1=replay, 2=mutate)

    Returns:
        Dict with aggregated metrics per branch
    """
    branch_names = {0: "random", 1: "replay", 2: "mutate"}
    aggregated = {name: {} for name in branch_names.values()}
    branch_counts = {name: 0 for name in branch_names.values()}

    for metrics, branch in zip(metrics_list, branches):
        branch_name = branch_names.get(branch, "unknown")
        if branch_name == "unknown":
            continue

        branch_counts[branch_name] += 1
        for key, value in metrics.items():
            if key not in aggregated[branch_name]:
                aggregated[branch_name][key] = []
            aggregated[branch_name][key].append(float(value))

    # Compute means
    result = {}
    for branch_name, metrics in aggregated.items():
        count = branch_counts[branch_name]
        if count == 0:
            continue
        for key, values in metrics.items():
            result[f"{branch_name}/{key}"] = np.mean(values)
            result[f"{branch_name}/{key}_std"] = np.std(values) if len(values) > 1 else 0.0

    return result


def apply_curriculum_prediction_gradient(
    train_state,
    curriculum_features: chex.Array,
    actual_level,
    env,
    env_params,
    sample_random_level,
    config: dict,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> Tuple[object, dict]:
    """
    Compute curriculum prediction loss and apply gradients.

    This function computes the prediction loss for the curriculum prediction head
    and applies gradients to the shared backbone. Unlike probe loss, these gradients
    DO affect the main network.

    Args:
        train_state: Current train state with network params
        curriculum_features: Features from get_curriculum_features()
        actual_level: The actual level that was generated
        env: The environment (for creating dummy observations)
        env_params: Environment parameters
        sample_random_level: Function to sample random levels (for dummy obs)
        config: Training configuration dict
        env_height: Environment height
        env_width: Environment width

    Returns:
        Updated train_state and metrics dict
    """
    from ..common.networks import ActorCriticWithCurriculumPrediction

    def prediction_loss_fn(params):
        # Create dummy observation via environment reset (matches original)
        rng = jax.random.PRNGKey(0)
        dummy_level = sample_random_level(rng)
        dummy_obs, _ = env.reset_to_level(rng, dummy_level, env_params)
        dummy_obs = jax.tree_util.tree_map(lambda x: x[None, None, ...], dummy_obs)
        dummy_dones = jnp.zeros((1, 1), dtype=bool)
        dummy_hidden = ActorCriticWithCurriculumPrediction.initialize_carry((1,))

        # Forward pass with prediction
        _, _, _, predictions = train_state.apply_fn(
            params,
            (dummy_obs, dummy_dones),
            dummy_hidden,
            curriculum_features=curriculum_features,
            predict_curriculum=True,
        )

        # Compute prediction loss
        loss, metrics = compute_curriculum_prediction_loss(
            predictions,
            actual_level,
            wall_weight=config.get("curriculum_wall_weight", 1.0),
            goal_weight=config.get("curriculum_goal_weight", 1.0),
            agent_pos_weight=config.get("curriculum_agent_pos_weight", 1.0),
            agent_dir_weight=config.get("curriculum_agent_dir_weight", 1.0),
            env_height=env_height,
            env_width=env_width,
        )

        # Scale by coefficient
        scaled_loss = config.get("curriculum_pred_coeff", 1.0) * loss
        return scaled_loss, metrics

    # Compute loss and gradients
    grad_fn = jax.value_and_grad(prediction_loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(train_state.params)

    # Apply gradients
    train_state = train_state.apply_gradients(grads=grads)

    metrics['prediction_loss_weighted'] = loss
    return train_state, metrics


def compute_calibration_metrics(
    predictions: dict,
    actual_level,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> dict:
    """
    Compute calibration metrics (Expected Calibration Error) for predictions.

    ECE measures how well-calibrated the predicted probabilities are:
    if we predict 70% probability, the actual frequency should be ~70%.

    Args:
        predictions: Dict from CurriculumPredictionHead
        actual_level: The actual Level
        n_bins: Number of bins for calibration computation
        env_height: Environment height
        env_width: Environment width

    Returns:
        Dict with calibration metrics
    """
    # Wall calibration: compute ECE for binary wall predictions
    wall_probs = jax.nn.sigmoid(predictions['wall_logits']).flatten()
    wall_targets = actual_level.wall_map.astype(jnp.float32).flatten()

    # Compute ECE for walls using a simpler approach
    # (The loop-based approach doesn't work well with JAX)
    # We use a binned approximation

    # Bin the predictions
    bin_boundaries = jnp.linspace(0, 1, n_bins + 1)

    def compute_bin_ece(bin_idx):
        lower = bin_boundaries[bin_idx]
        upper = bin_boundaries[bin_idx + 1]
        mask = (wall_probs >= lower) & (wall_probs < upper)
        bin_size = mask.sum()
        bin_confidence = jnp.where(bin_size > 0, (wall_probs * mask).sum() / jnp.maximum(bin_size, 1), 0.0)
        bin_accuracy = jnp.where(bin_size > 0, (wall_targets * mask).sum() / jnp.maximum(bin_size, 1), 0.0)
        return bin_size * jnp.abs(bin_confidence - bin_accuracy)

    bin_eces = jax.vmap(compute_bin_ece)(jnp.arange(n_bins))
    wall_ece = bin_eces.sum() / len(wall_probs)

    # Goal position: compute probability at actual location as calibration proxy
    goal_probs = jax.nn.softmax(predictions['goal_logits'])
    goal_idx = actual_level.goal_pos[1] * env_width + actual_level.goal_pos[0]
    goal_prob_at_actual = goal_probs[goal_idx]

    # Agent position: same approach
    agent_pos_probs = jax.nn.softmax(predictions['agent_pos_logits'])
    agent_pos_idx = actual_level.agent_pos[1] * env_width + actual_level.agent_pos[0]
    agent_pos_prob_at_actual = agent_pos_probs[agent_pos_idx]

    # Agent direction
    agent_dir_probs = jax.nn.softmax(predictions['agent_dir_logits'])
    agent_dir_prob_at_actual = agent_dir_probs[actual_level.agent_dir]

    return {
        'wall_ece': wall_ece,
        'goal_prob_at_actual': goal_prob_at_actual,
        'agent_pos_prob_at_actual': agent_pos_prob_at_actual,
        'agent_dir_prob_at_actual': agent_dir_prob_at_actual,
    }


# =============================================================================
# PROBE CORRELATION WITH PERFORMANCE
# =============================================================================

def compute_probe_correlation_with_performance(
    probe_accuracy_history: chex.Array,
    agent_returns_history: chex.Array,
    valid_samples: int,
) -> dict:
    """
    Compute correlation between probe accuracy and agent returns.

    This answers: does better environment understanding correlate with
    higher agent performance?

    Args:
        probe_accuracy_history: Buffer of probe accuracy values
        agent_returns_history: Buffer of agent return values
        valid_samples: Number of valid samples in buffers

    Returns:
        Dict with correlation coefficient and p-value proxy
    """
    if valid_samples < 10:
        return {
            'correlation': 0.0,
            'correlation_strength': 'insufficient_data',
            'n_samples': int(valid_samples),
        }

    buffer_size = probe_accuracy_history.shape[0]
    n = min(valid_samples, buffer_size)

    accuracy = probe_accuracy_history[:n]
    returns = agent_returns_history[:n]

    # Compute Pearson correlation
    acc_mean = accuracy.mean()
    ret_mean = returns.mean()
    acc_centered = accuracy - acc_mean
    ret_centered = returns - ret_mean

    numerator = (acc_centered * ret_centered).sum()
    denominator = jnp.sqrt((acc_centered ** 2).sum() * (ret_centered ** 2).sum()) + 1e-10
    correlation = numerator / denominator

    # Classify correlation strength
    abs_corr = jnp.abs(correlation)
    if abs_corr < 0.1:
        strength = 'negligible'
    elif abs_corr < 0.3:
        strength = 'weak'
    elif abs_corr < 0.5:
        strength = 'moderate'
    elif abs_corr < 0.7:
        strength = 'strong'
    else:
        strength = 'very_strong'

    return {
        'correlation': float(correlation),
        'correlation_strength': strength,
        'n_samples': int(n),
        'probe_accuracy_mean': float(acc_mean),
        'agent_returns_mean': float(ret_mean),
    }


# =============================================================================
# HIDDEN STATE STATISTICS
# =============================================================================

def compute_hidden_state_statistics(
    hstate_mean_by_branch: chex.Array,
    hstate_var_by_branch: chex.Array,
    hstate_count_by_branch: chex.Array,
) -> dict:
    """
    Compute statistics about hidden states across branches.

    Args:
        hstate_mean_by_branch: Mean hidden states per branch (3, hstate_dim)
        hstate_var_by_branch: Variance of hidden states per branch (3, hstate_dim)
        hstate_count_by_branch: Count of samples per branch (3,)

    Returns:
        Dict with hidden state statistics
    """
    branch_names = ['random', 'replay', 'mutate']
    stats = {}

    for i, name in enumerate(branch_names):
        count = hstate_count_by_branch[i]
        if count > 0:
            mean = hstate_mean_by_branch[i]
            var = hstate_var_by_branch[i]

            stats[f'hstate/{name}/mean_magnitude'] = float(jnp.abs(mean).mean())
            stats[f'hstate/{name}/mean_variance'] = float(var.mean())
            stats[f'hstate/{name}/max_magnitude'] = float(jnp.abs(mean).max())
            stats[f'hstate/{name}/count'] = int(count)

    # Cross-branch comparison
    total_count = hstate_count_by_branch.sum()
    if total_count > 0:
        # Compute average distance between branch means
        if hstate_count_by_branch[0] > 0 and hstate_count_by_branch[1] > 0:
            dr_replay_dist = jnp.sqrt(((hstate_mean_by_branch[0] - hstate_mean_by_branch[1]) ** 2).sum())
            stats['hstate/dr_replay_distance'] = float(dr_replay_dist)

        if hstate_count_by_branch[1] > 0 and hstate_count_by_branch[2] > 0:
            replay_mutate_dist = jnp.sqrt(((hstate_mean_by_branch[1] - hstate_mean_by_branch[2]) ** 2).sum())
            stats['hstate/replay_mutate_distance'] = float(replay_mutate_dist)

    return stats


# =============================================================================
# GREEDY MATCHING FOR DISTRIBUTIONAL COMPARISON
# =============================================================================

def compute_greedy_matching(
    predictions_batch: dict,
    levels_batch,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> chex.Array:
    """
    Compute greedy matching between predictions and actual levels.

    For each prediction, find the actual level with minimum loss.
    This provides a fair comparison when there's no 1-to-1 correspondence.

    Args:
        predictions_batch: Dict with batched predictions
        levels_batch: Batch of actual Levels
        env_height: Environment height
        env_width: Environment width

    Returns:
        matched_indices: Array of shape (batch_size,) with matched level indices
    """
    batch_size = predictions_batch['wall_logits'].shape[0]
    grid_size = env_height * env_width

    # Compute pairwise loss matrix
    # loss_matrix[i, j] = loss between prediction i and actual level j
    def compute_single_loss(pred_idx, actual_idx):
        # Wall loss
        wall_pred = jax.nn.sigmoid(predictions_batch['wall_logits'][pred_idx])
        wall_actual = levels_batch.wall_map[actual_idx].astype(jnp.float32)
        wall_loss = jnp.abs(wall_pred - wall_actual).mean()

        # Goal loss
        goal_probs = jax.nn.softmax(predictions_batch['goal_logits'][pred_idx])
        goal_idx = levels_batch.goal_pos[actual_idx, 1] * env_width + levels_batch.goal_pos[actual_idx, 0]
        goal_loss = -jnp.log(goal_probs[goal_idx] + 1e-10)

        return wall_loss + goal_loss * 0.1  # Weight wall more heavily

    # Compute loss matrix using nested vmap
    loss_matrix = jax.vmap(
        lambda p_idx: jax.vmap(lambda a_idx: compute_single_loss(p_idx, a_idx))(jnp.arange(batch_size))
    )(jnp.arange(batch_size))

    # Greedy matching: for each prediction, find best match
    # (Simple greedy, not optimal Hungarian but fast)
    matched_indices = loss_matrix.argmin(axis=1)

    return matched_indices


def compute_matched_accuracy_metrics(
    predictions_batch: dict,
    levels_batch,
    matched_indices: chex.Array,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> dict:
    """
    Compute accuracy metrics using greedy-matched pairs.

    Args:
        predictions_batch: Dict with batched predictions
        levels_batch: Batch of actual Levels
        matched_indices: Array of matched level indices
        env_height: Environment height
        env_width: Environment width

    Returns:
        Dict with matched accuracy metrics
    """
    batch_size = predictions_batch['wall_logits'].shape[0]
    grid_size = env_height * env_width

    # Gather matched levels
    def get_matched(arr, idx):
        if arr.ndim == 1:
            return arr[idx]
        elif arr.ndim == 2:
            return arr[idx]
        elif arr.ndim == 3:
            return arr[idx]
        return arr

    # Wall accuracy with matched pairs
    wall_probs = jax.nn.sigmoid(predictions_batch['wall_logits'])
    wall_correct_list = []
    goal_correct_list = []
    agent_pos_correct_list = []
    agent_dir_correct_list = []

    def compute_matched_accuracy(i):
        matched_idx = matched_indices[i]

        # Wall
        pred_wall = (wall_probs[i] > 0.5).astype(jnp.float32)
        actual_wall = levels_batch.wall_map[matched_idx].astype(jnp.float32)
        wall_acc = (pred_wall == actual_wall).mean()

        # Goal
        goal_probs = jax.nn.softmax(predictions_batch['goal_logits'][i])
        actual_goal_idx = levels_batch.goal_pos[matched_idx, 1] * env_width + levels_batch.goal_pos[matched_idx, 0]
        goal_correct = (goal_probs.argmax() == actual_goal_idx).astype(jnp.float32)

        # Agent pos
        agent_pos_probs = jax.nn.softmax(predictions_batch['agent_pos_logits'][i])
        actual_agent_idx = levels_batch.agent_pos[matched_idx, 1] * env_width + levels_batch.agent_pos[matched_idx, 0]
        agent_pos_correct = (agent_pos_probs.argmax() == actual_agent_idx).astype(jnp.float32)

        # Agent dir
        agent_dir_probs = jax.nn.softmax(predictions_batch['agent_dir_logits'][i])
        actual_dir = levels_batch.agent_dir[matched_idx]
        agent_dir_correct = (agent_dir_probs.argmax() == actual_dir).astype(jnp.float32)

        return wall_acc, goal_correct, agent_pos_correct, agent_dir_correct

    results = jax.vmap(compute_matched_accuracy)(jnp.arange(batch_size))
    wall_accs, goal_corrects, agent_pos_corrects, agent_dir_corrects = results

    return {
        'matched_wall_accuracy': float(wall_accs.mean()),
        'matched_goal_accuracy': float(goal_corrects.mean()),
        'matched_agent_pos_accuracy': float(agent_pos_corrects.mean()),
        'matched_agent_dir_accuracy': float(agent_dir_corrects.mean()),
        'mean_match_loss': float(wall_accs.mean() * 0.4 + goal_corrects.mean() * 0.2 +
                                  agent_pos_corrects.mean() * 0.2 + agent_dir_corrects.mean() * 0.2),
    }


# =============================================================================
# PAIRED-SPECIFIC METRICS
# =============================================================================

def compute_paired_regret_metrics(
    pro_returns: chex.Array,
    ant_returns: chex.Array,
) -> dict:
    """
    Compute comprehensive PAIRED regret metrics.

    Args:
        pro_returns: Protagonist returns per environment (batch_size,)
        ant_returns: Antagonist returns per environment (batch_size,)

    Returns:
        Dict with PAIRED regret metrics
    """
    # Per-instance regret
    per_instance_regret = ant_returns - pro_returns

    # Basic regret metrics
    est_regret = float(ant_returns.max() - pro_returns.mean())
    regret_variance = float(jnp.var(per_instance_regret))
    max_regret = float(per_instance_regret.max())
    min_regret = float(per_instance_regret.min())
    mean_regret = float(per_instance_regret.mean())

    # Solvability gap
    pro_solved = (pro_returns > 0).astype(jnp.float32)
    ant_solved = (ant_returns > 0).astype(jnp.float32)
    solvability_gap = float(ant_solved.mean() - pro_solved.mean())

    # Adversary success rate (levels where regret > 0)
    adversary_success_rate = float((per_instance_regret > 0).mean())

    return {
        'est_regret': est_regret,
        'regret_variance': regret_variance,
        'max_regret': max_regret,
        'min_regret': min_regret,
        'mean_regret': mean_regret,
        'solvability_gap': solvability_gap,
        'adversary_success_rate': adversary_success_rate,
        'pro_solve_rate': float(pro_solved.mean()),
        'ant_solve_rate': float(ant_solved.mean()),
    }


def compute_level_novelty(
    current_wall_maps: chex.Array,
    history_wall_maps: chex.Array,
    history_count: int,
) -> float:
    """
    Compute novelty of current levels compared to history.

    Novelty = mean minimum distance to historical levels.

    Args:
        current_wall_maps: Current level wall maps (batch, H, W)
        history_wall_maps: Historical wall maps buffer (buffer_size, H, W)
        history_count: Number of valid entries in history

    Returns:
        novelty: Mean novelty score (higher = more novel)
    """
    if history_count < 1:
        return 1.0  # First levels are maximally novel

    batch_size = current_wall_maps.shape[0]
    buffer_size = history_wall_maps.shape[0]
    n_history = min(history_count, buffer_size)

    # Flatten wall maps for distance computation
    current_flat = current_wall_maps.reshape(batch_size, -1).astype(jnp.float32)
    history_flat = history_wall_maps[:n_history].reshape(n_history, -1).astype(jnp.float32)

    # Compute pairwise Hamming distances (as fraction)
    def min_distance_to_history(current_level):
        distances = jnp.abs(current_level[None, :] - history_flat).mean(axis=1)
        return distances.min()

    min_distances = jax.vmap(min_distance_to_history)(current_flat)
    novelty = float(min_distances.mean())

    return novelty


def compute_paired_openendedness(
    novelty: float,
    learnability: float,
    regret: float,
) -> dict:
    """
    Compute PAIRED open-endedness metrics (OMNI-inspired).

    Open-endedness in PAIRED combines:
    - Novelty: How different are the generated levels
    - Learnability: Can the protagonist improve on these levels
    - Interestingness: Regret-weighted novelty

    Args:
        novelty: Level novelty score
        learnability: Protagonist improvement rate
        regret: Estimated regret

    Returns:
        Dict with open-endedness metrics
    """
    # Normalize metrics to [0, 1] range using sigmoid
    novelty_norm = 2.0 / (1.0 + np.exp(-10 * novelty)) - 1.0
    learnability_norm = 2.0 / (1.0 + np.exp(-learnability)) - 1.0

    # Regret-weighted interestingness
    interestingness = novelty * (1 - np.exp(-np.abs(regret)))

    # Open-endedness score (geometric mean when both positive)
    if novelty_norm > 0 and learnability_norm > 0:
        score = np.sqrt(novelty_norm * learnability_norm)
    else:
        score = 0.0

    # Classify regime
    if novelty > 0.05 and learnability > 0.1:
        regime = "open-ended"
    elif novelty > 0.05 and learnability <= 0.1:
        regime = "chaotic"
    elif novelty <= 0.05 and learnability > 0.1:
        regime = "converging"
    else:
        regime = "stagnant"

    return {
        'novelty': float(novelty),
        'learnability': float(learnability),
        'interestingness': float(interestingness),
        'openendedness_score': float(score),
        'regime': regime,
    }


def compute_level_feature_metrics(
    levels,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> dict:
    """
    Compute feature statistics of generated levels.

    Args:
        levels: Batch of Level objects
        env_height: Environment height
        env_width: Environment width

    Returns:
        Dict with level feature metrics
    """
    batch_size = levels.wall_map.shape[0]
    grid_size = env_height * env_width

    # Wall density
    wall_counts = levels.wall_map.sum(axis=(1, 2))
    wall_density = wall_counts / grid_size
    mean_wall_density = float(wall_density.mean())
    std_wall_density = float(wall_density.std())

    # Goal position entropy (how spread out are goals)
    goal_positions = levels.goal_pos  # (batch, 2)
    # Compute centroid and spread
    goal_centroid = goal_positions.mean(axis=0)
    goal_spread = jnp.sqrt(((goal_positions - goal_centroid) ** 2).sum(axis=1)).mean()

    # Agent position spread
    agent_positions = levels.agent_pos
    agent_centroid = agent_positions.mean(axis=0)
    agent_spread = jnp.sqrt(((agent_positions - agent_centroid) ** 2).sum(axis=1)).mean()

    # Direction distribution
    dir_counts = jnp.zeros(4).at[levels.agent_dir].add(1.0)
    dir_entropy = -jnp.sum((dir_counts / batch_size + 1e-10) * jnp.log(dir_counts / batch_size + 1e-10))

    return {
        'mean_wall_density': mean_wall_density,
        'std_wall_density': std_wall_density,
        'goal_spread': float(goal_spread),
        'agent_spread': float(agent_spread),
        'dir_entropy': float(dir_entropy),
        'mean_wall_count': float(wall_counts.mean()),
    }


# =============================================================================
# AGENT-CENTRIC METRICS (PAIRED)
# =============================================================================

def compute_policy_diversity(
    policy_logits: chex.Array,
    level_features: chex.Array,
    n_clusters: int = 5,
) -> dict:
    """
    Measure how differently agent behaves across level types (agent-centric novelty).

    This is an AGENT-CENTRIC metric: it measures policy diversity, not probe accuracy.
    Higher diversity = agent behaves differently on different level types.

    Args:
        policy_logits: Policy logits (n_samples, n_actions)
        level_features: Level feature vectors (n_samples, n_features)
        n_clusters: Number of level type clusters

    Returns:
        Dict with policy diversity metrics
    """
    from sklearn.cluster import KMeans

    n_samples = policy_logits.shape[0]
    if n_samples < n_clusters * 2:
        return {'error': 'Insufficient samples for clustering'}

    # Cluster levels by features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(np.array(level_features))

    # Compute policy entropy per cluster
    policy_probs = jax.nn.softmax(policy_logits, axis=-1)
    policy_entropy = -jnp.sum(policy_probs * jnp.log(policy_probs + 1e-10), axis=-1)

    cluster_entropies = {}
    cluster_mean_policies = {}

    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() > 0:
            cluster_entropies[c] = float(np.mean(policy_entropy[mask]))
            cluster_mean_policies[c] = np.mean(policy_probs[mask], axis=0)

    # Compute inter-cluster policy divergence (KL between mean policies)
    if len(cluster_mean_policies) >= 2:
        kl_divergences = []
        clusters = list(cluster_mean_policies.keys())
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                p = cluster_mean_policies[clusters[i]] + 1e-10
                q = cluster_mean_policies[clusters[j]] + 1e-10
                kl = float(np.sum(p * np.log(p / q)))
                kl_divergences.append(kl)
        mean_inter_cluster_kl = float(np.mean(kl_divergences))
    else:
        mean_inter_cluster_kl = 0.0

    # Compute entropy diversity across clusters
    entropy_values = list(cluster_entropies.values())
    entropy_diversity = float(np.std(entropy_values)) if len(entropy_values) > 1 else 0.0

    return {
        'mean_inter_cluster_kl': mean_inter_cluster_kl,
        'entropy_diversity': entropy_diversity,
        'cluster_entropies': cluster_entropies,
        'n_clusters': n_clusters,
        'interpretation': (
            "Higher inter_cluster_kl = more distinct policies for different level types. "
            "This is agent-centric novelty: how differently the agent behaves."
        ),
    }


def compute_value_calibration_by_regret(
    values: chex.Array,
    returns: chex.Array,
    regrets: chex.Array,
    n_bins: int = 10,
) -> dict:
    """
    V(s) calibration broken down by regret condition (agent-centric learnability).

    This is AGENT-CENTRIC: measures value function accuracy conditioned on regret.
    Key question: Does protagonist underestimate V(s) on high-regret levels?

    Args:
        values: Value predictions V(s) (n_samples,)
        returns: Actual returns (n_samples,)
        regrets: Regret values (ant_return - pro_return) (n_samples,)
        n_bins: Number of calibration bins

    Returns:
        Dict with regret-conditioned calibration metrics
    """
    values = np.array(values)
    returns = np.array(returns)
    regrets = np.array(regrets)

    # Compute regret terciles
    regret_percentiles = np.percentile(regrets, [33, 66])
    regret_tercile = np.digitize(regrets, regret_percentiles)
    tercile_names = ['low', 'medium', 'high']

    results = {}

    for tercile in [0, 1, 2]:
        mask = regret_tercile == tercile
        if mask.sum() < 5:
            continue

        v = values[mask]
        r = returns[mask]

        # Calibration metrics
        mae = float(np.mean(np.abs(v - r)))
        bias = float(np.mean(v - r))  # Positive = overestimate
        correlation = float(np.corrcoef(v, r)[0, 1]) if len(v) > 2 else 0.0

        # ECE
        v_min, v_max = v.min(), v.max()
        if v_max - v_min > 1e-6:
            bin_edges = np.linspace(v_min - 1e-5, v_max + 1e-5, n_bins + 1)
            ece = 0.0
            for i in range(n_bins):
                bin_mask = (v >= bin_edges[i]) & (v < bin_edges[i + 1])
                if bin_mask.sum() > 0:
                    bin_conf = v[bin_mask].mean()
                    bin_acc = r[bin_mask].mean()
                    ece += bin_mask.sum() * np.abs(bin_conf - bin_acc)
            ece = float(ece / len(v))
        else:
            ece = float(np.abs(v.mean() - r.mean()))

        results[tercile_names[tercile]] = {
            'mae': mae,
            'bias': bias,
            'correlation': correlation,
            'ece': ece,
            'n_samples': int(mask.sum()),
            'mean_regret': float(regrets[mask].mean()),
        }

    # Key diagnostic: does high regret correlate with value underestimation?
    if 'high' in results and 'low' in results:
        high_bias = results['high']['bias']
        low_bias = results['low']['bias']
        results['high_regret_underestimation'] = high_bias < low_bias
        results['bias_difference_high_vs_low'] = float(high_bias - low_bias)
        results['interpretation'] = (
            "Negative bias_difference = protagonist underestimates value on high-regret levels. "
            "This suggests adversary finds levels where protagonist's value model is poorly calibrated."
        )

    return results


def compute_regret_source_decomposition(
    pro_returns: chex.Array,
    ant_returns: chex.Array,
    regrets: chex.Array,
) -> dict:
    """
    Decompose: is regret from antagonist strength or protagonist weakness?

    This is AGENT-CENTRIC: directly analyzes returns, not probe predictions.

    Args:
        pro_returns: Protagonist returns per level (n_samples,)
        ant_returns: Antagonist returns per level (n_samples,)
        regrets: Regret values = ant_returns - pro_returns (n_samples,)

    Returns:
        Dict with regret source decomposition
    """
    pro_returns = np.array(pro_returns)
    ant_returns = np.array(ant_returns)
    regrets = np.array(regrets)

    # Normalize returns to [0, 1] for comparison
    all_returns = np.concatenate([pro_returns, ant_returns])
    min_ret, max_ret = all_returns.min(), all_returns.max()
    if max_ret - min_ret > 1e-6:
        pro_norm = (pro_returns - min_ret) / (max_ret - min_ret)
        ant_norm = (ant_returns - min_ret) / (max_ret - min_ret)
    else:
        pro_norm = pro_returns
        ant_norm = ant_returns

    # High regret episodes
    high_regret_mask = regrets > np.percentile(regrets, 66)

    # Source attribution: for high regret, is antagonist strong or protagonist weak?
    if high_regret_mask.sum() > 0:
        high_regret_pro = pro_norm[high_regret_mask].mean()
        high_regret_ant = ant_norm[high_regret_mask].mean()

        # Compare to overall means
        overall_pro = pro_norm.mean()
        overall_ant = ant_norm.mean()

        # Antagonist contribution: how much better is ant on high-regret levels?
        ant_contribution = float(high_regret_ant - overall_ant)

        # Protagonist contribution: how much worse is pro on high-regret levels?
        pro_contribution = float(overall_pro - high_regret_pro)

        # Relative attribution
        total_effect = ant_contribution + pro_contribution
        if total_effect > 1e-6:
            ant_attribution = ant_contribution / total_effect
            pro_attribution = pro_contribution / total_effect
        else:
            ant_attribution = 0.5
            pro_attribution = 0.5
    else:
        ant_contribution = 0.0
        pro_contribution = 0.0
        ant_attribution = 0.5
        pro_attribution = 0.5

    return {
        'antagonist_contribution': float(ant_contribution),
        'protagonist_contribution': float(pro_contribution),
        'antagonist_attribution_fraction': float(ant_attribution),
        'protagonist_attribution_fraction': float(pro_attribution),
        'regret_source': 'antagonist_strength' if ant_attribution > 0.6 else (
            'protagonist_weakness' if pro_attribution > 0.6 else 'mixed'
        ),
        'mean_pro_return': float(pro_returns.mean()),
        'mean_ant_return': float(ant_returns.mean()),
        'mean_regret': float(regrets.mean()),
        'interpretation': (
            f"Regret is primarily from {'antagonist finding exploits' if ant_attribution > 0.6 else ('protagonist failing' if pro_attribution > 0.6 else 'both factors')}. "
            f"Antagonist contributes {ant_attribution:.1%}, protagonist contributes {pro_attribution:.1%}."
        ),
    }


def compute_agent_novelty(
    policy_entropies: chex.Array,
    level_features: chex.Array,
    training_step: int,
    history_policy_entropies: chex.Array = None,
    history_level_features: chex.Array = None,
) -> dict:
    """
    Agent-centric novelty: policy diversity across level types.

    This replaces probe_loss-based novelty with actual behavioral metrics.

    Args:
        policy_entropies: Current policy entropies (n_samples,)
        level_features: Current level features (n_samples, n_features)
        training_step: Current training step
        history_policy_entropies: Historical policy entropies for comparison
        history_level_features: Historical level features

    Returns:
        Dict with agent-centric novelty metrics
    """
    policy_entropies = np.array(policy_entropies)
    level_features = np.array(level_features)

    # Current policy entropy statistics
    current_entropy_mean = float(policy_entropies.mean())
    current_entropy_std = float(policy_entropies.std())

    # Entropy conditioned on level difficulty (wall density as proxy)
    if level_features.ndim == 2 and level_features.shape[1] > 0:
        # Use first feature (typically wall density) as difficulty proxy
        difficulty = level_features[:, 0]
        difficulty_terciles = np.digitize(
            difficulty, np.percentile(difficulty, [33, 66])
        )

        entropy_by_difficulty = {}
        for t in [0, 1, 2]:
            mask = difficulty_terciles == t
            if mask.sum() > 0:
                entropy_by_difficulty[['easy', 'medium', 'hard'][t]] = float(
                    policy_entropies[mask].mean()
                )
    else:
        entropy_by_difficulty = {}

    # Compare to history if available
    novelty_score = 0.0
    if history_policy_entropies is not None and len(history_policy_entropies) > 0:
        history_mean = float(np.mean(history_policy_entropies))
        # Novelty = entropy shift (more uncertain on new levels = novel)
        novelty_score = current_entropy_mean - history_mean

    return {
        'entropy_mean': current_entropy_mean,
        'entropy_std': current_entropy_std,
        'entropy_by_difficulty': entropy_by_difficulty,
        'novelty_score': float(novelty_score),
        'training_step': training_step,
        'interpretation': (
            "Agent-centric novelty: how policy entropy varies across level types. "
            "Higher entropy on novel levels indicates exploration."
        ),
    }


def compute_agent_learnability(
    value_errors: chex.Array,
    returns: chex.Array,
    regrets: chex.Array,
    training_step: int,
    history_value_errors: chex.Array = None,
) -> dict:
    """
    Agent-centric learnability: value calibration + regret reduction.

    This replaces early_loss/late_loss with actual behavioral improvement metrics.

    Args:
        value_errors: |V(s) - actual_return| per level (n_samples,)
        returns: Agent returns per level (n_samples,)
        regrets: Regret values per level (n_samples,)
        training_step: Current training step
        history_value_errors: Historical value errors for trend

    Returns:
        Dict with agent-centric learnability metrics
    """
    value_errors = np.array(value_errors)
    returns = np.array(returns)
    regrets = np.array(regrets)

    # Current value calibration
    current_mae = float(value_errors.mean())

    # Return on high-regret levels
    high_regret_mask = regrets > np.percentile(regrets, 66)
    if high_regret_mask.sum() > 0:
        high_regret_return = float(returns[high_regret_mask].mean())
        high_regret_value_error = float(value_errors[high_regret_mask].mean())
    else:
        high_regret_return = float(returns.mean())
        high_regret_value_error = current_mae

    # Improvement over history
    learnability_score = 0.0
    if history_value_errors is not None and len(history_value_errors) > 0:
        history_mae = float(np.mean(history_value_errors))
        # Learnability = error reduction (lower is better, so positive score = improving)
        learnability_score = history_mae - current_mae

    return {
        'value_error_mae': current_mae,
        'high_regret_return': high_regret_return,
        'high_regret_value_error': high_regret_value_error,
        'learnability_score': float(learnability_score),
        'training_step': training_step,
        'interpretation': (
            "Agent-centric learnability: how well value function calibrates over time. "
            "Positive learnability_score = improving calibration."
        ),
    }


def compute_bilateral_cka(
    pro_hstates: chex.Array,
    ant_hstates: chex.Array,
) -> dict:
    """
    Compute CKA between protagonist and antagonist representations.

    This measures representational similarity between the two agents.

    Args:
        pro_hstates: Protagonist hidden states (n_samples, hidden_dim)
        ant_hstates: Antagonist hidden states (n_samples, hidden_dim)

    Returns:
        Dict with bilateral CKA metrics
    """
    pro_hstates = np.array(pro_hstates)
    ant_hstates = np.array(ant_hstates)

    # Ensure same number of samples
    n = min(len(pro_hstates), len(ant_hstates))
    pro_hstates = pro_hstates[:n]
    ant_hstates = ant_hstates[:n]

    if n < 5:
        return {'error': 'Insufficient samples for CKA'}

    # Compute linear CKA
    def _center_gram(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    def _linear_cka(X, Y):
        K = X @ X.T
        L = Y @ Y.T
        K_c = _center_gram(K)
        L_c = _center_gram(L)
        hsic = np.sum(K_c * L_c)
        norm_k = np.sqrt(np.sum(K_c * K_c))
        norm_l = np.sqrt(np.sum(L_c * L_c))
        return float(hsic / (norm_k * norm_l + 1e-10))

    cka_value = _linear_cka(pro_hstates, ant_hstates)

    return {
        'cka': cka_value,
        'n_samples': n,
        'interpretation': (
            f"CKA = {cka_value:.3f}. "
            "High CKA = similar representations. "
            "Low CKA = antagonist encodes different information than protagonist."
        ),
    }


def compute_information_gain_metrics(
    probe_metrics: dict,
    random_baselines: dict,
) -> dict:
    """
    Compute information gain compared to random baseline.

    Information gain measures how much better the probe is than random guessing.

    Args:
        probe_metrics: Dict with probe accuracy/loss metrics
        random_baselines: Dict with random baseline values

    Returns:
        Dict with information gain metrics
    """
    results = {}

    # Wall accuracy gain
    if 'wall_accuracy' in probe_metrics:
        wall_gain = probe_metrics['wall_accuracy'] - random_baselines['wall_accuracy']
        results['info_gain/wall_accuracy'] = float(wall_gain)
        results['info_gain/wall_relative'] = float(wall_gain / (1.0 - random_baselines['wall_accuracy']))

    # Goal accuracy gain
    if 'goal_accuracy' in probe_metrics:
        goal_gain = probe_metrics['goal_accuracy'] - random_baselines['goal_top1']
        results['info_gain/goal_accuracy'] = float(goal_gain)
        results['info_gain/goal_relative'] = float(goal_gain / (1.0 - random_baselines['goal_top1']))

    # Loss improvement
    if 'total_loss' in probe_metrics:
        loss_improvement = random_baselines['total_loss'] - probe_metrics['total_loss']
        results['info_gain/loss_improvement'] = float(loss_improvement)
        results['info_gain/loss_relative'] = float(loss_improvement / random_baselines['total_loss'])

    return results


# =============================================================================
# AGENT-CENTRIC OPEN-ENDEDNESS METRICS (from AgentTrackingState)
# =============================================================================

def compute_agent_novelty_from_tracking(
    agent_tracking,  # AgentTrackingState
    window_size: int = 500,
) -> Tuple[float, dict]:
    """
    Agent-centric novelty: policy diversity across level types.

    NOT probe-based. Measures how differently the agent behaves
    on different level types using policy entropy variance.

    Args:
        agent_tracking: AgentTrackingState with policy and level data
        window_size: Window for recent samples

    Returns:
        novelty: Policy entropy diversity (higher = more diverse behavior)
        details: Dict with per-cluster statistics
    """
    n_samples = min(agent_tracking.total_samples, agent_tracking.policy_entropy_history.shape[0])
    if n_samples < 10:
        return 0.0, {'error': 'Insufficient samples'}

    # Get recent data
    recent_n = min(n_samples, window_size)
    start_idx = max(0, n_samples - recent_n)

    policy_entropies = np.array(agent_tracking.policy_entropy_history[start_idx:n_samples])
    wall_densities = np.array(agent_tracking.level_wall_density_history[start_idx:n_samples])
    branches = np.array(agent_tracking.branch_history[start_idx:n_samples])

    # Cluster levels by wall density (as difficulty proxy)
    if len(wall_densities) > 3:
        density_terciles = np.digitize(
            wall_densities, np.percentile(wall_densities, [33, 66])
        )

        # Compute policy entropy variance across clusters
        entropy_by_cluster = []
        for cluster_id in [0, 1, 2]:
            mask = density_terciles == cluster_id
            if mask.sum() > 0:
                cluster_entropy = policy_entropies[mask].mean()
                entropy_by_cluster.append(cluster_entropy)

        # Novelty = variance in policy behavior across level types
        novelty = float(np.std(entropy_by_cluster)) if len(entropy_by_cluster) > 1 else 0.0
    else:
        novelty = 0.0
        entropy_by_cluster = []

    # Also compute entropy variance across branches
    entropy_by_branch = {}
    for branch in [0, 1, 2]:
        mask = branches == branch
        if mask.sum() > 0:
            entropy_by_branch[branch] = float(policy_entropies[mask].mean())

    branch_novelty = float(np.std(list(entropy_by_branch.values()))) if len(entropy_by_branch) > 1 else 0.0

    return novelty, {
        'num_clusters': len(entropy_by_cluster),
        'entropy_by_cluster': entropy_by_cluster,
        'entropy_by_branch': entropy_by_branch,
        'branch_novelty': branch_novelty,
        'mean_entropy': float(policy_entropies.mean()),
        'n_samples': n_samples,
    }


def compute_agent_learnability_from_tracking(
    agent_tracking,  # AgentTrackingState
    window_size: int = 500,
) -> Tuple[float, dict]:
    """
    Agent-centric learnability: V(s) calibration improvement.

    NOT probe-based. Measures how well the agent's value function
    predicts actual returns.

    Args:
        agent_tracking: AgentTrackingState with value and return data
        window_size: Window for comparison

    Returns:
        learnability: Value calibration improvement (higher = better learning)
        details: Dict with calibration statistics
    """
    n_samples = min(agent_tracking.total_samples, agent_tracking.value_predictions.shape[0])
    if n_samples < 20:
        return 0.0, {'error': 'Insufficient samples'}

    values = np.array(agent_tracking.value_predictions[:n_samples])
    returns = np.array(agent_tracking.actual_returns[:n_samples])

    # Split into early vs late training
    mid_idx = n_samples // 2

    # Compute calibration error: |V(s) - actual_return|
    early_error = np.abs(values[:mid_idx] - returns[:mid_idx]).mean()
    late_error = np.abs(values[mid_idx:] - returns[mid_idx:]).mean()

    # Learnability = improvement in calibration
    learnability = float(early_error - late_error)

    # Also compute correlation between V and returns
    if len(values) > 5:
        correlation = float(np.corrcoef(values, returns)[0, 1])
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0

    return learnability, {
        'early_calibration_error': float(early_error),
        'late_calibration_error': float(late_error),
        'improvement': learnability,
        'value_return_correlation': correlation,
        'mean_value': float(values.mean()),
        'mean_return': float(returns.mean()),
        'n_samples': n_samples,
    }


def compute_agent_openendedness_from_tracking(
    agent_tracking,  # AgentTrackingState
    window_size: int = 500,
) -> Tuple[float, str, dict]:
    """
    Agent-centric open-endedness score.

    Combines agent_novelty and agent_learnability using ACTUAL agent metrics,
    NOT probe predictions.

    Args:
        agent_tracking: AgentTrackingState with all agent data
        window_size: Window for computations

    Returns:
        score: Combined open-endedness score
        regime: Classification of training regime
        details: Dict with all metrics
    """
    novelty, nov_details = compute_agent_novelty_from_tracking(agent_tracking, window_size)
    learnability, learn_details = compute_agent_learnability_from_tracking(agent_tracking, window_size)

    # Classify regime
    if novelty > 0.1 and learnability > 0:
        regime = "open-ended"
    elif novelty > 0.1 and learnability <= 0:
        regime = "chaotic"
    elif novelty <= 0.1 and learnability > 0:
        regime = "converging"
    else:
        regime = "stagnant"

    # Compute score (geometric mean when both positive, otherwise 0)
    if novelty > 0 and learnability > 0:
        # Normalize
        novelty_norm = 2.0 / (1.0 + np.exp(-10 * novelty)) - 1.0
        learn_norm = 2.0 / (1.0 + np.exp(-learnability)) - 1.0
        score = float(np.sqrt(novelty_norm * learn_norm))
    else:
        score = 0.0

    return score, regime, {
        'novelty': novelty,
        'learnability': learnability,
        'regime': regime,
        **nov_details,
        **learn_details,
    }


def update_agent_tracking(
    agent_tracking,  # AgentTrackingState
    policy_entropy: float,
    value_prediction: float,
    actual_return: float,
    wall_density: float,
    goal_distance: float,
    branch: int,
    training_step: int,
    policy_kl: float = 0.0,
):
    """
    Update agent tracking state with new data.

    Args:
        agent_tracking: Current AgentTrackingState
        policy_entropy: Policy entropy for this step
        value_prediction: V(s) prediction
        actual_return: Actual episode return
        wall_density: Level wall density
        goal_distance: Manhattan distance agent to goal
        branch: Branch index (0=DR, 1=Replay, 2=Mutate)
        training_step: Current training step
        policy_kl: KL divergence from previous policy (optional)

    Returns:
        Updated AgentTrackingState
    """
    buffer_size = agent_tracking.policy_entropy_history.shape[0]
    ptr = agent_tracking.buffer_ptr % buffer_size

    return agent_tracking.replace(
        policy_entropy_history=agent_tracking.policy_entropy_history.at[ptr].set(policy_entropy),
        policy_kl_history=agent_tracking.policy_kl_history.at[ptr].set(policy_kl),
        value_predictions=agent_tracking.value_predictions.at[ptr].set(value_prediction),
        actual_returns=agent_tracking.actual_returns.at[ptr].set(actual_return),
        level_wall_density_history=agent_tracking.level_wall_density_history.at[ptr].set(wall_density),
        level_goal_distance_history=agent_tracking.level_goal_distance_history.at[ptr].set(goal_distance),
        branch_history=agent_tracking.branch_history.at[ptr].set(branch),
        training_step_history=agent_tracking.training_step_history.at[ptr].set(training_step),
        buffer_ptr=(ptr + 1) % buffer_size,
        total_samples=agent_tracking.total_samples + 1,
    )
