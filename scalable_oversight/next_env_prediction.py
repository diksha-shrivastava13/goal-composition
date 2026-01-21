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
from jaxued.environments.maze import Level, make_level_generator, make_level_mutator_minimax
from jaxued.environments.maze.env import Observation as MazeObservation
from jaxued.level_sampler import LevelSampler
from jaxued.utils import max_mc, positive_value_loss, compute_max_returns
from jaxued.wrappers import AutoReplayWrapper

# =============================================================================
# CURRICULUM PREDICTION CONSTANTS
# =============================================================================

# Default environment dimensions (for Maze)
DEFAULT_ENV_HEIGHT = 13
DEFAULT_ENV_WIDTH = 13

# Curriculum history tracking
DEFAULT_CURRICULUM_HISTORY_LENGTH = 64  # Number of recent levels to track
DEFAULT_CURRICULUM_HIDDEN_SIZE = 128

# Calibration computation
DEFAULT_CALIBRATION_BINS = 10

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
) -> chex.Array:
    """
    Extract feature vector from curriculum state for the encoder.

    Returns a flattened array of curriculum features suitable for
    input to a neural network.
    """
    history_length = curriculum_state.recent_wall_densities.shape[0]

    # Normalize training progress
    normalized_step = curriculum_state.training_step / 30000.0  # Assume 30k max steps
    # TODO: 30k max steps is an assumption here, this should be changed according to the steps we actually run for

    safe_training_step = jnp.maximum(curriculum_state.training_step, 1)
    replay_fraction = curriculum_state.total_replay_steps / safe_training_step

    # Buffer statistics (normalized)
    buffer_fill = curriculum_state.buffer_size / 4000.0  # Assume 4000 capacity
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
# CURRICULUM ENCODER AND PREDICTION NETWORK
# =============================================================================

class CurriculumEncoder(nn.Module):
    """
    Encodes curriculum history for next-level distribution prediction.

    This module processes the cross-episode curriculum state to produce
    an embedding that captures the structure of the regret-based curriculum.
    """
    hidden_size: int = DEFAULT_CURRICULUM_HIDDEN_SIZE
    history_length: int = DEFAULT_CURRICULUM_HISTORY_LENGTH

    @nn.compact
    def __call__(self, curriculum_features: chex.Array) -> chex.Array:
        """
        Encode curriculum features into a fixed-size embedding.

        Args:
            curriculum_features: Output from get_curriculum_features()

        Returns:
            curriculum_embedding: (hidden_size,) encoding of curriculum state
        """
        # First layer: process raw features
        x = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="curriculum_encoder_0"
        )(curriculum_features)
        x = nn.relu(x)

        # Second layer: compress to hidden size
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="curriculum_encoder_1"
        )(x)
        x = nn.relu(x)

        return x


class CurriculumPredictionHead(nn.Module):
    """
    Prediction head that outputs distributions over level features.

    Predicts:
    - wall_logits: (H, W) - per-cell probability of wall
    - goal_logits: (H, W) - distribution over goal positions
    - agent_pos_logits: (H, W) - distribution over agent spawn positions
    - agent_dir_logits: (4,) - distribution over agent directions
    """
    env_height: int = DEFAULT_ENV_HEIGHT
    env_width: int = DEFAULT_ENV_WIDTH

    @nn.compact
    def __call__(self, curriculum_embedding: chex.Array) -> dict:
        """
        Predict level feature distributions from curriculum embedding.

        Args:
            curriculum_embedding: Output from CurriculumEncoder

        Returns:
            Dictionary with prediction logits for each level component
        """
        # Shared hidden layer for all predictions
        hidden = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="pred_shared_0"
        )(curriculum_embedding)
        hidden = nn.relu(hidden)

        hidden = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="pred_shared_1"
        )(hidden)
        hidden = nn.relu(hidden)

        # Wall prediction: per-cell binary probability
        # Output logits for BCE loss
        wall_logits = nn.Dense(
            self.env_height * self.env_width,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="pred_wall"
        )(hidden)
        wall_logits = wall_logits.reshape(self.env_height, self.env_width)

        # Goal position: softmax over all cells
        goal_logits = nn.Dense(
            self.env_height * self.env_width,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="pred_goal"
        )(hidden)

        # Agent spawn position: softmax over all cells
        agent_pos_logits = nn.Dense(
            self.env_height * self.env_width,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="pred_agent_pos"
        )(hidden)

        # Agent direction: softmax over 4 directions
        agent_dir_logits = nn.Dense(
            4,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="pred_agent_dir"
        )(hidden)

        return {
            'wall_logits': wall_logits,           # (H, W)
            'goal_logits': goal_logits,           # (H * W,)
            'agent_pos_logits': agent_pos_logits, # (H * W,)
            'agent_dir_logits': agent_dir_logits, # (4,)
        }


class ActorCriticWithCurriculumPrediction(nn.Module):
    """
    Actor-critic with integrated curriculum prediction head sharing a backbone.

    This network processes both observations AND curriculum features through
    a shared encoder, then outputs:
    - Actor (policy distribution)
    - Critic (value estimate)
    - Curriculum prediction (distribution over next level features)

    The curriculum prediction is part of the agent's internal model, meaning
    the agent itself learns the theory-of-mind of the curriculum generator.

    The prediction output does NOT affect action selection - actions depend
    only on the current observation. But the shared backbone means the agent's
    representations encode curriculum understanding.
    """
    action_dim: int
    env_height: int = DEFAULT_ENV_HEIGHT
    env_width: int = DEFAULT_ENV_WIDTH
    curriculum_hidden_size: int = DEFAULT_CURRICULUM_HIDDEN_SIZE

    @nn.compact
    def __call__(
        self,
        inputs,
        hidden,
        curriculum_features: Optional[chex.Array] = None,
        predict_curriculum: bool = False,
    ):
        """
        Forward pass with optional curriculum prediction.

        Args:
            inputs: Tuple of (obs, dones) for actor-critic
            hidden: LSTM hidden state
            curriculum_features: Output from get_curriculum_features()
            predict_curriculum: Whether to compute curriculum predictions

        Returns:
            hidden: Updated LSTM hidden state
            pi: Action distribution (Categorical)
            value: Value estimate
            curriculum_predictions: Dict of prediction logits (if predict_curriculum=True, else None)
        """
        obs, dones = inputs

        # === OBSERVATION EMBEDDING ===
        img_embed = nn.Conv(
            16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            name="obs_conv"
        )(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(
            5,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="dir_embed"
        )(dir_embed)

        obs_embedding = jnp.append(img_embed, dir_embed, axis=-1)

        # === CURRICULUM EMBEDDING (if provided) ===
        if curriculum_features is not None:
            # Encode curriculum features
            curriculum_embed = nn.Dense(
                64,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name="curriculum_embed_0"
            )(curriculum_features)
            curriculum_embed = nn.relu(curriculum_embed)
            curriculum_embed = nn.Dense(
                32,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name="curriculum_embed_1"
            )(curriculum_embed)
            curriculum_embed = nn.relu(curriculum_embed)

            # Broadcast curriculum_embed to match obs_embedding batch dims
            # obs_embedding shape: (time, batch, features) or (batch, features)
            # curriculum_embed shape: (features,)
            curriculum_embed_broadcast = jnp.broadcast_to(
                curriculum_embed,
                (*obs_embedding.shape[:-1], curriculum_embed.shape[-1])
            )

            # Concatenate observation and curriculum embeddings
            combined_embedding = jnp.concatenate([obs_embedding, curriculum_embed_broadcast], axis=-1)
        else:
            # No curriculum features - pad with zeros to maintain consistent input size
            zero_pad = jnp.zeros((*obs_embedding.shape[:-1], 32))
            combined_embedding = jnp.concatenate([obs_embedding, zero_pad], axis=-1)

        # === SHARED LSTM BACKBONE ===
        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))(
            (combined_embedding, dones), initial_carry=hidden
        )

        # === ACTOR HEAD ===
        actor_hidden = nn.Dense(
            32,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="actor0"
        )(embedding)
        actor_hidden = nn.relu(actor_hidden)
        actor_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor1"
        )(actor_hidden)
        pi = distrax.Categorical(logits=actor_logits)

        # === CRITIC HEAD ===
        critic_hidden = nn.Dense(
            32,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="critic0"
        )(embedding)
        critic_hidden = nn.relu(critic_hidden)
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic1"
        )(critic_hidden)

        # === CURRICULUM PREDICTION HEAD ===
        # Predicts the NEXT level distribution based on curriculum history.
        # This uses only curriculum_features (cross-episode history), not the current
        # episode's LSTM embedding, because we're predicting what level comes NEXT
        # before we've seen it.
        curriculum_predictions = None
        if predict_curriculum and curriculum_features is not None:
            # Use the curriculum embedding computed earlier (shared with main network)
            # This is the "shared backbone" - the curriculum understanding is integrated
            # into the agent's representations
            pred_hidden = nn.Dense(
                128,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name="pred_hidden0"
            )(curriculum_embed)  # Uses curriculum_embed from shared encoding above
            pred_hidden = nn.relu(pred_hidden)

            pred_hidden = nn.Dense(
                128,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name="pred_hidden1"
            )(pred_hidden)
            pred_hidden = nn.relu(pred_hidden)

            # Wall prediction: per-cell probability
            wall_logits = nn.Dense(
                self.env_height * self.env_width,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="pred_wall"
            )(pred_hidden)
            wall_logits = wall_logits.reshape(self.env_height, self.env_width)

            # Goal position distribution
            goal_logits = nn.Dense(
                self.env_height * self.env_width,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="pred_goal"
            )(pred_hidden)

            # Agent position distribution
            agent_pos_logits = nn.Dense(
                self.env_height * self.env_width,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="pred_agent_pos"
            )(pred_hidden)

            # Agent direction distribution
            agent_dir_logits = nn.Dense(
                4,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="pred_agent_dir"
            )(pred_hidden)

            curriculum_predictions = {
                'wall_logits': wall_logits,
                'goal_logits': goal_logits,
                'agent_pos_logits': agent_pos_logits,
                'agent_dir_logits': agent_dir_logits,
            }

        return hidden, pi, jnp.squeeze(value, axis=-1), curriculum_predictions

    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(
            jax.random.PRNGKey(0), (*batch_dims, 256)
        )


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

    # Compute ECE for walls
    bin_boundaries = jnp.linspace(0, 1, n_bins + 1)
    wall_ece = 0.0
    total_samples = len(wall_probs)

    for i in range(n_bins):
        mask = (wall_probs >= bin_boundaries[i]) & (wall_probs < bin_boundaries[i + 1])
        bin_size = mask.sum()

        # Only compute if bin is non-empty
        bin_confidence = jnp.where(bin_size > 0, wall_probs[mask].mean(), 0.0)
        bin_accuracy = jnp.where(bin_size > 0, wall_targets[mask].mean(), 0.0)
        wall_ece += jnp.where(
            bin_size > 0,
            bin_size * jnp.abs(bin_confidence - bin_accuracy),
            0.0
        )

    wall_ece = wall_ece / total_samples

    # Goal position: compute top-k accuracy as calibration proxy
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
    history_lengths: list = [8, 16, 32, 64],
    buffer_size: int = 100,
) -> NoveltyLearnabilityState:
    """Create state for tracking novelty and learnability metrics."""
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

    # Compute empirical goal distribution from batch
    goal_counts = jnp.zeros(env_height * env_width)
    for i in range(batch_size):
        goal_idx = actual_levels_batch.goal_pos[i, 1] * env_width + actual_levels_batch.goal_pos[i, 0]
        goal_counts = goal_counts.at[goal_idx].add(1)
    empirical_goal_dist = goal_counts / batch_size + 1e-10  # Add small epsilon

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


def apply_curriculum_prediction_gradient(
    train_state: "TrainState",
    curriculum_features: chex.Array,
    actual_level,
    env,
    env_params,
    sample_random_level,
    config: dict,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> Tuple["TrainState", dict]:
    """
    Apply gradient update for the curriculum prediction loss.

    This is called after the main PPO update to train the prediction head
    on predicting the next level in the curriculum.

    Args:
        train_state: Current train state
        curriculum_features: Features from get_curriculum_features()
        actual_level: The actual level that was used
        env: The environment
        env_params: Environment parameters
        sample_random_level: Function to sample random levels (for dummy obs)
        config: Training config
        env_height: Environment height
        env_width: Environment width

    Returns:
        Updated train_state and metrics dict
    """
    def prediction_loss_fn(params):
        # Create dummy observation to call the network
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

    grad_fn = jax.value_and_grad(prediction_loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(train_state.params)

    # Apply gradients
    train_state = train_state.apply_gradients(grads=grads)

    return train_state, metrics


# =============================================================================
# LOGGING AND VISUALIZATION FOR CURRICULUM PREDICTION
# =============================================================================

def log_curriculum_prediction_metrics(
    metrics: dict,
    branch: int,
    update_count: int,
) -> dict:
    """
    Format curriculum prediction metrics for logging by branch type.

    Args:
        metrics: Dict from compute_curriculum_prediction_loss
        branch: Which branch (0=DR, 1=replay, 2=mutate)
        update_count: Current update step

    Returns:
        Dict formatted for wandb logging
    """
    branch_names = {0: "random", 1: "replay", 2: "mutate"}
    branch_name = branch_names.get(branch, "unknown")

    log_dict = {
        f"curriculum_pred/{branch_name}/wall_loss": metrics.get("wall_loss", 0.0),
        f"curriculum_pred/{branch_name}/goal_loss": metrics.get("goal_loss", 0.0),
        f"curriculum_pred/{branch_name}/agent_pos_loss": metrics.get("agent_pos_loss", 0.0),
        f"curriculum_pred/{branch_name}/agent_dir_loss": metrics.get("agent_dir_loss", 0.0),
        f"curriculum_pred/{branch_name}/total_loss": metrics.get("total_loss", 0.0),
        f"curriculum_pred/all/total_loss": metrics.get("total_loss", 0.0),
    }

    return log_dict


def create_wall_prediction_heatmap(
    predictions: dict,
    actual_level,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> np.ndarray:
    """
    Create a visualization comparing predicted vs actual wall maps.

    Args:
        predictions: Dict from CurriculumPredictionHead
        actual_level: The actual Level
        env_height: Environment height
        env_width: Environment width

    Returns:
        RGB image as numpy array
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Predicted wall probabilities
    wall_probs = jax.nn.sigmoid(predictions['wall_logits'])
    im0 = axes[0].imshow(np.array(wall_probs), cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Predicted Wall Probs')
    plt.colorbar(im0, ax=axes[0])

    # Actual wall map
    im1 = axes[1].imshow(np.array(actual_level.wall_map), cmap='binary', vmin=0, vmax=1)
    axes[1].set_title('Actual Wall Map')
    plt.colorbar(im1, ax=axes[1])

    # Difference (error)
    error = np.abs(np.array(wall_probs) - np.array(actual_level.wall_map.astype(jnp.float32)))
    im2 = axes[2].imshow(error, cmap='Reds', vmin=0, vmax=1)
    axes[2].set_title('Prediction Error')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    # Convert to image array
    fig.canvas.draw()
    # Use buffer_rgba() which is the modern API
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    # Convert RGBA to RGB
    img = img[:, :, :3]
    plt.close(fig)

    return img


def create_position_prediction_heatmap(
    predictions: dict,
    actual_level,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> np.ndarray:
    """
    Create a visualization comparing predicted vs actual goal/agent positions.

    Args:
        predictions: Dict from CurriculumPredictionHead
        actual_level: The actual Level
        env_height: Environment height
        env_width: Environment width

    Returns:
        RGB image as numpy array
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Goal position distribution
    goal_probs = jax.nn.softmax(predictions['goal_logits']).reshape(env_height, env_width)
    im0 = axes[0].imshow(np.array(goal_probs), cmap='hot')
    # Mark actual goal position
    actual_goal = actual_level.goal_pos
    axes[0].scatter(actual_goal[0], actual_goal[1], c='green', s=100, marker='*', label='Actual')
    axes[0].set_title('Goal Position Distribution')
    axes[0].legend()
    plt.colorbar(im0, ax=axes[0])

    # Agent position distribution
    agent_probs = jax.nn.softmax(predictions['agent_pos_logits']).reshape(env_height, env_width)
    im1 = axes[1].imshow(np.array(agent_probs), cmap='hot')
    # Mark actual agent position
    actual_agent = actual_level.agent_pos
    axes[1].scatter(actual_agent[0], actual_agent[1], c='blue', s=100, marker='^', label='Actual')
    axes[1].set_title('Agent Position Distribution')
    axes[1].legend()
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()

    # Convert to image array
    fig.canvas.draw()
    # Use buffer_rgba() which is the modern API
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    # Convert RGBA to RGB
    img = img[:, :, :3]
    plt.close(fig)

    return img


# =============================================================================
# JAXUED IMPORTS AND EXISTING CODE
# =============================================================================

class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1


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
    # Curriculum prediction state (cross-episode memory)
    curriculum_state: Optional[CurriculumState] = struct.field(pytree_node=True, default=None)
    # Track prediction metrics for logging
    pred_metrics_accumulator: Optional[dict] = struct.field(pytree_node=True, default=None)
    # Novelty and learnability tracking (from open-endedness paper)
    nl_state: Optional[NoveltyLearnabilityState] = struct.field(pytree_node=True, default=None)


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
        hstate, pi, value, _ = train_state.apply_fn(train_state.params, x, hstate)
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
    _, _, last_value, _ = train_state.apply_fn(train_state.params, x, hstate)
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
        hstate, pi, _, _ = train_state.apply_fn(train_state.params, x, hstate)
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
                _, pi, values_pred, _ = train_state.apply_fn(params, (obs, last_dones), init_hstate)
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
        }
    }


def compute_score(config, dones, values, max_returns, advantages):
    if config["score_function"] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config["score_function"] == "pvl":
        return positive_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")


# =============================================================================
# N-ENVIRONMENT PREDICTION AND VISUALIZATION
# =============================================================================

def evaluate_n_step_prediction(
    rng: chex.PRNGKey,
    train_state: TrainState,
    level_sampler: LevelSampler,
    sample_random_level,
    mutate_level,
    env,
    env_params,
    n_predictions: int = 10,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> dict:
    """
    Evaluate N-step prediction: predict N future environments and compare with actual.

    This evaluates the prediction head's ability to forecast multiple steps ahead
    by generating N environments using the curriculum and comparing predictions.

    Args:
        rng: Random key
        train_state: Current training state with prediction head
        level_sampler: Level sampler for curriculum
        sample_random_level: Function to sample random levels
        mutate_level: Function to mutate levels
        env: Environment
        env_params: Environment parameters
        n_predictions: Number of environments to predict
        env_height: Environment height
        env_width: Environment width

    Returns:
        Dict with prediction vs actual comparisons and metrics
    """
    results = {
        "predictions": [],
        "actuals": [],
        "wall_losses": [],
        "goal_losses": [],
        "agent_pos_losses": [],
        "agent_dir_losses": [],
        "total_losses": [],
    }

    sampler = train_state.sampler
    curriculum_state = train_state.curriculum_state

    for i in range(n_predictions):
        rng, rng_pred, rng_gen = jax.random.split(rng, 3)

        # Get prediction from the model
        curriculum_features = get_curriculum_features(curriculum_state)

        # Create proper Observation object with dummy values
        # Need (time=1, batch=1) dimensions for the network
        dummy_obs = MazeObservation(
            image=jnp.zeros((5, 5, 3)),  # Single observation
            agent_dir=jnp.array(0, dtype=jnp.uint8),
        )
        dummy_done = jnp.array(False)

        # Add time and batch dimensions: (time=1, batch=1, ...)
        dummy_obs = jax.tree_util.tree_map(lambda x: x[None, None, ...], dummy_obs)
        dummy_dones = dummy_done[None, None, ...]

        # Get prediction
        init_hstate = ActorCriticWithCurriculumPrediction(
            action_dim=7,
            env_height=env_height,
            env_width=env_width,
        ).initialize_carry((1,))

        _, _, _, predictions = train_state.apply_fn(
            train_state.params,
            (dummy_obs, dummy_dones),
            init_hstate,
            curriculum_features=curriculum_features[None, :],
            predict_curriculum=True,
        )

        # Squeeze batch dimension from predictions where needed
        # wall_logits is already (H, W), others are (batch, features)
        predictions = {
            'wall_logits': predictions['wall_logits'],  # Already (H, W)
            'goal_logits': predictions['goal_logits'].squeeze(0),  # (features,)
            'agent_pos_logits': predictions['agent_pos_logits'].squeeze(0),
            'agent_dir_logits': predictions['agent_dir_logits'].squeeze(0),
        }

        # Generate actual next environment
        # For evaluation, we use random generation to avoid JAX tracing issues
        # This tests the prediction head's ability to predict level features
        # In actual training, all three branches (random/replay/mutate) are used
        rng_gen, rng_level = jax.random.split(rng_gen)
        actual_level = sample_random_level(rng_level)
        branch = 0  # random for evaluation

        # Compute losses (returns tuple of loss and metrics dict)
        _, pred_metrics = compute_curriculum_prediction_loss(
            predictions,
            actual_level,
            env_height=env_height,
            env_width=env_width,
        )

        # Store results (predictions already squeezed above)
        results["predictions"].append({
            "wall_probs": np.array(jax.nn.sigmoid(predictions["wall_logits"])),
            "goal_probs": np.array(jax.nn.softmax(predictions["goal_logits"])),
            "agent_pos_probs": np.array(jax.nn.softmax(predictions["agent_pos_logits"])),
            "agent_dir_probs": np.array(jax.nn.softmax(predictions["agent_dir_logits"])),
        })
        results["actuals"].append({
            "wall_map": np.array(actual_level.wall_map),
            "goal_pos": np.array(actual_level.goal_pos),
            "agent_pos": np.array(actual_level.agent_pos),
            "agent_dir": int(actual_level.agent_dir),
            "branch": branch,
        })
        results["wall_losses"].append(float(pred_metrics["wall_loss"]))
        results["goal_losses"].append(float(pred_metrics["goal_loss"]))
        results["agent_pos_losses"].append(float(pred_metrics["agent_pos_loss"]))
        results["agent_dir_losses"].append(float(pred_metrics["agent_dir_loss"]))
        results["total_losses"].append(float(pred_metrics["total_loss"]))

        # Update curriculum state for next prediction
        curriculum_state = update_curriculum_state(
            curriculum_state,
            actual_level,
            branch,
            0.0,  # Dummy score
            {"size": sampler["size"], "mean_score": 0.0, "score_std": 0.0, "max_score": 0.0},
            env_height=env_height,
            env_width=env_width,
        )

    # Compute summary statistics
    results["mean_wall_loss"] = np.mean(results["wall_losses"])
    results["mean_goal_loss"] = np.mean(results["goal_losses"])
    results["mean_agent_pos_loss"] = np.mean(results["agent_pos_losses"])
    results["mean_agent_dir_loss"] = np.mean(results["agent_dir_losses"])
    results["mean_total_loss"] = np.mean(results["total_losses"])

    return results


def create_prediction_comparison_figure(
    prediction_results: dict,
    n_display: int = 5,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> plt.Figure:
    """
    Create a figure comparing predicted distributions vs actual environments.

    Shows side-by-side comparison of:
    - Predicted wall probability heatmap vs actual wall map
    - Predicted goal position distribution vs actual
    - Predicted agent position distribution vs actual

    Args:
        prediction_results: Output from evaluate_n_step_prediction
        n_display: Number of predictions to display
        env_height: Environment height
        env_width: Environment width

    Returns:
        Matplotlib figure
    """
    n_display = min(n_display, len(prediction_results["predictions"]))

    fig, axes = plt.subplots(n_display, 4, figsize=(16, 4 * n_display))
    if n_display == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_display):
        pred = prediction_results["predictions"][i]
        actual = prediction_results["actuals"][i]

        # Column 1: Predicted vs Actual Wall Map
        ax = axes[i, 0]
        wall_probs = pred["wall_probs"].reshape(env_height, env_width)
        ax.imshow(wall_probs, cmap='Reds', vmin=0, vmax=1)
        # Overlay actual walls as contours
        ax.contour(actual["wall_map"], levels=[0.5], colors='blue', linewidths=2)
        ax.set_title(f"Step {i+1}: Pred Walls (red) vs Actual (blue)")
        ax.set_xlabel(f"Loss: {prediction_results['wall_losses'][i]:.3f}")

        # Column 2: Predicted vs Actual Goal Position
        ax = axes[i, 1]
        goal_probs = pred["goal_probs"].reshape(env_height, env_width)
        ax.imshow(goal_probs, cmap='Greens', vmin=0, vmax=goal_probs.max())
        ax.scatter(actual["goal_pos"][0], actual["goal_pos"][1],
                   c='red', s=200, marker='*', label='Actual')
        ax.set_title(f"Goal: Pred (green) vs Actual (star)")
        ax.set_xlabel(f"Loss: {prediction_results['goal_losses'][i]:.3f}")

        # Column 3: Predicted vs Actual Agent Position
        ax = axes[i, 2]
        agent_probs = pred["agent_pos_probs"].reshape(env_height, env_width)
        ax.imshow(agent_probs, cmap='Blues', vmin=0, vmax=agent_probs.max())
        ax.scatter(actual["agent_pos"][0], actual["agent_pos"][1],
                   c='red', s=200, marker='^', label='Actual')
        ax.set_title(f"Agent Pos: Pred (blue) vs Actual (triangle)")
        ax.set_xlabel(f"Loss: {prediction_results['agent_pos_losses'][i]:.3f}")

        # Column 4: Agent Direction + Branch Info
        ax = axes[i, 3]
        dir_probs = pred["agent_dir_probs"]
        directions = ['Right', 'Down', 'Left', 'Up']
        colors = ['lightblue'] * 4
        colors[actual["agent_dir"]] = 'red'
        ax.bar(directions, dir_probs, color=colors)
        ax.set_title(f"Direction (actual={directions[actual['agent_dir']]})")
        branch_names = ['Random', 'Replay', 'Mutate']
        ax.set_xlabel(f"Branch: {branch_names[actual['branch']]}, Loss: {prediction_results['agent_dir_losses'][i]:.3f}")
        ax.set_ylim(0, 1)

    plt.tight_layout()
    return fig


def create_pareto_frontier_figure(
    novelty_history: list,
    learnability_history: list,
    update_steps: list,
) -> plt.Figure:
    """
    Create a Pareto frontier visualization of novelty vs learnability.

    Shows the trajectory through novelty-learnability space over training,
    highlighting the open-ended region where both are positive.

    Args:
        novelty_history: List of novelty values over training
        learnability_history: List of learnability values over training
        update_steps: List of update step numbers

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Novelty vs Learnability scatter with trajectory
    ax = axes[0]

    # Color by training progress
    colors = np.linspace(0, 1, len(novelty_history))
    scatter = ax.scatter(learnability_history, novelty_history, c=colors,
                         cmap='viridis', s=50, alpha=0.7)

    # Draw trajectory line
    ax.plot(learnability_history, novelty_history, 'k-', alpha=0.3, linewidth=1)

    # Mark start and end
    ax.scatter([learnability_history[0]], [novelty_history[0]],
               c='green', s=200, marker='o', label='Start', zorder=5)
    ax.scatter([learnability_history[-1]], [novelty_history[-1]],
               c='red', s=200, marker='s', label='End', zorder=5)

    # Draw quadrant regions
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Shade the open-ended region (both positive)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between([max(0, xlim[0]), xlim[1]], [0, 0], [ylim[1], ylim[1]],
                    alpha=0.1, color='green', label='Open-ended region')

    # Labels for quadrants
    ax.text(0.95, 0.95, 'Open-ended\n(ideal)', transform=ax.transAxes,
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
    ax.plot(update_steps, novelty_history, 'b-', label='Novelty', linewidth=2)
    ax.plot(update_steps, learnability_history, 'g-', label='Learnability', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(update_steps, 0, novelty_history,
                    where=[n > 0 for n in novelty_history], alpha=0.2, color='blue')
    ax.fill_between(update_steps, 0, learnability_history,
                    where=[l > 0 for l in learnability_history], alpha=0.2, color='green')
    ax.set_xlabel('Training Updates')
    ax.set_ylabel('Metric Value')
    ax.set_title('Novelty and Learnability Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_post_training_evaluation(
    train_state: TrainState,
    level_sampler: LevelSampler,
    sample_random_level,
    mutate_level,
    env,
    env_params,
    n_predictions: int = 20,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
    novelty_history: list = None,
    learnability_history: list = None,
    update_steps: list = None,
) -> dict:
    """
    Run post-training evaluation including N-step prediction and visualizations.

    Args:
        train_state: Final training state
        level_sampler: Level sampler
        sample_random_level: Random level generator
        mutate_level: Level mutator
        env: Environment
        env_params: Environment parameters
        n_predictions: Number of environments to predict
        env_height: Environment height
        env_width: Environment width
        novelty_history: History of novelty values (optional)
        learnability_history: History of learnability values (optional)
        update_steps: History of update steps (optional)

    Returns:
        Dict with all evaluation results and figures
    """
    rng = jax.random.PRNGKey(42)

    results = {}

    # N-step prediction evaluation
    print(f"Running {n_predictions}-step prediction evaluation...")
    prediction_results = evaluate_n_step_prediction(
        rng,
        train_state,
        level_sampler,
        sample_random_level,
        mutate_level,
        env,
        env_params,
        n_predictions=n_predictions,
        env_height=env_height,
        env_width=env_width,
    )
    results["prediction_results"] = prediction_results

    # Create comparison figure
    comparison_fig = create_prediction_comparison_figure(
        prediction_results,
        n_display=min(5, n_predictions),
        env_height=env_height,
        env_width=env_width,
    )
    results["comparison_figure"] = comparison_fig

    # Log to wandb
    wandb.log({
        "post_training/n_step_mean_wall_loss": prediction_results["mean_wall_loss"],
        "post_training/n_step_mean_goal_loss": prediction_results["mean_goal_loss"],
        "post_training/n_step_mean_agent_pos_loss": prediction_results["mean_agent_pos_loss"],
        "post_training/n_step_mean_agent_dir_loss": prediction_results["mean_agent_dir_loss"],
        "post_training/n_step_mean_total_loss": prediction_results["mean_total_loss"],
        "post_training/prediction_comparison": wandb.Image(comparison_fig),
    })

    # Pareto frontier visualization (if history available)
    if novelty_history and learnability_history and update_steps:
        pareto_fig = create_pareto_frontier_figure(
            novelty_history,
            learnability_history,
            update_steps,
        )
        results["pareto_figure"] = pareto_fig
        wandb.log({"post_training/pareto_frontier": wandb.Image(pareto_fig)})

    plt.close('all')

    print("Post-training evaluation complete.")
    print(f"  Mean prediction losses over {n_predictions} steps:")
    print(f"    Wall: {prediction_results['mean_wall_loss']:.4f}")
    print(f"    Goal: {prediction_results['mean_goal_loss']:.4f}")
    print(f"    Agent Pos: {prediction_results['mean_agent_pos_loss']:.4f}")
    print(f"    Agent Dir: {prediction_results['mean_agent_dir_loss']:.4f}")
    print(f"    Total: {prediction_results['mean_total_loss']:.4f}")

    return results


def main(config=None, project="JaxUED-minigrid-maze"):
    tags = []
    if not config["exploratory_grad_updates"]:
        tags.append("robust")
    if config["use_accel"]:
        tags.append("ACCEL")
    else:
        tags.append("PLR")

    run = wandb.init(config=config, project=project, group=config["run_name"], tags=tags)
    config = wandb.config

    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_lengths/*", step_metric="num_updates")

    # Curriculum prediction metrics (if enabled)
    if config.get("use_curriculum_prediction", False):
        wandb.define_metric("curriculum_pred/*", step_metric="num_updates")
        wandb.define_metric("curriculum_pred/random/*", step_metric="num_updates")
        wandb.define_metric("curriculum_pred/replay/*", step_metric="num_updates")
        wandb.define_metric("curriculum_pred/mutate/*", step_metric="num_updates")
        wandb.define_metric("curriculum_pred/all/*", step_metric="num_updates")
        wandb.define_metric("curriculum_pred/calibration/*", step_metric="num_updates")
        wandb.define_metric("curriculum_pred/divergence/*", step_metric="num_updates")
        wandb.define_metric("curriculum_pred/images/*", step_metric="num_updates")
        wandb.define_metric("curriculum_pred/openendedness/*", step_metric="num_updates")
        wandb.define_metric("curriculum_state/*", step_metric="num_updates")

    def log_eval(stats, train_state_info, train_state=None):
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

        # Curriculum prediction metrics (if enabled)
        if config.get("use_curriculum_prediction", False) and "pred" in stats:
            pred_metrics = stats["pred"]
            branches = pred_metrics.get("branch", jnp.array([]))

            # Log aggregated metrics across all branches
            log_dict.update({
                "curriculum_pred/all/wall_loss": float(pred_metrics.get("wall_loss", jnp.array(0.0)).mean()),
                "curriculum_pred/all/goal_loss": float(pred_metrics.get("goal_loss", jnp.array(0.0)).mean()),
                "curriculum_pred/all/agent_pos_loss": float(pred_metrics.get("agent_pos_loss", jnp.array(0.0)).mean()),
                "curriculum_pred/all/agent_dir_loss": float(pred_metrics.get("agent_dir_loss", jnp.array(0.0)).mean()),
                "curriculum_pred/all/total_loss": float(pred_metrics.get("total_loss", jnp.array(0.0)).mean()),
            })

            # Log per-branch metrics (Gap #1 fix)
            branch_names = {0: "random", 1: "replay", 2: "mutate"}
            for branch_id, branch_name in branch_names.items():
                mask = (branches == branch_id)
                count = mask.sum()
                if count > 0:
                    for metric_name in ["wall_loss", "goal_loss", "agent_pos_loss", "agent_dir_loss", "total_loss"]:
                        metric_vals = pred_metrics.get(metric_name, jnp.array([0.0]))
                        masked_mean = (metric_vals * mask).sum() / count
                        log_dict[f"curriculum_pred/{branch_name}/{metric_name}"] = float(masked_mean)
                    log_dict[f"curriculum_pred/{branch_name}/count"] = int(count)

        # Curriculum state metrics (if enabled)
        if config.get("use_curriculum_prediction", False) and train_state is not None and train_state.curriculum_state is not None:
            cs = train_state.curriculum_state
            log_dict.update({
                "curriculum_state/training_step": int(cs.training_step),
                "curriculum_state/total_replay_steps": int(cs.total_replay_steps),
                "curriculum_state/total_random_steps": int(cs.total_random_steps),
                "curriculum_state/replay_fraction": float(cs.total_replay_steps / max(cs.training_step, 1)),
                "curriculum_state/buffer_size": int(cs.buffer_size),
                "curriculum_state/buffer_mean_score": float(cs.buffer_mean_score),
                "curriculum_state/mean_wall_density": float(cs.recent_wall_densities.mean()),
            })

            # Periodically log visualizations and calibration metrics (Gap #2 and #3 fix)
            update_count = stats.get("update_count", 0)
            pred_eval_freq = config.get("curriculum_pred_eval_freq", 100)
            if update_count > 0 and update_count % pred_eval_freq == 0:
                try:
                    # Get current predictions for visualization
                    curriculum_features = get_curriculum_features(cs)

                    # Use the last level from replay batch as representative
                    representative_level = jax.tree_util.tree_map(
                        lambda x: x[0], train_state.replay_last_level_batch
                    )

                    # Create dummy observation to get predictions
                    rng_viz = jax.random.PRNGKey(update_count)
                    dummy_level = sample_random_level(rng_viz)
                    dummy_obs, _ = eval_env.reset_to_level(rng_viz, dummy_level, env_params)
                    dummy_obs = jax.tree_util.tree_map(lambda x: x[None, None, ...], dummy_obs)
                    dummy_dones = jnp.zeros((1, 1), dtype=bool)
                    dummy_hidden = ActorCriticWithCurriculumPrediction.initialize_carry((1,))

                    # Get predictions
                    _, _, _, predictions = train_state.apply_fn(
                        train_state.params,
                        (dummy_obs, dummy_dones),
                        dummy_hidden,
                        curriculum_features=curriculum_features,
                        predict_curriculum=True,
                    )

                    # Create and log visualization heatmaps (Gap #2)
                    wall_heatmap = create_wall_prediction_heatmap(
                        predictions, representative_level,
                        env_height=env_max_height, env_width=env_max_width
                    )
                    pos_heatmap = create_position_prediction_heatmap(
                        predictions, representative_level,
                        env_height=env_max_height, env_width=env_max_width
                    )
                    log_dict["curriculum_pred/images/wall_heatmap"] = wandb.Image(
                        wall_heatmap, caption=f"Wall Predictions (step {update_count})"
                    )
                    log_dict["curriculum_pred/images/position_heatmap"] = wandb.Image(
                        pos_heatmap, caption=f"Position Predictions (step {update_count})"
                    )

                    # Compute and log calibration metrics (Gap #3)
                    cal_metrics = compute_calibration_metrics(
                        predictions, representative_level,
                        env_height=env_max_height, env_width=env_max_width
                    )
                    log_dict.update({
                        "curriculum_pred/calibration/wall_ece": float(cal_metrics["wall_ece"]),
                        "curriculum_pred/calibration/goal_prob_at_actual": float(cal_metrics["goal_prob_at_actual"]),
                        "curriculum_pred/calibration/agent_pos_prob_at_actual": float(cal_metrics["agent_pos_prob_at_actual"]),
                        "curriculum_pred/calibration/agent_dir_prob_at_actual": float(cal_metrics["agent_dir_prob_at_actual"]),
                    })

                    # Compute divergence metrics using recent levels from buffer
                    # Use a batch of levels from the sampler for empirical distribution
                    sampler = train_state.sampler
                    if sampler["size"] >= 10:
                        # Get indices of top-10 scoring levels as representative batch
                        top_indices = jnp.argsort(sampler["scores"])[-10:]
                        batch_levels = level_sampler.get_levels(sampler, top_indices)
                        div_metrics = compute_distribution_divergence(
                            predictions, batch_levels,
                            env_height=env_max_height, env_width=env_max_width
                        )
                        log_dict.update({
                            "curriculum_pred/divergence/goal_kl": float(div_metrics["goal_kl"]),
                            "curriculum_pred/divergence/goal_js": float(div_metrics["goal_js"]),
                            "curriculum_pred/divergence/wall_density_error": float(div_metrics["wall_density_error"]),
                            "curriculum_pred/divergence/predicted_wall_density": float(div_metrics["predicted_wall_density"]),
                            "curriculum_pred/divergence/empirical_wall_density": float(div_metrics["empirical_wall_density"]),
                        })

                    # Compute and log novelty/learnability metrics (from open-endedness paper)
                    if train_state.nl_state is not None and train_state.nl_state.total_samples > 10:
                        novelty, novelty_details = compute_novelty(train_state.nl_state)
                        learnability, learnability_details = compute_learnability(train_state.nl_state)
                        openendedness, regime = compute_openendedness_score(novelty, learnability)

                        log_dict.update({
                            "curriculum_pred/openendedness/novelty": novelty,
                            "curriculum_pred/openendedness/learnability": learnability,
                            "curriculum_pred/openendedness/score": openendedness,
                            "curriculum_pred/openendedness/novelty_slope": novelty_details.get("novelty_slope", 0.0),
                            "curriculum_pred/openendedness/instantaneous_novelty": novelty_details.get("instantaneous_novelty", 0.0),
                            "curriculum_pred/openendedness/learnability_slope": learnability_details.get("learnability_slope", 0.0),
                        })

                except Exception as e:
                    print(f"Warning: Failed to compute visualization/calibration metrics: {e}")

        wandb.log(log_dict)

    # setup the environment
    env_max_height = 13
    env_max_width = 13
    env = Maze(max_height=env_max_height, max_width=env_max_width, agent_view_size=config["agent_view_size"], normalize_obs=True)
    eval_env = env
    sample_random_level = make_level_generator(env_max_height, env_max_width, config["n_walls"])
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

        # Choose network based on whether curriculum prediction is enabled
        if config.get("use_curriculum_prediction", False):
            network = ActorCriticWithCurriculumPrediction(
                action_dim=env.action_space(env_params).n,
                env_height=env_max_height,
                env_width=env_max_width,
                curriculum_hidden_size=config.get("curriculum_hidden_size", DEFAULT_CURRICULUM_HIDDEN_SIZE),
            )
            # Initialize with curriculum features for prediction head
            init_curriculum_features = get_curriculum_features(
                create_curriculum_state(config.get("curriculum_history_length", DEFAULT_CURRICULUM_HISTORY_LENGTH))
            )
            network_params = network.init(
                rng,
                init_x,
                ActorCriticWithCurriculumPrediction.initialize_carry((config["num_train_envs"],)),
                curriculum_features=init_curriculum_features,
                predict_curriculum=True,
            )
            # Initialize curriculum state
            curriculum_state = create_curriculum_state(
                config.get("curriculum_history_length", DEFAULT_CURRICULUM_HISTORY_LENGTH)
            )
            # Initialize prediction metrics accumulator
            pred_metrics_accumulator = {
                'random_nll': jnp.zeros(100),  # Rolling buffer for metrics
                'replay_nll': jnp.zeros(100),
                'mutate_nll': jnp.zeros(100),
                'random_count': 0,
                'replay_count': 0,
                'mutate_count': 0,
            }
        else:
            network = ActorCritic(env.action_space(env_params).n)
            network_params = network.init(rng, init_x, ActorCritic.initialize_carry((config["num_train_envs"],)))
            curriculum_state = None
            pred_metrics_accumulator = None

        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
        pholder_level = sample_random_level(jax.random.PRNGKey(0))
        sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
        pholder_level_batch = jax.tree_util.tree_map(lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0), pholder_level)

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
            curriculum_state=curriculum_state,
            pred_metrics_accumulator=pred_metrics_accumulator,
            nl_state=create_novelty_learnability_state() if config.get("use_curriculum_prediction", False) else None,
        )

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        """This is the main training loop. It basically calls either `on_new_levels`, `on_replay_levels`,
        or `on_mutate_levels` at every step."""

        # Helper to get correct initialize_carry based on config
        def get_init_carry():
            if config.get("use_curriculum_prediction", False):
                return ActorCriticWithCurriculumPrediction.initialize_carry((config["num_train_envs"],))
            else:
                return ActorCritic.initialize_carry((config["num_train_envs"],))

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
                get_init_carry(),
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
                get_init_carry(),
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

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=new_levels,
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
                get_init_carry(),
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
                get_init_carry(),
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

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.REPLAY,
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
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
                get_init_carry(),
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
                get_init_carry(),
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

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_mutation_updates=train_state.num_mutation_updates + 1,
                mutation_last_level_batch=child_levels,
            )
            return (rng, train_state), metrics

        rng, train_state = carry
        rng, rng_replay = jax.random.split(rng)

        # the train step makes a decision on which branch to take, either on_new, on_replay or on_mutate
        # on_mutate is only called if the replay branch has been taken before (as it uses train_state.update_state).

        if config["use_accel"]:
            s = train_state.update_state
            branch = (1 - s) * level_sampler.sample_replay_decision(train_state.sampler, rng_replay) + 2 * s
        else:
            branch = level_sampler.sample_replay_decision(train_state.sampler, rng_replay).astype(int)

        # =====================================================================
        # CURRICULUM PREDICTION (before branch execution)
        # =====================================================================
        curriculum_predictions = None
        curriculum_features = None
        if config.get("use_curriculum_prediction", False) and train_state.curriculum_state is not None:
            # Get curriculum features from current state
            curriculum_features = get_curriculum_features(train_state.curriculum_state)

            # Create dummy observation to call the network for prediction
            # The prediction only uses curriculum_features, not the observation,
            # but we need to call the full network to get properly initialized params
            rng, rng_dummy = jax.random.split(rng)
            dummy_level = sample_random_level(rng_dummy)
            dummy_obs, _ = env.reset_to_level(rng_dummy, dummy_level, env_params)
            # Add time and batch dimensions
            dummy_obs = jax.tree_util.tree_map(lambda x: x[None, None, ...], dummy_obs)
            dummy_dones = jnp.zeros((1, 1), dtype=bool)
            dummy_hidden = ActorCriticWithCurriculumPrediction.initialize_carry((1,))

            # Call network with predict_curriculum=True
            _, _, _, curriculum_predictions = train_state.apply_fn(
                train_state.params,
                (dummy_obs, dummy_dones),
                dummy_hidden,
                curriculum_features=curriculum_features,
                predict_curriculum=True,
            )

        # =====================================================================
        # EXECUTE BRANCH (unchanged logic)
        # =====================================================================
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
        # CURRICULUM PREDICTION LOSS AND STATE UPDATE (after branch execution)
        # =====================================================================
        if config.get("use_curriculum_prediction", False) and train_state.curriculum_state is not None:
            # Determine which level was used based on branch
            # Use first level from the batch as representative sample
            level_batch = jax.lax.switch(
                branch,
                [
                    lambda: train_state.dr_last_level_batch,
                    lambda: train_state.replay_last_level_batch,
                    lambda: train_state.mutation_last_level_batch,
                ],
            )
            # Get first level from batch
            representative_level = jax.tree_util.tree_map(lambda x: x[0], level_batch)

            # Compute mean score for this batch using mask-based approach (JAX-friendly)
            sampler = train_state.sampler
            # Create mask for valid entries
            valid_mask = jnp.arange(sampler["scores"].shape[0]) < sampler["size"]
            valid_count = jnp.maximum(valid_mask.sum(), 1)  # Avoid division by zero
            mean_score = (sampler["scores"] * valid_mask).sum() / valid_count

            # Compute std using mask
            score_var = ((sampler["scores"] - mean_score) ** 2 * valid_mask).sum() / valid_count
            score_std = jnp.sqrt(score_var)

            # Apply gradient update for prediction loss if curriculum features available
            if curriculum_features is not None:
                train_state, pred_metrics = apply_curriculum_prediction_gradient(
                    train_state,
                    curriculum_features,
                    representative_level,
                    env,
                    env_params,
                    sample_random_level,
                    config,
                    env_height=env_max_height,
                    env_width=env_max_width,
                )

                # Add prediction metrics to main metrics
                metrics["pred"] = pred_metrics
                metrics["pred"]["branch"] = branch

                # Update novelty/learnability tracking
                if train_state.nl_state is not None:
                    total_loss = pred_metrics.get("total_loss", jnp.array(0.0))
                    new_nl_state = update_novelty_learnability_state(
                        train_state.nl_state,
                        total_loss,  # Keep as JAX array, don't convert to Python float
                        history_length_idx=-1,  # Use full history
                    )
                    train_state = train_state.replace(nl_state=new_nl_state)

            # Update curriculum state with the actual level
            sampler_stats = {
                "size": sampler["size"],
                "mean_score": mean_score,
                "score_std": score_std,
                "max_score": sampler["scores"].max(),
            }

            new_curriculum_state = update_curriculum_state(
                train_state.curriculum_state,
                representative_level,
                branch,
                mean_score,  # Use mean score as representative
                sampler_stats,
                env_height=env_max_height,
                env_width=env_max_width,
            )

            train_state = train_state.replace(curriculum_state=new_curriculum_state)

        return (rng, train_state), metrics

    def eval(rng: chex.PRNGKey, train_state: TrainState):
        """This evaluates the current policy on the ste of evaluation levels specified by config["eval_levels"]."""

        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["eval_levels"])
        num_levels = len(config["eval_levels"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)

        # Use correct initialize_carry based on config
        if config.get("use_curriculum_prediction", False):
            init_hstate = ActorCriticWithCurriculumPrediction.initialize_carry((num_levels,))
        else:
            init_hstate = ActorCritic.initialize_carry((num_levels,))

        states, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            env_params,
            train_state,
            init_hstate,
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

        return (rng, train_state), metrics

    def eval_checkpoint(og_config):
        """This function is what is used to evaluate a saved checkpoint after training. It first loads the checkpoint and
        then runs evaluation."""
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))

        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, "config.json")) as f:
                config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_directory, "models"), checkpointers=ocp.PyTreeCheckpointer())

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

    if config["mode"] == "eval":
        return eval_checkpoint(config)

    # set up the train states
    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)

    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    # Track novelty/learnability history for Pareto frontier
    novelty_history = []
    learnability_history = []
    update_steps = []

    # and run the train_eval_sep function for the specified number of updates
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics["time_delta"] = curr_time - start_time
        log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler), train_state=runner_state[1])

        # Track novelty/learnability for Pareto frontier visualization
        if config.get("use_curriculum_prediction", False) and runner_state[1].nl_state is not None:
            if runner_state[1].nl_state.total_samples > 10:
                novelty, _ = compute_novelty(runner_state[1].nl_state)
                learnability, _ = compute_learnability(runner_state[1].nl_state)
                novelty_history.append(float(novelty))
                learnability_history.append(float(learnability))
                update_steps.append((eval_step + 1) * config["eval_freq"])

        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(eval_step, runner_state[1])
            checkpoint_manager.wait_until_finished()

    # Post-training evaluation for curriculum prediction
    if config.get("use_curriculum_prediction", False):
        print("\n" + "="*60)
        print("POST-TRAINING EVALUATION")
        print("="*60)

        final_train_state = runner_state[1]
        n_pred_eval = config.get("n_prediction_eval", 20)

        try:
            eval_results = run_post_training_evaluation(
                train_state=final_train_state,
                level_sampler=level_sampler,
                sample_random_level=sample_random_level,
                mutate_level=mutate_level,
                env=eval_env,
                env_params=env_params,
                n_predictions=n_pred_eval,
                env_height=env_max_height,
                env_width=env_max_width,
                novelty_history=novelty_history if novelty_history else None,
                learnability_history=learnability_history if learnability_history else None,
                update_steps=update_steps if update_steps else None,
            )
        except Exception as e:
            print(f"Warning: Post-training evaluation failed: {e}")

        # Save final checkpoint as wandb artifact (easy checkpoint saving)
        if config["checkpoint_save_interval"] > 0:
            try:
                artifact = wandb.Artifact(
                    name=f"model-{config.get('run_name', 'default')}-final",
                    type="model",
                    description=f"Final trained model after {config['num_updates']} updates"
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

    # env config
    group.add_argument("--agent_view_size", type=int, default=5)

    # dr
    group.add_argument("--n_walls", type=int, default=25)

    # Curriculum prediction (for theory-of-mind of env generator)
    curriculum_group = parser.add_argument_group("Curriculum prediction")
    curriculum_group.add_argument("--use_curriculum_prediction", action=argparse.BooleanOptionalAction, default=False,
                                   help="Enable curriculum prediction head for learning env generator ToM")
    curriculum_group.add_argument("--curriculum_history_length", type=int, default=64,
                                   help="Number of recent levels to track in curriculum history")
    curriculum_group.add_argument("--curriculum_hidden_size", type=int, default=128,
                                   help="Hidden size for curriculum encoder")
    curriculum_group.add_argument("--curriculum_pred_coeff", type=float, default=1.0,
                                   help="Coefficient for curriculum prediction loss in total loss")
    curriculum_group.add_argument("--curriculum_wall_weight", type=float, default=1.0,
                                   help="Weight for wall prediction loss component")
    curriculum_group.add_argument("--curriculum_goal_weight", type=float, default=1.0,
                                   help="Weight for goal position prediction loss component")
    curriculum_group.add_argument("--curriculum_agent_pos_weight", type=float, default=1.0,
                                   help="Weight for agent position prediction loss component")
    curriculum_group.add_argument("--curriculum_agent_dir_weight", type=float, default=1.0,
                                   help="Weight for agent direction prediction loss component")
    curriculum_group.add_argument("--curriculum_pred_eval_freq", type=int, default=100,
                                   help="Frequency (in updates) to log detailed curriculum prediction metrics")
    curriculum_group.add_argument("--n_prediction_eval", type=int, default=20,
                                   help="Number of environments to predict in post-training evaluation")

    config = vars(parser.parse_args())
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"])
    config["group_name"] = "".join([str(config[key]) for key in sorted([a.dest for a in parser._action_groups[2]._group_actions])])

    if config["mode"] == "eval":
        os.environ["WANDB_MODE"] = "disabled"

    wandb.login()
    main(config, project=config["project"])
