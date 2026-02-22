"""
Neural network architectures for curriculum awareness ablations.

Contains:
- ActorCritic: Standard PPO actor-critic with LSTM backbone
- ActorCriticPersistent: ActorCritic without hidden state reset
- ActorCriticWithContext: ActorCritic with additional context input
- ActorCriticWithCurriculumPrediction: ActorCritic with curriculum prediction head
- CurriculumProbe: Probe network for predicting level features from hidden state
- CurriculumEncoder: Encodes curriculum history for prediction
- CurriculumPredictionHead: Predicts next level distributions
"""

from typing import Sequence, Optional
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import chex

from jaxued.linen import ResetRNN
from .types import DEFAULT_ENV_HEIGHT, DEFAULT_ENV_WIDTH, DEFAULT_HSTATE_DIM

# Constants for curriculum prediction
DEFAULT_CURRICULUM_HIDDEN_SIZE = 128
DEFAULT_CURRICULUM_HISTORY_LENGTH = 64


# =============================================================================
# ACTOR-CRITIC NETWORKS
# =============================================================================

class ActorCritic(nn.Module):
    """
    Standard actor-critic with LSTM backbone.

    Uses ResetRNN which resets hidden state on episode boundary (done=True).
    This is the baseline architecture used by accel_probe.
    """
    action_dim: int

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs

        # CNN encoder for observations
        img_embed = nn.Conv(
            16, kernel_size=(3, 3), strides=(1, 1), padding="VALID"
        )(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        # Direction embedding
        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(
            5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
            name="scalar_embed"
        )(dir_embed)

        # Combine embeddings
        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        # LSTM with reset on done
        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))(
            (embedding, dones), initial_carry=hidden
        )

        # Actor head
        actor_mean = nn.Dense(
            32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0"
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0),
            name="actor1"
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        critic = nn.Dense(
            32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0"
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1"
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_dims):
        """Initialize LSTM hidden state (c, h) with zeros."""
        return nn.OptimizedLSTMCell(features=256).initialize_carry(
            jax.random.PRNGKey(0), (*batch_dims, 256)
        )


class ActorCriticPersistent(nn.Module):
    """
    Actor-critic with PERSISTENT LSTM (no reset on episode boundary).

    Key difference from ActorCritic: hidden state carries across episodes.
    Used by persistent_lstm agent to test emergent curriculum awareness.

    Uses nn.scan for proper Flax module handling inside scan.
    """
    action_dim: int

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs

        # CNN encoder
        img_embed = nn.Conv(
            16, kernel_size=(3, 3), strides=(1, 1), padding="VALID"
        )(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        # Direction embedding
        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(
            5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
            name="scalar_embed"
        )(dir_embed)

        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        # LSTM WITHOUT reset - state persists across episodes
        # Use nn.scan for proper Flax module handling
        # Note: we ignore 'dones' here - no reset happens
        class LSTMCell(nn.Module):
            features: int = 256

            @nn.compact
            def __call__(self, carry, x):
                return nn.OptimizedLSTMCell(features=self.features)(carry, x)

        # Handle batch dimension
        if embedding.ndim == 3:  # (time, batch, features)
            ScanLSTM = nn.scan(
                LSTMCell,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=0,
                out_axes=0,
            )
            hidden, embedding = ScanLSTM(name="lstm")(hidden, embedding)
        else:  # (batch, features) - single step
            hidden, embedding = LSTMCell(name="lstm")(hidden, embedding)

        # Actor head
        actor_mean = nn.Dense(
            32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0"
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0),
            name="actor1"
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        critic = nn.Dense(
            32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0"
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1"
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(
            jax.random.PRNGKey(0), (*batch_dims, 256)
        )


class AdversaryActorCritic(nn.Module):
    """
    Adversary network for PAIRED level generation.

    Takes MazeEditor observations (image, time, random_z) and outputs
    edit actions to construct maze levels. The adversary learns to
    create levels that maximize regret (antagonist - protagonist performance).

    Architecture:
    - Conv128 encoder for image observation
    - Time embedding for sequential generation
    - Random Z input for diversity
    - LSTM for sequential decision making
    - Actor/Critic heads
    """
    action_dim: int
    max_timesteps: int = 50

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs

        # CNN encoder with larger capacity for level generation
        img_embed = nn.Conv(
            128, kernel_size=(3, 3), strides=(1, 1), padding="VALID"
        )(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        # Time embedding for sequential generation
        time_value = nn.Embed(
            self.max_timesteps + 1, 10,
            name="time_embed",
            embedding_init=orthogonal(1.0)
        )(jnp.clip(obs.time, None, self.max_timesteps))

        # Random Z for diversity (passed through from obs)
        random_z_value = obs.random_z

        # Combine all embeddings
        embedding = jnp.concatenate((img_embed, time_value, random_z_value), axis=-1)

        # LSTM with reset on done
        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))(
            (embedding, dones), initial_carry=hidden
        )

        # Actor head
        actor_mean = nn.Dense(
            32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0"
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0),
            name="actor1"
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        critic = nn.Dense(
            32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0"
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1"
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_dims):
        """Initialize LSTM hidden state (c, h) with zeros."""
        return nn.OptimizedLSTMCell(features=256).initialize_carry(
            jax.random.PRNGKey(0), (*batch_dims, 256)
        )


class ActorCriticWithContext(nn.Module):
    """
    Actor-critic with additional context vector input.

    The context vector is concatenated with the observation embedding
    before being fed to the LSTM. Used by context_vector and episodic_memory agents.
    """
    action_dim: int
    context_dim: int = 64

    @nn.compact
    def __call__(self, inputs, hidden, context=None):
        obs, dones = inputs

        # CNN encoder
        img_embed = nn.Conv(
            16, kernel_size=(3, 3), strides=(1, 1), padding="VALID"
        )(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        # Direction embedding
        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(
            5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
            name="scalar_embed"
        )(dir_embed)

        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        # Always concatenate context (default to zeros if not provided).
        # This ensures consistent LSTM input shape regardless of whether
        # context is passed, matching the pattern in ActorCriticWithCurriculumPrediction.
        if context is not None:
            # Ensure context has right shape for broadcasting
            if context.ndim == 1:
                context = context[None, :]  # Add batch dim
            if embedding.ndim == 3:  # (time, batch, features)
                # Expand context for time dimension
                context = jnp.broadcast_to(
                    context[None, :, :],
                    (embedding.shape[0], embedding.shape[1], context.shape[-1])
                )
        else:
            # No context provided - pad with zeros to maintain consistent input size
            context = jnp.zeros((*embedding.shape[:-1], self.context_dim))
        embedding = jnp.concatenate([embedding, context], axis=-1)

        # LSTM with reset
        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))(
            (embedding, dones), initial_carry=hidden
        )

        # Actor head
        actor_mean = nn.Dense(
            32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0"
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0),
            name="actor1"
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        critic = nn.Dense(
            32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0"
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1"
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(
            jax.random.PRNGKey(0), (*batch_dims, 256)
        )


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
    """
    env_height: int = DEFAULT_ENV_HEIGHT
    env_width: int = DEFAULT_ENV_WIDTH
    use_episode_context: bool = True

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
            episode_return: Return from last episode, shape (batch,) or (batch, 1)
            episode_solved: Whether agent solved, shape (batch,) or (batch, 1)
            episode_length: Number of steps in episode, shape (batch,) or (batch, 1)

        Returns:
            Dict with prediction logits for each level component
        """
        # Build input features
        features = [hidden_state]

        if self.use_episode_context:
            if episode_return is not None:
                ret = episode_return.reshape(-1, 1) if episode_return.ndim == 1 else episode_return
                features.append(ret)
            if episode_solved is not None:
                solved = episode_solved.astype(jnp.float32).reshape(-1, 1)
                features.append(solved)
            if episode_length is not None:
                length_norm = (episode_length.astype(jnp.float32) / 256.0).reshape(-1, 1)
                features.append(length_norm)

        x = jnp.concatenate(features, axis=-1) if len(features) > 1 else hidden_state

        # Probe encoder
        x = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
            name="probe_encoder_0"
        )(x)
        x = nn.relu(x)

        x = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
            name="probe_encoder_1"
        )(x)
        x = nn.relu(x)

        # Prediction heads

        # Wall prediction: per-cell probability
        wall_logits = nn.Dense(
            self.env_height * self.env_width,
            kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="probe_wall"
        )(x)
        wall_logits = wall_logits.reshape(-1, self.env_height, self.env_width)

        # Goal position: softmax over all cells
        goal_logits = nn.Dense(
            self.env_height * self.env_width,
            kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="probe_goal"
        )(x)

        # Agent spawn position
        agent_pos_logits = nn.Dense(
            self.env_height * self.env_width,
            kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="probe_agent_pos"
        )(x)

        # Agent direction: 4 directions
        agent_dir_logits = nn.Dense(
            4, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="probe_agent_dir"
        )(x)

        return {
            'wall_logits': wall_logits,
            'goal_logits': goal_logits,
            'agent_pos_logits': agent_pos_logits,
            'agent_dir_logits': agent_dir_logits,
        }


# =============================================================================
# EPISODE ENCODER (for episodic memory)
# =============================================================================

class EpisodeEncoder(nn.Module):
    """
    Encodes episode summary (return, length, solved, final obs embedding)
    into a fixed-size embedding for episodic memory.
    """
    embed_dim: int = 64

    @nn.compact
    def __call__(
        self,
        episode_return: chex.Array,
        episode_length: chex.Array,
        episode_solved: chex.Array,
        final_obs_embedding: Optional[chex.Array] = None,
    ) -> chex.Array:
        """
        Args:
            episode_return: shape (batch,)
            episode_length: shape (batch,)
            episode_solved: shape (batch,)
            final_obs_embedding: shape (batch, obs_embed_dim), optional

        Returns:
            Episode embedding, shape (batch, embed_dim)
        """
        # Normalize inputs
        ret_norm = episode_return.reshape(-1, 1)  # Assume already in [0, 1]
        length_norm = (episode_length.astype(jnp.float32) / 256.0).reshape(-1, 1)
        solved = episode_solved.astype(jnp.float32).reshape(-1, 1)

        features = [ret_norm, length_norm, solved]
        if final_obs_embedding is not None:
            features.append(final_obs_embedding)

        x = jnp.concatenate(features, axis=-1)

        x = nn.Dense(
            self.embed_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)

        x = nn.Dense(
            self.embed_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(x)

        return x


# =============================================================================
# CONTEXT UPDATE NETWORK (for context_vector)
# =============================================================================

class ContextUpdateNetwork(nn.Module):
    """
    Learns to update context vector based on episode summary.
    Alternative to simple EMA update.
    """
    context_dim: int = 64

    @nn.compact
    def __call__(
        self,
        current_context: chex.Array,
        episode_return: chex.Array,
        episode_length: chex.Array,
        episode_solved: chex.Array,
    ) -> chex.Array:
        """
        Args:
            current_context: shape (batch, context_dim)
            episode_return: shape (batch,)
            episode_length: shape (batch,)
            episode_solved: shape (batch,)

        Returns:
            Updated context, shape (batch, context_dim)
        """
        ret_norm = episode_return.reshape(-1, 1)
        length_norm = (episode_length.astype(jnp.float32) / 256.0).reshape(-1, 1)
        solved = episode_solved.astype(jnp.float32).reshape(-1, 1)

        episode_features = jnp.concatenate([ret_norm, length_norm, solved], axis=-1)

        # Combine with current context
        x = jnp.concatenate([current_context, episode_features], axis=-1)

        x = nn.Dense(
            self.context_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)

        # Gated update: learn how much to update
        gate = nn.Dense(
            self.context_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(x)
        gate = jax.nn.sigmoid(gate)

        new_context = nn.Dense(
            self.context_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(x)

        # Interpolate between old and new
        updated_context = gate * new_context + (1 - gate) * current_context

        return updated_context


# =============================================================================
# CURRICULUM ENCODER AND PREDICTION NETWORKS
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
        curriculum_embed = None
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
        if predict_curriculum and curriculum_embed is not None:
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
