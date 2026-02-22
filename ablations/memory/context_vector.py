"""
Context vector memory mechanism.

A compressed context vector that persists across episodes and is
updated via EMA (or learned update rule) after each episode.

This tests whether a compressed summary of episode history
is sufficient for curriculum awareness.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from flax import struct

from ..common.types import ContextState, create_context_state


class ContextVectorWrapper:
    """
    Wrapper for context vector memory mechanism.

    Maintains a fixed-size context vector that is:
    1. Concatenated with observations as input to the policy
    2. Updated after each episode based on episode summary
    """

    def __init__(
        self,
        context_dim: int = 64,
        decay: float = 0.9,
        use_learned_update: bool = False,
    ):
        self.context_dim = context_dim
        self.decay = decay
        self.use_learned_update = use_learned_update

    def initialize_context(self, batch_size: int = 1) -> ContextState:
        """Initialize context state with zeros."""
        return create_context_state(self.context_dim)

    def update_context_ema(
        self,
        context_state: ContextState,
        episode_return: chex.Array,
        episode_length: chex.Array,
        episode_solved: chex.Array,
        final_hstate: chex.ArrayTree = None,
    ) -> ContextState:
        """
        Update context vector using EMA.

        Args:
            context_state: Current context state
            episode_return: Return from completed episode
            episode_length: Length of completed episode
            episode_solved: Whether episode was solved
            final_hstate: Final LSTM hidden state (optional, for richer summary)

        Returns:
            Updated context state
        """
        # Create episode embedding
        episode_features = jnp.array([
            episode_return,
            episode_length / 256.0,  # Normalize
            episode_solved.astype(jnp.float32),
        ])

        # If we have final hidden state, include projection of it
        if final_hstate is not None:
            h_c, h_h = final_hstate
            hstate_flat = jnp.concatenate([h_c.flatten(), h_h.flatten()])
            # Simple projection to context_dim
            hstate_proj = hstate_flat[:self.context_dim - 3]
            episode_embedding = jnp.concatenate([episode_features, hstate_proj])
        else:
            # Pad to context_dim
            episode_embedding = jnp.zeros(self.context_dim)
            episode_embedding = episode_embedding.at[:3].set(episode_features)

        # EMA update
        new_context = (
            self.decay * context_state.context_vector +
            (1 - self.decay) * episode_embedding
        )

        return context_state.replace(
            context_vector=new_context,
            episode_count=context_state.episode_count + 1,
        )

    def update_context_learned(
        self,
        context_state: ContextState,
        update_network,
        update_params,
        episode_return: chex.Array,
        episode_length: chex.Array,
        episode_solved: chex.Array,
    ) -> ContextState:
        """
        Update context vector using learned update network.

        Args:
            context_state: Current context state
            update_network: Flax module for context update
            update_params: Parameters for update network
            episode_return: Return from completed episode
            episode_length: Length of completed episode
            episode_solved: Whether episode was solved

        Returns:
            Updated context state
        """
        new_context = update_network.apply(
            update_params,
            context_state.context_vector[None, ...],
            episode_return[None],
            episode_length[None],
            episode_solved[None],
        )[0]

        return context_state.replace(
            context_vector=new_context,
            episode_count=context_state.episode_count + 1,
        )

    def get_context_for_policy(
        self,
        context_state: ContextState,
        batch_size: int = 1,
    ) -> chex.Array:
        """
        Get context vector formatted for policy input.

        Args:
            context_state: Current context state
            batch_size: Batch size for broadcasting

        Returns:
            Context vector, shape (batch_size, context_dim)
        """
        context = context_state.context_vector
        if context.ndim == 1:
            context = jnp.broadcast_to(context[None, :], (batch_size, self.context_dim))
        return context


def create_context_state(context_dim: int = 64) -> ContextState:
    """Initialize context state with zeros."""
    return ContextState(
        context_vector=jnp.zeros(context_dim),
        episode_count=0,
    )


class ContextUpdateNetwork(nn.Module):
    """
    Learned context update network.

    Takes current context + episode summary and outputs new context.
    Uses gated update to learn how much to change.
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
        Update context based on episode summary.

        Args:
            current_context: shape (batch, context_dim)
            episode_return: shape (batch,)
            episode_length: shape (batch,)
            episode_solved: shape (batch,)

        Returns:
            Updated context, shape (batch, context_dim)
        """
        # Normalize inputs
        ret_norm = episode_return.reshape(-1, 1)
        length_norm = (episode_length.astype(jnp.float32) / 256.0).reshape(-1, 1)
        solved = episode_solved.astype(jnp.float32).reshape(-1, 1)

        episode_features = jnp.concatenate([ret_norm, length_norm, solved], axis=-1)
        x = jnp.concatenate([current_context, episode_features], axis=-1)

        x = nn.Dense(self.context_dim)(x)
        x = nn.relu(x)

        # Gated update
        gate = nn.sigmoid(nn.Dense(self.context_dim)(x))
        new_content = nn.Dense(self.context_dim)(x)

        updated_context = gate * new_content + (1 - gate) * current_context

        return updated_context
