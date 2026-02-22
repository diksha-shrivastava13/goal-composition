"""
Episodic buffer memory mechanism.

Maintains a discrete buffer of recent episode summaries.
At episode start, retrieves relevant memories via attention.
Retrieved context is concatenated with observation embedding.

This tests whether discrete episode retrieval is sufficient
for curriculum awareness.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from flax import struct

from ..common.types import EpisodicMemoryState, create_episodic_memory_state


class EpisodicBufferWrapper:
    """
    Wrapper for episodic buffer memory mechanism.

    Maintains a circular buffer of episode summaries and retrieves
    relevant memories at the start of each episode.
    """

    def __init__(
        self,
        buffer_size: int = 64,
        embed_dim: int = 64,
        top_k: int = 8,
    ):
        self.buffer_size = buffer_size
        self.embed_dim = embed_dim
        self.top_k = top_k

    def initialize_memory(self) -> EpisodicMemoryState:
        """Initialize empty episodic memory buffer."""
        return create_episodic_memory_state(self.buffer_size, self.embed_dim)

    def add_episode(
        self,
        memory_state: EpisodicMemoryState,
        episode_embedding: chex.Array,
        episode_return: float,
        episode_length: int,
        episode_solved: bool,
    ) -> EpisodicMemoryState:
        """
        Add episode summary to buffer.

        Args:
            memory_state: Current memory state
            episode_embedding: Embedding of the episode, shape (embed_dim,)
            episode_return: Return from episode
            episode_length: Length of episode
            episode_solved: Whether episode was solved

        Returns:
            Updated memory state
        """
        ptr = memory_state.pointer % self.buffer_size

        new_embeddings = memory_state.episode_embeddings.at[ptr].set(episode_embedding)
        new_returns = memory_state.episode_returns.at[ptr].set(episode_return)
        new_lengths = memory_state.episode_lengths.at[ptr].set(episode_length)
        new_solved = memory_state.episode_solved.at[ptr].set(episode_solved)

        new_size = jnp.minimum(memory_state.size + 1, self.buffer_size)

        return memory_state.replace(
            episode_embeddings=new_embeddings,
            episode_returns=new_returns,
            episode_lengths=new_lengths,
            episode_solved=new_solved,
            pointer=(ptr + 1) % self.buffer_size,
            size=new_size,
        )

    def retrieve_context(
        self,
        memory_state: EpisodicMemoryState,
        query_embedding: chex.Array,
        aggregation: str = "mean",
    ) -> chex.Array:
        """
        Retrieve context from memory using query embedding.

        Args:
            memory_state: Current memory state
            query_embedding: Query for retrieval, shape (embed_dim,)
            aggregation: How to combine retrieved memories ("mean", "attention")

        Returns:
            Retrieved context, shape (embed_dim,)
        """
        # Compute similarities (works even when size==0 due to masking)
        similarities = memory_state.episode_embeddings @ query_embedding

        # Mask invalid entries
        valid_mask = jnp.arange(self.buffer_size) < memory_state.size
        similarities = jnp.where(valid_mask, similarities, -jnp.inf)

        # Get top-k indices (always use fixed top_k for JIT compatibility)
        top_k_indices = jnp.argsort(-similarities)[:self.top_k]

        # Retrieve memories
        retrieved = memory_state.episode_embeddings[top_k_indices]

        if aggregation == "mean":
            # Simple average
            context = retrieved.mean(axis=0)
        elif aggregation == "attention":
            # Attention-weighted average
            weights = jax.nn.softmax(similarities[top_k_indices])
            context = (retrieved * weights[:, None]).sum(axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Return zeros when buffer is empty
        return jnp.where(memory_state.size > 0, context, jnp.zeros(self.embed_dim))

    def retrieve_context_with_metadata(
        self,
        memory_state: EpisodicMemoryState,
        query_embedding: chex.Array,
    ) -> dict:
        """
        Retrieve context with additional metadata about retrieved episodes.

        Returns context plus statistics about retrieved memories.
        """
        similarities = memory_state.episode_embeddings @ query_embedding
        valid_mask = jnp.arange(self.buffer_size) < memory_state.size
        similarities = jnp.where(valid_mask, similarities, -jnp.inf)

        top_k_indices = jnp.argsort(-similarities)[:self.top_k]

        retrieved = memory_state.episode_embeddings[top_k_indices]
        weights = jax.nn.softmax(similarities[top_k_indices])
        context = (retrieved * weights[:, None]).sum(axis=0)

        # Metadata about retrieved episodes
        retrieved_returns = memory_state.episode_returns[top_k_indices]
        retrieved_lengths = memory_state.episode_lengths[top_k_indices]
        retrieved_solved = memory_state.episode_solved[top_k_indices]

        is_empty = memory_state.size == 0
        zeros = jnp.zeros(self.embed_dim)

        return {
            "context": jnp.where(is_empty, zeros, context),
            "mean_return": jnp.where(is_empty, 0.0, (retrieved_returns * weights).sum()),
            "mean_length": jnp.where(is_empty, 0.0, (retrieved_lengths.astype(jnp.float32) * weights).sum()),
            "solve_rate": jnp.where(is_empty, 0.0, (retrieved_solved.astype(jnp.float32) * weights).sum()),
            "num_retrieved": jnp.where(is_empty, 0, jnp.minimum(self.top_k, memory_state.size)),
        }


def create_episodic_memory_state(
    buffer_size: int = 64,
    embed_dim: int = 64,
) -> EpisodicMemoryState:
    """Initialize empty episodic memory buffer."""
    return EpisodicMemoryState(
        episode_embeddings=jnp.zeros((buffer_size, embed_dim)),
        episode_returns=jnp.zeros(buffer_size),
        episode_lengths=jnp.zeros(buffer_size, dtype=jnp.int32),
        episode_solved=jnp.zeros(buffer_size, dtype=jnp.bool_),
        pointer=0,
        size=0,
    )


class EpisodeEncoder(nn.Module):
    """
    Encodes episode summary into embedding for storage in buffer.
    """
    embed_dim: int = 64

    @nn.compact
    def __call__(
        self,
        episode_return: chex.Array,
        episode_length: chex.Array,
        episode_solved: chex.Array,
        final_hstate: chex.ArrayTree = None,
    ) -> chex.Array:
        """
        Encode episode summary.

        Args:
            episode_return: shape (batch,) or scalar
            episode_length: shape (batch,) or scalar
            episode_solved: shape (batch,) or scalar
            final_hstate: Final LSTM state (optional)

        Returns:
            Episode embedding, shape (batch, embed_dim) or (embed_dim,)
        """
        # Handle scalar vs batched input
        is_scalar = episode_return.ndim == 0

        if is_scalar:
            ret = episode_return.reshape(1, 1)
            length = (episode_length.astype(jnp.float32) / 256.0).reshape(1, 1)
            solved = episode_solved.astype(jnp.float32).reshape(1, 1)
        else:
            ret = episode_return.reshape(-1, 1)
            length = (episode_length.astype(jnp.float32) / 256.0).reshape(-1, 1)
            solved = episode_solved.astype(jnp.float32).reshape(-1, 1)

        features = [ret, length, solved]

        if final_hstate is not None:
            h_c, h_h = final_hstate
            if h_c.ndim == 1:
                h_c = h_c[None, :]
                h_h = h_h[None, :]
            hstate_flat = jnp.concatenate([h_c, h_h], axis=-1)
            # Project to smaller dimension
            hstate_proj = nn.Dense(32)(hstate_flat)
            features.append(hstate_proj)

        x = jnp.concatenate(features, axis=-1)

        x = nn.Dense(self.embed_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.embed_dim)(x)

        if is_scalar:
            x = x[0]  # Remove batch dimension

        return x


class QueryEncoder(nn.Module):
    """
    Encodes current observation into query for memory retrieval.
    """
    embed_dim: int = 64

    @nn.compact
    def __call__(
        self,
        obs_embedding: chex.Array,
        current_hstate: chex.ArrayTree = None,
    ) -> chex.Array:
        """
        Encode query for memory retrieval.

        Args:
            obs_embedding: Observation embedding from CNN
            current_hstate: Current LSTM state (optional)

        Returns:
            Query embedding, shape (embed_dim,)
        """
        features = [obs_embedding]

        if current_hstate is not None:
            h_c, h_h = current_hstate
            if h_c.ndim == 2:
                hstate_flat = jnp.concatenate([h_c[0], h_h[0]], axis=-1)
            else:
                hstate_flat = jnp.concatenate([h_c, h_h], axis=-1)
            features.append(hstate_flat)

        x = jnp.concatenate(features, axis=-1)

        x = nn.Dense(self.embed_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.embed_dim)(x)

        return x
