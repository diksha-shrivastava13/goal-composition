"""
PAIRED + Episodic Memory Agent.

PAIRED agent with discrete episodic memory buffer that stores episode
summaries for retrieval. Tests whether episodic memory can support
curriculum awareness in PAIRED training.
"""

import jax
import jax.numpy as jnp
import chex
from typing import Optional

from ..common.networks import ActorCriticWithContext
from ..common.types import EpisodicMemoryState, create_episodic_memory_state
from ..memory.episodic_buffer import EpisodicBufferWrapper
from .paired_base import PAIREDBaseAgent, PAIREDTrainState


def flatten_hstate(hstate):
    """Flatten LSTM hidden state for use as query/embedding."""
    h_c, h_h = hstate
    return jnp.concatenate([h_c, h_h], axis=-1)


class PAIREDEpisodicMemoryAgent(PAIREDBaseAgent):
    """
    PAIRED agent with episodic memory buffer.

    Maintains a discrete buffer of recent episode summaries (returns,
    lengths, solved flags, embeddings) for retrieval-based context.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.memory_wrapper = EpisodicBufferWrapper(
            buffer_size=config.get("memory_buffer_size", 64),
            embed_dim=config.get("context_dim", 64),
            top_k=config.get("memory_top_k", 8),
        )

    def get_actor_critic_class(self) -> type:
        """Use ActorCriticWithContext (context comes from episodic retrieval)."""
        return ActorCriticWithContext

    def initialize_hidden_state(self, batch_size: int) -> chex.ArrayTree:
        """Initialize LSTM hidden state to zeros."""
        return ActorCriticWithContext.initialize_carry((batch_size,))

    def create_train_state(self, rng: chex.PRNGKey):
        """Create train state with episodic memory initialized."""
        train_state = super().create_train_state(rng)

        # Initialize episodic memory
        buffer_size = self.config.get("memory_buffer_size", 64)
        embed_dim = self.config.get("context_dim", 64)
        episodic_memory = create_episodic_memory_state(buffer_size, embed_dim)

        return train_state.replace(memory_state=episodic_memory)

    def _get_student_context(self, train_state: PAIREDTrainState) -> Optional[chex.Array]:
        """Get context from episodic memory for protagonist/antagonist rollouts."""
        if train_state.memory_state is None:
            return None

        embed_dim = self.config.get("context_dim", 64)
        num_envs = self.config["num_train_envs"]

        # Use a zero query when no hstate is available yet
        query = jnp.zeros(embed_dim)
        context = self.memory_wrapper.retrieve_context(
            train_state.memory_state, query, aggregation="attention"
        )

        # Broadcast to batch
        return jnp.broadcast_to(context[None, :], (num_envs, embed_dim))

    def _update_memory_after_rollouts(
        self,
        train_state: PAIREDTrainState,
        pro_extras: dict,
        ant_extras: dict,
    ):
        """Update episodic buffer using protagonist episode statistics."""
        if train_state.memory_state is None:
            return train_state.memory_state

        # Compute episode stats from protagonist rollout
        pro_rewards = pro_extras["rewards"]
        pro_dones = pro_extras["dones"]
        episode_return = pro_rewards.sum(axis=0).mean()
        episode_length = pro_dones.sum(axis=0).mean()
        episode_solved = (pro_rewards.sum(axis=0) > 0).mean() > 0.5

        # Create episode embedding from protagonist final hidden state
        final_hstate = pro_extras["final_hstate"]
        hstate_flat = flatten_hstate(final_hstate)
        embed_dim = self.config.get("context_dim", 64)

        episode_features = jnp.array([
            episode_return,
            episode_length / 256.0,
            episode_solved.astype(jnp.float32),
        ])
        hstate_proj = hstate_flat[0, :embed_dim - 3]
        episode_embedding = jnp.concatenate([episode_features, hstate_proj])

        return self.memory_wrapper.add_episode(
            train_state.memory_state,
            episode_embedding,
            episode_return,
            episode_length.astype(jnp.int32),
            episode_solved,
        )