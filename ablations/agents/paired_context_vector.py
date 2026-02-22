"""
PAIRED + Context Vector Agent.

PAIRED agent with compressed context vector that summarizes episode history
via EMA updates. Tests whether compressed context can support curriculum
awareness in PAIRED training.
"""

import jax
import jax.numpy as jnp
import chex
from typing import Optional

from ..common.networks import ActorCriticWithContext
from ..common.types import ContextState, create_context_state
from ..memory.context_vector import ContextVectorWrapper
from .paired_base import PAIREDBaseAgent, PAIREDTrainState


class PAIREDContextVectorAgent(PAIREDBaseAgent):
    """
    PAIRED agent with context vector memory.

    Uses a compressed context vector (updated via EMA after each episode)
    that is concatenated with observations before the LSTM.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.context_wrapper = ContextVectorWrapper(
            context_dim=config.get("context_dim", 64),
            decay=config.get("context_decay", 0.9),
        )

    def get_actor_critic_class(self) -> type:
        """Use ActorCriticWithContext."""
        return ActorCriticWithContext

    def initialize_hidden_state(self, batch_size: int) -> chex.ArrayTree:
        """Initialize LSTM hidden state to zeros."""
        return ActorCriticWithContext.initialize_carry((batch_size,))

    def create_train_state(self, rng: chex.PRNGKey):
        """Create train state with context state initialized."""
        train_state = super().create_train_state(rng)

        # Initialize context state
        context_dim = self.config.get("context_dim", 64)
        context_state = create_context_state(context_dim)

        return train_state.replace(memory_state=context_state)

    def _get_student_context(self, train_state: PAIREDTrainState) -> Optional[chex.Array]:
        """Get context vector for protagonist/antagonist rollouts."""
        if train_state.memory_state is None:
            return None
        return self.context_wrapper.get_context_for_policy(
            train_state.memory_state, self.config["num_train_envs"]
        )

    def _update_memory_after_rollouts(
        self,
        train_state: PAIREDTrainState,
        pro_extras: dict,
        ant_extras: dict,
    ):
        """Update context vector via EMA using protagonist episode statistics."""
        if train_state.memory_state is None:
            return train_state.memory_state

        # Use protagonist rollout stats to update context
        pro_rewards = pro_extras["rewards"]
        pro_dones = pro_extras["dones"]
        episode_return = pro_rewards.sum(axis=0).mean()
        episode_length = pro_dones.sum(axis=0).mean()
        episode_solved = (pro_rewards.sum(axis=0) > 0).mean() > 0.5

        return self.context_wrapper.update_context_ema(
            train_state.memory_state,
            episode_return,
            episode_length,
            episode_solved,
            pro_extras.get("final_hstate"),
        )