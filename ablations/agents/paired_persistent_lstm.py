"""
PAIRED + Persistent LSTM Agent.

PAIRED agent with persistent hidden state that carries across episodes.
Tests whether persistent memory enables emergent curriculum awareness
in the PAIRED training paradigm.
"""

import jax
import jax.numpy as jnp
import chex

from ..common.networks import ActorCriticPersistent
from .paired_base import PAIREDBaseAgent, PAIREDTrainState


class PAIREDPersistentLSTMAgent(PAIREDBaseAgent):
    """
    PAIRED agent with persistent LSTM hidden state.

    Unlike standard PAIRED, the hidden state is NOT reset at episode
    boundaries. This allows the agent to maintain cross-episode memory,
    potentially enabling emergent curriculum awareness.

    The protagonist's final hidden state is stored in memory_state after
    each train_step and restored as init_hstate for the next step.
    """

    def get_actor_critic_class(self) -> type:
        """Use ActorCriticPersistent (no reset on done)."""
        return ActorCriticPersistent

    def initialize_hidden_state(self, batch_size: int) -> chex.ArrayTree:
        """Initialize LSTM hidden state to zeros."""
        return ActorCriticPersistent.initialize_carry((batch_size,))

    def create_train_state(self, rng: chex.PRNGKey) -> PAIREDTrainState:
        """Create train state with persistent hidden state initialized in memory_state."""
        train_state = super().create_train_state(rng)

        # Store initial hidden state in memory_state for persistence across train_steps
        init_hstate = self.initialize_hidden_state(self.config["num_train_envs"])
        return PAIREDTrainState(
            update_count=train_state.update_count,
            pro_train_state=train_state.pro_train_state,
            ant_train_state=train_state.ant_train_state,
            adv_train_state=train_state.adv_train_state,
            memory_state=init_hstate,
            agent_tracking=train_state.agent_tracking,
            probe_params=train_state.probe_params,
            probe_opt_state=train_state.probe_opt_state,
            probe_tracking=train_state.probe_tracking,
            hstate_samples=train_state.hstate_samples,
            hstate_sample_branches=train_state.hstate_sample_branches,
            hstate_sample_ptr=train_state.hstate_sample_ptr,
            pro_returns_history=train_state.pro_returns_history,
            ant_returns_history=train_state.ant_returns_history,
            regret_history=train_state.regret_history,
            training_steps_history=train_state.training_steps_history,
            novelty_history=train_state.novelty_history,
            learnability_history=train_state.learnability_history,
            level_history_wall_maps=train_state.level_history_wall_maps,
            history_ptr=train_state.history_ptr,
            history_total=train_state.history_total,
        )

    def _get_student_init_hstate(self, train_state: PAIREDTrainState) -> chex.ArrayTree:
        """Restore persistent hidden state from memory_state."""
        if train_state.memory_state is not None:
            return train_state.memory_state
        return super()._get_student_init_hstate(train_state)

    def _update_memory_after_rollouts(
        self,
        train_state: PAIREDTrainState,
        pro_extras: dict,
        ant_extras: dict,
    ):
        """Store protagonist's final hidden state for persistence across train_steps."""
        return pro_extras["final_hstate"]