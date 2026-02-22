"""
PAIRED + Standard Probe Agent.

Baseline PAIRED agent with standard LSTM that resets on episode boundary.
Uses separate probe network (with stop_gradient) to analyze hidden state.
"""

import jax
import jax.numpy as jnp
import chex

from ..common.networks import ActorCritic
from .paired_base import PAIREDBaseAgent


class PAIREDAccelProbeAgent(PAIREDBaseAgent):
    """
    PAIRED agent with standard ActorCritic (reset on done).

    This is the baseline PAIRED agent - hidden state resets at episode
    boundary, matching the standard ACCEL probe setup.
    """

    def get_actor_critic_class(self) -> type:
        """Use standard ActorCritic with reset on done."""
        return ActorCritic

    def initialize_hidden_state(self, batch_size: int) -> chex.ArrayTree:
        """Initialize LSTM hidden state to zeros."""
        return ActorCritic.initialize_carry((batch_size,))
