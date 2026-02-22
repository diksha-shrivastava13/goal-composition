"""
Reset RNN memory mechanism.

Standard behavior: LSTM hidden state resets to zeros on episode boundary (done=True).
This is the baseline memory mechanism used by accel_probe.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import chex

from jaxued.linen import ResetRNN


class ResetRNNWrapper:
    """
    Wrapper for standard ResetRNN behavior.

    Hidden state resets to zeros when done=True.
    """

    def __init__(self, features: int = 256):
        self.features = features

    def create_cell(self) -> nn.Module:
        """Create the RNN module."""
        return ResetRNN(nn.OptimizedLSTMCell(features=self.features))

    def initialize_carry(self, batch_dims: tuple) -> chex.ArrayTree:
        """Initialize hidden state with zeros."""
        return nn.OptimizedLSTMCell(features=self.features).initialize_carry(
            jax.random.PRNGKey(0), (*batch_dims, self.features)
        )

    def reset_carry(self, carry: chex.ArrayTree, done: chex.Array) -> chex.ArrayTree:
        """
        Reset hidden state where done=True.

        This is handled internally by ResetRNN, but exposed here for API consistency.
        """
        h_c, h_h = carry
        reset_carry = self.initialize_carry((h_c.shape[0],))

        # Reset where done=True
        new_h_c = jnp.where(done[:, None], reset_carry[0], h_c)
        new_h_h = jnp.where(done[:, None], reset_carry[1], h_h)

        return (new_h_c, new_h_h)


def create_reset_rnn_state(features: int = 256, batch_size: int = 1) -> dict:
    """
    Create state dictionary for reset RNN.

    For reset RNN, state is just the current hidden state.
    """
    wrapper = ResetRNNWrapper(features=features)
    return {
        "hstate": wrapper.initialize_carry((batch_size,)),
        "wrapper": wrapper,
    }
