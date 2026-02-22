"""
Persistent RNN memory mechanism.

Key difference: LSTM hidden state does NOT reset on episode boundary.
Hidden state persists across episodes and branches.

This tests whether emergent curriculum awareness develops when the agent
has memory capacity but no explicit curriculum information.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import chex


class PersistentRNNWrapper:
    """
    Wrapper for persistent (non-resetting) RNN behavior.

    Hidden state carries across episodes - never automatically reset.
    """

    def __init__(self, features: int = 256):
        self.features = features

    def initialize_carry(self, batch_dims: tuple) -> chex.ArrayTree:
        """Initialize hidden state with zeros."""
        return nn.OptimizedLSTMCell(features=self.features).initialize_carry(
            jax.random.PRNGKey(0), (*batch_dims, self.features)
        )

    def process_sequence(
        self,
        params,
        embedding: chex.Array,
        carry: chex.ArrayTree,
        dones: chex.Array,  # Ignored - no reset
    ) -> tuple:
        """
        Process sequence without reset on done.

        Args:
            params: Network parameters (not used directly here)
            embedding: Input embeddings, shape (time, batch, features) or (batch, features)
            carry: LSTM hidden state
            dones: Done flags (IGNORED - no reset)

        Returns:
            (new_carry, output_sequence)
        """
        lstm_cell = nn.OptimizedLSTMCell(features=self.features)

        if embedding.ndim == 3:  # (time, batch, features)
            def scan_fn(carry, emb):
                new_carry, out = lstm_cell.apply({"params": {}}, carry, emb)
                return new_carry, out

            final_carry, outputs = jax.lax.scan(scan_fn, carry, embedding)
            return final_carry, outputs
        else:  # (batch, features) - single step
            new_carry, output = lstm_cell.apply({"params": {}}, carry, embedding)
            return new_carry, output

    def reset_carry_manual(
        self,
        carry: chex.ArrayTree,
        reset_mask: chex.Array,
    ) -> chex.ArrayTree:
        """
        Manually reset hidden state for specific environments.

        This can be called explicitly (e.g., at start of new trial)
        but is NOT called automatically on episode done.

        Args:
            carry: Current hidden state
            reset_mask: Boolean mask, shape (batch,), True where reset needed

        Returns:
            Updated hidden state
        """
        h_c, h_h = carry
        zero_carry = self.initialize_carry((h_c.shape[0],))

        new_h_c = jnp.where(reset_mask[:, None], zero_carry[0], h_c)
        new_h_h = jnp.where(reset_mask[:, None], zero_carry[1], h_h)

        return (new_h_c, new_h_h)


def create_persistent_rnn_state(features: int = 256, batch_size: int = 1) -> dict:
    """
    Create state dictionary for persistent RNN.

    The key difference from reset_rnn is that this state persists
    across episodes and should be carried between branches.
    """
    wrapper = PersistentRNNWrapper(features=features)
    return {
        "hstate": wrapper.initialize_carry((batch_size,)),
        "wrapper": wrapper,
        # Track how many episodes this state has seen (for analysis)
        "episodes_seen": 0,
        "branches_seen": [],  # List of branch IDs
    }


class PersistentLSTMCell(nn.Module):
    """
    Custom LSTM cell that can be used with persistent state.

    This is a thin wrapper around OptimizedLSTMCell that makes
    the non-resetting behavior explicit in the architecture.
    """
    features: int = 256

    @nn.compact
    def __call__(
        self,
        carry: chex.ArrayTree,
        inputs: chex.Array,
        done: chex.Array = None,  # Explicitly ignored
    ) -> tuple:
        """
        Process single step without reset.

        Args:
            carry: LSTM hidden state
            inputs: Input tensor
            done: Done flag (IGNORED - included for API compatibility)

        Returns:
            (new_carry, output)
        """
        lstm = nn.OptimizedLSTMCell(features=self.features)
        new_carry, output = lstm(carry, inputs)
        return new_carry, output
