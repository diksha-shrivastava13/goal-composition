"""
Persistent LSTM Agent - Emergent Memory Test.

Key difference: LSTM hidden state does NOT reset on episode boundary.
Hidden state persists across episodes and branches.

This tests the core hypothesis: "Given memory capacity (but no explicit
curriculum info), does the agent spontaneously develop curriculum awareness?"

AGENT-CENTRIC DESIGN:
- Persistent hidden state is part of the agent (memory)
- Probe is EXTERNAL and OPTIONAL (for interpretability only)
- Agent training works with or without probe
"""

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import chex

from ..common.types import (
    PersistentMemoryTrainState,
    UpdateState,
    create_probe_tracking_state,
    create_pareto_history_state,
    create_agent_tracking_state,
    DEFAULT_HSTATE_DIM,
)
from ..common.networks import ActorCriticPersistent, CurriculumProbe
from ..common.utils import flatten_hstate
from ..memory.persistent_rnn import PersistentRNNWrapper
from .base import BaseAgent


class PersistentLSTMAgent(BaseAgent):
    """
    ACCEL agent with persistent (non-resetting) LSTM.

    Memory: Persists across episodes and branches
    Probe: EXTERNAL (optional, for interpretability only)

    Key difference from accel_probe: hidden state carries information
    from previous episodes, allowing potential emergent curriculum awareness.

    AGENT-CENTRIC DESIGN:
    - get_hidden_state_for_rollout() returns persistent state
    - update_hidden_state_after_rollout() stores state for next rollout
    - Probe is NOT part of agent training
    - Agent works with or without probe
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.memory_wrapper = PersistentRNNWrapper(features=256)

    def get_actor_critic_class(self) -> type:
        return ActorCriticPersistent

    def initialize_hidden_state(self, batch_size: int) -> chex.ArrayTree:
        return self.memory_wrapper.initialize_carry((batch_size,))

    def get_hidden_state_for_rollout(
        self,
        train_state: PersistentMemoryTrainState,
        branch: int,
    ) -> chex.ArrayTree:
        """
        Get hidden state for rollout - PERSISTENT across branches.

        Unlike accel_probe which returns fresh zeros, this returns
        the persistent hidden state from the train state.
        """
        if train_state.persistent_hstate is not None:
            return train_state.persistent_hstate
        return self.initialize_hidden_state(self.config["num_train_envs"])

    def update_hidden_state_after_rollout(
        self,
        train_state: PersistentMemoryTrainState,
        final_hstate: chex.ArrayTree,
        branch: int,
    ) -> PersistentMemoryTrainState:
        """
        Store final hidden state for next rollout.

        This is the key difference - state persists.
        """
        return train_state.replace(persistent_hstate=final_hstate)

    def create_train_state(self, rng: chex.PRNGKey) -> PersistentMemoryTrainState:
        """Create train state with persistent memory."""
        config = self.config
        rng, rng_net, rng_probe = jax.random.split(rng, 3)

        # Initialize network (using persistent variant)
        dummy_level = self.sample_random_level(rng)
        obs, _ = self.env.reset_to_level(rng, dummy_level, self.env_params)
        obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(
                jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...],
                256, axis=0
            ),
            obs,
        )

        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
        network = ActorCriticPersistent(self.env.action_space(self.env_params).n)
        network_params = network.init(
            rng_net, init_x, self.initialize_hidden_state(config["num_train_envs"])
        )

        def linear_schedule(count):
            frac = (
                1.0 - (count // (config["num_minibatches"] * config["epoch_ppo"]))
                / config["num_updates"]
            )
            return config["lr"] * frac

        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )

        pholder_level = self.sample_random_level(jax.random.PRNGKey(0))
        sampler = self.level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
        pholder_level_batch = jax.tree_util.tree_map(
            lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0),
            pholder_level,
        )

        # Initialize probe
        probe_params = None
        probe_opt_state = None
        probe_tracking = None
        pareto_history = None

        if config.get("use_probe", True):
            probe = CurriculumProbe(env_height=13, env_width=13, use_episode_context=True)
            dummy_hstate = jnp.zeros((1, DEFAULT_HSTATE_DIM))
            probe_params = probe.init(
                rng_probe, dummy_hstate,
                episode_return=jnp.zeros(1),
                episode_solved=jnp.zeros(1),
                episode_length=jnp.zeros(1, dtype=jnp.int32),
            )

            probe_tx = optax.adam(learning_rate=config.get("probe_lr", 1e-3))
            probe_opt_state = probe_tx.init(probe_params)

            probe_tracking = create_probe_tracking_state(
                buffer_size=config.get("probe_tracking_buffer_size", 500),
                hstate_dim=DEFAULT_HSTATE_DIM,
            )
            pareto_history = create_pareto_history_state()

        # Initialize persistent hidden state
        persistent_hstate = self.initialize_hidden_state(config["num_train_envs"])

        # Initialize DR continuous rollout state
        rng, rng_dr = jax.random.split(rng)
        dr_hstate, dr_obs, dr_env_state = self.initialize_dr_state(rng_dr)

        return PersistentMemoryTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=UpdateState.DR,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            mutation_last_level_batch=pholder_level_batch,
            agent_tracking=create_agent_tracking_state(
                buffer_size=config.get("agent_tracking_buffer_size", 1000)
            ),
            probe_params=probe_params,
            probe_opt_state=probe_opt_state,
            current_hstate=self.initialize_hidden_state(config["num_train_envs"]),
            probe_tracking=probe_tracking,
            pareto_history=pareto_history,
            persistent_hstate=persistent_hstate,
            hstate_samples=jnp.zeros((500, DEFAULT_HSTATE_DIM)) if config.get("use_probe", True) else None,
            hstate_sample_branches=jnp.zeros(500, dtype=jnp.int32) if config.get("use_probe", True) else None,
            last_hstate=dr_hstate,
            last_obs=dr_obs,
            last_env_state=dr_env_state,
        )

    # NOTE: update_probe() is inherited from BaseAgent.
    # Persistent memory is handled via get_hidden_state_for_rollout() and
    # update_hidden_state_after_rollout() which are called by base class.
    # The probe is EXTERNAL and OPTIONAL.
    # Memory (persistent LSTM state) and probe are now decoupled.
