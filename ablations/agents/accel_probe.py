"""
ACCEL Probe Agent - No Memory Baseline.

Standard PPO with LSTM that resets on episode boundary.
Probe attached to test what's implicitly encoded in weights.

This is the baseline for testing: "What curriculum structure is
encoded in weights alone, without explicit memory?"

AGENT-CENTRIC DESIGN:
- This is the baseline: no explicit memory beyond LSTM
- Probe is EXTERNAL and OPTIONAL (for interpretability only)
- Agent training works with or without probe
"""

import jax
import jax.numpy as jnp
import optax
import chex

from ..common.types import (
    ProbeTrainState,
    UpdateState,
    create_probe_tracking_state,
    create_pareto_history_state,
    create_agent_tracking_state,
    DEFAULT_HSTATE_DIM,
)
from ..common.networks import ActorCritic, CurriculumProbe
from .base import BaseAgent


class AccelProbeAgent(BaseAgent):
    """
    ACCEL agent with probe for curriculum awareness testing.

    Memory: Reset per episode (standard ResetRNN)
    Probe: EXTERNAL (optional, for interpretability only)

    AGENT-CENTRIC DESIGN:
    - This is the baseline: LSTM resets per episode
    - No explicit curriculum memory
    - Probe is NOT part of agent training
    - Agent works with or without probe
    """

    def get_actor_critic_class(self) -> type:
        return ActorCritic

    def initialize_hidden_state(self, batch_size: int) -> chex.ArrayTree:
        return ActorCritic.initialize_carry((batch_size,))

    def create_train_state(self, rng: chex.PRNGKey) -> ProbeTrainState:
        """Create train state with probe."""
        config = self.config
        rng, rng_net, rng_probe = jax.random.split(rng, 3)

        # Initialize network
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
        network = ActorCritic(self.env.action_space(self.env_params).n)
        network_params = network.init(
            rng_net, init_x, self.initialize_hidden_state(config["num_train_envs"])
        )

        # Optimizer with schedule
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

        # Initialize level sampler
        pholder_level = self.sample_random_level(jax.random.PRNGKey(0))
        sampler = self.level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
        pholder_level_batch = jax.tree_util.tree_map(
            lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0),
            pholder_level,
        )

        # Initialize probe if enabled
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

        # Initialize DR continuous rollout state
        rng, rng_dr = jax.random.split(rng)
        dr_hstate, dr_obs, dr_env_state = self.initialize_dr_state(rng_dr)

        return ProbeTrainState.create(
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
            hstate_samples=jnp.zeros((500, DEFAULT_HSTATE_DIM)) if config.get("use_probe", True) else None,
            hstate_sample_branches=jnp.zeros(500, dtype=jnp.int32) if config.get("use_probe", True) else None,
            last_hstate=dr_hstate,
            last_obs=dr_obs,
            last_env_state=dr_env_state,
        )

    # NOTE: update_probe() is inherited from BaseAgent.
    # This is the baseline agent - no explicit memory beyond LSTM.
    # The probe is EXTERNAL and OPTIONAL.
