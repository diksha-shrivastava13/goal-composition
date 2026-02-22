"""
Episodic Memory Agent - Discrete Buffer Test.

Maintains a discrete buffer of recent episode summaries.
At episode start, retrieves relevant memories via attention.

This tests: "Is discrete episode retrieval sufficient for
curriculum awareness?"

AGENT-CENTRIC DESIGN:
- Episodic buffer updates are part of the agent (memory)
- Probe is EXTERNAL and OPTIONAL (for interpretability only)
- Agent training works with or without probe
"""

import jax
import jax.numpy as jnp
import optax
import chex

from ..common.types import (
    EpisodicMemoryTrainState,
    UpdateState,
    create_probe_tracking_state,
    create_pareto_history_state,
    create_agent_tracking_state,
    DEFAULT_HSTATE_DIM,
)
from ..common.networks import ActorCriticWithContext, CurriculumProbe
from ..common.utils import flatten_hstate
from ..common.training import (
    sample_trajectories_rnn_with_context,
    update_actor_critic_rnn_with_context,
)
from ..memory.episodic_buffer import (
    EpisodicBufferWrapper,
    create_episodic_memory_state,
    EpisodeEncoder,
)
from .base import BaseAgent


class EpisodicMemoryAgent(BaseAgent):
    """
    ACCEL agent with episodic memory buffer.

    Memory: Discrete buffer of episode summaries + attention retrieval
    Probe: EXTERNAL (optional, for interpretability only)

    At each episode:
    1. Retrieve relevant context from buffer via attention
    2. Use context as input to policy
    3. After episode, encode and store episode summary

    AGENT-CENTRIC DESIGN:
    - on_episode_complete() handles memory updates (episodic buffer)
    - Probe is NOT part of agent training
    - Agent works with or without probe
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.memory_wrapper = EpisodicBufferWrapper(
            buffer_size=config.get("memory_buffer_size", 64),
            embed_dim=config.get("context_dim", 64),
            top_k=config.get("memory_top_k", 8),
        )

    def get_actor_critic_class(self) -> type:
        return ActorCriticWithContext

    def initialize_hidden_state(self, batch_size: int) -> chex.ArrayTree:
        return ActorCriticWithContext.initialize_carry((batch_size,))

    def on_episode_complete(
        self,
        train_state: EpisodicMemoryTrainState,
        episode_return: chex.Array,
        episode_length: chex.Array,
        episode_solved: chex.Array,
        final_hstate: chex.ArrayTree,
    ) -> EpisodicMemoryTrainState:
        """
        Add completed episode to memory buffer.
        """
        if train_state.episodic_memory is None:
            return train_state

        # Encode episode
        hstate_flat = flatten_hstate(final_hstate)

        # Simple embedding: concatenate episode stats with projected hidden state
        embed_dim = self.config.get("context_dim", 64)

        # Use first sample from batch for memory
        episode_features = jnp.array([
            episode_return.mean(),
            episode_length.mean() / 256.0,
            episode_solved.mean(),
        ])

        # Project hidden state to remaining dimensions
        hstate_proj = hstate_flat[0, :embed_dim - 3]
        episode_embedding = jnp.concatenate([episode_features, hstate_proj])

        # Add to buffer
        new_memory = self.memory_wrapper.add_episode(
            train_state.episodic_memory,
            episode_embedding,
            episode_return.mean(),
            episode_length.mean().astype(jnp.int32),
            episode_solved.mean() > 0.5,
        )

        return train_state.replace(episodic_memory=new_memory)

    def get_context_from_memory(
        self,
        train_state: EpisodicMemoryTrainState,
        query_hstate: chex.ArrayTree,
    ) -> chex.Array:
        """
        Retrieve context from episodic memory.
        """
        if train_state.episodic_memory is None:
            return jnp.zeros((self.config["num_train_envs"], self.config.get("context_dim", 64)))

        # Use current hidden state as query
        hstate_flat = flatten_hstate(query_hstate)
        embed_dim = self.config.get("context_dim", 64)

        # Create query from hidden state (use first sample)
        query = hstate_flat[0, :embed_dim]

        # Retrieve context
        context = self.memory_wrapper.retrieve_context(
            train_state.episodic_memory,
            query,
            aggregation="attention",
        )

        # Broadcast to batch
        return jnp.broadcast_to(context[None, :], (self.config["num_train_envs"], embed_dim))

    def create_train_state(self, rng: chex.PRNGKey) -> EpisodicMemoryTrainState:
        """Create train state with episodic memory."""
        config = self.config
        rng, rng_net, rng_probe = jax.random.split(rng, 3)

        # Initialize network with context
        dummy_level = self.sample_random_level(rng)
        obs, _ = self.env.reset_to_level(rng, dummy_level, self.env_params)
        obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(
                jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...],
                256, axis=0
            ),
            obs,
        )

        context_dim = config.get("context_dim", 64)
        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
        # Initialize WITH context so LSTM params expect (obs_dim + context_dim) input.
        # The network defaults to zero-context when context=None is passed (e.g., eval).
        # During training, real context is threaded via _do_sample_trajectories hook.
        dummy_context = jnp.zeros((config["num_train_envs"], context_dim))
        network = ActorCriticWithContext(
            self.env.action_space(self.env_params).n,
            context_dim=context_dim,
        )
        network_params = network.init(
            rng_net, init_x, self.initialize_hidden_state(config["num_train_envs"]),
            context=dummy_context,
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

        # Initialize episodic memory
        episodic_memory = create_episodic_memory_state(
            buffer_size=config.get("memory_buffer_size", 64),
            embed_dim=context_dim,
        )

        # Initialize DR continuous rollout state
        rng, rng_dr = jax.random.split(rng)
        dr_hstate, dr_obs, dr_env_state = self.initialize_dr_state(rng_dr)

        return EpisodicMemoryTrainState.create(
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
            episodic_memory=episodic_memory,
            hstate_samples=jnp.zeros((500, DEFAULT_HSTATE_DIM)) if config.get("use_probe", True) else None,
            hstate_sample_branches=jnp.zeros(500, dtype=jnp.int32) if config.get("use_probe", True) else None,
            last_hstate=dr_hstate,
            last_obs=dr_obs,
            last_env_state=dr_env_state,
        )

    def _do_sample_trajectories(self, rng, train_state, init_hstate, init_obs, init_env_state):
        """Override to pass episodic memory context to the network during rollout."""
        context = self.get_context_from_memory(train_state, init_hstate)
        return sample_trajectories_rnn_with_context(
            rng, self.env, self.env_params, train_state,
            init_hstate, init_obs, init_env_state,
            self.config["num_train_envs"], self.config["num_steps"],
            context=context,
        )

    def _do_update_actor_critic(self, rng, train_state, init_hstate, batch, update_grad=True):
        """Override to pass episodic memory context to the network during PPO update."""
        config = self.config
        context = self.get_context_from_memory(train_state, init_hstate)
        return update_actor_critic_rnn_with_context(
            rng, train_state, init_hstate, batch,
            config["num_train_envs"], config["num_steps"],
            config["num_minibatches"], config["epoch_ppo"],
            config["clip_eps"], config["entropy_coeff"], config["critic_coeff"],
            update_grad=update_grad,
            context=context,
        )

    # NOTE: update_probe() is inherited from BaseAgent.
    # Episodic memory updates happen in on_episode_complete() which is called
    # by the base class training loop. The probe is EXTERNAL and OPTIONAL.
    # Memory (episodic buffer) and probe are now decoupled.
