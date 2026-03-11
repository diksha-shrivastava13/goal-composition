"""
PAIRED Base Agent for curriculum awareness ablations.

Implements the PAIRED (Protagonist Antagonist Induced Regret Environment Design)
training paradigm with support for different memory architectures.

PAIRED uses three networks:
- Protagonist: Standard agent learning to solve levels
- Antagonist: Agent finding exploits/shortcuts in levels
- Adversary: Level generator maximizing regret (antagonist - protagonist)

AGENT-CENTRIC DESIGN:
- PAIRED has natural 1-to-1 correspondence (protagonist[i], antagonist[i], level[i])
- Per-instance metrics ARE meaningful for PAIRED
- Probe is OPTIONAL and EXTERNAL (for interpretability only)
- Agent-centric metrics focus on regret dynamics and value calibration
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import time
import os

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import chex
import wandb
import numpy as np
from flax.training.train_state import TrainState as FlaxTrainState

from jaxued.environments.underspecified_env import EnvParams
from jaxued.environments.maze import Level
from jaxued.environments.maze.env_editor import MazeEditor
from jaxued.utils import compute_max_mean_returns_epcount
from ..common.types import (
    PAIREDTrainState,
    ProbeTrackingState,
    AgentTrackingState,
    create_probe_tracking_state,
    create_agent_tracking_state,
    DEFAULT_ENV_HEIGHT,
    DEFAULT_ENV_WIDTH,
)
from ..common.networks import ActorCritic, AdversaryActorCritic, CurriculumProbe
from ..common.training import (
    compute_gae,
    sample_trajectories_rnn,
    sample_trajectories_rnn_with_context,
    update_actor_critic_rnn,
    evaluate_rnn,
)
from ..common.environment import (
    setup_environment,
    get_eval_levels,
)
from ..common.metrics import (
    compute_probe_loss_batch,
    compute_per_instance_calibration_batch,
    compute_distributional_calibration_metrics,
    compute_paired_regret_metrics,
    compute_level_novelty,
    compute_paired_openendedness,
    compute_level_feature_metrics,
    compute_learnability,
    compute_novelty,
    compute_openendedness_score,
    compute_distribution_divergence,
    compute_random_baselines,
    compute_greedy_matching,
    compute_matched_accuracy_metrics,
    compute_probe_correlation_with_performance,
    compute_hidden_state_statistics,
    # Agent-centric metrics
    compute_agent_novelty_from_tracking,
    compute_agent_learnability_from_tracking,
    compute_agent_openendedness_from_tracking,
    update_agent_tracking,
    # Displacement metrics
    compute_batch_displacement_metrics,
)
from ..common.visualization import (
    create_regret_dynamics_plot,
    create_paired_dashboard,
    create_adversary_level_analysis,
    create_paired_openendedness_plot,
    create_probe_loss_by_branch_plot,
    create_hidden_state_tsne_plot,
    create_wall_prediction_heatmap,
    create_position_prediction_heatmap,
    create_information_content_dashboard,
    create_novelty_learnability_plot,
    create_correlation_scatter_plot,
    build_curriculum_pred_log_dict,
)
from ..common.utils import setup_checkpointing, flatten_hstate


class PAIREDBaseAgent(ABC):
    """
    Base class for PAIRED training with any memory architecture.

    Key differences from BaseAgent:
    - Uses PAIREDTrainState with 3 networks (protagonist, antagonist, adversary)
    - Different train_step with regret-based reward computation
    - Uses MazeEditor env for adversary level generation
    - No level sampler (levels come from adversary)

    AGENT-CENTRIC DESIGN:
    - PAIRED has natural 1-to-1 correspondence: (protagonist[i], antagonist[i], level[i])
    - Per-instance metrics ARE meaningful (unlike DR/Replay in ACCEL)
    - Probe is OPTIONAL and EXTERNAL
    - Agent-centric metrics track:
      * Protagonist value calibration
      * Regret dynamics (antagonist - protagonist)
      * Level difficulty features
    """

    def __init__(self, config: dict, probe_runner=None):
        self.config = config
        self.probe_runner = probe_runner  # Optional, for interpretability only
        self.setup_environment()
        # No level sampler needed - adversary generates levels

    def setup_environment(self):
        """Setup maze env for students and MazeEditor for adversary."""
        config = self.config

        # Setup standard maze environment
        self.env, self.eval_env, self.sample_random_level, self.env_renderer, _ = \
            setup_environment(
                max_height=13,
                max_width=13,
                agent_view_size=config["agent_view_size"],
                normalize_obs=True,
                n_walls=config["n_walls"],
            )

        # env is already wrapped with AutoReplayWrapper by setup_environment()
        self.env_params = self.env.default_params

        # Setup MazeEditor for adversary
        self.adv_env = MazeEditor(
            self.env._env,  # Unwrap to get base Maze
            random_z_dimensions=config.get("adv_random_z_dimension", 16),
            zero_out_random_z=config.get("adv_zero_out_random_z", False),
        )
        self.adv_env_params = self.adv_env.default_params

    def sample_empty_level(self) -> Level:
        """Create empty level template for adversary to edit."""
        w, h = self.env._env.max_width, self.env._env.max_height
        return Level(
            wall_map=jnp.zeros((h, w), dtype=jnp.bool_),
            width=w,
            height=h,
            # These values are overwritten by adversary
            goal_pos=jnp.array([0, 0], dtype=jnp.uint32),
            agent_pos=jnp.array([1, 1], dtype=jnp.uint32),
            agent_dir=jnp.array(0, dtype=jnp.uint8),
        )

    @abstractmethod
    def get_actor_critic_class(self) -> type:
        """Return the ActorCritic class to use for protagonist/antagonist."""
        pass

    def _get_student_init_kwargs(self) -> dict:
        """Return extra kwargs for student network init. Override for networks needing extra args."""
        return {}

    @abstractmethod
    def initialize_hidden_state(self, batch_size: int) -> chex.ArrayTree:
        """Initialize hidden state for rollouts."""
        pass

    def _get_student_init_hstate(self, train_state: PAIREDTrainState) -> chex.ArrayTree:
        """Get initial hidden state for protagonist/antagonist rollouts.

        Default: fresh zeros each step. Override in PersistentLSTM to
        carry hidden state across train_steps.
        """
        actor_critic_cls = self.get_actor_critic_class()
        return actor_critic_cls.initialize_carry((self.config["num_train_envs"],))

    def _get_student_context(self, train_state: PAIREDTrainState) -> Optional[chex.Array]:
        """Get context for protagonist/antagonist rollouts. Override in memory agents."""
        return None

    def _update_memory_after_rollouts(
        self,
        train_state: PAIREDTrainState,
        pro_extras: dict,
        ant_extras: dict,
    ):
        """Update memory state after rollouts. Override in memory agents. Returns updated memory_state."""
        return train_state.memory_state

    def create_train_state(self, rng: chex.PRNGKey) -> PAIREDTrainState:
        """Create initial PAIRED train state with 3 networks."""
        config = self.config

        def create_network_state(
            rng: chex.PRNGKey,
            env,
            env_params,
            network_cls,
            prefix: str,
            network_kwargs: dict = None,
            init_kwargs: dict = None,
        ) -> FlaxTrainState:
            """Create train state for a single network."""
            if network_kwargs is None:
                network_kwargs = {}
            if init_kwargs is None:
                init_kwargs = {}

            def linear_schedule(count):
                frac = (
                    1.0 -
                    (count // (config[f"{prefix}num_minibatches"] * config[f"{prefix}epoch_ppo"]))
                    / config["num_updates"]
                )
                return config[f"{prefix}lr"] * frac

            # Get sample observation for initialization
            empty_level = self.sample_empty_level()
            obs, _ = env.reset_to_level(rng, empty_level, env_params)
            obs = jax.tree_util.tree_map(
                lambda x: jnp.repeat(
                    jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...],
                    256, axis=0
                ),
                obs,
            )
            init_x = (obs, jnp.zeros((256, config["num_train_envs"])))

            network = network_cls(env.action_space(env_params).n, **network_kwargs)
            rng, init_rng = jax.random.split(rng)
            network_params = network.init(
                init_rng, init_x,
                network_cls.initialize_carry((config["num_train_envs"],)),
                **init_kwargs,
            )

            tx = optax.chain(
                optax.clip_by_global_norm(config[f"{prefix}max_grad_norm"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )

            return FlaxTrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )

        rng_pro, rng_ant, rng_adv = jax.random.split(rng, 3)

        # Create protagonist and antagonist (same architecture)
        actor_critic_cls = self.get_actor_critic_class()
        student_init_kwargs = self._get_student_init_kwargs()
        pro_train_state = create_network_state(
            rng_pro, self.env, self.env_params, actor_critic_cls, "student_",
            init_kwargs=student_init_kwargs,
        )
        ant_train_state = create_network_state(
            rng_ant, self.env, self.env_params, actor_critic_cls, "student_",
            init_kwargs=student_init_kwargs,
        )

        # Create adversary
        adv_train_state = create_network_state(
            rng_adv, self.adv_env, self.adv_env_params, AdversaryActorCritic, "adv_",
            network_kwargs={"max_timesteps": config.get("adv_num_steps", 50)}
        )

        # Initialize probe if enabled
        probe_params = None
        probe_opt_state = None
        probe_tracking = None

        if config.get("use_probe", True):
            rng, probe_rng = jax.random.split(rng)
            probe = CurriculumProbe(env_height=13, env_width=13, use_episode_context=True)

            # Initialize probe with dummy input
            dummy_hstate = jnp.zeros((1, 512))  # Flattened LSTM state
            probe_params = probe.init(
                probe_rng, dummy_hstate,
                episode_return=jnp.zeros(1),
                episode_solved=jnp.zeros(1),
                episode_length=jnp.zeros(1),
            )
            probe_tx = optax.adam(learning_rate=config.get("probe_lr", 1e-3))
            probe_opt_state = probe_tx.init(probe_params)
            probe_tracking = create_probe_tracking_state(
                buffer_size=config.get("probe_tracking_buffer_size", 500),
                batch_size=config["num_train_envs"],
            )

        # Initialize PAIRED-specific history tracking
        history_size = config.get("paired_history_size", 500)

        # Initialize agent tracking (must not be None for scan pytree consistency)
        agent_tracking = create_agent_tracking_state(
            buffer_size=config.get("agent_tracking_buffer_size", 1000)
        )

        # Initialize last_adversary_level with batched empty levels
        # (must not be None for scan pytree consistency)
        empty_level = self.sample_empty_level()
        num_envs = config["num_train_envs"]
        init_last_adv_level = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(jnp.asarray(x), (num_envs, *jnp.asarray(x).shape)),
            empty_level,
        )

        return PAIREDTrainState(
            update_count=0,
            pro_train_state=pro_train_state,
            ant_train_state=ant_train_state,
            adv_train_state=adv_train_state,
            memory_state=None,
            agent_tracking=agent_tracking,
            probe_params=probe_params,
            probe_opt_state=probe_opt_state,
            probe_tracking=probe_tracking,
            # PAIRED history tracking
            pro_returns_history=jnp.zeros(history_size),
            ant_returns_history=jnp.zeros(history_size),
            regret_history=jnp.zeros(history_size),
            training_steps_history=jnp.zeros(history_size, dtype=jnp.int32),
            novelty_history=jnp.zeros(history_size),
            learnability_history=jnp.zeros(history_size),
            level_history_wall_maps=jnp.zeros((min(history_size, 100), 13, 13), dtype=jnp.bool_),
            history_ptr=0,
            history_total=0,
            last_adversary_level=init_last_adv_level,
        )

    def train(self):
        """Main PAIRED training loop."""
        config = self.config

        # Setup wandb
        tags = ["PAIRED", config["agent_type"]]
        run = wandb.init(
            config=config,
            project=config.get("project", "curriculum-awareness-ablation"),
            group=config["run_name"],
            tags=tags,
        )

        # Define metrics
        wandb.define_metric("num_updates")
        wandb.define_metric("num_env_steps")
        wandb.define_metric("solve_rate/*", step_metric="num_updates")
        wandb.define_metric("paired/*", step_metric="num_updates")
        wandb.define_metric("probe/*", step_metric="num_updates")
        wandb.define_metric("return/*", step_metric="num_updates")

        # Create train state
        rng = jax.random.PRNGKey(config["seed"])
        rng_init, rng_train = jax.random.split(rng)
        train_state = self.create_train_state(rng_init)

        # Setup checkpointing
        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager = setup_checkpointing(
                config, config["run_name"], config["seed"]
            )

        # Training loop
        runner_state = (rng_train, train_state)
        for eval_step in range(config["num_updates"] // config["eval_freq"]):
            start_time = time.time()

            # Train for eval_freq steps
            runner_state, metrics = self.train_and_eval_step(runner_state)

            # Log
            curr_time = time.time()
            metrics["time_delta"] = curr_time - start_time
            self.log_metrics(metrics, runner_state[1])

            # Checkpoint
            if config["checkpoint_save_interval"] > 0:
                checkpoint_manager.save(
                    eval_step,
                    {"train_state": runner_state[1]},
                )
                checkpoint_manager.wait_until_finished()

        return runner_state[1]

    def train_and_eval_step(
        self,
        runner_state: Tuple[chex.PRNGKey, PAIREDTrainState],
    ) -> Tuple[Tuple[chex.PRNGKey, PAIREDTrainState], dict]:
        """Run training for eval_freq steps, then evaluate."""
        config = self.config

        # Train
        (rng, train_state), metrics = jax.lax.scan(
            self.train_step, runner_state, None, config["eval_freq"]
        )

        # Eval (on protagonist)
        rng, rng_eval = jax.random.split(rng)
        eval_metrics = self.evaluate(rng_eval, train_state)
        metrics.update(eval_metrics)

        metrics["update_count"] = train_state.update_count
        metrics["max_updates"] = jnp.float32(self.config.get("num_updates", 50000))

        return (rng, train_state), metrics

    def train_step(
        self,
        carry: Tuple[chex.PRNGKey, PAIREDTrainState],
        _,
    ) -> Tuple[Tuple[chex.PRNGKey, PAIREDTrainState], dict]:
        """PAIRED training step."""
        rng, train_state = carry
        config = self.config

        pro_train_state = train_state.pro_train_state
        ant_train_state = train_state.ant_train_state
        adv_train_state = train_state.adv_train_state

        # 1. Adversary rollout (level generation)
        rng, rng_adv = jax.random.split(rng)
        empty_levels = jax.tree_util.tree_map(
            lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0),
            self.sample_empty_level()
        )

        adv_rollout, adv_extras = self._rollout(
            rng_adv, self.adv_env, self.adv_env_params,
            adv_train_state,
            AdversaryActorCritic.initialize_carry((config["num_train_envs"],)),
            empty_levels,
            config.get("adv_num_steps", 50),
            "adv_"
        )

        # Extract generated levels from adversary's final state
        levels = adv_extras["last_env_state"].level

        # Compute displacement from previous adversary-generated levels
        # On first step, last_adversary_level is None so compare with self (zero displacement)
        prev_levels = train_state.last_adversary_level
        displacement = compute_batch_displacement_metrics(levels, prev_levels)

        # 2. Protagonist rollout
        rng, rng_pro = jax.random.split(rng)
        student_context = self._get_student_context(train_state)
        student_init_hstate = self._get_student_init_hstate(train_state)
        pro_rollout, pro_extras = self._rollout(
            rng_pro, self.env, self.env_params,
            pro_train_state,
            student_init_hstate,
            levels,
            config.get("student_num_steps", 256),
            "student_",
            context=student_context,
        )
        pro_mean_returns, pro_max_returns, pro_eps = compute_max_mean_returns_epcount(
            pro_extras["dones"], pro_extras["rewards"]
        )

        # 3. Antagonist rollout
        rng, rng_ant = jax.random.split(rng)
        ant_rollout, ant_extras = self._rollout(
            rng_ant, self.env, self.env_params,
            ant_train_state,
            student_init_hstate,
            levels,
            config.get("student_num_steps", 256),
            "student_",
            context=student_context,
        )
        ant_mean_returns, ant_max_returns, ant_eps = compute_max_mean_returns_epcount(
            ant_extras["dones"], ant_extras["rewards"]
        )

        # 4. Compute regret reward for adversary
        est_regret = ant_max_returns - pro_mean_returns
        obs, actions, dones, log_probs, values, targets, advantages = adv_rollout
        rewards = jnp.zeros_like(values).at[-1].set(est_regret)
        advantages, targets = compute_gae(
            config.get("adv_gamma", 0.995),
            config.get("adv_gae_lambda", 0.98),
            adv_extras["last_value"], values, rewards, dones
        )
        adv_rollout = (obs, actions, dones.at[-1].set(True), log_probs, values, targets, advantages)

        # 5. Update all three networks
        rng, rng_pro_update, rng_ant_update, rng_adv_update = jax.random.split(rng, 4)

        (rng_pro_update, pro_train_state), pro_losses = update_actor_critic_rnn(
            rng_pro_update, pro_train_state,
            student_init_hstate,
            pro_rollout,
            config["num_train_envs"], config.get("student_num_steps", 256),
            config.get("student_num_minibatches", 1),
            config.get("student_epoch_ppo", 5),
            config.get("student_clip_eps", 0.2),
            config.get("student_entropy_coeff", 1e-3),
            config.get("student_critic_coeff", 0.5),
            update_grad=True,
        )

        (rng_ant_update, ant_train_state), ant_losses = update_actor_critic_rnn(
            rng_ant_update, ant_train_state,
            student_init_hstate,
            ant_rollout,
            config["num_train_envs"], config.get("student_num_steps", 256),
            config.get("student_num_minibatches", 1),
            config.get("student_epoch_ppo", 5),
            config.get("student_clip_eps", 0.2),
            config.get("student_entropy_coeff", 1e-3),
            config.get("student_critic_coeff", 0.5),
            update_grad=True,
        )

        (rng_adv_update, adv_train_state), adv_losses = update_actor_critic_rnn(
            rng_adv_update, adv_train_state,
            AdversaryActorCritic.initialize_carry((config["num_train_envs"],)),
            adv_rollout,
            config["num_train_envs"], config.get("adv_num_steps", 50),
            config.get("adv_num_minibatches", 1),
            config.get("adv_epoch_ppo", 5),
            config.get("adv_clip_eps", 0.2),
            config.get("adv_entropy_coeff", 1e-3),
            config.get("adv_critic_coeff", 0.5),
            update_grad=True,
        )

        # 6. Agent-centric tracking (NOT probe-based)
        agent_tracking = self._update_agent_tracking(
            train_state.agent_tracking,
            pro_extras["values"] if "values" in pro_extras else None,
            pro_extras["rewards"],
            pro_extras["dones"],
            levels,
            train_state.update_count,
            pro_mean_returns,
            ant_max_returns,
            est_regret,
        )

        # 7. Update probe if applicable (EXTERNAL interpretability tool)
        probe_params = train_state.probe_params
        probe_opt_state = train_state.probe_opt_state
        probe_tracking = train_state.probe_tracking

        probe_loss = jnp.float32(0.0)
        if probe_params is not None:
            rng, probe_rng = jax.random.split(rng)
            probe_params, probe_opt_state, probe_tracking, probe_loss = self._update_probe(
                probe_rng, probe_params, probe_opt_state, probe_tracking,
                pro_extras["final_hstate"], levels,
                pro_extras["rewards"], pro_extras["dones"],
                train_state.update_count,
            )

        metrics = {
            "pro_losses": jax.tree_util.tree_map(lambda x: x.mean(), pro_losses),
            "ant_losses": jax.tree_util.tree_map(lambda x: x.mean(), ant_losses),
            "adv_losses": jax.tree_util.tree_map(lambda x: x.mean(), adv_losses),
            "mean_num_blocks": levels.wall_map.sum() / config["num_train_envs"],
            "pro_mean_returns": pro_mean_returns,
            "ant_max_returns": ant_max_returns,
            "pro_max_returns": pro_max_returns,
            "ant_mean_returns": ant_mean_returns,
            "est_regret": est_regret,
            "pro_eps": pro_eps,
            "ant_eps": ant_eps,
            "displacement": displacement,
            "branch": jnp.int32(3),  # adversary branch
            "probe_loss": probe_loss,
        }

        # Update PAIRED-specific history tracking
        history_size = train_state.pro_returns_history.shape[0] if train_state.pro_returns_history is not None else 500
        history_ptr = train_state.history_ptr % history_size
        history_total = train_state.history_total

        new_pro_returns_history = train_state.pro_returns_history
        new_ant_returns_history = train_state.ant_returns_history
        new_regret_history = train_state.regret_history
        new_training_steps_history = train_state.training_steps_history
        new_novelty_history = train_state.novelty_history
        new_learnability_history = train_state.learnability_history
        new_level_history = train_state.level_history_wall_maps

        if new_pro_returns_history is not None:
            new_pro_returns_history = new_pro_returns_history.at[history_ptr].set(pro_mean_returns.mean())
            new_ant_returns_history = new_ant_returns_history.at[history_ptr].set(ant_max_returns.mean())
            new_regret_history = new_regret_history.at[history_ptr].set(est_regret.mean())
            new_training_steps_history = new_training_steps_history.at[history_ptr].set(train_state.update_count + 1)

            # Compute novelty from level history
            if new_level_history is not None:
                level_history_size = new_level_history.shape[0]
                level_ptr = history_ptr % level_history_size
                # Store first level's wall map for novelty computation
                new_level_history = new_level_history.at[level_ptr].set(levels.wall_map[0])
                novelty = compute_level_novelty(
                    levels.wall_map,
                    new_level_history,
                    jnp.minimum(history_total, level_history_size),
                )
                new_novelty_history = new_novelty_history.at[history_ptr].set(novelty)

            # Compute learnability from probe tracking
            if probe_tracking is not None:
                learnability, _ = compute_learnability(
                    probe_tracking.loss_history,
                    probe_tracking.training_step_history,
                    probe_tracking.total_samples,
                    probe_tracking.current_training_step,
                )
                # Only write if we have enough samples (traced condition)
                learnability = jnp.where(probe_tracking.total_samples > 10, learnability, 0.0)
                new_learnability_history = new_learnability_history.at[history_ptr].set(learnability)

            history_ptr = (history_ptr + 1) % history_size
            history_total = history_total + 1

        # Update memory state (context vector EMA, episodic buffer, etc.)
        updated_memory = self._update_memory_after_rollouts(
            train_state, pro_extras, ant_extras,
        )

        train_state = PAIREDTrainState(
            update_count=train_state.update_count + 1,
            pro_train_state=pro_train_state,
            ant_train_state=ant_train_state,
            adv_train_state=adv_train_state,
            memory_state=updated_memory,
            # Agent-centric tracking (NOT probe)
            agent_tracking=agent_tracking,
            # Probe is EXTERNAL interpretability tool
            probe_params=probe_params,
            probe_opt_state=probe_opt_state,
            probe_tracking=probe_tracking,
            hstate_samples=train_state.hstate_samples,
            hstate_sample_branches=train_state.hstate_sample_branches,
            hstate_sample_ptr=train_state.hstate_sample_ptr,
            # Updated history
            pro_returns_history=new_pro_returns_history,
            ant_returns_history=new_ant_returns_history,
            regret_history=new_regret_history,
            training_steps_history=new_training_steps_history,
            novelty_history=new_novelty_history,
            learnability_history=new_learnability_history,
            level_history_wall_maps=new_level_history,
            history_ptr=history_ptr,
            history_total=history_total,
            # Displacement tracking
            last_adversary_level=levels,
        )

        return (rng, train_state), metrics

    def _rollout(
        self,
        rng: chex.PRNGKey,
        env,
        env_params: EnvParams,
        train_state: FlaxTrainState,
        init_hstate: chex.ArrayTree,
        levels,
        num_steps: int,
        prefix: str,
        context: Optional[chex.Array] = None,
    ) -> Tuple[Tuple, dict]:
        """Run rollout and return trajectory + extras."""
        config = self.config

        rng, rng_reset = jax.random.split(rng)
        init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
            jax.random.split(rng_reset, config["num_train_envs"]),
            levels,
            env_params,
        )

        if context is not None:
            (
                (rng, final_hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn_with_context(
                rng, env, env_params, train_state,
                init_hstate, init_obs, init_env_state,
                config["num_train_envs"], num_steps, context,
            )
        else:
            (
                (rng, final_hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng, env, env_params, train_state,
                init_hstate, init_obs, init_env_state,
                config["num_train_envs"], num_steps,
                track_positions=hasattr(init_env_state, 'agent_pos'),
            )

        advantages, targets = compute_gae(
            config.get(f"{prefix}gamma", 0.995),
            config.get(f"{prefix}gae_lambda", 0.98),
            last_value, values, rewards, dones,
        )

        rollout = (obs, actions, dones, log_probs, values, targets, advantages)
        extras = {
            "dones": dones,
            "rewards": rewards,
            "values": values,  # For agent-centric tracking
            "last_value": last_value,
            "last_env_state": last_env_state,
            "final_hstate": final_hstate,
        }

        return rollout, extras

    def _update_agent_tracking(
        self,
        agent_tracking: Optional[AgentTrackingState],
        values: Optional[chex.Array],
        rewards: chex.Array,
        dones: chex.Array,
        levels,
        training_step: int,
        pro_mean_returns: chex.Array,
        ant_max_returns: chex.Array,
        est_regret: chex.Array,
    ) -> AgentTrackingState:
        """
        Update agent-centric tracking for PAIRED protagonist.

        This tracks AGENT metrics (not probe):
        - Protagonist value calibration
        - Regret dynamics
        - Level difficulty features

        Args:
            agent_tracking: Current tracking state (or None to initialize)
            values: Value predictions (if available)
            rewards: Rewards from rollout
            dones: Done flags
            levels: Levels from adversary
            training_step: Current training step
            pro_mean_returns: Protagonist mean returns
            ant_max_returns: Antagonist max returns
            est_regret: Estimated regret

        Returns:
            Updated agent tracking state
        """
        config = self.config

        # Initialize if needed
        if agent_tracking is None:
            agent_tracking = create_agent_tracking_state(
                buffer_size=config.get("agent_tracking_buffer_size", 1000)
            )

        # Compute episode statistics
        episode_returns = rewards.sum(axis=0)
        mean_return = episode_returns.mean()

        # Value calibration (if available)
        if values is not None:
            mean_value = values.mean()
            policy_entropy_proxy = values.std()
        else:
            mean_value = jnp.float32(0.0)
            policy_entropy_proxy = jnp.float32(0.0)

        # Level features
        wall_density = levels.wall_map.sum() / (13 * 13 * config["num_train_envs"])
        goal_distances = jnp.abs(levels.goal_pos[:, 0] - levels.agent_pos[:, 0]) + \
                        jnp.abs(levels.goal_pos[:, 1] - levels.agent_pos[:, 1])
        mean_goal_distance = goal_distances.mean()

        # Update tracking
        agent_tracking = update_agent_tracking(
            agent_tracking,
            policy_entropy=policy_entropy_proxy,
            value_prediction=mean_value,
            actual_return=mean_return,
            wall_density=wall_density,
            goal_distance=mean_goal_distance,
            branch=0,  # PAIRED doesn't have branches like ACCEL
            training_step=training_step,
            policy_kl=0.0,  # Would need previous policy
        )

        return agent_tracking

    def _update_probe(
        self,
        rng: chex.PRNGKey,
        probe_params: chex.ArrayTree,
        probe_opt_state: chex.ArrayTree,
        probe_tracking: ProbeTrackingState,
        hstate: chex.ArrayTree,
        levels: Level,
        rewards: chex.Array,
        dones: chex.Array,
        training_step: int,
        tier_targets: dict = None,
    ) -> Tuple[chex.ArrayTree, chex.ArrayTree, ProbeTrackingState]:
        """
        Update probe network on protagonist hidden state.

        NOTE: The probe is an EXTERNAL interpretability tool, NOT part of the agent.
        It can be disabled without affecting training.

        PAIRED has 1-to-1 correspondence: protagonist[i] and antagonist[i]
        evaluate on the same level[i], so per-instance metrics ARE meaningful.
        This is different from DR/Replay branches in ACCEL where no correspondence exists.
        """
        config = self.config

        # Flatten hidden state
        hstate_flat = flatten_hstate(hstate)
        hstate_flat = jax.lax.stop_gradient(hstate_flat)

        # Compute episode statistics
        episode_returns = rewards.sum(axis=0)
        episode_lengths = dones.sum(axis=0).astype(jnp.int32)
        episode_solved = (episode_returns > 0).astype(jnp.float32)

        # Create probe
        probe = CurriculumProbe(env_height=13, env_width=13, use_episode_context=True)

        # Forward pass to get predictions for metrics
        predictions = probe.apply(
            probe_params, hstate_flat,
            episode_return=episode_returns,
            episode_solved=episode_solved,
            episode_length=episode_lengths,
        )

        # Compute loss and gradients
        def probe_loss_fn(params):
            preds = probe.apply(
                params, hstate_flat,
                episode_return=episode_returns,
                episode_solved=episode_solved,
                episode_length=episode_lengths,
            )
            loss, loss_dict = compute_probe_loss_batch(
                preds, levels,
                tier_targets=tier_targets,
                is_paired=True,
                tier1_weight=config.get("tier1_weight", 1.0),
                tier2_weight=config.get("tier2_weight", 1.0),
                tier3_weight=config.get("tier3_weight", 1.0),
            )
            return loss, loss_dict

        (loss, loss_components), grads = jax.value_and_grad(
            probe_loss_fn, has_aux=True
        )(probe_params)

        # Update probe params
        probe_tx = optax.adam(learning_rate=config.get("probe_lr", 1e-3))
        updates, new_opt_state = probe_tx.update(grads, probe_opt_state, probe_params)
        new_probe_params = optax.apply_updates(probe_params, updates)

        # PAIRED has 1-to-1 correspondence, so compute per-instance metrics
        per_instance_metrics = compute_per_instance_calibration_batch(predictions, levels)
        dist_calibration = compute_distributional_calibration_metrics(predictions, levels)

        # Update tracking state
        if probe_tracking is not None:
            buffer_size = probe_tracking.loss_history.shape[0]
            ptr = probe_tracking.buffer_ptr % buffer_size

            new_loss_history = probe_tracking.loss_history.at[ptr].set(loss)
            new_step_history = probe_tracking.training_step_history.at[ptr].set(training_step)

            # Update agent returns for correlation tracking
            mean_return = episode_returns.mean()
            new_agent_returns = probe_tracking.agent_returns_history.at[ptr].set(mean_return)

            # Update probe accuracy history (using combined accuracy as proxy)
            new_probe_accuracy = probe_tracking.probe_accuracy_history.at[ptr].set(
                per_instance_metrics['combined_accuracy']
            )

            # Per-instance tracking (valid for PAIRED due to 1-to-1 correspondence)
            pi_ptr = probe_tracking.per_instance_ptr % buffer_size
            new_per_instance_wall = probe_tracking.per_instance_wall_accuracy_history.at[pi_ptr].set(
                per_instance_metrics['wall_accuracy']
            )
            new_per_instance_goal = probe_tracking.per_instance_goal_accuracy_history.at[pi_ptr].set(
                per_instance_metrics['goal_accuracy']
            )
            new_per_instance_agent_pos = probe_tracking.per_instance_agent_pos_accuracy_history.at[pi_ptr].set(
                per_instance_metrics['agent_pos_accuracy']
            )
            new_per_instance_agent_dir = probe_tracking.per_instance_agent_dir_accuracy_history.at[pi_ptr].set(
                per_instance_metrics['agent_dir_accuracy']
            )
            new_per_instance_combined = probe_tracking.per_instance_combined_accuracy_history.at[pi_ptr].set(
                per_instance_metrics['combined_accuracy']
            )
            new_per_instance_loss = probe_tracking.per_instance_loss_history.at[pi_ptr].set(loss)
            new_is_valid = probe_tracking.is_per_instance_valid.at[pi_ptr].set(True)

            # Tier loss tracking
            new_tier1_loss = probe_tracking.tier1_loss_history.at[ptr].set(
                loss_components.get('tier1_loss', jnp.float32(0.0)))
            new_tier2_loss = probe_tracking.tier2_loss_history.at[ptr].set(
                loss_components.get('tier2_loss', jnp.float32(0.0)))
            new_tier3_loss = probe_tracking.tier3_loss_history.at[ptr].set(
                loss_components.get('tier3_loss', jnp.float32(0.0)))
            new_t1_regret_loss = probe_tracking.tier1_regret_loss_history.at[ptr].set(
                loss_components.get('tier1/regret', jnp.float32(0.0)))
            new_t1_difficulty_loss = probe_tracking.tier1_difficulty_loss_history.at[ptr].set(
                loss_components.get('tier1/difficulty', jnp.float32(0.0)))
            new_t1_branch_loss = probe_tracking.tier1_branch_loss_history.at[ptr].set(
                loss_components.get('tier1/branch', jnp.float32(0.0)))
            new_t1_score_loss = probe_tracking.tier1_score_loss_history.at[ptr].set(
                loss_components.get('tier1/score', jnp.float32(0.0)))
            new_t2_return_loss = probe_tracking.tier2_return_loss_history.at[ptr].set(
                loss_components.get('tier2/return', jnp.float32(0.0)))
            new_t2_novelty_loss = probe_tracking.tier2_novelty_loss_history.at[ptr].set(
                loss_components.get('tier2/novelty', jnp.float32(0.0)))
            new_t2_unusualness_loss = probe_tracking.tier2_unusualness_loss_history.at[ptr].set(
                loss_components.get('tier2/unusualness', jnp.float32(0.0)))
            new_t3_drift_loss = probe_tracking.tier3_drift_loss_history.at[ptr].set(
                loss_components.get('tier3/drift', jnp.float32(0.0)))

            probe_tracking = probe_tracking.replace(
                loss_history=new_loss_history,
                training_step_history=new_step_history,
                buffer_ptr=(ptr + 1) % buffer_size,
                total_samples=probe_tracking.total_samples + 1,
                current_training_step=training_step + 1,
                agent_returns_history=new_agent_returns,
                probe_accuracy_history=new_probe_accuracy,
                # Per-instance (valid for PAIRED)
                per_instance_wall_accuracy_history=new_per_instance_wall,
                per_instance_goal_accuracy_history=new_per_instance_goal,
                per_instance_agent_pos_accuracy_history=new_per_instance_agent_pos,
                per_instance_agent_dir_accuracy_history=new_per_instance_agent_dir,
                per_instance_combined_accuracy_history=new_per_instance_combined,
                per_instance_loss_history=new_per_instance_loss,
                per_instance_ptr=pi_ptr + 1,
                per_instance_total=probe_tracking.per_instance_total + 1,
                is_per_instance_valid=new_is_valid,
                # Tier losses
                tier1_loss_history=new_tier1_loss,
                tier2_loss_history=new_tier2_loss,
                tier3_loss_history=new_tier3_loss,
                tier1_regret_loss_history=new_t1_regret_loss,
                tier1_difficulty_loss_history=new_t1_difficulty_loss,
                tier1_branch_loss_history=new_t1_branch_loss,
                tier1_score_loss_history=new_t1_score_loss,
                tier2_return_loss_history=new_t2_return_loss,
                tier2_novelty_loss_history=new_t2_novelty_loss,
                tier2_unusualness_loss_history=new_t2_unusualness_loss,
                tier3_drift_loss_history=new_t3_drift_loss,
            )

        return new_probe_params, new_opt_state, probe_tracking, loss

    def evaluate(
        self,
        rng: chex.PRNGKey,
        train_state: PAIREDTrainState,
    ) -> dict:
        """Evaluate protagonist on eval levels, capturing frames for animations."""
        config = self.config

        rng, rng_reset = jax.random.split(rng)
        levels = get_eval_levels(config["eval_levels"])
        num_levels = len(config["eval_levels"])

        init_obs, init_env_state = jax.vmap(
            self.eval_env.reset_to_level, (0, 0, None)
        )(jax.random.split(rng_reset, num_levels), levels, self.env_params)

        actor_critic_cls = self.get_actor_critic_class()
        states, rewards, episode_lengths = evaluate_rnn(
            rng, self.eval_env, self.env_params, train_state.pro_train_state,
            actor_critic_cls.initialize_carry((num_levels,)),
            init_obs, init_env_state,
            self.env_params.max_steps_in_episode,
        )

        mask = jnp.arange(self.env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)

        # Render frames for animations
        try:
            images = jax.vmap(jax.vmap(
                self.env_renderer.render_state, (0, None)
            ), (0, None))(states, self.env_params)
            frames = images.transpose(0, 1, 4, 2, 3)
        except Exception:
            frames = None

        result = {
            "eval_solve_rates": jnp.where(cum_rewards > 0, 1., 0.),
            "eval_returns": cum_rewards,
            "eval_ep_lengths": episode_lengths,
        }
        if frames is not None:
            result["eval_animation"] = (frames, episode_lengths)
        return result

    def _build_log_dict(self, metrics: dict, train_state: PAIREDTrainState) -> dict:
        """Build the wandb log dict for PAIRED agents. Subclasses override this to add metrics."""
        config = self.config

        update_count = metrics["update_count"]
        # Note: 2x because PAIRED runs both protagonist and antagonist rollouts per update.
        # This makes raw sps values not directly comparable with non-PAIRED agents.
        env_steps = 2 * update_count * config["num_train_envs"] * config.get("student_num_steps", 256)

        log_dict = {
            "num_updates": update_count,
            "num_env_steps": env_steps,
            "sps": env_steps / metrics.get("time_delta", 1),
        }

        # =====================================================================
        # CORE PAIRED METRICS (three-entity)
        # =====================================================================
        pro_mean_returns = metrics.get("pro_mean_returns", jnp.zeros(1))
        ant_max_returns = metrics.get("ant_max_returns", jnp.zeros(1))
        pro_max_returns = metrics.get("pro_max_returns", jnp.zeros(1))
        ant_mean_returns = metrics.get("ant_mean_returns", jnp.zeros(1))
        est_regret = metrics.get("est_regret", jnp.zeros(1))

        # Protagonist metrics
        log_dict["paired/protagonist/mean_returns"] = float(pro_mean_returns.mean())
        log_dict["paired/protagonist/max_returns"] = float(pro_max_returns.mean())
        log_dict["paired/protagonist/episodes"] = float(metrics.get("pro_eps", jnp.zeros(1)).mean())

        # Antagonist metrics
        log_dict["paired/antagonist/mean_returns"] = float(ant_mean_returns.mean())
        log_dict["paired/antagonist/max_returns"] = float(ant_max_returns.mean())
        log_dict["paired/antagonist/episodes"] = float(metrics.get("ant_eps", jnp.zeros(1)).mean())

        # Backward-compatible flat keys
        log_dict["paired/pro_mean_returns"] = log_dict["paired/protagonist/mean_returns"]
        log_dict["paired/ant_max_returns"] = log_dict["paired/antagonist/max_returns"]
        log_dict["paired/pro_max_returns"] = log_dict["paired/protagonist/max_returns"]
        log_dict["paired/ant_mean_returns"] = log_dict["paired/antagonist/mean_returns"]
        log_dict["paired/est_regret"] = float(est_regret.mean())
        log_dict["paired/mean_num_blocks"] = float(metrics.get("mean_num_blocks", jnp.zeros(1)).mean())

        # Comprehensive regret metrics
        regret_metrics = compute_paired_regret_metrics(pro_mean_returns, ant_max_returns)
        log_dict["paired/regret/variance"] = regret_metrics['regret_variance']
        log_dict["paired/regret/max"] = regret_metrics['max_regret']
        log_dict["paired/regret/min"] = regret_metrics['min_regret']
        log_dict["paired/regret/mean"] = regret_metrics['mean_regret']
        log_dict["paired/regret/solvability_gap"] = regret_metrics['solvability_gap']
        # Backward-compatible flat keys
        log_dict["paired/regret_variance"] = regret_metrics['regret_variance']
        log_dict["paired/max_regret"] = regret_metrics['max_regret']
        log_dict["paired/mean_regret"] = regret_metrics['mean_regret']

        # =====================================================================
        # ADVERSARY METRICS (level generator)
        # =====================================================================
        log_dict["paired/adversary/success_rate"] = regret_metrics['adversary_success_rate']
        log_dict["paired/adversary/wall_density"] = float(metrics.get("mean_num_blocks", jnp.zeros(1)).mean()) / 169.0

        # Adversary-generated level features
        if train_state.level_history_wall_maps is not None and train_state.history_total > 0:
            history_valid = min(train_state.history_total, train_state.level_history_wall_maps.shape[0])
            recent_walls = train_state.level_history_wall_maps[:history_valid]
            log_dict["paired/adversary/mean_wall_density"] = float(recent_walls.astype(jnp.float32).mean())
            log_dict["paired/adversary/wall_density_std"] = float(recent_walls.astype(jnp.float32).std())

        # =====================================================================
        # THREE-ENTITY DYNAMICS & COOPERATION METRICS
        # =====================================================================
        pro_ret = float(pro_mean_returns.mean())
        ant_ret = float(ant_max_returns.mean())
        ant_mean_ret = float(ant_mean_returns.mean())

        # Core gap metrics
        log_dict["paired/dynamics/pro_vs_ant_gap"] = ant_ret - pro_ret
        log_dict["paired/dynamics/ant_advantage"] = ant_ret - pro_ret

        # Antagonist-adversary cooperation: adversary generates levels that
        # the antagonist solves but protagonist doesn't. High regret with high
        # antagonist success indicates effective ant-adv cooperation.
        pro_solve = regret_metrics['pro_solve_rate']
        ant_solve = regret_metrics['ant_solve_rate']
        log_dict["paired/protagonist/solve_rate"] = pro_solve
        log_dict["paired/antagonist/solve_rate"] = ant_solve

        # Cooperation score: how much better antagonist does than protagonist
        # on adversary-generated levels. High value = strong ant-adv cooperation.
        cooperation_score = ant_solve - pro_solve
        log_dict["paired/cooperation/ant_adv_score"] = cooperation_score
        log_dict["paired/cooperation/exploitation_gap"] = cooperation_score

        # Misalignment: regret should come from genuine difficulty, not from
        # adversary-antagonist collusion. Track whether antagonist returns
        # are close to optimal or just better than protagonist.
        if ant_ret > 0:
            exploitation_ratio = pro_ret / max(ant_ret, 1e-6)
            log_dict["paired/cooperation/pro_ant_ratio"] = exploitation_ratio
        else:
            log_dict["paired/cooperation/pro_ant_ratio"] = 0.0

        # =====================================================================
        # EVAL METRICS
        # =====================================================================
        if "eval_solve_rates" in metrics:
            solve_rates = metrics["eval_solve_rates"]
            returns = metrics["eval_returns"]
            for i, name in enumerate(config["eval_levels"]):
                log_dict[f"solve_rate/{name}"] = float(solve_rates[i])
                log_dict[f"return/{name}"] = float(returns[i])
            log_dict["solve_rate/mean"] = float(solve_rates.mean())
            log_dict["return/mean"] = float(returns.mean())
        if "eval_ep_lengths" in metrics:
            log_dict["eval_ep_lengths/mean"] = float(metrics["eval_ep_lengths"].mean())

        # =====================================================================
        # EVAL ANIMATIONS
        # =====================================================================
        if "eval_animation" in metrics:
            try:
                frames, ep_lengths = metrics["eval_animation"]
                for i, level_name in enumerate(config["eval_levels"]):
                    level_frames = frames[:, i]
                    ep_len = int(ep_lengths[i])
                    level_frames = np.array(level_frames[:ep_len])
                    log_dict[f"animations/{level_name}"] = wandb.Video(
                        level_frames, fps=4, format="gif"
                    )
            except Exception:
                pass

        # =====================================================================
        # ADVERSARY LEVEL IMAGES (every eval call)
        # =====================================================================
        if update_count > 0 and train_state.level_history_wall_maps is not None and train_state.history_total > 0:
            try:
                latest_idx = (train_state.history_ptr - 1) % train_state.level_history_wall_maps.shape[0]
                latest_wall_map = train_state.level_history_wall_maps[latest_idx]
                log_dict["images/adversary_latest_level"] = wandb.Image(
                    np.array(latest_wall_map), caption="Latest Adversary Level"
                )
            except Exception:
                pass

        # =====================================================================
        # PROTAGONIST PROBE METRICS (full per-component, calibration, divergence)
        # =====================================================================
        if train_state.probe_tracking is not None:
            probe_tracking = train_state.probe_tracking
            buffer_size = probe_tracking.loss_history.shape[0]
            valid_samples = jnp.minimum(probe_tracking.total_samples, buffer_size)

            if valid_samples > 0:
                # Per-component distributional losses
                log_dict["probe/protagonist/dist_loss/wall"] = float(probe_tracking.wall_loss_history[:valid_samples].mean())
                log_dict["probe/protagonist/dist_loss/goal"] = float(probe_tracking.goal_loss_history[:valid_samples].mean())
                log_dict["probe/protagonist/dist_loss/agent_pos"] = float(probe_tracking.agent_pos_loss_history[:valid_samples].mean())
                log_dict["probe/protagonist/dist_loss/agent_dir"] = float(probe_tracking.agent_dir_loss_history[:valid_samples].mean())
                log_dict["probe/protagonist/dist_loss/total"] = float(probe_tracking.loss_history[:valid_samples].mean())
                # Also log as flat keys for backward compat
                log_dict["probe/dist_loss/total"] = log_dict["probe/protagonist/dist_loss/total"]
                log_dict["probe/total_loss"] = log_dict["probe/protagonist/dist_loss/total"]

                # Distributional calibration
                wall_cal_vals = probe_tracking.dist_calibration_wall_history[0, :valid_samples]
                goal_cal_vals = probe_tracking.dist_calibration_goal_history[0, :valid_samples]
                wall_acc_vals = probe_tracking.dist_accuracy_wall_history[0, :valid_samples]
                goal_mode_vals = probe_tracking.dist_accuracy_goal_mode_match_history[0, :valid_samples]
                agent_pos_mode_vals = probe_tracking.dist_accuracy_agent_pos_history[0, :valid_samples]
                agent_dir_mode_vals = probe_tracking.dist_accuracy_agent_dir_history[0, :valid_samples]
                combined_vals = probe_tracking.dist_accuracy_combined_history[0, :valid_samples]

                log_dict["probe/protagonist/dist_calibration/wall"] = float(wall_cal_vals.mean())
                log_dict["probe/protagonist/dist_calibration/goal"] = float(goal_cal_vals.mean())
                log_dict["probe/protagonist/dist_accuracy/wall"] = float(wall_acc_vals.mean())
                log_dict["probe/protagonist/dist_accuracy/goal_mode_match"] = float(goal_mode_vals.mean())
                log_dict["probe/protagonist/dist_accuracy/agent_pos_mode_match"] = float(agent_pos_mode_vals.mean())
                log_dict["probe/protagonist/dist_accuracy/agent_dir_mode_match"] = float(agent_dir_mode_vals.mean())
                log_dict["probe/protagonist/dist_accuracy/combined"] = float(combined_vals.mean())

                # Distribution divergence from protagonist probe
                try:
                    last_preds = {
                        'wall_logits': probe_tracking.last_predictions_wall_logits,
                        'goal_logits': probe_tracking.last_predictions_goal_logits,
                        'agent_pos_logits': probe_tracking.last_predictions_agent_pos_logits,
                        'agent_dir_logits': probe_tracking.last_predictions_agent_dir_logits,
                    }

                    class LevelBatch:
                        def __init__(self, wall_map, goal_pos, agent_pos, agent_dir):
                            self.wall_map = wall_map
                            self.goal_pos = goal_pos
                            self.agent_pos = agent_pos
                            self.agent_dir = agent_dir

                    last_levels = LevelBatch(
                        wall_map=probe_tracking.last_levels_wall_map,
                        goal_pos=probe_tracking.last_levels_goal_pos,
                        agent_pos=probe_tracking.last_levels_agent_pos,
                        agent_dir=probe_tracking.last_levels_agent_dir,
                    )
                    div_metrics = compute_distribution_divergence(
                        last_preds, last_levels,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["probe/protagonist/divergence/goal_kl"] = float(div_metrics['goal_kl'])
                    log_dict["probe/protagonist/divergence/goal_js"] = float(div_metrics['goal_js'])
                    log_dict["probe/protagonist/divergence/wall_density_error"] = float(div_metrics['wall_density_error'])
                except Exception:
                    pass

                # Info gain vs random baselines
                try:
                    random_baselines = compute_random_baselines(DEFAULT_ENV_HEIGHT, DEFAULT_ENV_WIDTH)
                    wall_acc = float(wall_acc_vals.mean())
                    goal_mode = float(goal_mode_vals.mean())
                    total_loss = float(probe_tracking.loss_history[:valid_samples].mean())
                    log_dict["probe/protagonist/info_gain/wall_vs_random"] = wall_acc - random_baselines['wall_accuracy']
                    log_dict["probe/protagonist/info_gain/goal_vs_random"] = goal_mode - random_baselines['goal_top1']
                    log_dict["probe/protagonist/info_gain/total_loss_improvement"] = random_baselines['total_loss'] - total_loss
                except Exception:
                    pass

            # Per-instance metrics (meaningful for PAIRED: natural 1-to-1 correspondence)
            pi_total = probe_tracking.per_instance_total
            if pi_total > 0:
                pi_valid = jnp.minimum(pi_total, buffer_size)
                log_dict["paired/per_instance/wall_accuracy"] = float(
                    probe_tracking.per_instance_wall_accuracy_history[:pi_valid].mean()
                )
                log_dict["paired/per_instance/goal_accuracy"] = float(
                    probe_tracking.per_instance_goal_accuracy_history[:pi_valid].mean()
                )
                log_dict["paired/per_instance/agent_pos_accuracy"] = float(
                    probe_tracking.per_instance_agent_pos_accuracy_history[:pi_valid].mean()
                )
                log_dict["paired/per_instance/agent_dir_accuracy"] = float(
                    probe_tracking.per_instance_agent_dir_accuracy_history[:pi_valid].mean()
                )
                log_dict["paired/per_instance/combined_accuracy"] = float(
                    probe_tracking.per_instance_combined_accuracy_history[:pi_valid].mean()
                )
                log_dict["paired/per_instance/loss"] = float(
                    probe_tracking.per_instance_loss_history[:pi_valid].mean()
                )

            # Novelty and learnability
            if valid_samples > 10:
                learnability, learn_details = compute_learnability(
                    probe_tracking.loss_history,
                    probe_tracking.training_step_history,
                    probe_tracking.total_samples,
                    probe_tracking.current_training_step,
                )
                novelty, novelty_details = compute_novelty(
                    probe_tracking.loss_history,
                    probe_tracking.training_step_history,
                    probe_tracking.total_samples,
                    probe_tracking.current_training_step,
                )
                oe_score, regime = compute_openendedness_score(novelty, learnability)

                log_dict["probe/protagonist/dynamics/novelty"] = novelty
                log_dict["probe/protagonist/dynamics/learnability"] = learnability
                log_dict["probe/protagonist/dynamics/openendedness_score"] = oe_score

            # Correlation
            correlation_stats = compute_probe_correlation_with_performance(
                probe_tracking.probe_accuracy_history,
                probe_tracking.agent_returns_history,
                probe_tracking.total_samples,
            )
            log_dict["probe/protagonist/correlation/probe_return"] = correlation_stats['correlation']

            # Samples
            log_dict["probe/protagonist/samples/total"] = int(probe_tracking.total_samples)

        # =====================================================================
        # AGENT-CENTRIC METRICS (protagonist)
        # =====================================================================
        if train_state.agent_tracking is not None:
            agent_tracking = train_state.agent_tracking

            if agent_tracking.total_samples > 10:
                agent_novelty, novelty_details = compute_agent_novelty_from_tracking(
                    agent_tracking, window_size=500
                )
                log_dict["agent/protagonist/novelty"] = agent_novelty

                agent_learnability, learn_details = compute_agent_learnability_from_tracking(
                    agent_tracking, window_size=500
                )
                log_dict["agent/protagonist/learnability"] = agent_learnability
                log_dict["agent/protagonist/value_calibration_error"] = learn_details.get('late_calibration_error', 0.0)

                agent_oe_score, regime, _ = compute_agent_openendedness_from_tracking(
                    agent_tracking, window_size=500
                )
                log_dict["agent/protagonist/openendedness_score"] = agent_oe_score
                log_dict["agent/protagonist/regime"] = regime
                # Backward compat
                log_dict["agent/novelty"] = agent_novelty
                log_dict["agent/learnability"] = agent_learnability
                log_dict["agent/openendedness_score"] = agent_oe_score

            log_dict["agent/protagonist/total_samples"] = int(agent_tracking.total_samples)

        # =====================================================================
        # PAIRED-SPECIFIC OPEN-ENDEDNESS (from history)
        # =====================================================================
        if train_state.history_total > 10:
            history_valid = min(train_state.history_total, train_state.novelty_history.shape[0])

            recent_novelty = float(train_state.novelty_history[train_state.history_ptr - 1]) if train_state.history_ptr > 0 else 0.0
            recent_learnability = float(train_state.learnability_history[train_state.history_ptr - 1]) if train_state.history_ptr > 0 else 0.0
            recent_regret = float(train_state.regret_history[train_state.history_ptr - 1]) if train_state.history_ptr > 0 else 0.0

            openendedness = compute_paired_openendedness(recent_novelty, recent_learnability, recent_regret)
            log_dict["paired/openendedness/novelty"] = openendedness['novelty']
            log_dict["paired/openendedness/learnability"] = openendedness['learnability']
            log_dict["paired/openendedness/interestingness"] = openendedness['interestingness']
            log_dict["paired/openendedness/score"] = openendedness['openendedness_score']

        # =====================================================================
        # DISPLACEMENT + ZONE METRICS (curriculum_pred/)
        # =====================================================================
        try:
            curriculum_log = build_curriculum_pred_log_dict(metrics, train_state)
            log_dict.update(curriculum_log)
        except Exception:
            pass

        # =====================================================================
        # VISUALIZATIONS (every eval call)
        # =====================================================================
        self._log_paired_visualizations(train_state, metrics, log_dict)

        return log_dict

    def log_metrics(self, metrics: dict, train_state: PAIREDTrainState):
        """Log all PAIRED metrics to wandb. Calls _build_log_dict then logs."""
        log_dict = self._build_log_dict(metrics, train_state)
        if self.config.get("use_wandb", True):
            wandb.log(log_dict)

    def _log_paired_visualizations(
        self,
        train_state: PAIREDTrainState,
        metrics: dict,
        log_dict: dict,
    ):
        """Log PAIRED-specific visualizations."""
        try:
            # Regret dynamics plot (protagonist vs antagonist returns over time)
            if train_state.history_total > 10:
                history_valid = min(train_state.history_total, train_state.pro_returns_history.shape[0])
                try:
                    regret_plot = create_regret_dynamics_plot(
                        np.array(train_state.pro_returns_history[:history_valid]),
                        np.array(train_state.ant_returns_history[:history_valid]),
                        np.array(train_state.training_steps_history[:history_valid]),
                    )
                    log_dict["paired/images/regret_dynamics"] = wandb.Image(regret_plot)
                except Exception:
                    pass

                # PAIRED open-endedness plot
                if train_state.novelty_history is not None:
                    try:
                        oe_plot = create_paired_openendedness_plot(
                            np.array(train_state.novelty_history[:history_valid]),
                            np.array(train_state.learnability_history[:history_valid]),
                            np.array(train_state.regret_history[:history_valid]),
                        )
                        log_dict["paired/images/openendedness"] = wandb.Image(oe_plot)
                    except Exception:
                        pass

            # Protagonist probe visualizations
            if train_state.probe_tracking is not None and train_state.probe_tracking.total_samples > 10:
                probe_tracking = train_state.probe_tracking

                # Wall prediction heatmap
                try:
                    last_preds = {
                        'wall_logits': probe_tracking.last_predictions_wall_logits,
                        'goal_logits': probe_tracking.last_predictions_goal_logits,
                        'agent_pos_logits': probe_tracking.last_predictions_agent_pos_logits,
                        'agent_dir_logits': probe_tracking.last_predictions_agent_dir_logits,
                    }

                    class LevelBatch:
                        def __init__(self, wall_map, goal_pos, agent_pos, agent_dir):
                            self.wall_map = wall_map
                            self.goal_pos = goal_pos
                            self.agent_pos = agent_pos
                            self.agent_dir = agent_dir

                    last_levels = LevelBatch(
                        wall_map=probe_tracking.last_levels_wall_map,
                        goal_pos=probe_tracking.last_levels_goal_pos,
                        agent_pos=probe_tracking.last_levels_agent_pos,
                        agent_dir=probe_tracking.last_levels_agent_dir,
                    )

                    wall_heatmap = create_wall_prediction_heatmap(
                        last_preds, last_levels,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["probe/protagonist/images/wall_heatmap"] = wandb.Image(
                        wall_heatmap, caption="Protagonist Wall Prediction"
                    )

                    pos_heatmap = create_position_prediction_heatmap(
                        last_preds, last_levels,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["probe/protagonist/images/position_heatmap"] = wandb.Image(
                        pos_heatmap, caption="Protagonist Position Prediction"
                    )
                except Exception:
                    pass

                # Information dashboard
                try:
                    probe_metrics_for_dashboard = {
                        'wall_dist_accuracy': log_dict.get("probe/protagonist/dist_accuracy/wall", 0.0),
                        'goal_dist_mode_match': log_dict.get("probe/protagonist/dist_accuracy/goal_mode_match", 0.0),
                        'agent_pos_dist_mode_match': log_dict.get("probe/protagonist/dist_accuracy/agent_pos_mode_match", 0.0),
                        'agent_dir_dist_mode_match': log_dict.get("probe/protagonist/dist_accuracy/agent_dir_mode_match", 0.0),
                        'wall_loss': log_dict.get("probe/protagonist/dist_loss/wall", 0.0),
                        'goal_loss': log_dict.get("probe/protagonist/dist_loss/goal", 0.0),
                        'agent_pos_loss': log_dict.get("probe/protagonist/dist_loss/agent_pos", 0.0),
                        'agent_dir_loss': log_dict.get("probe/protagonist/dist_loss/agent_dir", 0.0),
                    }
                    info_dashboard = create_information_content_dashboard(
                        probe_metrics_for_dashboard, probe_tracking,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["probe/protagonist/images/information_dashboard"] = wandb.Image(
                        info_dashboard, caption="Protagonist Information Content"
                    )
                except Exception:
                    pass

                # Novelty-learnability plot
                try:
                    learnability, _ = compute_learnability(
                        probe_tracking.loss_history,
                        probe_tracking.training_step_history,
                        probe_tracking.total_samples,
                        probe_tracking.current_training_step,
                    )
                    novelty, _ = compute_novelty(
                        probe_tracking.loss_history,
                        probe_tracking.training_step_history,
                        probe_tracking.total_samples,
                        probe_tracking.current_training_step,
                    )
                    oe_score, regime = compute_openendedness_score(novelty, learnability)
                    nl_plot = create_novelty_learnability_plot(novelty, learnability, oe_score, regime)
                    log_dict["probe/protagonist/images/novelty_learnability"] = wandb.Image(nl_plot)
                except Exception:
                    pass

                # Correlation scatter
                if probe_tracking.total_samples > 50:
                    try:
                        corr_plot = create_correlation_scatter_plot(
                            probe_tracking.probe_accuracy_history,
                            probe_tracking.agent_returns_history,
                            probe_tracking.total_samples,
                        )
                        log_dict["probe/protagonist/images/correlation"] = wandb.Image(corr_plot)
                    except Exception:
                        pass

            # Hidden state t-SNE (protagonist, expensive)
            eval_freq = self.config.get("eval_freq", 250)
            tsne_freq = self.config.get("tsne_freq", 10)
            update_count = metrics.get("update_count", 0)
            if update_count % (tsne_freq * eval_freq) == 0:
                if train_state.hstate_samples is not None and train_state.hstate_sample_ptr > 50:
                    try:
                        tsne_plot = create_hidden_state_tsne_plot(
                            train_state.hstate_samples,
                            train_state.hstate_sample_branches,
                            max_samples=min(int(train_state.hstate_sample_ptr), 500),
                        )
                        log_dict["paired/images/hidden_state_tsne"] = wandb.Image(tsne_plot)
                    except Exception:
                        pass

        except Exception as e:
            print(f"Warning: Failed to create PAIRED visualizations: {e}")
