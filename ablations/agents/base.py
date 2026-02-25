"""
Base agent class for curriculum awareness ablations.

Provides shared training loop and infrastructure for all agent variants.

AGENT-CENTRIC DESIGN:
- The probe is an interpretability tool, NOT part of the agent
- Probe updates are OPTIONAL and don't affect agent training
- Agent-centric metrics (policy entropy, value calibration) are primary
- Probe-based metrics are secondary, for interpretability only
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable
import time
import os
import json

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import chex
import wandb
import numpy as np

from jaxued.environments.underspecified_env import EnvParams, UnderspecifiedEnv
from jaxued.environments.maze import Level
from jaxued.level_sampler import LevelSampler
from jaxued.utils import compute_max_returns

from ..common.types import (
    BaseTrainState,
    ProbeTrainState,
    UpdateState,
    TrainingMethod,
    ProbeTrackingState,
    ParetoHistoryState,
    AgentTrackingState,
    VisualizationData,
    create_probe_tracking_state,
    create_pareto_history_state,
    create_agent_tracking_state,
    create_visualization_data,
    DEFAULT_HSTATE_DIM,
    DEFAULT_ENV_HEIGHT,
    DEFAULT_ENV_WIDTH,
)
from ..common.networks import ActorCritic, CurriculumProbe
from ..common.training import (
    compute_gae,
    sample_trajectories_rnn,
    update_actor_critic_rnn,
    evaluate_rnn,
)
from ..common.environment import (
    setup_environment,
    setup_level_sampler,
    compute_score,
    get_eval_levels,
)
from ..common.metrics import (
    compute_probe_loss_batch,
    compute_distributional_calibration_metrics,
    compute_per_instance_calibration_batch,
    compute_random_baselines,
    compute_learnability,
    compute_novelty,
    compute_openendedness_score,
    compute_probe_correlation_with_performance,
    compute_hidden_state_statistics,
    compute_information_gain_metrics,
    compute_greedy_matching,
    compute_matched_accuracy_metrics,
    compute_distribution_divergence,
    # Agent-centric metrics
    compute_agent_novelty_from_tracking,
    compute_agent_learnability_from_tracking,
    compute_agent_openendedness_from_tracking,
    update_agent_tracking,
    # Displacement metrics
    compute_batch_displacement_metrics,
    # Zone decomposition
    compute_visited_mask_from_positions,
)
from ..common.visualization import (
    create_probe_loss_by_branch_plot,
    create_hidden_state_tsne_plot,
    create_pareto_trajectory_plot,
    create_novelty_learnability_plot,
    create_curriculum_trajectory_plot,
    create_per_cell_correlation_heatmap,
    create_gradient_flow_visualization,
    create_prediction_confidence_histogram,
    create_information_content_dashboard,
    create_replay_to_mutate_heatmap,
    create_correlation_scatter_plot,
    create_batch_wall_prediction_summary,
    create_batch_position_prediction_summary,
    create_wall_prediction_heatmap,
    create_position_prediction_heatmap,
    create_matched_pairs_visualization,
    build_curriculum_pred_log_dict,
)
from ..common.utils import (
    setup_checkpointing,
    train_state_to_log_dict,
    flatten_hstate,
)
from ..common.probe_runner import ProbeRunner, ProbeAnalysisState


class BaseAgent(ABC):
    """
    Abstract base class for all agent variants.

    Implements the shared ACCEL training loop with hooks for
    agent-specific behavior. Also supports DR (Domain Randomization) mode.

    AGENT-CENTRIC DESIGN:
    - Probe is EXTERNAL: not part of training, optional for interpretability
    - Agent tracking: policy entropy, value calibration, returns
    - Matched pairs: greedy matching for DR/Replay, 1-to-1 for R->M
    """

    def __init__(
        self,
        config: dict,
        probe_runner: Optional[ProbeRunner] = None,
        training_experiments: Optional[list] = None,
    ):
        """
        Initialize base agent.

        Args:
            config: Configuration dictionary
            probe_runner: Optional ProbeRunner for interpretability analysis.
                         If None and use_probe=True, creates one internally.
            training_experiments: Optional list of TrainingTimeExperiment instances
                         that will have their training_hook() called during training.
        """
        self.config = config
        self.training_method = TrainingMethod(config.get("training_method", "accel"))
        self.probe_runner = probe_runner
        self.probe_analysis_state = None
        self.training_experiments = training_experiments or []

        self.setup_environment()
        # Always set up level sampler (needed for train state initialization).
        # In DR mode, the sampler exists but isn't used during training.
        self.setup_level_sampler()

    def setup_environment(self):
        """Setup environment and level generation."""
        self.env, self.eval_env, self.sample_random_level, self.env_renderer, self.mutate_level = \
            setup_environment(
                max_height=13,
                max_width=13,
                agent_view_size=self.config["agent_view_size"],
                normalize_obs=True,
                n_walls=self.config["n_walls"],
            )
        self.env_params = self.env.default_params

    def setup_level_sampler(self):
        """Setup PLR level sampler."""
        self.level_sampler = setup_level_sampler(
            capacity=self.config["level_buffer_capacity"],
            replay_prob=self.config["replay_prob"],
            staleness_coeff=self.config["staleness_coeff"],
            minimum_fill_ratio=self.config["minimum_fill_ratio"],
            prioritization=self.config["prioritization"],
            temperature=self.config["temperature"],
            top_k=self.config["top_k"],
            duplicate_check=self.config["buffer_duplicate_check"],
        )

    def initialize_dr_state(self, rng: chex.PRNGKey) -> tuple:
        """
        Initialize continuous rollout state for DR mode.

        Returns:
            (init_hstate, init_obs, init_env_state) for the first DR step.
        """
        config = self.config
        rng, rng_levels, rng_reset = jax.random.split(rng, 3)
        new_levels = jax.vmap(self.sample_random_level)(
            jax.random.split(rng_levels, config["num_train_envs"])
        )
        init_obs, init_env_state = jax.vmap(self.env.reset_to_level, in_axes=(0, 0, None))(
            jax.random.split(rng_reset, config["num_train_envs"]),
            new_levels,
            self.env_params,
        )
        init_hstate = self.initialize_hidden_state(config["num_train_envs"])
        return init_hstate, init_obs, init_env_state

    @abstractmethod
    def create_train_state(self, rng: chex.PRNGKey) -> BaseTrainState:
        """Create initial train state. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_actor_critic_class(self) -> type:
        """Return the ActorCritic class to use."""
        pass

    @abstractmethod
    def initialize_hidden_state(self, batch_size: int) -> chex.ArrayTree:
        """Initialize hidden state for rollouts."""
        pass

    def get_hidden_state_for_rollout(
        self,
        train_state: BaseTrainState,
        branch: int,
    ) -> chex.ArrayTree:
        """
        Get hidden state for starting a rollout.

        Default: fresh zeros (reset per branch).
        Override for persistent memory variants.
        """
        return self.initialize_hidden_state(self.config["num_train_envs"])

    def update_hidden_state_after_rollout(
        self,
        train_state: BaseTrainState,
        final_hstate: chex.ArrayTree,
        branch: int,
    ) -> BaseTrainState:
        """
        Update train state with final hidden state after rollout.

        Default: no-op (don't persist).
        Override for persistent memory variants.
        """
        return train_state

    def on_episode_complete(
        self,
        train_state: BaseTrainState,
        episode_return: chex.Array,
        episode_length: chex.Array,
        episode_solved: chex.Array,
        final_hstate: chex.ArrayTree,
    ) -> BaseTrainState:
        """
        Hook called when episodes complete during rollout.

        Default: no-op.
        Override for memory variants that need episode summaries.
        """
        return train_state

    def train(self):
        """Main training loop."""
        config = self.config

        # Setup wandb
        tags = []
        training_method = config.get("training_method", "accel")
        tags.append(training_method.upper())  # ACCEL, PLR, ROBUST_PLR, DR, PAIRED
        if training_method in ["plr", "robust_plr"]:
            if not config["exploratory_grad_updates"]:
                tags.append("robust")
        tags.append(config["agent_type"])

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
        wandb.define_metric("level_sampler/*", step_metric="num_updates")
        wandb.define_metric("probe/*", step_metric="num_updates")
        wandb.define_metric("return/*", step_metric="num_updates")
        wandb.define_metric("curriculum_pred/*", step_metric="num_updates")
        wandb.define_metric("novelty_learnability/*", step_metric="num_updates")

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
        total_steps = 0

        for eval_step in range(config["num_updates"] // config["eval_freq"]):
            start_time = time.time()

            # Train for eval_freq steps
            runner_state, metrics = self.train_and_eval_step(runner_state)
            total_steps += config["eval_freq"]

            # Log
            curr_time = time.time()
            metrics["time_delta"] = curr_time - start_time
            self.log_metrics(metrics, runner_state[1])

            # Call training-time experiment hooks
            self._call_training_hooks(runner_state[1], metrics, total_steps)

            # Checkpoint
            if config["checkpoint_save_interval"] > 0:
                save_args = {"params": runner_state[1].params}
                if hasattr(runner_state[1], 'probe_params') and runner_state[1].probe_params is not None:
                    save_args["probe_params"] = runner_state[1].probe_params
                checkpoint_manager.save(eval_step, save_args)
                checkpoint_manager.wait_until_finished()

        # Finalize training-time experiments
        self._finalize_training_experiments()

        return runner_state[1]

    def _call_training_hooks(
        self,
        train_state: BaseTrainState,
        metrics: dict,
        step: int,
    ):
        """
        Call training hooks for all registered training-time experiments.

        Args:
            train_state: Current training state
            metrics: Training metrics from this step
            step: Current total training step
        """
        for experiment in self.training_experiments:
            if hasattr(experiment, 'training_hook'):
                try:
                    hook_data = experiment.training_hook(train_state, metrics, step)
                    # Log hook data to wandb if present
                    if hook_data and self.config.get("use_wandb", True):
                        prefixed = {f"exp/{experiment.name}/{k}": v for k, v in hook_data.items()}
                        wandb.log(prefixed, step=step)
                except Exception as e:
                    print(f"Warning: training_hook failed for {experiment.name}: {e}")

    def _finalize_training_experiments(self):
        """
        Finalize training-time experiments after training completes.

        Calls analyze() and visualize() on each experiment.
        """
        for experiment in self.training_experiments:
            try:
                print(f"Finalizing experiment: {experiment.name}")
                results = experiment.analyze()
                viz = experiment.visualize()
                experiment.save()
            except Exception as e:
                print(f"Warning: finalize failed for {experiment.name}: {e}")

    def train_and_eval_step(
        self,
        runner_state: Tuple[chex.PRNGKey, BaseTrainState],
    ) -> Tuple[Tuple[chex.PRNGKey, BaseTrainState], dict]:
        """Run training for eval_freq steps, then evaluate."""
        config = self.config

        # Train
        (rng, train_state), metrics = jax.lax.scan(
            self.train_step, runner_state, None, config["eval_freq"]
        )

        # Eval
        rng, rng_eval = jax.random.split(rng)
        eval_metrics = self.evaluate(rng_eval, train_state)
        metrics.update(eval_metrics)

        metrics["update_count"] = (
            train_state.num_dr_updates +
            train_state.num_replay_updates +
            train_state.num_mutation_updates
        )
        metrics["max_updates"] = jnp.float32(self.config.get("num_updates", 50000))

        return (rng, train_state), metrics

    def train_step(
        self,
        carry: Tuple[chex.PRNGKey, BaseTrainState],
        _,
    ) -> Tuple[Tuple[chex.PRNGKey, BaseTrainState], dict]:
        """Single training step - dispatches to appropriate branch."""
        rng, train_state = carry
        config = self.config

        # DR mode: pure domain randomization with continuous rollouts
        if self.training_method == TrainingMethod.DR:
            return self.on_dr_step(rng, train_state)

        rng, rng_replay = jax.random.split(rng)

        # Decide which branch to take
        if config["use_accel"]:
            s = train_state.update_state
            branch = (
                (1 - s) * self.level_sampler.sample_replay_decision(train_state.sampler, rng_replay) +
                2 * s
            )
        else:
            branch = self.level_sampler.sample_replay_decision(train_state.sampler, rng_replay).astype(int)

        return jax.lax.switch(
            branch,
            [self.on_new_levels, self.on_replay_levels, self.on_mutate_levels],
            rng, train_state,
        )

    def _do_sample_trajectories(self, rng, train_state, init_hstate, init_obs, init_env_state):
        """Sample trajectories using RNN policy.

        Override in subclasses to customize rollout (e.g., pass context).
        """
        return sample_trajectories_rnn(
            rng, self.env, self.env_params, train_state,
            init_hstate, init_obs, init_env_state,
            self.config["num_train_envs"], self.config["num_steps"],
        )

    def _do_update_actor_critic(self, rng, train_state, init_hstate, batch, update_grad=True):
        """PPO update step.

        Override in subclasses to customize PPO update (e.g., pass context).
        """
        config = self.config
        return update_actor_critic_rnn(
            rng, train_state, init_hstate, batch,
            config["num_train_envs"], config["num_steps"],
            config["num_minibatches"], config["epoch_ppo"],
            config["clip_eps"], config["entropy_coeff"], config["critic_coeff"],
            update_grad=update_grad,
        )

    def on_new_levels(
        self,
        rng: chex.PRNGKey,
        train_state: BaseTrainState,
    ) -> Tuple[Tuple[chex.PRNGKey, BaseTrainState], dict]:
        """DR branch: sample new random levels."""
        config = self.config
        sampler = train_state.sampler

        # Generate levels
        rng, rng_levels, rng_reset = jax.random.split(rng, 3)
        new_levels = jax.vmap(self.sample_random_level)(
            jax.random.split(rng_levels, config["num_train_envs"])
        )

        # Reset environments
        init_obs, init_env_state = jax.vmap(self.env.reset_to_level, in_axes=(0, 0, None))(
            jax.random.split(rng_reset, config["num_train_envs"]),
            new_levels,
            self.env_params,
        )

        # Get hidden state for this branch
        init_hstate = self.get_hidden_state_for_rollout(train_state, branch=0)

        # Rollout
        (rng, final_hstate, last_obs, last_env_state, last_value), traj = \
            self._do_sample_trajectories(
                rng, train_state, init_hstate, init_obs, init_env_state,
            )

        obs, actions, rewards, dones, log_probs, values, info = traj

        # Compute advantages and scores
        advantages, targets = compute_gae(
            config["gamma"], config["gae_lambda"],
            last_value, values, rewards, dones,
        )
        max_returns = compute_max_returns(dones, rewards)
        scores = compute_score(config, dones, values, max_returns, advantages)

        # Add levels to buffer
        sampler, _ = self.level_sampler.insert_batch(
            sampler, new_levels, scores, {"max_return": max_returns}
        )

        # Update policy (only if exploratory_grad_updates)
        (rng, train_state), losses = self._do_update_actor_critic(
            rng, train_state, init_hstate,
            (obs, actions, dones, log_probs, values, targets, advantages),
            update_grad=config["exploratory_grad_updates"],
        )

        # Update hidden state
        train_state = self.update_hidden_state_after_rollout(train_state, final_hstate, branch=0)

        # Agent-centric tracking (NOT probe)
        train_state = self._update_agent_tracking(
            train_state, obs, values, rewards, dones, new_levels, branch=0
        )

        # Store visualization data for DR branch (for matched pairs)
        train_state = self._store_visualization_data(
            train_state, final_hstate, new_levels, rewards, dones, branch=0
        )

        # Update probe (if enabled)
        rng, rng_probe = jax.random.split(rng)
        train_state, probe_loss = self.update_probe(
            rng_probe, train_state, final_hstate, new_levels,
            branch=0, rewards=rewards, dones=dones, is_replay_to_mutate=False,
            traj_info=info,
        )

        # Update memory (context vector / episodic buffer) after rollout
        episode_return = rewards.sum(axis=0)
        episode_length = dones.sum(axis=0)
        episode_solved = (episode_return > 0).astype(jnp.float32)
        train_state = self.on_episode_complete(
            train_state, episode_return, episode_length, episode_solved, final_hstate,
        )

        # Compute displacement from previous DR batch
        displacement = compute_batch_displacement_metrics(
            new_levels, train_state.dr_last_level_batch
        )

        metrics = {
            "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
            "mean_num_blocks": new_levels.wall_map.sum() / config["num_train_envs"],
            "displacement": displacement,
            "branch": jnp.int32(0),
            "probe_loss": probe_loss,
        }

        train_state = train_state.replace(
            sampler=sampler,
            update_state=UpdateState.DR,
            num_dr_updates=train_state.num_dr_updates + 1,
            dr_last_level_batch=new_levels,
            training_step=train_state.training_step + 1,
        )

        return (rng, train_state), metrics

    def on_replay_levels(
        self,
        rng: chex.PRNGKey,
        train_state: BaseTrainState,
    ) -> Tuple[Tuple[chex.PRNGKey, BaseTrainState], dict]:
        """Replay branch: sample from buffer and update."""
        config = self.config
        sampler = train_state.sampler

        # Sample levels from buffer
        rng, rng_levels, rng_reset = jax.random.split(rng, 3)
        sampler, (level_inds, levels) = self.level_sampler.sample_replay_levels(
            sampler, rng_levels, config["num_train_envs"]
        )

        init_obs, init_env_state = jax.vmap(self.env.reset_to_level, in_axes=(0, 0, None))(
            jax.random.split(rng_reset, config["num_train_envs"]),
            levels,
            self.env_params,
        )

        init_hstate = self.get_hidden_state_for_rollout(train_state, branch=1)

        (rng, final_hstate, last_obs, last_env_state, last_value), traj = \
            self._do_sample_trajectories(
                rng, train_state, init_hstate, init_obs, init_env_state,
            )

        obs, actions, rewards, dones, log_probs, values, info = traj

        advantages, targets = compute_gae(
            config["gamma"], config["gae_lambda"],
            last_value, values, rewards, dones,
        )
        max_returns = jnp.maximum(
            self.level_sampler.get_levels_extra(sampler, level_inds)["max_return"],
            compute_max_returns(dones, rewards),
        )
        scores = compute_score(config, dones, values, max_returns, advantages)
        sampler = self.level_sampler.update_batch(sampler, level_inds, scores, {"max_return": max_returns})

        (rng, train_state), losses = self._do_update_actor_critic(
            rng, train_state, init_hstate,
            (obs, actions, dones, log_probs, values, targets, advantages),
            update_grad=True,
        )

        train_state = self.update_hidden_state_after_rollout(train_state, final_hstate, branch=1)

        # Agent-centric tracking (NOT probe)
        train_state = self._update_agent_tracking(
            train_state, obs, values, rewards, dones, levels, branch=1
        )

        # Store visualization data for Replay branch (for matched pairs)
        train_state = self._store_visualization_data(
            train_state, final_hstate, levels, rewards, dones, branch=1
        )

        # Update probe (if enabled)
        rng, rng_probe = jax.random.split(rng)
        train_state, probe_loss = self.update_probe(
            rng_probe, train_state, final_hstate, levels,
            branch=1, rewards=rewards, dones=dones, is_replay_to_mutate=False,
            traj_info=info,
        )

        # Update memory (context vector / episodic buffer) after rollout
        episode_return = rewards.sum(axis=0)
        episode_length = dones.sum(axis=0)
        episode_solved = (episode_return > 0).astype(jnp.float32)
        train_state = self.on_episode_complete(
            train_state, episode_return, episode_length, episode_solved, final_hstate,
        )

        # Compute displacement from previous replay batch
        displacement = compute_batch_displacement_metrics(
            levels, train_state.replay_last_level_batch
        )

        metrics = {
            "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
            "mean_num_blocks": levels.wall_map.sum() / config["num_train_envs"],
            "displacement": displacement,
            "branch": jnp.int32(1),
            "probe_loss": probe_loss,
        }

        train_state = train_state.replace(
            sampler=sampler,
            update_state=UpdateState.REPLAY,
            num_replay_updates=train_state.num_replay_updates + 1,
            replay_last_level_batch=levels,
            training_step=train_state.training_step + 1,
        )

        return (rng, train_state), metrics

    def on_mutate_levels(
        self,
        rng: chex.PRNGKey,
        train_state: BaseTrainState,
    ) -> Tuple[Tuple[chex.PRNGKey, BaseTrainState], dict]:
        """Mutate branch: mutate previous replay levels."""
        config = self.config
        sampler = train_state.sampler

        rng, rng_mutate, rng_reset = jax.random.split(rng, 3)

        # Mutate previous replay levels
        parent_levels = train_state.replay_last_level_batch
        child_levels = jax.vmap(self.mutate_level, (0, 0, None))(
            jax.random.split(rng_mutate, config["num_train_envs"]),
            parent_levels,
            config["num_edits"],
        )

        init_obs, init_env_state = jax.vmap(self.env.reset_to_level, in_axes=(0, 0, None))(
            jax.random.split(rng_reset, config["num_train_envs"]),
            child_levels,
            self.env_params,
        )

        init_hstate = self.get_hidden_state_for_rollout(train_state, branch=2)

        (rng, final_hstate, last_obs, last_env_state, last_value), traj = \
            self._do_sample_trajectories(
                rng, train_state, init_hstate, init_obs, init_env_state,
            )

        obs, actions, rewards, dones, log_probs, values, info = traj

        advantages, targets = compute_gae(
            config["gamma"], config["gae_lambda"],
            last_value, values, rewards, dones,
        )
        max_returns = compute_max_returns(dones, rewards)
        scores = compute_score(config, dones, values, max_returns, advantages)
        sampler, _ = self.level_sampler.insert_batch(sampler, child_levels, scores, {"max_return": max_returns})

        (rng, train_state), losses = self._do_update_actor_critic(
            rng, train_state, init_hstate,
            (obs, actions, dones, log_probs, values, targets, advantages),
            update_grad=config["exploratory_grad_updates"],
        )

        train_state = self.update_hidden_state_after_rollout(train_state, final_hstate, branch=2)

        # Agent-centric tracking (NOT probe)
        train_state = self._update_agent_tracking(
            train_state, obs, values, rewards, dones, child_levels, branch=2
        )

        # Store visualization data for R->M (1-to-1 correspondence)
        train_state = self._store_visualization_data(
            train_state, final_hstate, child_levels, rewards, dones, branch=2,
            is_replay_to_mutate=True, parent_levels=parent_levels
        )

        # Update probe (if enabled)
        rng, rng_probe = jax.random.split(rng)
        train_state, probe_loss = self.update_probe(
            rng_probe, train_state, final_hstate, child_levels,
            branch=2, rewards=rewards, dones=dones, is_replay_to_mutate=True,
            traj_info=info,
        )

        # Update memory (context vector / episodic buffer) after rollout
        episode_return = rewards.sum(axis=0)
        episode_length = dones.sum(axis=0)
        episode_solved = (episode_return > 0).astype(jnp.float32)
        train_state = self.on_episode_complete(
            train_state, episode_return, episode_length, episode_solved, final_hstate,
        )

        # Compute displacement from previous mutation batch
        displacement = compute_batch_displacement_metrics(
            child_levels, train_state.mutation_last_level_batch
        )

        metrics = {
            "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
            "mean_num_blocks": child_levels.wall_map.sum() / config["num_train_envs"],
            "displacement": displacement,
            "branch": jnp.int32(2),
            "probe_loss": probe_loss,
        }

        train_state = train_state.replace(
            sampler=sampler,
            update_state=UpdateState.MUTATION,
            num_mutation_updates=train_state.num_mutation_updates + 1,
            mutation_last_level_batch=child_levels,
            training_step=train_state.training_step + 1,
        )

        return (rng, train_state), metrics

    def on_dr_step(
        self,
        rng: chex.PRNGKey,
        train_state: BaseTrainState,
    ) -> Tuple[Tuple[chex.PRNGKey, BaseTrainState], dict]:
        """
        Pure Domain Randomization training step.

        Continuous rollouts with auto-reset on episode end. Hidden state and
        env state persist across train_step calls for continuous exploration.

        This is the simplest baseline: random levels, no buffer, no replay.
        """
        config = self.config

        # Continue from previous state (initialized in create_train_state for DR mode)
        init_hstate = train_state.last_hstate
        init_obs = train_state.last_obs
        init_env_state = train_state.last_env_state

        # Rollout with auto-reset handled by environment done signals
        (rng, final_hstate, last_obs, last_env_state, last_value), traj = \
            self._do_sample_trajectories(
                rng, train_state, init_hstate, init_obs, init_env_state,
            )

        obs, actions, rewards, dones, log_probs, values, info = traj

        # Compute advantages
        advantages, targets = compute_gae(
            config["gamma"], config["gae_lambda"],
            last_value, values, rewards, dones,
        )

        # Update policy - always apply gradients in DR mode
        (rng, train_state), losses = self._do_update_actor_critic(
            rng, train_state, init_hstate,
            (obs, actions, dones, log_probs, values, targets, advantages),
            update_grad=True,
        )

        # Update hidden state for continuity
        train_state = self.update_hidden_state_after_rollout(train_state, final_hstate, branch=0)

        # Get levels from env state for tracking (if available)
        current_levels = info.get("level", None) if isinstance(info, dict) else None

        # Agent-centric tracking (if levels available)
        if current_levels is not None:
            train_state = self._update_agent_tracking(
                train_state, obs, values, rewards, dones, current_levels, branch=0
            )

        # Update probe (if enabled and levels available)
        rng, rng_probe = jax.random.split(rng)
        probe_loss = jnp.float32(0.0)
        if current_levels is not None:
            train_state, probe_loss = self.update_probe(
                rng_probe, train_state, final_hstate, current_levels,
                branch=0, rewards=rewards, dones=dones, is_replay_to_mutate=False,
                traj_info=info,
            )

        # Update memory (context vector / episodic buffer) after rollout
        episode_return = rewards.sum(axis=0)
        episode_length = dones.sum(axis=0)
        episode_solved = (episode_return > 0).astype(jnp.float32)
        train_state = self.on_episode_complete(
            train_state, episode_return, episode_length, episode_solved, final_hstate,
        )

        # For pure DR mode, displacement is zeros (no previous level batch to compare)
        _zero = jnp.float32(0.0)
        displacement = {
            'pairwise/wall_map_hamming/mean': _zero, 'pairwise/wall_map_hamming/min': _zero,
            'pairwise/wall_map_hamming/max': _zero, 'pairwise/goal_pos_l2/mean': _zero,
            'pairwise/goal_pos_l2/min': _zero, 'pairwise/goal_pos_l2/max': _zero,
            'pairwise/agent_pos_l2/mean': _zero, 'pairwise/agent_pos_l2/min': _zero,
            'pairwise/agent_pos_l2/max': _zero, 'pairwise/agent_dir_changed/mean': _zero,
            'pairwise/agent_dir_changed/min': _zero, 'pairwise/agent_dir_changed/max': _zero,
            'batch/wall_density_shift': _zero, 'batch/wall_iou': _zero,
            'batch/goal_centroid_shift': _zero, 'batch/agent_pos_centroid_shift': _zero,
            'batch/agent_dir_distribution_shift': _zero,
        }

        metrics = {
            "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
            "mean_return": rewards.sum(axis=0).mean(),
            "displacement": displacement,
            "branch": jnp.int32(0),
            "probe_loss": probe_loss,
        }

        # Update train state with continuous rollout state
        train_state = train_state.replace(
            update_state=UpdateState.DR,
            num_dr_updates=train_state.num_dr_updates + 1,
            training_step=train_state.training_step + 1,
            last_hstate=final_hstate,
            last_obs=last_obs,
            last_env_state=last_env_state,
        )

        return (rng, train_state), metrics

    def update_probe(
        self,
        rng: chex.PRNGKey,
        train_state: ProbeTrainState,
        hstate: chex.ArrayTree,
        levels: Level,
        branch: int,
        rewards: chex.Array,
        dones: chex.Array,
        is_replay_to_mutate: bool = False,
        traj_info=None,
    ) -> tuple:
        """
        Update probe network on current hidden state.

        This method trains a separate probe network to predict level features
        from the agent's hidden state. The probe uses stop_gradient to avoid
        affecting agent learning.

        Args:
            rng: Random key (unused currently)
            train_state: Current train state with probe params
            hstate: LSTM hidden state tuple (c, h)
            levels: Batch of levels used in this rollout
            branch: Which branch (0=DR, 1=replay, 2=mutate)
            rewards: Rewards from rollout, shape (num_steps, batch_size)
            dones: Done flags from rollout, shape (num_steps, batch_size)
            is_replay_to_mutate: Whether this is R->M transition (1-to-1 correspondence)

        Returns:
            Tuple of (updated train state, probe loss scalar)
        """
        config = self.config

        if train_state.probe_params is None:
            return train_state, jnp.float32(0.0)

        # Flatten hidden state: (c, h) -> (batch, 512)
        hstate_flat = flatten_hstate(hstate)

        # Stop gradient - probe doesn't affect agent
        hstate_flat = jax.lax.stop_gradient(hstate_flat)

        # Compute episode statistics for context
        episode_returns = rewards.sum(axis=0)  # Sum over time
        episode_lengths = dones.sum(axis=0).astype(jnp.int32)
        episode_solved = (episode_returns > 0).astype(jnp.float32)

        # Probe forward pass
        probe = CurriculumProbe(env_height=13, env_width=13, use_episode_context=True)
        predictions = probe.apply(
            train_state.probe_params,
            hstate_flat,
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
            loss, loss_dict = compute_probe_loss_batch(preds, levels)
            return loss, loss_dict

        (loss, loss_components), grads = jax.value_and_grad(
            probe_loss_fn, has_aux=True
        )(train_state.probe_params)

        # Compute zone-decomposed wall losses if trajectory info available
        zone_wall_loss_observed = jnp.float32(0.0)
        zone_wall_loss_adjacent = jnp.float32(0.0)
        zone_wall_loss_distant = jnp.float32(0.0)
        zone_wall_accuracy_observed = jnp.float32(0.0)
        zone_wall_accuracy_adjacent = jnp.float32(0.0)
        zone_wall_accuracy_distant = jnp.float32(0.0)

        if traj_info is not None and 'agent_pos' in traj_info and 'agent_dir' in traj_info:
            agent_positions = traj_info['agent_pos']  # (T, N, 2)
            agent_dirs = traj_info['agent_dir']        # (T, N)
            agent_view_size = self.config.get("agent_view_size", 5)

            observed_mask, adjacent_mask, distant_mask = compute_visited_mask_from_positions(
                agent_positions, agent_dirs, agent_view_size
            )

            wall_targets = levels.wall_map.astype(jnp.float32)  # (B, H, W)
            wall_bce = optax.sigmoid_binary_cross_entropy(
                predictions['wall_logits'], wall_targets
            )  # (B, H, W)
            wall_correct = ((jax.nn.sigmoid(predictions['wall_logits']) > 0.5) == wall_targets).astype(jnp.float32)

            def _zone_metrics(mask):
                mask_f = mask.astype(jnp.float32)
                safe_count = jnp.maximum(mask_f.sum(), 1.0)
                per_env_loss = (wall_bce * mask_f[None]).sum(axis=(-2, -1)) / safe_count
                per_env_acc = (wall_correct * mask_f[None]).sum(axis=(-2, -1)) / safe_count
                return per_env_loss.mean(), per_env_acc.mean()

            zone_wall_loss_observed, zone_wall_accuracy_observed = _zone_metrics(observed_mask)
            zone_wall_loss_adjacent, zone_wall_accuracy_adjacent = _zone_metrics(adjacent_mask)
            zone_wall_loss_distant, zone_wall_accuracy_distant = _zone_metrics(distant_mask)

        # Update probe params
        probe_tx = optax.adam(learning_rate=config.get("probe_lr", 1e-3))
        updates, new_opt_state = probe_tx.update(
            grads, train_state.probe_opt_state, train_state.probe_params
        )
        new_probe_params = optax.apply_updates(train_state.probe_params, updates)

        # Compute distributional calibration metrics for all branches
        dist_calibration = compute_distributional_calibration_metrics(predictions, levels)

        # Compute per-instance metrics for R->M transition (where 1-to-1 correspondence exists)
        per_instance_metrics = None
        if is_replay_to_mutate:
            per_instance_metrics = compute_per_instance_calibration_batch(predictions, levels)

        # Update tracking state
        probe_tracking = train_state.probe_tracking
        if probe_tracking is not None:
            # Update loss history
            buffer_size = probe_tracking.loss_history.shape[0]
            ptr = probe_tracking.buffer_ptr % buffer_size

            new_loss_history = probe_tracking.loss_history.at[ptr].set(loss)
            new_step_history = probe_tracking.training_step_history.at[ptr].set(
                train_state.training_step
            )

            # Update branch history
            branch_ptr = probe_tracking.branch_ptrs[branch] % buffer_size
            new_branch_loss = probe_tracking.branch_loss_history.at[branch, branch_ptr].set(loss)
            new_branch_ptrs = probe_tracking.branch_ptrs.at[branch].set(branch_ptr + 1)
            new_branch_counts = probe_tracking.branch_sample_counts.at[branch].add(1)

            # Update distributional calibration tracking
            new_dist_cal_wall = probe_tracking.dist_calibration_wall_history.at[branch, branch_ptr].set(
                dist_calibration['wall_dist_calibration']
            )
            new_dist_cal_goal = probe_tracking.dist_calibration_goal_history.at[branch, branch_ptr].set(
                dist_calibration['goal_dist_calibration']
            )
            new_dist_acc_wall = probe_tracking.dist_accuracy_wall_history.at[branch, branch_ptr].set(
                dist_calibration['wall_dist_accuracy']
            )
            new_dist_acc_goal = probe_tracking.dist_accuracy_goal_mode_match_history.at[branch, branch_ptr].set(
                dist_calibration['goal_dist_mode_match']
            )

            # Update agent returns for correlation tracking
            mean_return = episode_returns.mean()
            new_agent_returns = probe_tracking.agent_returns_history.at[ptr].set(mean_return)

            # Update per-instance tracking for R->M
            per_instance_ptr = probe_tracking.per_instance_ptr
            per_instance_total = probe_tracking.per_instance_total
            new_per_instance_wall = probe_tracking.per_instance_wall_accuracy_history
            new_per_instance_goal = probe_tracking.per_instance_goal_accuracy_history
            new_per_instance_agent_pos = probe_tracking.per_instance_agent_pos_accuracy_history
            new_per_instance_agent_dir = probe_tracking.per_instance_agent_dir_accuracy_history
            new_per_instance_combined = probe_tracking.per_instance_combined_accuracy_history
            new_per_instance_loss = probe_tracking.per_instance_loss_history
            new_is_valid = probe_tracking.is_per_instance_valid
            new_r2m_wall = probe_tracking.replay_to_mutate_wall_accuracy
            new_r2m_goal = probe_tracking.replay_to_mutate_goal_accuracy
            new_r2m_loss = probe_tracking.replay_to_mutate_loss
            new_r2m_count = probe_tracking.replay_to_mutate_count
            new_r2m_ptr = probe_tracking.replay_to_mutate_ptr

            # Initialize R→M visualization data placeholders
            new_r2m_wall_logits = probe_tracking.last_r2m_wall_logits
            new_r2m_goal_logits = probe_tracking.last_r2m_goal_logits
            new_r2m_wall_map = probe_tracking.last_r2m_wall_map
            new_r2m_goal_pos = probe_tracking.last_r2m_goal_pos
            new_r2m_agent_pos = probe_tracking.last_r2m_agent_pos
            new_r2m_valid = probe_tracking.last_r2m_valid

            if is_replay_to_mutate and per_instance_metrics is not None:
                pi_ptr = per_instance_ptr % buffer_size
                new_per_instance_wall = new_per_instance_wall.at[pi_ptr].set(
                    per_instance_metrics['wall_accuracy']
                )
                new_per_instance_goal = new_per_instance_goal.at[pi_ptr].set(
                    per_instance_metrics['goal_accuracy']
                )
                new_per_instance_agent_pos = new_per_instance_agent_pos.at[pi_ptr].set(
                    per_instance_metrics['agent_pos_accuracy']
                )
                new_per_instance_agent_dir = new_per_instance_agent_dir.at[pi_ptr].set(
                    per_instance_metrics['agent_dir_accuracy']
                )
                new_per_instance_combined = new_per_instance_combined.at[pi_ptr].set(
                    per_instance_metrics['combined_accuracy']
                )
                new_per_instance_loss = new_per_instance_loss.at[pi_ptr].set(loss)
                new_is_valid = new_is_valid.at[pi_ptr].set(True)
                per_instance_ptr = pi_ptr + 1
                per_instance_total = per_instance_total + 1

                # Update R->M specific tracking
                r2m_ptr = new_r2m_ptr % buffer_size
                new_r2m_wall = new_r2m_wall.at[r2m_ptr].set(per_instance_metrics['wall_accuracy'])
                new_r2m_goal = new_r2m_goal.at[r2m_ptr].set(per_instance_metrics['goal_accuracy'])
                new_r2m_loss = new_r2m_loss.at[r2m_ptr].set(loss)
                new_r2m_count = new_r2m_count + 1
                new_r2m_ptr = r2m_ptr + 1

                # Store R→M visualization data (predictions and target levels)
                new_r2m_wall_logits = predictions['wall_logits']
                new_r2m_goal_logits = predictions['goal_logits']
                new_r2m_wall_map = levels.wall_map
                new_r2m_goal_pos = levels.goal_pos
                new_r2m_agent_pos = levels.agent_pos
                new_r2m_valid = True

            # Per-component loss histories
            new_wall_loss = probe_tracking.wall_loss_history.at[ptr].set(loss_components['wall_loss'])
            new_goal_loss = probe_tracking.goal_loss_history.at[ptr].set(loss_components['goal_loss'])
            new_agent_pos_loss = probe_tracking.agent_pos_loss_history.at[ptr].set(loss_components['agent_pos_loss'])
            new_agent_dir_loss = probe_tracking.agent_dir_loss_history.at[ptr].set(loss_components['agent_dir_loss'])

            # Per-branch per-component losses
            new_branch_wall_loss = probe_tracking.branch_wall_loss_history.at[branch, branch_ptr].set(loss_components['wall_loss'])
            new_branch_goal_loss = probe_tracking.branch_goal_loss_history.at[branch, branch_ptr].set(loss_components['goal_loss'])
            new_branch_agent_pos_loss = probe_tracking.branch_agent_pos_loss_history.at[branch, branch_ptr].set(loss_components['agent_pos_loss'])
            new_branch_agent_dir_loss = probe_tracking.branch_agent_dir_loss_history.at[branch, branch_ptr].set(loss_components['agent_dir_loss'])

            # Per-branch zone-decomposed wall losses
            new_branch_zone_wall_loss_observed = probe_tracking.branch_zone_wall_loss_observed.at[branch, branch_ptr].set(zone_wall_loss_observed)
            new_branch_zone_wall_loss_adjacent = probe_tracking.branch_zone_wall_loss_adjacent.at[branch, branch_ptr].set(zone_wall_loss_adjacent)
            new_branch_zone_wall_loss_distant = probe_tracking.branch_zone_wall_loss_distant.at[branch, branch_ptr].set(zone_wall_loss_distant)
            new_branch_zone_wall_accuracy_observed = probe_tracking.branch_zone_wall_accuracy_observed.at[branch, branch_ptr].set(zone_wall_accuracy_observed)
            new_branch_zone_wall_accuracy_adjacent = probe_tracking.branch_zone_wall_accuracy_adjacent.at[branch, branch_ptr].set(zone_wall_accuracy_adjacent)
            new_branch_zone_wall_accuracy_distant = probe_tracking.branch_zone_wall_accuracy_distant.at[branch, branch_ptr].set(zone_wall_accuracy_distant)

            # Per-branch aggregated accuracy (from dist_calibration)
            new_dist_acc_agent_pos = probe_tracking.dist_accuracy_agent_pos_history.at[branch, branch_ptr].set(
                dist_calibration.get('agent_pos_dist_mode_match', 0.0)
            )
            new_dist_acc_agent_dir = probe_tracking.dist_accuracy_agent_dir_history.at[branch, branch_ptr].set(
                dist_calibration.get('agent_dir_dist_mode_match', 0.0)
            )
            new_dist_acc_combined = probe_tracking.dist_accuracy_combined_history.at[branch, branch_ptr].set(
                dist_calibration.get('combined_accuracy', 0.0)
            )

            # Probe accuracy for correlation tracking (combined distributional accuracy)
            combined_acc = dist_calibration.get('combined_accuracy', 0.0)
            new_probe_accuracy = probe_tracking.probe_accuracy_history.at[ptr].set(combined_acc)

            probe_tracking = probe_tracking.replace(
                loss_history=new_loss_history,
                training_step_history=new_step_history,
                branch_loss_history=new_branch_loss,
                branch_ptrs=new_branch_ptrs,
                branch_sample_counts=new_branch_counts,
                buffer_ptr=(ptr + 1) % buffer_size,
                total_samples=probe_tracking.total_samples + 1,
                current_training_step=train_state.training_step + 1,
                agent_returns_history=new_agent_returns,
                probe_accuracy_history=new_probe_accuracy,
                # Distributional calibration
                dist_calibration_wall_history=new_dist_cal_wall,
                dist_calibration_goal_history=new_dist_cal_goal,
                dist_accuracy_wall_history=new_dist_acc_wall,
                dist_accuracy_goal_mode_match_history=new_dist_acc_goal,
                dist_accuracy_agent_pos_history=new_dist_acc_agent_pos,
                dist_accuracy_agent_dir_history=new_dist_acc_agent_dir,
                dist_accuracy_combined_history=new_dist_acc_combined,
                # Per-component losses
                wall_loss_history=new_wall_loss,
                goal_loss_history=new_goal_loss,
                agent_pos_loss_history=new_agent_pos_loss,
                agent_dir_loss_history=new_agent_dir_loss,
                branch_wall_loss_history=new_branch_wall_loss,
                branch_goal_loss_history=new_branch_goal_loss,
                branch_agent_pos_loss_history=new_branch_agent_pos_loss,
                branch_agent_dir_loss_history=new_branch_agent_dir_loss,
                # Zone-decomposed wall losses
                branch_zone_wall_loss_observed=new_branch_zone_wall_loss_observed,
                branch_zone_wall_loss_adjacent=new_branch_zone_wall_loss_adjacent,
                branch_zone_wall_loss_distant=new_branch_zone_wall_loss_distant,
                branch_zone_wall_accuracy_observed=new_branch_zone_wall_accuracy_observed,
                branch_zone_wall_accuracy_adjacent=new_branch_zone_wall_accuracy_adjacent,
                branch_zone_wall_accuracy_distant=new_branch_zone_wall_accuracy_distant,
                # Per-instance tracking
                per_instance_wall_accuracy_history=new_per_instance_wall,
                per_instance_goal_accuracy_history=new_per_instance_goal,
                per_instance_agent_pos_accuracy_history=new_per_instance_agent_pos,
                per_instance_agent_dir_accuracy_history=new_per_instance_agent_dir,
                per_instance_combined_accuracy_history=new_per_instance_combined,
                per_instance_loss_history=new_per_instance_loss,
                per_instance_ptr=per_instance_ptr,
                per_instance_total=per_instance_total,
                is_per_instance_valid=new_is_valid,
                # R->M specific
                replay_to_mutate_wall_accuracy=new_r2m_wall,
                replay_to_mutate_goal_accuracy=new_r2m_goal,
                replay_to_mutate_loss=new_r2m_loss,
                replay_to_mutate_count=new_r2m_count,
                replay_to_mutate_ptr=new_r2m_ptr,
                # R->M visualization data
                last_r2m_wall_logits=new_r2m_wall_logits,
                last_r2m_goal_logits=new_r2m_goal_logits,
                last_r2m_wall_map=new_r2m_wall_map,
                last_r2m_goal_pos=new_r2m_goal_pos,
                last_r2m_agent_pos=new_r2m_agent_pos,
                last_r2m_valid=new_r2m_valid,
                # Last predictions/levels for divergence and heatmaps
                last_predictions_wall_logits=predictions['wall_logits'],
                last_predictions_goal_logits=predictions['goal_logits'],
                last_predictions_agent_pos_logits=predictions['agent_pos_logits'],
                last_predictions_agent_dir_logits=predictions['agent_dir_logits'],
                last_levels_wall_map=levels.wall_map,
                last_levels_goal_pos=levels.goal_pos,
                last_levels_agent_pos=levels.agent_pos,
                last_levels_agent_dir=levels.agent_dir,
                last_branch=branch,
            )

        # Update hidden state samples for t-SNE visualization
        hstate_samples = train_state.hstate_samples
        hstate_sample_branches = train_state.hstate_sample_branches
        hstate_sample_ptr = train_state.hstate_sample_ptr

        if hstate_samples is not None:
            # Store first sample from batch
            sample_idx = hstate_sample_ptr % hstate_samples.shape[0]
            hstate_samples = hstate_samples.at[sample_idx].set(hstate_flat[0])
            hstate_sample_branches = hstate_sample_branches.at[sample_idx].set(branch)
            hstate_sample_ptr = hstate_sample_ptr + 1

        return train_state.replace(
            probe_params=new_probe_params,
            probe_opt_state=new_opt_state,
            probe_tracking=probe_tracking,
            current_hstate=hstate,
            current_branch=branch,
            hstate_samples=hstate_samples,
            hstate_sample_branches=hstate_sample_branches,
            hstate_sample_ptr=hstate_sample_ptr,
        ), loss

    # =========================================================================
    # AGENT-CENTRIC TRACKING (NOT PROBE-BASED)
    # =========================================================================

    def _update_agent_tracking(
        self,
        train_state: BaseTrainState,
        obs: chex.Array,
        values: chex.Array,
        rewards: chex.Array,
        dones: chex.Array,
        levels,
        branch: int,
    ) -> BaseTrainState:
        """
        Update agent-centric tracking state.

        This tracks AGENT metrics (not probe):
        - Policy entropy
        - Value calibration (V(s) vs actual return)
        - Level difficulty features
        - Per-branch statistics

        Args:
            train_state: Current train state
            obs: Observations (unused for now)
            values: Value predictions from rollout
            rewards: Rewards from rollout
            dones: Done flags from rollout
            levels: Levels played in this rollout
            branch: Branch index (0=DR, 1=Replay, 2=Mutate)

        Returns:
            Updated train state with agent tracking
        """
        config = self.config

        agent_tracking = train_state.agent_tracking

        # Compute episode statistics
        episode_returns = rewards.sum(axis=0)  # (batch_size,)
        mean_return = episode_returns.mean()

        # Compute mean value prediction for this rollout
        mean_value = values.mean()

        # Compute level features
        wall_density = levels.wall_map.sum() / (13 * 13 * config["num_train_envs"])

        # Goal distance (Manhattan) - approximate
        goal_distances = jnp.abs(levels.goal_pos[:, 0] - levels.agent_pos[:, 0]) + \
                        jnp.abs(levels.goal_pos[:, 1] - levels.agent_pos[:, 1])
        mean_goal_distance = goal_distances.mean()

        # Policy entropy would require access to policy output, which we don't have here
        # For now, use value variance as proxy for policy uncertainty
        policy_entropy_proxy = values.std()

        # Update tracking state
        agent_tracking = update_agent_tracking(
            agent_tracking,
            policy_entropy=policy_entropy_proxy,
            value_prediction=mean_value,
            actual_return=mean_return,
            wall_density=wall_density,
            goal_distance=mean_goal_distance,
            branch=branch,
            training_step=train_state.training_step,
            policy_kl=0.0,  # Would need previous policy to compute
        )

        return train_state.replace(agent_tracking=agent_tracking)

    def _store_visualization_data(
        self,
        train_state: BaseTrainState,
        hstate: chex.ArrayTree,
        levels,
        rewards: chex.Array,
        dones: chex.Array,
        branch: int,
        is_replay_to_mutate: bool = False,
        parent_levels=None,
    ) -> BaseTrainState:
        """
        Store visualization data for matched pairs.

        For DR and Replay branches: stores data for greedy matching
        For R->M: stores data with 1-to-1 correspondence

        Args:
            train_state: Current train state
            hstate: Hidden state after rollout
            levels: Levels from this branch
            rewards: Rewards from rollout
            dones: Done flags
            branch: Branch index
            is_replay_to_mutate: Whether this is R->M transition
            parent_levels: Parent levels for R->M (for correspondence)

        Returns:
            Updated train state with visualization data
        """
        config = self.config

        # Skip if probe runner not available
        if self.probe_runner is None:
            return train_state

        # Initialize visualization data if needed
        if train_state.visualization_data is None:
            viz_data = create_visualization_data(
                batch_size=config["num_train_envs"],
                env_height=13,
                env_width=13,
            )
        else:
            viz_data = train_state.visualization_data

        # Flatten hidden state
        hstate_flat = flatten_hstate(hstate)
        hstate_flat = jax.lax.stop_gradient(hstate_flat)

        # Episode context
        episode_returns = rewards.sum(axis=0)
        episode_solved = (episode_returns > 0).astype(jnp.float32)
        episode_lengths = dones.sum(axis=0).astype(jnp.int32)

        # Get predictions from probe runner
        try:
            predictions = self.probe_runner.evaluate(
                hstate_flat,
                episode_return=episode_returns,
                episode_solved=episode_solved,
                episode_length=episode_lengths,
            )

            # Store based on branch
            if branch == 0:  # DR
                viz_data = viz_data.replace(
                    last_dr_predictions_wall=predictions['wall_logits'],
                    last_dr_predictions_goal=predictions['goal_logits'],
                    last_dr_levels_wall=levels.wall_map,
                    last_dr_levels_goal=levels.goal_pos,
                    dr_valid=True,
                )
            elif branch == 1:  # Replay
                viz_data = viz_data.replace(
                    last_replay_predictions_wall=predictions['wall_logits'],
                    last_replay_predictions_goal=predictions['goal_logits'],
                    last_replay_levels_wall=levels.wall_map,
                    last_replay_levels_goal=levels.goal_pos,
                    replay_valid=True,
                )
            # R->M visualization is handled in probe_tracking (1-to-1 correspondence)

        except Exception as e:
            # Don't fail training due to probe errors
            pass

        return train_state.replace(visualization_data=viz_data)

    def run_probe_analysis(
        self,
        rng: chex.PRNGKey,
        train_state: BaseTrainState,
        levels,
        num_episodes: int = 10,
    ) -> dict:
        """
        Run probe analysis OUTSIDE the training loop.

        This method is for interpretability experiments only.
        It does NOT affect agent training.

        Args:
            rng: Random key
            train_state: Current train state
            levels: Levels to analyze
            num_episodes: Number of episodes per level

        Returns:
            Dict with probe analysis results
        """
        if self.probe_runner is None:
            return {'error': 'No probe runner available'}

        config = self.config
        results = {
            'wall_accuracies': [],
            'goal_accuracies': [],
            'per_level_results': [],
        }

        # Run agent on each level and analyze
        for i, level in enumerate(levels):
            rng, rng_rollout = jax.random.split(rng)

            # Reset to level
            level_batch = jax.tree_util.tree_map(
                lambda x: jnp.array([x]).repeat(1, axis=0), level
            )
            init_obs, init_env_state = self.env.reset_to_level(
                rng_rollout, level, self.env_params
            )

            # Run episode
            hstate = self.initialize_hidden_state(1)
            obs_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)
            env_state_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_env_state)

            (rng_rollout, final_hstate, _, _, _), traj = sample_trajectories_rnn(
                rng_rollout, self.env, self.env_params, train_state,
                hstate, obs_batch, env_state_batch,
                1, config["num_steps"],
            )

            _, _, rewards, dones, _, _, _ = traj

            # Analyze with probe
            eval_result = self.probe_runner.evaluate_per_instance(
                final_hstate, level_batch,
                episode_return=rewards.sum(axis=0),
                episode_solved=(rewards.sum(axis=0) > 0).astype(jnp.float32),
                episode_length=dones.sum(axis=0).astype(jnp.int32),
            )

            results['wall_accuracies'].append(
                float(eval_result['per_instance_metrics']['wall_accuracy'])
            )
            results['goal_accuracies'].append(
                float(eval_result['per_instance_metrics']['goal_accuracy'])
            )
            results['per_level_results'].append(eval_result)

        # Aggregate
        results['mean_wall_accuracy'] = np.mean(results['wall_accuracies'])
        results['mean_goal_accuracy'] = np.mean(results['goal_accuracies'])

        return results

    def evaluate(
        self,
        rng: chex.PRNGKey,
        train_state: BaseTrainState,
    ) -> dict:
        """Evaluate agent on eval levels, capturing frames for animations."""
        config = self.config

        rng, rng_reset = jax.random.split(rng)
        levels = get_eval_levels(config["eval_levels"])
        num_levels = len(config["eval_levels"])

        init_obs, init_env_state = jax.vmap(
            self.eval_env.reset_to_level, (0, 0, None)
        )(jax.random.split(rng_reset, num_levels), levels, self.env_params)

        states, rewards, episode_lengths = evaluate_rnn(
            rng, self.eval_env, self.env_params, train_state,
            self.initialize_hidden_state(num_levels),
            init_obs, init_env_state,
            self.env_params.max_steps_in_episode,
        )

        mask = jnp.arange(self.env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)

        # Render frames for animations: states shape is (max_steps, num_levels, ...)
        # render_state produces (H, W, C) images
        try:
            images = jax.vmap(jax.vmap(
                self.env_renderer.render_state, (0, None)
            ), (0, None))(states, self.env_params)
            # images: (max_steps, num_levels, H, W, C) -> (max_steps, num_levels, C, H, W)
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

    def _build_log_dict(self, metrics: dict, train_state: BaseTrainState) -> dict:
        """Build the log dict with all base metrics.

        Subclasses can override to add additional metrics before wandb.log.
        All metrics are logged every eval call (no sub-sampling with modulo checks).
        """
        config = self.config

        update_count = metrics["update_count"]
        env_steps = update_count * config["num_train_envs"] * config["num_steps"]
        log_dict = {
            "num_updates": update_count,
            "num_env_steps": env_steps,
            "sps": env_steps / metrics.get("time_delta", 1),
        }

        # Eval metrics
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

        # Level sampler metrics
        sampler_info = train_state_to_log_dict(train_state, self.level_sampler)
        log_dict.update(sampler_info["log"])

        # =====================================================================
        # LEVEL IMAGES (every eval call)
        # =====================================================================
        if update_count > 0:
            try:
                sampler = train_state.sampler
                highest_scoring = self.level_sampler.get_levels(
                    sampler, sampler["scores"].argmax()
                )
                highest_scoring_img = self.env_renderer.render_level(
                    highest_scoring, self.env_params
                )
                log_dict["images/highest_scoring_level"] = wandb.Image(
                    np.array(highest_scoring_img), caption="Highest Scoring Level"
                )

                weights = self.level_sampler.level_weights(sampler)
                highest_weighted = self.level_sampler.get_levels(
                    sampler, weights.argmax()
                )
                highest_weighted_img = self.env_renderer.render_level(
                    highest_weighted, self.env_params
                )
                log_dict["images/highest_weighted_level"] = wandb.Image(
                    np.array(highest_weighted_img), caption="Highest Weighted Level"
                )
            except Exception:
                pass

            # Per-branch level images
            branch_level_map = {
                "dr": train_state.dr_last_level_batch,
                "replay": train_state.replay_last_level_batch,
                "mutation": train_state.mutation_last_level_batch,
            }
            branch_update_map = {
                "dr": train_state.num_dr_updates,
                "replay": train_state.num_replay_updates,
                "mutation": train_state.num_mutation_updates,
            }
            for branch_name, levels_batch in branch_level_map.items():
                if levels_batch is not None and branch_update_map[branch_name] > 0:
                    try:
                        branch_imgs = jax.vmap(
                            self.env_renderer.render_level, (0, None)
                        )(levels_batch, self.env_params)
                        log_dict[f"images/{branch_name}_levels"] = [
                            wandb.Image(np.array(img)) for img in branch_imgs[:4]
                        ]
                    except Exception:
                        pass

        # =====================================================================
        # EVAL ANIMATIONS (wandb.Video)
        # =====================================================================
        if "eval_animation" in metrics:
            try:
                frames, ep_lengths = metrics["eval_animation"]
                for i, level_name in enumerate(config["eval_levels"]):
                    level_frames = frames[:, i]  # (max_steps, C, H, W)
                    ep_len = int(ep_lengths[i])
                    level_frames = np.array(level_frames[:ep_len])
                    log_dict[f"animations/{level_name}"] = wandb.Video(
                        level_frames, fps=4, format="gif"
                    )
            except Exception:
                pass

        # =====================================================================
        # PROBE METRICS (for agents with probes)
        # =====================================================================
        if hasattr(train_state, 'probe_tracking') and train_state.probe_tracking is not None:
            probe_tracking = train_state.probe_tracking
            buffer_size = probe_tracking.loss_history.shape[0]
            valid_samples = jnp.minimum(probe_tracking.total_samples, buffer_size)

            if valid_samples > 0:
                # --- Per-component distributional losses (Cat 4) ---
                log_dict["probe/dist_loss/wall"] = float(probe_tracking.wall_loss_history[:valid_samples].mean())
                log_dict["probe/dist_loss/goal"] = float(probe_tracking.goal_loss_history[:valid_samples].mean())
                log_dict["probe/dist_loss/agent_pos"] = float(probe_tracking.agent_pos_loss_history[:valid_samples].mean())
                log_dict["probe/dist_loss/agent_dir"] = float(probe_tracking.agent_dir_loss_history[:valid_samples].mean())
                log_dict["probe/dist_loss/total"] = float(probe_tracking.loss_history[:valid_samples].mean())

                # Legacy per-instance loss keys (backward compatibility)
                log_dict["probe/all/wall_loss"] = log_dict["probe/dist_loss/wall"]
                log_dict["probe/all/goal_loss"] = log_dict["probe/dist_loss/goal"]
                log_dict["probe/all/agent_pos_loss"] = log_dict["probe/dist_loss/agent_pos"]
                log_dict["probe/all/agent_dir_loss"] = log_dict["probe/dist_loss/agent_dir"]
                log_dict["probe/all/total_loss"] = log_dict["probe/dist_loss/total"]
                log_dict["probe/total_loss"] = log_dict["probe/dist_loss/total"]

                # --- Aggregated distributional calibration & accuracy (Cat 5) ---
                branch_names = {0: "random", 1: "replay", 2: "mutate"}

                # Aggregated across all branches
                all_wall_cal = []
                all_goal_cal = []
                all_wall_acc = []
                all_goal_mode = []
                all_agent_pos_mode = []
                all_agent_dir_mode = []
                all_combined = []

                for branch_idx, branch_name in branch_names.items():
                    branch_ptr = int(probe_tracking.branch_ptrs[branch_idx])
                    if branch_ptr > 0:
                        branch_valid = jnp.minimum(branch_ptr, buffer_size)

                        # Per-branch total loss
                        branch_total = probe_tracking.branch_loss_history[branch_idx, :branch_valid].mean()
                        log_dict[f"probe/{branch_name}/dist_loss/total"] = float(branch_total)

                        # Per-branch per-component losses
                        bw = float(probe_tracking.branch_wall_loss_history[branch_idx, :branch_valid].mean())
                        bg = float(probe_tracking.branch_goal_loss_history[branch_idx, :branch_valid].mean())
                        bap = float(probe_tracking.branch_agent_pos_loss_history[branch_idx, :branch_valid].mean())
                        bad = float(probe_tracking.branch_agent_dir_loss_history[branch_idx, :branch_valid].mean())
                        log_dict[f"probe/{branch_name}/dist_loss/wall"] = bw
                        log_dict[f"probe/{branch_name}/wall_loss"] = bw
                        log_dict[f"probe/{branch_name}/goal_loss"] = bg
                        log_dict[f"probe/{branch_name}/agent_pos_loss"] = bap
                        log_dict[f"probe/{branch_name}/agent_dir_loss"] = bad
                        log_dict[f"probe/{branch_name}/total_loss"] = float(branch_total)

                        # Per-branch distributional calibration
                        wall_cal = float(probe_tracking.dist_calibration_wall_history[branch_idx, :branch_valid].mean())
                        goal_cal = float(probe_tracking.dist_calibration_goal_history[branch_idx, :branch_valid].mean())
                        wall_acc = float(probe_tracking.dist_accuracy_wall_history[branch_idx, :branch_valid].mean())
                        goal_mode = float(probe_tracking.dist_accuracy_goal_mode_match_history[branch_idx, :branch_valid].mean())
                        agent_pos_mode = float(probe_tracking.dist_accuracy_agent_pos_history[branch_idx, :branch_valid].mean())
                        agent_dir_mode = float(probe_tracking.dist_accuracy_agent_dir_history[branch_idx, :branch_valid].mean())
                        combined = float(probe_tracking.dist_accuracy_combined_history[branch_idx, :branch_valid].mean())

                        log_dict[f"probe/{branch_name}/dist_calibration/wall"] = wall_cal
                        log_dict[f"probe/{branch_name}/dist_calibration/goal"] = goal_cal
                        log_dict[f"probe/{branch_name}/dist_accuracy/wall"] = wall_acc
                        log_dict[f"probe/{branch_name}/dist_accuracy/goal_mode_match"] = goal_mode
                        log_dict[f"probe/{branch_name}/dist_accuracy/agent_pos_mode_match"] = agent_pos_mode
                        log_dict[f"probe/{branch_name}/dist_accuracy/combined"] = combined

                        all_wall_cal.append(wall_cal)
                        all_goal_cal.append(goal_cal)
                        all_wall_acc.append(wall_acc)
                        all_goal_mode.append(goal_mode)
                        all_agent_pos_mode.append(agent_pos_mode)
                        all_agent_dir_mode.append(agent_dir_mode)
                        all_combined.append(combined)

                # Aggregated calibration/accuracy across branches
                if all_wall_cal:
                    log_dict["probe/dist_calibration/wall"] = sum(all_wall_cal) / len(all_wall_cal)
                    log_dict["probe/dist_calibration/goal"] = sum(all_goal_cal) / len(all_goal_cal)
                    log_dict["probe/dist_accuracy/wall"] = sum(all_wall_acc) / len(all_wall_acc)
                    log_dict["probe/dist_accuracy/goal_mode_match"] = sum(all_goal_mode) / len(all_goal_mode)
                    log_dict["probe/dist_accuracy/agent_pos_mode_match"] = sum(all_agent_pos_mode) / len(all_agent_pos_mode)
                    log_dict["probe/dist_accuracy/agent_dir_mode_match"] = sum(all_agent_dir_mode) / len(all_agent_dir_mode)
                    log_dict["probe/dist_accuracy/combined"] = sum(all_combined) / len(all_combined)

                # --- R→M per-instance metrics (Cat 5 continued) ---
                r2m_count = probe_tracking.replay_to_mutate_count
                if r2m_count > 0:
                    r2m_valid = jnp.minimum(r2m_count, buffer_size)
                    log_dict["probe/replay_to_mutate/per_instance_loss"] = float(
                        probe_tracking.replay_to_mutate_loss[:r2m_valid].mean()
                    )
                    log_dict["probe/replay_to_mutate/per_instance_wall_loss"] = float(
                        probe_tracking.replay_to_mutate_wall_accuracy[:r2m_valid].mean()
                    )
                    log_dict["probe/replay_to_mutate/count"] = int(r2m_count)

                    # Per-instance calibration from tracking
                    pi_total = probe_tracking.per_instance_total
                    if pi_total > 0:
                        pi_valid = jnp.minimum(pi_total, buffer_size)
                        log_dict["probe/replay_to_mutate/wall_accuracy"] = float(
                            probe_tracking.per_instance_wall_accuracy_history[:pi_valid].mean()
                        )
                        log_dict["probe/replay_to_mutate/goal_accuracy"] = float(
                            probe_tracking.per_instance_goal_accuracy_history[:pi_valid].mean()
                        )
                        log_dict["probe/replay_to_mutate/agent_pos_accuracy"] = float(
                            probe_tracking.per_instance_agent_pos_accuracy_history[:pi_valid].mean()
                        )
                        log_dict["probe/replay_to_mutate/agent_dir_accuracy"] = float(
                            probe_tracking.per_instance_agent_dir_accuracy_history[:pi_valid].mean()
                        )
                        log_dict["probe/replay_to_mutate/combined_accuracy"] = float(
                            probe_tracking.per_instance_combined_accuracy_history[:pi_valid].mean()
                        )

                # --- Distribution divergence (Cat 6) ---
                try:
                    last_preds = {
                        'wall_logits': probe_tracking.last_predictions_wall_logits,
                        'goal_logits': probe_tracking.last_predictions_goal_logits,
                        'agent_pos_logits': probe_tracking.last_predictions_agent_pos_logits,
                        'agent_dir_logits': probe_tracking.last_predictions_agent_dir_logits,
                    }
                    # Use last levels batch as empirical distribution estimate
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
                    divergence_metrics = compute_distribution_divergence(
                        last_preds, last_levels,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["probe/divergence/goal_kl"] = float(divergence_metrics['goal_kl'])
                    log_dict["probe/divergence/goal_js"] = float(divergence_metrics['goal_js'])
                    log_dict["probe/divergence/agent_pos_kl"] = float(divergence_metrics['agent_pos_kl'])
                    log_dict["probe/divergence/wall_density_error"] = float(divergence_metrics['wall_density_error'])
                    log_dict["probe/divergence/empirical_wall_density"] = float(divergence_metrics['empirical_wall_density'])
                    log_dict["probe/divergence/predicted_wall_density"] = float(divergence_metrics['predicted_wall_density'])
                except Exception as e:
                    pass

                # --- Information gain vs random baselines (Cat 7) ---
                try:
                    random_baselines = compute_random_baselines(DEFAULT_ENV_HEIGHT, DEFAULT_ENV_WIDTH)
                    wall_acc = log_dict.get("probe/dist_accuracy/wall", 0.0)
                    goal_mode = log_dict.get("probe/dist_accuracy/goal_mode_match", 0.0)
                    total_loss = log_dict.get("probe/dist_loss/total", 0.0)
                    log_dict["probe/info_gain/wall_vs_random"] = wall_acc - random_baselines['wall_accuracy']
                    log_dict["probe/info_gain/goal_vs_random"] = goal_mode - random_baselines['goal_top1']
                    log_dict["probe/info_gain/total_loss_improvement"] = random_baselines['total_loss'] - total_loss
                except Exception:
                    pass

                # --- Matched/batch accuracy with greedy matching (Cat 8) ---
                try:
                    last_preds = {
                        'wall_logits': probe_tracking.last_predictions_wall_logits,
                        'goal_logits': probe_tracking.last_predictions_goal_logits,
                        'agent_pos_logits': probe_tracking.last_predictions_agent_pos_logits,
                        'agent_dir_logits': probe_tracking.last_predictions_agent_dir_logits,
                    }
                    last_levels = LevelBatch(
                        wall_map=probe_tracking.last_levels_wall_map,
                        goal_pos=probe_tracking.last_levels_goal_pos,
                        agent_pos=probe_tracking.last_levels_agent_pos,
                        agent_dir=probe_tracking.last_levels_agent_dir,
                    )
                    matched_indices, match_losses = compute_greedy_matching(
                        last_preds, last_levels,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    matched_metrics = compute_matched_accuracy_metrics(
                        last_preds, last_levels, matched_indices,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["probe/matched/wall_accuracy"] = float(matched_metrics['matched_wall_accuracy'])
                    log_dict["probe/matched/goal_accuracy"] = float(matched_metrics['matched_goal_accuracy'])
                    log_dict["probe/matched/agent_pos_accuracy"] = float(matched_metrics['matched_agent_pos_accuracy'])
                    log_dict["probe/matched/dir_accuracy"] = float(matched_metrics['matched_dir_accuracy'])
                    log_dict["probe/matched/mean_match_loss"] = float(match_losses.mean())
                except Exception:
                    pass

            # --- Novelty, learnability, open-endedness (Cat 5) ---
            if probe_tracking.total_samples > 10:
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

                log_dict["probe/dynamics/novelty"] = novelty
                log_dict["probe/dynamics/learnability"] = learnability
                log_dict["probe/dynamics/openendedness_score"] = oe_score
                log_dict["probe/dynamics/regime"] = regime
                log_dict["probe/dynamics/novelty_slope"] = novelty_details.get("novelty_slope", 0.0)
                log_dict["probe/dynamics/instantaneous_novelty"] = novelty_details.get("instantaneous_novelty", 0.0)
                log_dict["probe/dynamics/early_loss"] = learn_details.get("early_loss", 0.0)
                log_dict["probe/dynamics/late_loss"] = learn_details.get("late_loss", 0.0)
                # Also log under dynamics/ for consistency
                log_dict["dynamics/novelty"] = novelty
                log_dict["dynamics/learnability"] = learnability
                log_dict["dynamics/openendedness_score"] = oe_score

            # --- Correlation with agent performance ---
            correlation_stats = compute_probe_correlation_with_performance(
                probe_tracking.probe_accuracy_history,
                probe_tracking.agent_returns_history,
                probe_tracking.total_samples,
            )
            log_dict["probe/correlation/probe_return_correlation"] = correlation_stats['correlation']
            log_dict["probe/correlation/mean_probe_accuracy"] = correlation_stats.get('probe_accuracy_mean', 0.0)
            log_dict["probe/correlation/mean_agent_return"] = correlation_stats.get('agent_returns_mean', 0.0)

            # --- Hidden state statistics ---
            hstate_stats = compute_hidden_state_statistics(
                probe_tracking.hstate_mean_by_branch,
                probe_tracking.hstate_var_by_branch,
                probe_tracking.hstate_count_by_branch,
            )
            for key, value in hstate_stats.items():
                log_dict[f"probe/hstate/{key}"] = value

            # --- Branch sample counts ---
            log_dict["probe/samples/random_count"] = int(probe_tracking.branch_sample_counts[0])
            log_dict["probe/samples/replay_count"] = int(probe_tracking.branch_sample_counts[1])
            log_dict["probe/samples/mutate_count"] = int(probe_tracking.branch_sample_counts[2])
            log_dict["probe/samples/total"] = int(probe_tracking.total_samples)

            # --- Zone-decomposed wall losses ---
            if hasattr(probe_tracking, 'branch_zone_wall_loss_observed'):
                for b, bname in enumerate(["random", "replay", "mutate"]):
                    n_samples = int(probe_tracking.branch_sample_counts[b])
                    if n_samples > 0:
                        ptr = int(probe_tracking.branch_ptrs[b])
                        n = min(n_samples, buffer_size)
                        start = max(0, ptr - n)
                        log_dict[f"probe/{bname}/zone_loss/observed"] = float(
                            probe_tracking.branch_zone_wall_loss_observed[b, start:ptr].mean()
                        )
                        log_dict[f"probe/{bname}/zone_loss/adjacent"] = float(
                            probe_tracking.branch_zone_wall_loss_adjacent[b, start:ptr].mean()
                        )
                        log_dict[f"probe/{bname}/zone_loss/distant"] = float(
                            probe_tracking.branch_zone_wall_loss_distant[b, start:ptr].mean()
                        )
                        log_dict[f"probe/{bname}/zone_accuracy/observed"] = float(
                            probe_tracking.branch_zone_wall_accuracy_observed[b, start:ptr].mean()
                        )
                        log_dict[f"probe/{bname}/zone_accuracy/adjacent"] = float(
                            probe_tracking.branch_zone_wall_accuracy_adjacent[b, start:ptr].mean()
                        )
                        log_dict[f"probe/{bname}/zone_accuracy/distant"] = float(
                            probe_tracking.branch_zone_wall_accuracy_distant[b, start:ptr].mean()
                        )

        # =====================================================================
        # AGENT-CENTRIC METRICS (NOT probe-based)
        # =====================================================================
        if train_state.agent_tracking is not None:
            agent_tracking = train_state.agent_tracking

            if agent_tracking.total_samples > 10:
                agent_novelty, novelty_details = compute_agent_novelty_from_tracking(
                    agent_tracking, window_size=500
                )
                log_dict["agent/novelty"] = agent_novelty
                log_dict["agent/novelty_details/entropy_mean"] = novelty_details.get('mean_entropy', 0.0)
                log_dict["agent/novelty_details/branch_novelty"] = novelty_details.get('branch_novelty', 0.0)

                agent_learnability, learn_details = compute_agent_learnability_from_tracking(
                    agent_tracking, window_size=500
                )
                log_dict["agent/learnability"] = agent_learnability
                log_dict["agent/learnability_details/early_error"] = learn_details.get('early_calibration_error', 0.0)
                log_dict["agent/learnability_details/late_error"] = learn_details.get('late_calibration_error', 0.0)
                log_dict["agent/learnability_details/correlation"] = learn_details.get('value_return_correlation', 0.0)

                agent_oe_score, regime, oe_details = compute_agent_openendedness_from_tracking(
                    agent_tracking, window_size=500
                )
                log_dict["agent/openendedness_score"] = agent_oe_score
                log_dict["agent/regime"] = regime

            log_dict["agent/total_samples"] = int(agent_tracking.total_samples)

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
        self._log_periodic_visualizations(train_state, update_count, log_dict)

        return log_dict

    def log_metrics(self, metrics: dict, train_state: BaseTrainState):
        """Log metrics to wandb. Override _build_log_dict to extend."""
        log_dict = self._build_log_dict(metrics, train_state)
        wandb.log(log_dict)

    def _log_periodic_visualizations(
        self,
        train_state: BaseTrainState,
        update_count: int,
        log_dict: dict,
    ):
        """Log visualizations to wandb. Called every eval."""
        try:
            if not (hasattr(train_state, 'probe_tracking') and train_state.probe_tracking is not None):
                return
            probe_tracking = train_state.probe_tracking
            if probe_tracking.total_samples < 10:
                return

            # Helper for building predictions/levels from tracking state
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

            # --- Wall prediction heatmap (Cat 9) ---
            try:
                wall_heatmap = create_wall_prediction_heatmap(
                    last_preds, last_levels,
                    env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                )
                log_dict["probe/images/wall_prediction_heatmap"] = wandb.Image(
                    wall_heatmap, caption="Wall Prediction from Hidden State"
                )
            except Exception:
                pass

            # --- Position prediction heatmap (Cat 9) ---
            try:
                pos_heatmap = create_position_prediction_heatmap(
                    last_preds, last_levels,
                    env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                )
                log_dict["probe/images/position_prediction_heatmap"] = wandb.Image(
                    pos_heatmap, caption="Position Prediction from Hidden State"
                )
            except Exception:
                pass

            # --- Batch wall prediction summary (Cat 9) ---
            try:
                batch_wall_summary = create_batch_wall_prediction_summary(
                    last_preds, last_levels,
                    env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH,
                    n_samples=4
                )
                log_dict["probe/images/batch_wall_summary"] = wandb.Image(
                    batch_wall_summary, caption="Batch Wall Prediction (shows variance & samples)"
                )
            except Exception:
                pass

            # --- Batch position prediction summary (Cat 9) ---
            try:
                batch_pos_summary = create_batch_position_prediction_summary(
                    last_preds, last_levels,
                    env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                )
                log_dict["probe/images/batch_position_summary"] = wandb.Image(
                    batch_pos_summary, caption="Batch Position Prediction (shows all actuals)"
                )
            except Exception:
                pass

            # --- Matched pairs visualization with greedy matching (Cat 9) ---
            try:
                matched_indices, _ = compute_greedy_matching(
                    last_preds, last_levels,
                    env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                )
                matched_pairs_viz = create_matched_pairs_visualization(
                    last_preds, last_levels, matched_indices,
                    n_pairs=4,
                    env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                )
                log_dict["probe/images/matched_pairs"] = wandb.Image(
                    matched_pairs_viz, caption="Greedy-Matched Prediction/Actual Pairs"
                )
            except Exception:
                pass

            # --- R→M heatmap (when we have R→M data) ---
            if probe_tracking.last_r2m_valid:
                try:
                    r2m_preds = {
                        'wall_logits': probe_tracking.last_r2m_wall_logits,
                        'goal_logits': probe_tracking.last_r2m_goal_logits,
                    }
                    class R2MLevels:
                        def __init__(self, wall_map, goal_pos, agent_pos):
                            self.wall_map = wall_map
                            self.goal_pos = goal_pos
                            self.agent_pos = agent_pos

                    r2m_levels = R2MLevels(
                        wall_map=probe_tracking.last_r2m_wall_map,
                        goal_pos=probe_tracking.last_r2m_goal_pos,
                        agent_pos=probe_tracking.last_r2m_agent_pos,
                    )
                    r2m_heatmap = create_replay_to_mutate_heatmap(
                        r2m_preds, r2m_levels, n_samples=4
                    )
                    log_dict["probe/images/replay_to_mutate_heatmap"] = wandb.Image(
                        r2m_heatmap, caption="Replay→Mutate: Per-Instance Correspondence"
                    )
                except Exception:
                    pass

            # --- Probe loss by branch plot ---
            try:
                branch_plot = create_probe_loss_by_branch_plot(
                    probe_tracking.branch_loss_history,
                    probe_tracking.branch_ptrs,
                )
                log_dict["probe/images/loss_by_branch"] = wandb.Image(
                    branch_plot, caption="Probe Loss by Curriculum Branch"
                )
            except Exception:
                pass

            # --- Information content dashboard ---
            try:
                probe_metrics_for_dashboard = {
                    'wall_dist_accuracy': log_dict.get("probe/dist_accuracy/wall", 0.0),
                    'goal_dist_mode_match': log_dict.get("probe/dist_accuracy/goal_mode_match", 0.0),
                    'agent_pos_dist_mode_match': log_dict.get("probe/dist_accuracy/agent_pos_mode_match", 0.0),
                    'agent_dir_dist_mode_match': log_dict.get("probe/dist_accuracy/agent_dir_mode_match", 0.0),
                    'wall_loss': log_dict.get("probe/dist_loss/wall", 0.0),
                    'goal_loss': log_dict.get("probe/dist_loss/goal", 0.0),
                    'agent_pos_loss': log_dict.get("probe/dist_loss/agent_pos", 0.0),
                    'agent_dir_loss': log_dict.get("probe/dist_loss/agent_dir", 0.0),
                }
                info_dashboard = create_information_content_dashboard(
                    probe_metrics_for_dashboard, probe_tracking,
                    env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                )
                log_dict["probe/images/information_dashboard"] = wandb.Image(
                    info_dashboard, caption="Information Content Dashboard"
                )
            except Exception:
                pass

            # --- Novelty-learnability plot ---
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
                log_dict["probe/images/novelty_learnability"] = wandb.Image(
                    nl_plot, caption="Novelty-Learnability Space"
                )
            except Exception:
                pass

            # --- Correlation scatter plot ---
            if probe_tracking.total_samples > 50:
                try:
                    corr_plot = create_correlation_scatter_plot(
                        probe_tracking.probe_accuracy_history,
                        probe_tracking.agent_returns_history,
                        probe_tracking.total_samples,
                    )
                    log_dict["probe/images/accuracy_return_correlation"] = wandb.Image(
                        corr_plot, caption="Probe Accuracy vs Agent Return"
                    )
                except Exception:
                    pass

            # --- Hidden state t-SNE (expensive, do every ~10 evals) ---
            eval_freq = self.config.get("eval_freq", 250)
            tsne_freq = self.config.get("tsne_freq", 10)
            if update_count % (tsne_freq * eval_freq) == 0:
                if hasattr(train_state, 'hstate_samples') and train_state.hstate_samples is not None:
                    n_samples = int(train_state.hstate_sample_ptr)
                    if n_samples >= 50:
                        try:
                            tsne_plot = create_hidden_state_tsne_plot(
                                train_state.hstate_samples[:n_samples],
                                train_state.hstate_sample_branches[:n_samples],
                                max_samples=500,
                            )
                            log_dict["probe/images/hidden_state_tsne"] = wandb.Image(
                                tsne_plot, caption="Hidden State t-SNE by Branch"
                            )
                        except Exception:
                            pass

            # --- Pareto trajectory ---
            if hasattr(train_state, 'pareto_history') and train_state.pareto_history is not None:
                if train_state.pareto_history.num_checkpoints > 1:
                    try:
                        pareto_plot = create_pareto_trajectory_plot(train_state.pareto_history)
                        log_dict["probe/images/pareto_trajectory"] = wandb.Image(
                            pareto_plot, caption="Novelty-Learnability Trajectory"
                        )
                    except Exception:
                        pass

        except Exception as e:
            print(f"Warning: Failed to create visualizations: {e}")
