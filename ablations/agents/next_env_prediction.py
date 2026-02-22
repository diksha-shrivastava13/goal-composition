"""
Next Environment Prediction Agent - Upper-bound Baseline.

Agent with explicit curriculum info fed INTO the network backbone.
This represents the theoretical upper-bound: if an agent had perfect
knowledge of the curriculum structure, how well could it predict
and leverage this information?

Key differences from probe agents:
1. CurriculumState tracks cross-episode history
2. Curriculum features fed INTO the LSTM backbone
3. Prediction loss computed (gradients flow through backbone)
4. Uses NoveltyLearnabilityState for tracking open-endedness

AGENT-CENTRIC DESIGN:
- This agent HAS integrated prediction (unlike probe agents)
- Predictions are PART of the agent (gradients flow through backbone)
- This is the upper-bound: explicit curriculum awareness
- No separate probe needed - prediction is built-in
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import optax
import chex
import wandb
import numpy as np

from ..common.types import (
    NextEnvPredictionTrainState,
    UpdateState,
    DEFAULT_ENV_HEIGHT,
    DEFAULT_ENV_WIDTH,
    create_agent_tracking_state,
)
from ..common.networks import ActorCriticWithCurriculumPrediction
from ..common.curriculum_state import (
    CurriculumState,
    NoveltyLearnabilityState,
    create_curriculum_state,
    create_novelty_learnability_state,
    update_curriculum_state,
    get_curriculum_features,
    update_novelty_learnability_state,
    compute_learnability as compute_learnability_from_nl_state,
    compute_novelty as compute_novelty_from_nl_state,
    compute_openendedness_score as compute_openendedness_from_nl_state,
)
from ..common.metrics import (
    compute_curriculum_prediction_loss,
    compute_calibration_metrics,
    compute_distribution_divergence,
)
from ..common.visualization import (
    log_curriculum_prediction_metrics,
    create_prediction_comparison_figure,
    create_pareto_frontier_figure,
    create_curriculum_trajectory_plot,
    create_wall_prediction_heatmap,
    create_position_prediction_heatmap,
)
from ..common.training import compute_gae
from ..common.environment import compute_score
from ..common.utils import train_state_to_log_dict
from .base import BaseAgent

from jaxued.utils import compute_max_returns


class NextEnvPredictionAgent(BaseAgent):
    """
    Agent with explicit curriculum info (upper-bound baseline).

    This agent receives curriculum history features as input to its
    network backbone. It also has a prediction head that learns to
    predict the next level, with gradients flowing through the shared
    backbone.

    Key architecture:
    - ActorCriticWithCurriculumPrediction network
    - CurriculumState for cross-episode memory
    - NoveltyLearnabilityState for open-endedness tracking

    AGENT-CENTRIC DESIGN:
    - Curriculum prediction is PART of the agent (not external probe)
    - Prediction head shares backbone with policy/value heads
    - No separate ProbeRunner needed - predictions are integrated
    - This is the upper-bound for what's possible with explicit info
    """

    def get_actor_critic_class(self) -> type:
        return ActorCriticWithCurriculumPrediction

    def initialize_hidden_state(self, batch_size: int) -> chex.ArrayTree:
        return ActorCriticWithCurriculumPrediction.initialize_carry((batch_size,))

    def create_train_state(self, rng: chex.PRNGKey) -> NextEnvPredictionTrainState:
        """Create train state with curriculum prediction components."""
        config = self.config
        rng, rng_net = jax.random.split(rng)

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

        # Dummy curriculum features for initialization
        history_length = config.get("curriculum_history_length", 64)
        dummy_curriculum_state = create_curriculum_state(history_length=history_length)
        dummy_curriculum_features = get_curriculum_features(dummy_curriculum_state)

        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
        network = ActorCriticWithCurriculumPrediction(
            action_dim=self.env.action_space(self.env_params).n,
            env_height=DEFAULT_ENV_HEIGHT,
            env_width=DEFAULT_ENV_WIDTH,
        )
        network_params = network.init(
            rng_net,
            init_x,
            self.initialize_hidden_state(config["num_train_envs"]),
            curriculum_features=dummy_curriculum_features,
            predict_curriculum=True,
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

        # Initialize curriculum state
        curriculum_state = create_curriculum_state(
            history_length=config.get("curriculum_history_length", 64)
        )

        # Initialize novelty/learnability tracking
        nl_state = create_novelty_learnability_state(
            history_lengths=[8, 16, 32, 64],
            buffer_size=config.get("nl_buffer_size", 100),
        )

        # Initialize DR continuous rollout state
        rng, rng_dr = jax.random.split(rng)
        dr_hstate, dr_obs, dr_env_state = self.initialize_dr_state(rng_dr)

        return NextEnvPredictionTrainState.create(
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
            curriculum_state=curriculum_state,
            nl_state=nl_state,
            pred_metrics_accumulator=None,
            last_hstate=dr_hstate,
            last_obs=dr_obs,
            last_env_state=dr_env_state,
        )

    def sample_trajectories_with_curriculum(
        self,
        rng: chex.PRNGKey,
        train_state: NextEnvPredictionTrainState,
        init_hstate: chex.ArrayTree,
        init_obs,
        init_env_state,
        curriculum_features: chex.Array,
        num_envs: int,
        num_steps: int,
    ):
        """
        Sample trajectories with curriculum features fed to the network.

        This is the key difference from standard trajectory sampling:
        the curriculum features are passed to the network at each step.
        """
        config = self.config

        def sample_step(carry, _):
            rng, hstate, obs, env_state, last_done = carry
            rng, rng_action, rng_step = jax.random.split(rng, 3)

            # Add time dimension for network
            x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, last_done))

            # Pass curriculum features to network
            hstate, pi, value, _ = train_state.apply_fn(
                train_state.params, x, hstate,
                curriculum_features=curriculum_features,
                predict_curriculum=False,  # Don't need predictions during rollout
            )

            action = pi.sample(seed=rng_action)
            log_prob = pi.log_prob(action)
            value, action, log_prob = (
                value.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
            )

            # Step environment
            next_obs, env_state, reward, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(jax.random.split(rng_step, num_envs), env_state, action, self.env_params)

            carry = (rng, hstate, next_obs, env_state, done)
            return carry, (obs, action, reward, done, log_prob, value, info)

        (rng, hstate, last_obs, last_env_state, last_done), traj = jax.lax.scan(
            sample_step,
            (rng, init_hstate, init_obs, init_env_state, jnp.zeros(num_envs, dtype=bool)),
            None,
            length=num_steps,
        )

        # Get final value estimate
        x = jax.tree_util.tree_map(lambda x: x[None, ...], (last_obs, last_done))
        _, _, last_value, _ = train_state.apply_fn(
            train_state.params, x, hstate,
            curriculum_features=curriculum_features,
            predict_curriculum=False,
        )

        return (rng, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj

    def update_actor_critic_with_curriculum(
        self,
        rng: chex.PRNGKey,
        train_state: NextEnvPredictionTrainState,
        init_hstate: chex.ArrayTree,
        batch: chex.ArrayTree,
        curriculum_features: chex.Array,
        num_envs: int,
        n_steps: int,
        n_minibatch: int,
        n_epochs: int,
        clip_eps: float,
        entropy_coeff: float,
        critic_coeff: float,
        update_grad: bool = True,
    ) -> Tuple[Tuple[chex.PRNGKey, NextEnvPredictionTrainState], chex.ArrayTree]:
        """
        PPO update with curriculum features passed to the network.
        """
        obs, actions, dones, log_probs, values, targets, advantages = batch
        last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
        batch = obs, actions, last_dones, log_probs, values, targets, advantages

        def update_epoch(carry, _):
            def update_minibatch(train_state, minibatch):
                init_hstate_mb, obs, actions, last_dones, log_probs, values, targets, advantages = minibatch

                def loss_fn(params):
                    _, pi, values_pred, _ = train_state.apply_fn(
                        params, (obs, last_dones), init_hstate_mb,
                        curriculum_features=curriculum_features,
                        predict_curriculum=False,
                    )
                    log_probs_pred = pi.log_prob(actions)
                    entropy = pi.entropy().mean()

                    ratio = jnp.exp(log_probs_pred - log_probs)
                    A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                    l_clip = (-jnp.minimum(
                        ratio * A,
                        jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A
                    )).mean()

                    values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                    l_vf = 0.5 * jnp.maximum(
                        (values_pred - targets) ** 2,
                        (values_pred_clipped - targets) ** 2
                    ).mean()

                    loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy
                    return loss, (l_vf, l_clip, entropy)

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                loss, grads = grad_fn(train_state.params)
                if update_grad:
                    train_state = train_state.apply_gradients(grads=grads)
                return train_state, loss

            rng, train_state = carry
            rng, rng_perm = jax.random.split(rng)
            permutation = jax.random.permutation(rng_perm, num_envs)

            minibatches = (
                jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0)
                    .reshape(n_minibatch, -1, *x.shape[1:]),
                    init_hstate,
                ),
                *jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1)
                    .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                    .swapaxes(0, 1),
                    batch,
                ),
            )
            train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
            return (rng, train_state), losses

        return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)

    def compute_prediction_loss_and_update(
        self,
        train_state: NextEnvPredictionTrainState,
        curriculum_features: chex.Array,
        levels,  # The actual levels that were used
        branch: int,
    ) -> Tuple[NextEnvPredictionTrainState, dict]:
        """
        Compute prediction loss and update novelty/learnability tracking.

        Unlike probe agents where gradients don't affect the main network,
        here we compute the prediction loss but don't apply gradients separately
        (they're already included in the PPO update through shared backbone).
        """
        config = self.config

        # Get predictions from the network
        dummy_obs = jax.tree_util.tree_map(
            lambda x: x[None, None, ...], self.env.reset_to_level(
                jax.random.PRNGKey(0),
                self.sample_random_level(jax.random.PRNGKey(0)),
                self.env_params
            )[0]
        )
        dummy_dones = jnp.zeros((1, 1), dtype=bool)
        dummy_hidden = self.initialize_hidden_state(1)

        _, _, _, predictions = train_state.apply_fn(
            train_state.params,
            (dummy_obs, dummy_dones),
            dummy_hidden,
            curriculum_features=curriculum_features,
            predict_curriculum=True,
        )

        # Compute prediction loss for the first level in batch
        level_0 = jax.tree_util.tree_map(lambda x: x[0], levels)
        pred_loss, pred_metrics = compute_curriculum_prediction_loss(
            predictions,
            level_0,
            wall_weight=config.get("curriculum_wall_weight", 1.0),
            goal_weight=config.get("curriculum_goal_weight", 1.0),
            agent_pos_weight=config.get("curriculum_agent_pos_weight", 1.0),
            agent_dir_weight=config.get("curriculum_agent_dir_weight", 1.0),
        )

        # Compute calibration metrics
        calibration = compute_calibration_metrics(predictions, level_0)
        pred_metrics.update(calibration)
        pred_metrics["branch"] = branch

        # Update novelty/learnability tracking
        nl_state = train_state.nl_state
        if nl_state is not None:
            nl_state = update_novelty_learnability_state(
                nl_state,
                pred_loss,
                history_length_idx=-1,  # Use full history
            )

        train_state = train_state.replace(nl_state=nl_state)

        return train_state, pred_metrics

    def on_new_levels(
        self,
        rng: chex.PRNGKey,
        train_state: NextEnvPredictionTrainState,
    ) -> Tuple[Tuple[chex.PRNGKey, NextEnvPredictionTrainState], dict]:
        """DR branch: sample new random levels with curriculum prediction."""
        config = self.config
        sampler = train_state.sampler

        # Get curriculum features
        curriculum_features = get_curriculum_features(
            train_state.curriculum_state,
            max_training_steps=config["num_updates"],
            max_buffer_capacity=config["level_buffer_capacity"],
        )

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

        # Get hidden state
        init_hstate = self.get_hidden_state_for_rollout(train_state, branch=0)

        # Rollout with curriculum features
        (rng, final_hstate, last_obs, last_env_state, last_value), traj = \
            self.sample_trajectories_with_curriculum(
                rng, train_state, init_hstate, init_obs, init_env_state,
                curriculum_features, config["num_train_envs"], config["num_steps"],
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

        # Update policy
        (rng, train_state), losses = self.update_actor_critic_with_curriculum(
            rng, train_state, init_hstate,
            (obs, actions, dones, log_probs, values, targets, advantages),
            curriculum_features,
            config["num_train_envs"], config["num_steps"],
            config["num_minibatches"], config["epoch_ppo"],
            config["clip_eps"], config["entropy_coeff"], config["critic_coeff"],
            update_grad=config["exploratory_grad_updates"],
        )

        # Compute prediction loss and update tracking
        train_state, pred_metrics = self.compute_prediction_loss_and_update(
            train_state, curriculum_features, new_levels, branch=0
        )

        # Update curriculum state
        sampler_stats = {
            'size': sampler['size'],
            'mean_score': sampler['scores'].mean(),
            'max_score': sampler['scores'].max(),
        }
        level_0 = jax.tree_util.tree_map(lambda x: x[0], new_levels)
        curriculum_state = update_curriculum_state(
            train_state.curriculum_state,
            level_0,
            branch=0,
            score=scores[0],
            sampler_stats=sampler_stats,
        )

        metrics = {
            "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
            "mean_num_blocks": new_levels.wall_map.sum() / config["num_train_envs"],
            "pred_metrics": pred_metrics,
        }

        train_state = train_state.replace(
            sampler=sampler,
            update_state=UpdateState.DR,
            num_dr_updates=train_state.num_dr_updates + 1,
            dr_last_level_batch=new_levels,
            training_step=train_state.training_step + 1,
            curriculum_state=curriculum_state,
        )

        return (rng, train_state), metrics

    def on_replay_levels(
        self,
        rng: chex.PRNGKey,
        train_state: NextEnvPredictionTrainState,
    ) -> Tuple[Tuple[chex.PRNGKey, NextEnvPredictionTrainState], dict]:
        """Replay branch: sample from buffer with curriculum prediction."""
        config = self.config
        sampler = train_state.sampler

        # Get curriculum features
        curriculum_features = get_curriculum_features(
            train_state.curriculum_state,
            max_training_steps=config["num_updates"],
            max_buffer_capacity=config["level_buffer_capacity"],
        )

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
            self.sample_trajectories_with_curriculum(
                rng, train_state, init_hstate, init_obs, init_env_state,
                curriculum_features, config["num_train_envs"], config["num_steps"],
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

        (rng, train_state), losses = self.update_actor_critic_with_curriculum(
            rng, train_state, init_hstate,
            (obs, actions, dones, log_probs, values, targets, advantages),
            curriculum_features,
            config["num_train_envs"], config["num_steps"],
            config["num_minibatches"], config["epoch_ppo"],
            config["clip_eps"], config["entropy_coeff"], config["critic_coeff"],
            update_grad=True,
        )

        # Compute prediction loss and update tracking
        train_state, pred_metrics = self.compute_prediction_loss_and_update(
            train_state, curriculum_features, levels, branch=1
        )

        # Update curriculum state
        sampler_stats = {
            'size': sampler['size'],
            'mean_score': sampler['scores'].mean(),
            'max_score': sampler['scores'].max(),
        }
        level_0 = jax.tree_util.tree_map(lambda x: x[0], levels)
        curriculum_state = update_curriculum_state(
            train_state.curriculum_state,
            level_0,
            branch=1,
            score=scores[0],
            sampler_stats=sampler_stats,
        )

        metrics = {
            "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
            "mean_num_blocks": levels.wall_map.sum() / config["num_train_envs"],
            "pred_metrics": pred_metrics,
        }

        train_state = train_state.replace(
            sampler=sampler,
            update_state=UpdateState.REPLAY,
            num_replay_updates=train_state.num_replay_updates + 1,
            replay_last_level_batch=levels,
            training_step=train_state.training_step + 1,
            curriculum_state=curriculum_state,
        )

        return (rng, train_state), metrics

    def on_mutate_levels(
        self,
        rng: chex.PRNGKey,
        train_state: NextEnvPredictionTrainState,
    ) -> Tuple[Tuple[chex.PRNGKey, NextEnvPredictionTrainState], dict]:
        """Mutate branch: mutate previous replay levels with curriculum prediction."""
        config = self.config
        sampler = train_state.sampler

        # Get curriculum features
        curriculum_features = get_curriculum_features(
            train_state.curriculum_state,
            max_training_steps=config["num_updates"],
            max_buffer_capacity=config["level_buffer_capacity"],
        )

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
            self.sample_trajectories_with_curriculum(
                rng, train_state, init_hstate, init_obs, init_env_state,
                curriculum_features, config["num_train_envs"], config["num_steps"],
            )

        obs, actions, rewards, dones, log_probs, values, info = traj

        advantages, targets = compute_gae(
            config["gamma"], config["gae_lambda"],
            last_value, values, rewards, dones,
        )
        max_returns = compute_max_returns(dones, rewards)
        scores = compute_score(config, dones, values, max_returns, advantages)
        sampler, _ = self.level_sampler.insert_batch(sampler, child_levels, scores, {"max_return": max_returns})

        (rng, train_state), losses = self.update_actor_critic_with_curriculum(
            rng, train_state, init_hstate,
            (obs, actions, dones, log_probs, values, targets, advantages),
            curriculum_features,
            config["num_train_envs"], config["num_steps"],
            config["num_minibatches"], config["epoch_ppo"],
            config["clip_eps"], config["entropy_coeff"], config["critic_coeff"],
            update_grad=config["exploratory_grad_updates"],
        )

        # Compute prediction loss and update tracking
        train_state, pred_metrics = self.compute_prediction_loss_and_update(
            train_state, curriculum_features, child_levels, branch=2
        )

        # Update curriculum state
        sampler_stats = {
            'size': sampler['size'],
            'mean_score': sampler['scores'].mean(),
            'max_score': sampler['scores'].max(),
        }
        level_0 = jax.tree_util.tree_map(lambda x: x[0], child_levels)
        curriculum_state = update_curriculum_state(
            train_state.curriculum_state,
            level_0,
            branch=2,
            score=scores[0],
            sampler_stats=sampler_stats,
        )

        metrics = {
            "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
            "mean_num_blocks": child_levels.wall_map.sum() / config["num_train_envs"],
            "pred_metrics": pred_metrics,
        }

        train_state = train_state.replace(
            sampler=sampler,
            update_state=UpdateState.MUTATION,
            num_mutation_updates=train_state.num_mutation_updates + 1,
            mutation_last_level_batch=child_levels,
            training_step=train_state.training_step + 1,
            curriculum_state=curriculum_state,
        )

        return (rng, train_state), metrics

    def on_dr_step(
        self,
        rng: chex.PRNGKey,
        train_state: NextEnvPredictionTrainState,
    ) -> Tuple[Tuple[chex.PRNGKey, NextEnvPredictionTrainState], dict]:
        """
        DR training step with curriculum prediction.

        Overrides base on_dr_step to use curriculum-aware sampling and update.
        Continuous rollouts with auto-reset, same as base, but threads
        curriculum_features through the network.
        """
        config = self.config

        # Get curriculum features
        curriculum_features = get_curriculum_features(
            train_state.curriculum_state,
            max_training_steps=config["num_updates"],
            max_buffer_capacity=config["level_buffer_capacity"],
        )

        # Continue from previous state
        init_hstate = train_state.last_hstate
        init_obs = train_state.last_obs
        init_env_state = train_state.last_env_state

        # Rollout with curriculum features
        (rng, final_hstate, last_obs, last_env_state, last_value), traj = \
            self.sample_trajectories_with_curriculum(
                rng, train_state, init_hstate, init_obs, init_env_state,
                curriculum_features, config["num_train_envs"], config["num_steps"],
            )

        obs, actions, rewards, dones, log_probs, values, info = traj

        # Compute advantages
        advantages, targets = compute_gae(
            config["gamma"], config["gae_lambda"],
            last_value, values, rewards, dones,
        )

        # Update policy with curriculum features
        (rng, train_state), losses = self.update_actor_critic_with_curriculum(
            rng, train_state, init_hstate,
            (obs, actions, dones, log_probs, values, targets, advantages),
            curriculum_features,
            config["num_train_envs"], config["num_steps"],
            config["num_minibatches"], config["epoch_ppo"],
            config["clip_eps"], config["entropy_coeff"], config["critic_coeff"],
            update_grad=True,
        )

        # Update hidden state for continuity
        train_state = self.update_hidden_state_after_rollout(train_state, final_hstate, branch=0)

        # Get levels from env state for prediction tracking (if available)
        current_levels = info.get("level", None) if isinstance(info, dict) else None

        # Compute prediction loss and update tracking (if levels available)
        pred_metrics = {}
        if current_levels is not None:
            train_state, pred_metrics = self.compute_prediction_loss_and_update(
                train_state, curriculum_features, current_levels, branch=0
            )

            # Update curriculum state
            sampler_stats = {
                'size': jnp.int32(0),
                'mean_score': jnp.float32(0.0),
                'max_score': jnp.float32(0.0),
            }
            level_0 = jax.tree_util.tree_map(lambda x: x[0], current_levels)
            curriculum_state = update_curriculum_state(
                train_state.curriculum_state,
                level_0,
                branch=0,
                score=jnp.float32(0.0),
                sampler_stats=sampler_stats,
            )
            train_state = train_state.replace(curriculum_state=curriculum_state)

        metrics = {
            "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
            "mean_return": rewards.sum(axis=0).mean(),
            "pred_metrics": pred_metrics,
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

    def _build_log_dict(self, metrics: dict, train_state: NextEnvPredictionTrainState) -> dict:
        """Extend base log dict with curriculum prediction metrics."""
        log_dict = super()._build_log_dict(metrics, train_state)

        update_count = metrics["update_count"]

        # Curriculum prediction metrics from NoveltyLearnabilityState
        if hasattr(train_state, 'nl_state') and train_state.nl_state is not None:
            nl_state = train_state.nl_state

            learnability, learn_details = compute_learnability_from_nl_state(nl_state)
            novelty, novelty_details = compute_novelty_from_nl_state(nl_state)
            oe_score, regime = compute_openendedness_from_nl_state(novelty, learnability)

            log_dict["curriculum_pred/openendedness/novelty"] = novelty
            log_dict["curriculum_pred/openendedness/learnability"] = learnability
            log_dict["curriculum_pred/openendedness/score"] = oe_score
            # Also log under old keys for backward compat
            log_dict["novelty_learnability/novelty"] = novelty
            log_dict["novelty_learnability/learnability"] = learnability
            log_dict["novelty_learnability/openendedness_score"] = oe_score
            regime_map = {"open-ended": 3, "chaotic": 2, "converging": 1, "stagnant": 0}
            log_dict["novelty_learnability/regime_code"] = regime_map.get(regime, -1)

            for key, value in learn_details.items():
                log_dict[f"novelty_learnability/{key}"] = value
            for key, value in novelty_details.items():
                log_dict[f"novelty_learnability/{key}"] = value

        # Curriculum state metrics
        if hasattr(train_state, 'curriculum_state') and train_state.curriculum_state is not None:
            cs = train_state.curriculum_state
            log_dict["curriculum_state/training_step"] = int(cs.training_step)
            log_dict["curriculum_state/buffer_size"] = int(cs.buffer_size)
            log_dict["curriculum_state/buffer_mean_score"] = float(cs.buffer_mean_score)
            log_dict["curriculum_state/total_replay_steps"] = int(cs.total_replay_steps)
            log_dict["curriculum_state/total_random_steps"] = int(cs.total_random_steps)
            log_dict["curriculum_state/replay_fraction"] = float(
                cs.total_replay_steps / max(int(cs.training_step), 1)
            )

            # Rolling statistics
            if cs.head_pointer > 0 or cs.history_filled:
                log_dict["curriculum_state/mean_wall_density"] = float(cs.recent_wall_densities.mean())
                log_dict["curriculum_state/recent_mean_score"] = float(cs.recent_scores.mean())

        # Per-branch prediction metrics from train step
        pred_metrics = metrics.get("pred_metrics", {})
        if pred_metrics:
            branch = int(pred_metrics.get("branch", 0))
            formatted = log_curriculum_prediction_metrics(
                pred_metrics, branch, update_count,
            )
            log_dict.update(formatted)

        # Curriculum prediction visualizations, calibration, and divergence
        self._log_curriculum_visualizations(train_state, update_count, log_dict)

        return log_dict

    def _log_curriculum_visualizations(
        self,
        train_state: NextEnvPredictionTrainState,
        update_count: int,
        log_dict: dict,
    ):
        """Log curriculum prediction visualizations, calibration, and divergence."""
        try:
            # Curriculum trajectory plot
            if hasattr(train_state, 'curriculum_state') and train_state.curriculum_state is not None:
                cs = train_state.curriculum_state

                try:
                    traj_plot = create_curriculum_trajectory_plot(cs)
                    log_dict["curriculum_pred/trajectory"] = wandb.Image(traj_plot)
                except Exception:
                    pass

                # --- Wall and position prediction heatmaps (Cat 10) ---
                try:
                    curriculum_features = get_curriculum_features(cs)

                    # Use last replay level as representative
                    representative_level = jax.tree_util.tree_map(
                        lambda x: x[0], train_state.replay_last_level_batch
                    )

                    # Create dummy observation for forward pass to get predictions
                    rng_viz = jax.random.PRNGKey(update_count)
                    dummy_level = self.sample_random_level(rng_viz)
                    dummy_obs, _ = self.eval_env.reset_to_level(rng_viz, dummy_level, self.env_params)
                    dummy_obs = jax.tree_util.tree_map(lambda x: x[None, None, ...], dummy_obs)
                    dummy_dones = jnp.zeros((1, 1), dtype=bool)
                    dummy_hidden = self.initialize_hidden_state(1)

                    _, _, _, predictions = train_state.apply_fn(
                        train_state.params,
                        (dummy_obs, dummy_dones),
                        dummy_hidden,
                        curriculum_features=curriculum_features,
                        predict_curriculum=True,
                    )

                    # Wall prediction heatmap
                    wall_heatmap = create_wall_prediction_heatmap(
                        predictions, representative_level,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["curriculum_pred/images/wall_heatmap"] = wandb.Image(
                        wall_heatmap, caption=f"Wall Predictions (step {update_count})"
                    )

                    # Position prediction heatmap
                    pos_heatmap = create_position_prediction_heatmap(
                        predictions, representative_level,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["curriculum_pred/images/position_heatmap"] = wandb.Image(
                        pos_heatmap, caption=f"Position Predictions (step {update_count})"
                    )

                    # --- Calibration metrics (Cat 10) ---
                    cal_metrics = compute_calibration_metrics(
                        predictions, representative_level,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["curriculum_pred/calibration/wall_ece"] = float(cal_metrics["wall_ece"])
                    log_dict["curriculum_pred/calibration/goal_prob_at_actual"] = float(cal_metrics["goal_prob_at_actual"])
                    log_dict["curriculum_pred/calibration/agent_pos_prob_at_actual"] = float(cal_metrics["agent_pos_prob_at_actual"])
                    log_dict["curriculum_pred/calibration/agent_dir_prob_at_actual"] = float(cal_metrics["agent_dir_prob_at_actual"])

                    # --- Divergence metrics using sampler levels (Cat 10) ---
                    sampler = train_state.sampler
                    if sampler["size"] >= 10:
                        top_indices = jnp.argsort(sampler["scores"])[-10:]
                        batch_levels = self.level_sampler.get_levels(sampler, top_indices)
                        div_metrics = compute_distribution_divergence(
                            predictions, batch_levels,
                            env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                        )
                        log_dict["curriculum_pred/divergence/goal_kl"] = float(div_metrics["goal_kl"])
                        log_dict["curriculum_pred/divergence/goal_js"] = float(div_metrics["goal_js"])
                        log_dict["curriculum_pred/divergence/wall_density_error"] = float(div_metrics["wall_density_error"])
                        log_dict["curriculum_pred/divergence/predicted_wall_density"] = float(div_metrics["predicted_wall_density"])
                        log_dict["curriculum_pred/divergence/empirical_wall_density"] = float(div_metrics["empirical_wall_density"])

                except Exception as e:
                    pass

            # Pareto frontier plot from NL state
            if hasattr(train_state, 'nl_state') and train_state.nl_state is not None:
                nl_state = train_state.nl_state
                if nl_state.total_samples > 10:
                    try:
                        n_samples = min(nl_state.total_samples, nl_state.loss_over_time.shape[0])
                        losses = np.array(nl_state.loss_over_time[:n_samples])

                        window = 20
                        if n_samples >= window * 2:
                            novelty_history = []
                            learnability_history = []
                            update_steps = []

                            for i in range(window, n_samples, window):
                                early_loss = losses[i-window:i-window//2].mean()
                                late_loss = losses[i-window//2:i].mean()
                                learnability_history.append(early_loss - late_loss)

                                recent = losses[max(0, i-window):i]
                                if len(recent) > 1:
                                    trend = np.polyfit(np.arange(len(recent)), recent, 1)[0]
                                    novelty_history.append(trend)
                                else:
                                    novelty_history.append(0.0)
                                update_steps.append(i)

                            if len(novelty_history) > 2:
                                pareto_plot = create_pareto_frontier_figure(
                                    np.array(novelty_history),
                                    np.array(learnability_history),
                                    np.array(update_steps),
                                )
                                log_dict["novelty_learnability/pareto_frontier"] = wandb.Image(pareto_plot)
                    except Exception:
                        pass

        except Exception as e:
            print(f"Warning: Failed to create curriculum visualizations: {e}")
