"""
PAIRED + Next Environment Prediction Agent.

PAIRED agent with integrated curriculum prediction head sharing the backbone.
In PAIRED, the "curriculum" is the adversary's level generation policy, so
curriculum features track adversary-generated level statistics.

Unlike the basic PAIRED agents, this variant:
1. Threads curriculum_features through protagonist/antagonist rollouts
2. Uses ActorCriticWithCurriculumPrediction (4-output: hstate, pi, value, predictions)
3. Applies curriculum prediction gradient after PPO updates
4. Tracks curriculum state and novelty/learnability metrics

This mirrors the non-PAIRED NextEnvPredictionAgent but adapted for the
three-agent PAIRED paradigm.
"""

import jax
import jax.numpy as jnp
import chex
import wandb
import numpy as np
from typing import Tuple, Optional

from jaxued.environments.underspecified_env import EnvParams

from ..common.networks import ActorCriticWithCurriculumPrediction
from ..common.curriculum_state import (
    CurriculumState,
    create_curriculum_state,
    update_curriculum_state,
    get_curriculum_features,
    NoveltyLearnabilityState,
    create_novelty_learnability_state,
    update_novelty_learnability_state,
    compute_learnability as compute_learnability_from_nl_state,
    compute_novelty as compute_novelty_from_nl_state,
    compute_openendedness_score as compute_openendedness_from_nl_state,
)
from ..common.training import (
    sample_trajectories_rnn_with_curriculum,
    compute_gae,
)
from ..common.metrics import (
    compute_calibration_metrics,
    compute_distribution_divergence,
    compute_batch_displacement_metrics,
)
from ..common.visualization import (
    create_wall_prediction_heatmap,
    create_position_prediction_heatmap,
    create_curriculum_trajectory_plot,
    create_pareto_frontier_figure,
)
from ..common.types import create_agent_tracking_state, DEFAULT_ENV_HEIGHT, DEFAULT_ENV_WIDTH
from .paired_base import PAIREDBaseAgent, PAIREDTrainState


class PAIREDNextEnvPredictionAgent(PAIREDBaseAgent):
    """
    PAIRED agent with integrated curriculum prediction network.

    Uses ActorCriticWithCurriculumPrediction which has a prediction head
    sharing the backbone with the policy. Curriculum features are built
    from adversary-generated level statistics and threaded through the
    protagonist and antagonist rollouts.

    Memory state stores:
    - curriculum_state: Tracks adversary-generated level statistics
    - nl_state: Tracks prediction loss for novelty/learnability metrics
    """

    def get_actor_critic_class(self) -> type:
        """Use ActorCriticWithCurriculumPrediction."""
        return ActorCriticWithCurriculumPrediction

    def initialize_hidden_state(self, batch_size: int) -> chex.ArrayTree:
        """Initialize LSTM hidden state to zeros."""
        return ActorCriticWithCurriculumPrediction.initialize_carry((batch_size,))

    def create_train_state(self, rng: chex.PRNGKey) -> PAIREDTrainState:
        """Create train state with curriculum tracking in memory_state."""
        train_state = super().create_train_state(rng)

        # Initialize curriculum tracking state
        curriculum_state = create_curriculum_state()
        nl_state = create_novelty_learnability_state()
        memory = {
            "curriculum_state": curriculum_state,
            "nl_state": nl_state,
        }

        return PAIREDTrainState(
            update_count=train_state.update_count,
            pro_train_state=train_state.pro_train_state,
            ant_train_state=train_state.ant_train_state,
            adv_train_state=train_state.adv_train_state,
            memory_state=memory,
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

    def _get_curriculum_features(self, train_state: PAIREDTrainState) -> Optional[chex.Array]:
        """Extract curriculum features from memory state."""
        if train_state.memory_state is None:
            return None
        curriculum_state = train_state.memory_state.get("curriculum_state")
        if curriculum_state is None:
            return None
        return get_curriculum_features(curriculum_state)

    def _rollout(
        self,
        rng: chex.PRNGKey,
        env,
        env_params: EnvParams,
        train_state,
        init_hstate: chex.ArrayTree,
        levels,
        num_steps: int,
        prefix: str,
        context: Optional[chex.Array] = None,
        curriculum_features: Optional[chex.Array] = None,
    ) -> Tuple[Tuple, dict]:
        """Run rollout with curriculum features passed to the network.

        Overrides PAIREDBaseAgent._rollout to use
        sample_trajectories_rnn_with_curriculum when curriculum_features
        are provided. Falls back to standard rollout for the adversary
        (which doesn't use curriculum features).
        """
        config = self.config

        rng, rng_reset = jax.random.split(rng)
        init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
            jax.random.split(rng_reset, config["num_train_envs"]),
            levels,
            env_params,
        )

        if curriculum_features is not None:
            # Use curriculum-aware trajectory sampling
            (
                (rng, final_hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn_with_curriculum(
                rng, env, env_params, train_state,
                init_hstate, init_obs, init_env_state,
                config["num_train_envs"], num_steps,
                curriculum_features,
            )
        elif context is not None:
            # Fall back to context-based sampling (shouldn't happen for this agent)
            from ..common.training import sample_trajectories_rnn_with_context
            (
                (rng, final_hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn_with_context(
                rng, env, env_params, train_state,
                init_hstate, init_obs, init_env_state,
                config["num_train_envs"], num_steps, context,
            )
        else:
            # Standard sampling (used for adversary rollout)
            from ..common.training import sample_trajectories_rnn
            (
                (rng, final_hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng, env, env_params, train_state,
                init_hstate, init_obs, init_env_state,
                config["num_train_envs"], num_steps,
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
            "values": values,
            "last_value": last_value,
            "last_env_state": last_env_state,
            "final_hstate": final_hstate,
        }

        return rollout, extras

    def train_step(self, carry, _):
        """Override train_step to thread curriculum features through rollouts.

        This is the key difference: we extract curriculum_features from memory_state
        and pass them to protagonist/antagonist rollouts via the overridden _rollout.
        """
        from ..common.training import update_actor_critic_rnn
        from ..common.networks import AdversaryActorCritic
        from jaxued.utils import compute_max_mean_returns_epcount

        rng, train_state = carry
        config = self.config
        rng, *step_rngs = jax.random.split(rng, 10)

        pro_train_state = train_state.pro_train_state
        ant_train_state = train_state.ant_train_state
        adv_train_state = train_state.adv_train_state

        # Get curriculum features BEFORE rollouts
        curriculum_features = self._get_curriculum_features(train_state)

        # 1. Adversary rollout (no curriculum features — adversary uses standard network)
        empty_levels = jax.tree_util.tree_map(
            lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0),
            self.sample_empty_level()
        )

        adv_rollout, adv_extras = self._rollout(
            step_rngs[0], self.adv_env, self.adv_env_params,
            adv_train_state,
            AdversaryActorCritic.initialize_carry((config["num_train_envs"],)),
            empty_levels,
            config.get("adv_num_steps", 50),
            "adv_"
        )

        levels = adv_extras["last_env_state"].level

        # Compute displacement from previous adversary-generated levels
        prev_levels = train_state.last_adversary_level if train_state.last_adversary_level is not None else levels
        displacement = compute_batch_displacement_metrics(levels, prev_levels)

        # 2. Protagonist rollout WITH curriculum features
        student_init_hstate = self._get_student_init_hstate(train_state)
        pro_rollout, pro_extras = self._rollout(
            step_rngs[1], self.env, self.env_params,
            pro_train_state,
            student_init_hstate,
            levels,
            config.get("student_num_steps", 256),
            "student_",
            curriculum_features=curriculum_features,
        )
        pro_mean_returns, pro_max_returns, pro_eps = compute_max_mean_returns_epcount(
            pro_extras["dones"], pro_extras["rewards"]
        )

        # 3. Antagonist rollout WITH curriculum features
        ant_rollout, ant_extras = self._rollout(
            step_rngs[2], self.env, self.env_params,
            ant_train_state,
            student_init_hstate,
            levels,
            config.get("student_num_steps", 256),
            "student_",
            curriculum_features=curriculum_features,
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

        # 5. PPO updates for all three networks
        (_, pro_train_state), pro_losses = update_actor_critic_rnn(
            step_rngs[3], pro_train_state,
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

        (_, ant_train_state), ant_losses = update_actor_critic_rnn(
            step_rngs[4], ant_train_state,
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

        (_, adv_train_state), adv_losses = update_actor_critic_rnn(
            step_rngs[5], adv_train_state,
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

        # 6. Apply curriculum prediction gradient (protagonist only)
        if curriculum_features is not None:
            from ..common.metrics import apply_curriculum_prediction_gradient
            pro_train_state, pred_metrics = apply_curriculum_prediction_gradient(
                pro_train_state,
                curriculum_features,
                levels,
                self.env,
                self.env_params,
                self.sample_random_level,
                config,
            )

        # 7. Agent-centric tracking
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

        # 8. Update probe if applicable
        probe_params = train_state.probe_params
        probe_opt_state = train_state.probe_opt_state
        probe_tracking = train_state.probe_tracking

        probe_loss = jnp.float32(0.0)
        if probe_params is not None:
            probe_params, probe_opt_state, probe_tracking, probe_loss = self._update_probe(
                step_rngs[6], probe_params, probe_opt_state, probe_tracking,
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

        # 9. Update memory (curriculum state + nl_state)
        updated_memory = self._update_curriculum_memory(
            train_state, levels, pro_extras,
        )

        # 10. History tracking (same as base PAIRED)
        from ..common.metrics import compute_learnability, compute_level_novelty
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

            if new_level_history is not None:
                level_history_size = new_level_history.shape[0]
                level_ptr = history_ptr % level_history_size
                new_level_history = new_level_history.at[level_ptr].set(levels.wall_map[0])
                novelty = compute_level_novelty(
                    levels.wall_map,
                    new_level_history,
                    min(history_total, level_history_size),
                )
                new_novelty_history = new_novelty_history.at[history_ptr].set(novelty)

            if probe_tracking is not None and probe_tracking.total_samples > 10:
                learnability, _ = compute_learnability(
                    probe_tracking.loss_history,
                    probe_tracking.training_step_history,
                    probe_tracking.total_samples,
                    probe_tracking.current_training_step,
                )
                new_learnability_history = new_learnability_history.at[history_ptr].set(learnability)

            history_ptr = (history_ptr + 1) % history_size
            history_total = history_total + 1

        train_state = PAIREDTrainState(
            update_count=train_state.update_count + 1,
            pro_train_state=pro_train_state,
            ant_train_state=ant_train_state,
            adv_train_state=adv_train_state,
            memory_state=updated_memory,
            agent_tracking=agent_tracking,
            probe_params=probe_params,
            probe_opt_state=probe_opt_state,
            probe_tracking=probe_tracking,
            hstate_samples=train_state.hstate_samples,
            hstate_sample_branches=train_state.hstate_sample_branches,
            hstate_sample_ptr=train_state.hstate_sample_ptr,
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

    def _update_curriculum_memory(
        self,
        train_state: PAIREDTrainState,
        levels,
        pro_extras: dict,
    ):
        """Update curriculum state with adversary-generated level statistics.

        Unlike base _update_memory_after_rollouts, this takes the actual levels
        so update_curriculum_state can extract wall density, goal/agent positions.
        """
        if train_state.memory_state is None:
            return train_state.memory_state

        memory = train_state.memory_state
        curriculum_state = memory["curriculum_state"]

        pro_rewards = pro_extras["rewards"]
        episode_return = pro_rewards.sum(axis=0).mean()

        # Use the first level as representative for curriculum state update
        representative_level = jax.tree_util.tree_map(lambda x: x[0], levels)

        # Update curriculum state with actual level data
        # In PAIRED, the adversary IS the curriculum — use branch=0
        curriculum_state = update_curriculum_state(
            curriculum_state,
            level=representative_level,
            branch=jnp.int32(0),
            score=episode_return,
            sampler_stats={
                'size': jnp.int32(train_state.update_count + 1),
                'mean_score': episode_return,
                'max_score': episode_return,
            },
        )

        return {
            "curriculum_state": curriculum_state,
            "nl_state": memory["nl_state"],
        }

    def _build_log_dict(self, metrics: dict, train_state: PAIREDTrainState) -> dict:
        """Extend PAIRED log dict with curriculum prediction metrics."""
        # Get all standard PAIRED + probe metrics from parent
        log_dict = super()._build_log_dict(metrics, train_state)

        config = self.config
        update_count = metrics["update_count"]

        memory = train_state.memory_state
        if memory is None:
            return log_dict

        # Curriculum state metrics
        curriculum_state = memory.get("curriculum_state")
        if curriculum_state is not None:
            log_dict["curriculum_state/training_step"] = int(curriculum_state.training_step)
            log_dict["curriculum_state/buffer_size"] = int(curriculum_state.buffer_size)
            log_dict["curriculum_state/buffer_mean_score"] = float(curriculum_state.buffer_mean_score)
            log_dict["curriculum_state/total_replay_steps"] = int(curriculum_state.total_replay_steps)
            log_dict["curriculum_state/total_random_steps"] = int(curriculum_state.total_random_steps)
            log_dict["curriculum_state/replay_fraction"] = float(
                curriculum_state.total_replay_steps / max(int(curriculum_state.training_step), 1)
            )
            if curriculum_state.head_pointer > 0 or curriculum_state.history_filled:
                log_dict["curriculum_state/mean_wall_density"] = float(
                    curriculum_state.recent_wall_densities.mean()
                )

            # Curriculum prediction heatmaps and calibration
            try:
                curriculum_features = get_curriculum_features(curriculum_state)

                # Get representative level from history
                if train_state.level_history_wall_maps is not None and train_state.history_total > 0:
                    latest_idx = (train_state.history_ptr - 1) % train_state.level_history_wall_maps.shape[0]

                    # Create dummy obs for forward pass
                    rng_viz = jax.random.PRNGKey(update_count)
                    dummy_level = self.sample_random_level(rng_viz)
                    dummy_obs, _ = self.eval_env.reset_to_level(rng_viz, dummy_level, self.env_params)
                    dummy_obs = jax.tree_util.tree_map(lambda x: x[None, None, ...], dummy_obs)
                    dummy_dones = jnp.zeros((1, 1), dtype=bool)
                    dummy_hidden = self.initialize_hidden_state(1)

                    _, _, _, predictions = train_state.pro_train_state.apply_fn(
                        train_state.pro_train_state.params,
                        (dummy_obs, dummy_dones),
                        dummy_hidden,
                        curriculum_features=curriculum_features,
                        predict_curriculum=True,
                    )

                    # Use the latest adversary-generated level as representative
                    representative_level = dummy_level  # fallback

                    wall_heatmap = create_wall_prediction_heatmap(
                        predictions, representative_level,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["curriculum_pred/images/wall_heatmap"] = wandb.Image(
                        wall_heatmap, caption=f"Curriculum Wall Predictions (step {update_count})"
                    )

                    pos_heatmap = create_position_prediction_heatmap(
                        predictions, representative_level,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["curriculum_pred/images/position_heatmap"] = wandb.Image(
                        pos_heatmap, caption=f"Curriculum Position Predictions (step {update_count})"
                    )

                    # Calibration metrics
                    cal_metrics = compute_calibration_metrics(
                        predictions, representative_level,
                        env_height=DEFAULT_ENV_HEIGHT, env_width=DEFAULT_ENV_WIDTH
                    )
                    log_dict["curriculum_pred/calibration/wall_ece"] = float(cal_metrics["wall_ece"])
                    log_dict["curriculum_pred/calibration/goal_prob_at_actual"] = float(cal_metrics["goal_prob_at_actual"])
                    log_dict["curriculum_pred/calibration/agent_pos_prob_at_actual"] = float(cal_metrics["agent_pos_prob_at_actual"])
                    log_dict["curriculum_pred/calibration/agent_dir_prob_at_actual"] = float(cal_metrics["agent_dir_prob_at_actual"])

            except Exception:
                pass

        # NL state metrics
        nl_state = memory.get("nl_state")
        if nl_state is not None and nl_state.total_samples > 10:
            learnability, learn_details = compute_learnability_from_nl_state(nl_state)
            novelty, novelty_details = compute_novelty_from_nl_state(nl_state)
            oe_score, regime = compute_openendedness_from_nl_state(novelty, learnability)

            log_dict["curriculum_pred/openendedness/novelty"] = novelty
            log_dict["curriculum_pred/openendedness/learnability"] = learnability
            log_dict["curriculum_pred/openendedness/score"] = oe_score
            for key, value in novelty_details.items():
                log_dict[f"curriculum_pred/openendedness/{key}"] = value

        # =====================================================================
        # PREDICTION-SPECIFIC: Spearman correlations (adversary branch)
        # =====================================================================
        try:
            displacement = metrics.get("displacement", None)
            if displacement is not None and nl_state is not None and nl_state.total_samples > 3:
                from scipy.stats import spearmanr

                # Pair displacement with recent prediction losses from NL state
                n = min(int(nl_state.total_samples), nl_state.loss_over_time.shape[0])
                if n >= 3:
                    pred_losses = np.array(nl_state.loss_over_time[:n])
                    # Use aggregated displacement means as summary scalars
                    wall_disp_vals = displacement.get("pairwise/wall_map_hamming/mean")
                    if wall_disp_vals is not None:
                        wall_disp_np = np.array(wall_disp_vals)
                        # Use eval window displacement if same length
                        disp_len = len(wall_disp_np)
                        pred_recent = pred_losses[-disp_len:] if n >= disp_len else pred_losses
                        use_len = min(disp_len, len(pred_recent))
                        if use_len >= 3:
                            rho, p = spearmanr(wall_disp_np[:use_len], pred_recent[:use_len])
                            if np.isfinite(rho):
                                log_dict["curriculum_pred/adversary/spearman/wall_vs_pred_loss"] = float(rho)

                    goal_disp_vals = displacement.get("pairwise/goal_pos_l2/mean")
                    if goal_disp_vals is not None:
                        goal_disp_np = np.array(goal_disp_vals)
                        disp_len = len(goal_disp_np)
                        pred_recent = pred_losses[-disp_len:] if n >= disp_len else pred_losses
                        use_len = min(disp_len, len(pred_recent))
                        if use_len >= 3:
                            rho, _ = spearmanr(goal_disp_np[:use_len], pred_recent[:use_len])
                            if np.isfinite(rho):
                                log_dict["curriculum_pred/adversary/spearman/goal_vs_pred_loss"] = float(rho)
        except Exception:
            pass

        # --- Scatter plots + bar chart (temporal colour gradient) ---
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            displacement = metrics.get("displacement", None)
            pred_metrics_scatter = metrics.get("pred_metrics", {})

            if displacement is not None and pred_metrics_scatter:
                # Lazy-init persistent accumulator
                if not hasattr(self, '_scatter_accum'):
                    from collections import defaultdict
                    self._scatter_accum = defaultdict(list)

                branch_id = 3  # adversary
                branch_name = "adversary"
                current_update = float(metrics.get("update_count", 0))
                max_updates = self.config.get("num_updates", 50000)

                scatter_pairings = [
                    ("pairwise/wall_map_hamming/mean", "wall_loss", "wall"),
                    ("pairwise/goal_pos_l2/mean", "goal_loss", "goal"),
                    ("pairwise/agent_pos_l2/mean", "agent_pos_loss", "agent_pos"),
                    ("pairwise/agent_dir_changed/mean", "agent_dir_loss", "agent_dir"),
                ]

                # Accumulate points from this eval window
                for disp_key, loss_key, var_name in scatter_pairings:
                    disp_vals = displacement.get(disp_key)
                    loss_vals = pred_metrics_scatter.get(loss_key)
                    if (disp_vals is None or not hasattr(disp_vals, 'shape') or not disp_vals.shape
                            or loss_vals is None or not hasattr(loss_vals, 'shape') or not loss_vals.shape):
                        continue
                    d_arr = np.array(disp_vals)
                    l_arr = np.array(loss_vals)
                    for i in range(len(d_arr)):
                        self._scatter_accum[(branch_id, var_name)].append(
                            [float(d_arr[i]), float(l_arr[i]), current_update]
                        )

                # Log scatter plots (continuous vars) and bar chart (agent_dir)
                for disp_key, loss_key, var_name in scatter_pairings:
                    points = self._scatter_accum.get((branch_id, var_name))
                    if not points or len(points) < 2:
                        continue
                    arr = np.array(points)
                    xs, ys, steps = arr[:, 0], arr[:, 1], arr[:, 2]

                    # Compute Spearman rank correlation
                    def _rankdata(a):
                        order = np.argsort(a)
                        ranks = np.empty_like(order, dtype=float)
                        ranks[order] = np.arange(1, len(a) + 1, dtype=float)
                        return ranks
                    n = len(xs)
                    rx, ry = _rankdata(xs), _rankdata(ys)
                    mx, my = rx.mean(), ry.mean()
                    cov = ((rx - mx) * (ry - my)).sum()
                    denom = max(np.sqrt(((rx - mx)**2).sum() * ((ry - my)**2).sum()), 1e-8)
                    spearman = float(cov / denom)

                    if var_name == "agent_dir":
                        unchanged_losses = ys[xs == 0.0]
                        changed_losses = ys[xs > 0.0]
                        bar_data = []
                        if len(unchanged_losses) > 0:
                            bar_data.append(["dir_unchanged", float(unchanged_losses.mean())])
                        if len(changed_losses) > 0:
                            bar_data.append(["dir_changed", float(changed_losses.mean())])
                        if bar_data:
                            bar_table = wandb.Table(data=bar_data, columns=["condition", "mean_loss"])
                            log_dict[f"curriculum_pred/{branch_name}/scatter/{var_name}"] = wandb.plot.bar(
                                bar_table, "condition", "mean_loss",
                                title=f"{branch_name}: agent_dir_loss by direction change (rho={spearman:.3f}, n={n})"
                            )
                    else:
                        # Scatter plot with temporal colour gradient
                        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
                        sc = ax.scatter(xs, ys, c=steps, cmap='Reds', s=12, alpha=0.7,
                                        vmin=0, vmax=max_updates, edgecolors='none')
                        cbar = plt.colorbar(sc, ax=ax)
                        cbar.set_label('Training Update')
                        ax.set_xlabel(f'{var_name} displacement')
                        ax.set_ylabel(loss_key)
                        ax.set_title(f"{branch_name}: {var_name} vs {loss_key} (rho={spearman:.3f}, n={n})")
                        fig.tight_layout()
                        log_dict[f"curriculum_pred/{branch_name}/scatter/{var_name}"] = wandb.Image(fig)
                        plt.close(fig)
        except Exception:
            pass

        return log_dict
