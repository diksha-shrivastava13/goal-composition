"""
Training utilities for curriculum awareness ablations.

Contains:
- compute_gae: Generalized Advantage Estimation
- sample_trajectories_rnn: Trajectory collection with RNN
- update_actor_critic_rnn: PPO update step
- evaluate_rnn: Evaluation rollout
"""

from typing import Any, Tuple, Callable
import jax
import jax.numpy as jnp
import chex

from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv

from .networks import ActorCritic


# =============================================================================
# GENERALIZED ADVANTAGE ESTIMATION
# =============================================================================

def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """
    Compute Generalized Advantage Estimation.

    Args:
        gamma: Discount factor
        lambd: GAE lambda parameter
        last_value: Value estimate at final state, shape (num_envs,)
        values: Value estimates, shape (num_steps, num_envs)
        rewards: Rewards, shape (num_steps, num_envs)
        dones: Done flags, shape (num_steps, num_envs)

    Returns:
        advantages: shape (num_steps, num_envs)
        targets: value targets, shape (num_steps, num_envs)
    """
    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values


# =============================================================================
# TRAJECTORY SAMPLING
# =============================================================================

def sample_trajectories_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
) -> Tuple[
    Tuple[chex.PRNGKey, chex.ArrayTree, Observation, EnvState, chex.Array],
    Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]
]:
    """
    Sample trajectories using RNN policy.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: TrainState with params and apply_fn
        init_hstate: Initial LSTM hidden state
        init_obs: Initial observations
        init_env_state: Initial environment states
        num_envs: Number of parallel environments
        max_episode_length: Number of steps to collect

    Returns:
        Final carry: (rng, hstate, last_obs, last_env_state, last_value)
        Trajectory: (obs, actions, rewards, dones, log_probs, values, info)
            info includes 'agent_pos' (T, N, 2) and 'agent_dir' (T, N) from env_state
    """
    def sample_step(carry, _):
        rng, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        # Extract agent positions/directions before stepping
        step_agent_pos = env_state.agent_pos  # (N, 2)
        step_agent_dir = env_state.agent_dir  # (N,)

        # Add time dimension for network
        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)

        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        # Step environment
        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        # Inject agent position/direction into info
        info['agent_pos'] = step_agent_pos
        info['agent_dir'] = step_agent_dir

        carry = (rng, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, info)

    (rng, hstate, last_obs, last_env_state, last_done), traj = jax.lax.scan(
        sample_step,
        (rng, init_hstate, init_obs, init_env_state, jnp.zeros(num_envs, dtype=bool)),
        None,
        length=max_episode_length,
    )

    # Get final value estimate
    x = jax.tree_util.tree_map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)

    return (rng, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj


def sample_trajectories_rnn_with_context(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
    context: chex.Array,
) -> Tuple[
    Tuple[chex.PRNGKey, chex.ArrayTree, Observation, EnvState, chex.Array],
    Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]
]:
    """
    Sample trajectories with additional context vector.

    Used by context_vector and episodic_memory agents.
    """
    def sample_step(carry, _):
        rng, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        # Extract agent positions/directions before stepping
        step_agent_pos = env_state.agent_pos  # (N, 2)
        step_agent_dir = env_state.agent_dir  # (N,)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, last_done))
        # Pass context to network
        hstate, pi, value = train_state.apply_fn(
            train_state.params, x, hstate, context=context
        )

        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        # Inject agent position/direction into info
        info['agent_pos'] = step_agent_pos
        info['agent_dir'] = step_agent_dir

        carry = (rng, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, info)

    (rng, hstate, last_obs, last_env_state, last_done), traj = jax.lax.scan(
        sample_step,
        (rng, init_hstate, init_obs, init_env_state, jnp.zeros(num_envs, dtype=bool)),
        None,
        length=max_episode_length,
    )

    x = jax.tree_util.tree_map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(
        train_state.params, x, hstate, context=context
    )

    return (rng, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj


def sample_trajectories_rnn_with_curriculum(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state,
    init_hstate: chex.ArrayTree,
    init_obs,
    init_env_state,
    num_envs: int,
    max_episode_length: int,
    curriculum_features: chex.Array,
) -> Tuple[
    Tuple[chex.PRNGKey, chex.ArrayTree, object, object, chex.Array],
    Tuple[object, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]
]:
    """
    Sample trajectories with curriculum features passed to the network.

    Used by PAIREDNextEnvPredictionAgent. Unlike sample_trajectories_rnn_with_context
    (which passes context=), this passes curriculum_features= and handles the
    4-value return from ActorCriticWithCurriculumPrediction.
    """
    def sample_step(carry, _):
        rng, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        # Extract agent positions/directions before stepping
        step_agent_pos = env_state.agent_pos  # (N, 2)
        step_agent_dir = env_state.agent_dir  # (N,)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, last_done))
        hstate, pi, value, _ = train_state.apply_fn(
            train_state.params, x, hstate,
            curriculum_features=curriculum_features,
            predict_curriculum=False,
        )

        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        # Inject agent position/direction into info
        info['agent_pos'] = step_agent_pos
        info['agent_dir'] = step_agent_dir

        carry = (rng, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, info)

    (rng, hstate, last_obs, last_env_state, last_done), traj = jax.lax.scan(
        sample_step,
        (rng, init_hstate, init_obs, init_env_state, jnp.zeros(num_envs, dtype=bool)),
        None,
        length=max_episode_length,
    )

    x = jax.tree_util.tree_map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value, _ = train_state.apply_fn(
        train_state.params, x, hstate,
        curriculum_features=curriculum_features,
        predict_curriculum=False,
    )

    return (rng, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj


# =============================================================================
# PPO UPDATE
# =============================================================================

def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool = True,
) -> Tuple[Tuple[chex.PRNGKey, object], chex.ArrayTree]:
    """
    PPO update step for RNN policy.

    Args:
        rng: Random key
        train_state: TrainState with params, apply_fn, and tx
        init_hstate: Initial hidden state for minibatches
        batch: Trajectory data (obs, actions, dones, log_probs, values, targets, advantages)
        num_envs: Number of environments
        n_steps: Number of steps per trajectory
        n_minibatch: Number of minibatches
        n_epochs: Number of PPO epochs
        clip_eps: PPO clipping epsilon
        entropy_coeff: Entropy bonus coefficient
        critic_coeff: Value loss coefficient
        update_grad: Whether to apply gradients (False for exploratory updates)

    Returns:
        (rng, updated_train_state), loss_metrics
    """
    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages

    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate_mb, obs, actions, last_dones, log_probs, values, targets, advantages = minibatch

            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(
                    params, (obs, last_dones), init_hstate_mb
                )
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                # PPO clipped loss
                ratio = jnp.exp(log_probs_pred - log_probs)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                l_clip = (-jnp.minimum(
                    ratio * A,
                    jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A
                )).mean()

                # Value loss with clipping
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

        # Create minibatches
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


def update_actor_critic_rnn_with_context(
    rng: chex.PRNGKey,
    train_state,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool = True,
    context: chex.Array = None,
) -> Tuple[Tuple[chex.PRNGKey, Any], chex.ArrayTree]:
    """PPO update with context passed to apply_fn.

    Identical to update_actor_critic_rnn but passes context kwarg.
    Used by context_vector and episodic_memory agents.
    """
    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages

    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate_mb, obs, actions, last_dones, log_probs, values, targets, advantages = minibatch

            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(
                    params, (obs, last_dones), init_hstate_mb, context=context
                )
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                # PPO clipped loss
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


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Evaluate RNN policy on a set of levels.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: TrainState with params and apply_fn
        init_hstate: Initial hidden state
        init_obs: Initial observations for each level
        init_env_state: Initial environment states
        max_episode_length: Maximum steps per episode

    Returns:
        states: All environment states, shape (max_steps, num_levels, ...)
        rewards: All rewards, shape (max_steps, num_levels)
        episode_lengths: Episode lengths, shape (num_levels,)
    """
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]

    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        result = train_state.apply_fn(train_state.params, x, hstate)
        hstate, pi = result[0], result[1]
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)

        next_mask = mask & ~done
        episode_length += mask

        return (rng, hstate, obs, next_state, done, next_mask, episode_length), (state, reward)

    (_, _, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (
            rng,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_levels, dtype=bool),
            jnp.ones(num_levels, dtype=bool),
            jnp.zeros(num_levels, dtype=jnp.int32),
        ),
        None,
        length=max_episode_length,
    )

    return states, rewards, episode_lengths


# =============================================================================
# POST-TRAINING N-ENVIRONMENT PREDICTIONS
# =============================================================================

def evaluate_n_env_predictions(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state,
    probe_network,
    sample_random_level: Callable,
    n_envs: int = 100,
    num_steps: int = 256,
    env_height: int = 13,
    env_width: int = 13,
) -> dict:
    """
    Generate post-training predictions for N environments.

    This evaluates the probe's ability to predict level features across
    many randomly generated levels, providing aggregate statistics.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: TrainState with agent params and probe params
        probe_network: CurriculumProbe network
        sample_random_level: Function to generate random levels
        n_envs: Number of environments to evaluate
        num_steps: Steps per episode to generate hidden state
        env_height: Environment height
        env_width: Environment width

    Returns:
        Dict with prediction metrics across N environments
    """
    from .networks import ActorCritic
    from .metrics import compute_per_instance_calibration_batch

    results = {
        'wall_accuracy': [],
        'goal_accuracy': [],
        'agent_pos_accuracy': [],
        'agent_dir_accuracy': [],
        'wall_predictions': [],
        'actual_walls': [],
    }

    for i in range(n_envs):
        rng, rng_level, rng_reset, rng_rollout = jax.random.split(rng, 4)

        # Generate level
        level = sample_random_level(rng_level)

        # Run agent to get hidden state
        init_obs, init_env_state = env.reset_to_level(rng_reset, level, env_params)
        init_obs_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)
        init_env_state_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_env_state)
        hstate = ActorCritic.initialize_carry((1,))

        obs = init_obs_batch
        env_state = init_env_state_batch
        done = jnp.zeros(1, dtype=bool)

        # Rollout to get final hidden state
        for _ in range(num_steps):
            rng_rollout, rng_action = jax.random.split(rng_rollout)
            x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
            result = train_state.apply_fn(train_state.params, x, hstate)
            hstate, pi = result[0], result[1]
            action = pi.sample(seed=rng_action).squeeze(0)
            obs, env_state, _, done, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(jax.random.split(rng_action, 1), env_state, action, env_params)

        # Flatten hidden state and make prediction
        h_c, h_h = hstate
        hstate_flat = jnp.concatenate([h_c[0], h_h[0]], axis=-1)

        # Get prediction from probe
        probe_input = jax.lax.stop_gradient(hstate_flat[None, ...])
        predictions = probe_network.apply(train_state.probe_params, probe_input)

        # Compute per-instance metrics
        level_batch = jax.tree_util.tree_map(lambda x: x[None, ...], level)
        metrics = compute_per_instance_calibration_batch(
            predictions, level_batch, env_height, env_width
        )

        results['wall_accuracy'].append(float(metrics['wall_accuracy']))
        results['goal_accuracy'].append(float(metrics['goal_accuracy']))
        results['agent_pos_accuracy'].append(float(metrics['agent_pos_accuracy']))
        results['agent_dir_accuracy'].append(float(metrics['agent_dir_accuracy']))

        # Store raw predictions for visualization
        wall_pred = jax.nn.sigmoid(predictions['wall_logits'][0])
        results['wall_predictions'].append(wall_pred)
        results['actual_walls'].append(level.wall_map)

    # Aggregate statistics
    import numpy as np
    summary = {
        'mean_wall_accuracy': np.mean(results['wall_accuracy']),
        'std_wall_accuracy': np.std(results['wall_accuracy']),
        'mean_goal_accuracy': np.mean(results['goal_accuracy']),
        'std_goal_accuracy': np.std(results['goal_accuracy']),
        'mean_agent_pos_accuracy': np.mean(results['agent_pos_accuracy']),
        'std_agent_pos_accuracy': np.std(results['agent_pos_accuracy']),
        'mean_agent_dir_accuracy': np.mean(results['agent_dir_accuracy']),
        'std_agent_dir_accuracy': np.std(results['agent_dir_accuracy']),
        'n_envs': n_envs,
        'raw_results': results,
    }

    return summary


def evaluate_n_step_prediction(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state,
    level_sampler,
    sample_random_level: Callable,
    mutate_level: Callable,
    probe_runner=None,  # Optional ProbeRunner for predictions
    n_steps: int = 20,
    num_envs: int = 32,
    num_rollout_steps: int = 256,
    env_height: int = 13,
    env_width: int = 13,
) -> dict:
    """
    N-STEP sequential prediction through curriculum.

    Different from N-ENV:
    - N-ENV: predict N independent random environments
    - N-STEP: predict step 1→2→3→...→N sequentially through curriculum

    This tests curriculum prediction (theory-of-mind of environment generator).
    It follows the actual ACCEL training loop: DR → Replay → Mutate cycle.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: Agent train state
        level_sampler: PLR level sampler
        sample_random_level: Function to generate random levels
        mutate_level: Function to mutate levels
        probe_runner: Optional ProbeRunner for making predictions
        n_steps: Number of curriculum steps to predict through
        num_envs: Number of parallel environments per step
        num_rollout_steps: Steps per rollout
        env_height: Environment height
        env_width: Environment width

    Returns:
        Dict with per-step prediction results and aggregate metrics
    """
    from .networks import ActorCritic
    from .metrics import compute_per_instance_calibration_batch
    import numpy as np

    results = {
        'per_step_wall_accuracy': [],
        'per_step_goal_accuracy': [],
        'per_step_branch': [],
        'per_step_returns': [],
        'hidden_states': [],
        'levels': [],
    }

    # Initialize
    hstate = ActorCritic.initialize_carry((num_envs,))
    sampler = train_state.sampler if hasattr(train_state, 'sampler') else None

    # Track last levels for mutation
    last_replay_levels = None

    for step in range(n_steps):
        rng, rng_step, rng_rollout = jax.random.split(rng, 3)

        # Determine branch based on ACCEL cycle
        if step == 0 or (step % 3 == 0):
            # DR branch - new random levels
            branch = 0
            levels = jax.vmap(sample_random_level)(
                jax.random.split(rng_step, num_envs)
            )
        elif step % 3 == 1 and sampler is not None:
            # Replay branch - sample from buffer
            branch = 1
            rng_step, rng_sample = jax.random.split(rng_step)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(
                sampler, rng_sample, num_envs
            )
            last_replay_levels = levels
        elif step % 3 == 2 and last_replay_levels is not None:
            # Mutate branch - mutate previous replay levels
            branch = 2
            levels = jax.vmap(mutate_level, (0, 0, None))(
                jax.random.split(rng_step, num_envs),
                last_replay_levels,
                3,  # num_edits
            )
        else:
            # Fallback to DR if no sampler
            branch = 0
            levels = jax.vmap(sample_random_level)(
                jax.random.split(rng_step, num_envs)
            )

        # Run agent on levels to get hidden state
        rng_rollout, rng_reset = jax.random.split(rng_rollout)
        init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
            jax.random.split(rng_reset, num_envs),
            levels,
            env_params,
        )

        # Run rollout
        (rng_rollout, final_hstate, last_obs, last_env_state, last_value), traj = \
            sample_trajectories_rnn(
                rng_rollout, env, env_params, train_state,
                hstate, init_obs, init_env_state,
                num_envs, num_rollout_steps,
            )

        obs, actions, rewards, dones, log_probs, values, info = traj

        # Compute returns
        episode_returns = rewards.sum(axis=0)

        # Make predictions if probe_runner available
        if probe_runner is not None:
            try:
                # Use per-instance evaluation (valid for R->M, greedy matching for others)
                if branch == 2 and last_replay_levels is not None:
                    # R->M has 1-to-1 correspondence
                    eval_result = probe_runner.evaluate_per_instance(
                        final_hstate, levels,
                        episode_return=episode_returns,
                        episode_solved=(episode_returns > 0).astype(jnp.float32),
                        episode_length=dones.sum(axis=0).astype(jnp.int32),
                    )
                    wall_accuracy = eval_result['per_instance_metrics']['wall_accuracy']
                    goal_accuracy = eval_result['per_instance_metrics']['goal_accuracy']
                else:
                    # DR/Replay need greedy matching
                    eval_result = probe_runner.evaluate_batch_with_matching(
                        final_hstate, levels,
                        episode_return=episode_returns,
                        episode_solved=(episode_returns > 0).astype(jnp.float32),
                        episode_length=dones.sum(axis=0).astype(jnp.int32),
                    )
                    wall_accuracy = eval_result['matched_metrics']['matched_wall_accuracy']
                    goal_accuracy = eval_result['matched_metrics']['matched_goal_accuracy']

                results['per_step_wall_accuracy'].append(float(wall_accuracy))
                results['per_step_goal_accuracy'].append(float(goal_accuracy))
            except Exception as e:
                # If probe evaluation fails, record NaN
                results['per_step_wall_accuracy'].append(float('nan'))
                results['per_step_goal_accuracy'].append(float('nan'))
        else:
            results['per_step_wall_accuracy'].append(float('nan'))
            results['per_step_goal_accuracy'].append(float('nan'))

        results['per_step_branch'].append(branch)
        results['per_step_returns'].append(float(episode_returns.mean()))
        results['hidden_states'].append(final_hstate)
        results['levels'].append(levels)

        # Update hidden state for next step
        hstate = final_hstate

        # Update sampler if available
        if sampler is not None and hasattr(level_sampler, 'insert_batch'):
            # Compute scores for buffer update
            from jaxued.utils import compute_max_returns
            max_returns = compute_max_returns(dones, rewards)
            scores = max_returns  # Simple scoring
            sampler, _ = level_sampler.insert_batch(
                sampler, levels, scores, {"max_return": max_returns}
            )

    # Aggregate metrics
    wall_accuracies = [x for x in results['per_step_wall_accuracy'] if not np.isnan(x)]
    goal_accuracies = [x for x in results['per_step_goal_accuracy'] if not np.isnan(x)]

    results['mean_wall_accuracy'] = np.mean(wall_accuracies) if wall_accuracies else float('nan')
    results['mean_goal_accuracy'] = np.mean(goal_accuracies) if goal_accuracies else float('nan')

    # Prediction improvement: first half vs second half
    if len(wall_accuracies) >= 4:
        mid = len(wall_accuracies) // 2
        first_half = np.mean(wall_accuracies[:mid])
        second_half = np.mean(wall_accuracies[mid:])
        results['prediction_improvement'] = float(second_half - first_half)
    else:
        results['prediction_improvement'] = 0.0

    # Per-branch accuracy
    results['per_branch_wall_accuracy'] = {}
    results['per_branch_returns'] = {}
    for branch in [0, 1, 2]:
        branch_mask = [b == branch for b in results['per_step_branch']]
        branch_wall = [acc for acc, m in zip(results['per_step_wall_accuracy'], branch_mask)
                       if m and not np.isnan(acc)]
        branch_returns = [ret for ret, m in zip(results['per_step_returns'], branch_mask) if m]

        if branch_wall:
            results['per_branch_wall_accuracy'][branch] = float(np.mean(branch_wall))
        if branch_returns:
            results['per_branch_returns'][branch] = float(np.mean(branch_returns))

    return results


def run_post_training_evaluation(
    train_state,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    sample_random_level: Callable,
    probe_network,
    n_predictions: int = 20,
    env_height: int = 13,
    env_width: int = 13,
    novelty_history: list = None,
    learnability_history: list = None,
    update_steps: list = None,
    log_to_wandb: bool = True,
    # New parameters for N-STEP
    level_sampler=None,
    mutate_level: Callable = None,
    probe_runner=None,
    run_n_step: bool = True,
    n_step_steps: int = 20,
) -> dict:
    """
    Run comprehensive post-training evaluation including N-step prediction and visualizations.

    This orchestrates:
    1. N-environment prediction evaluation
    2. Comparison figure generation
    3. Wandb logging (if enabled)
    4. Pareto frontier visualization (if history available)

    Args:
        train_state: Final training state with agent and probe params
        env: Environment
        env_params: Environment parameters
        sample_random_level: Random level generator
        probe_network: CurriculumProbe network
        n_predictions: Number of environments to predict
        env_height: Environment height
        env_width: Environment width
        novelty_history: History of novelty values (optional)
        learnability_history: History of learnability values (optional)
        update_steps: History of update steps (optional)
        log_to_wandb: Whether to log to wandb
        level_sampler: Optional level sampler for N-STEP evaluation
        mutate_level: Optional mutation function for N-STEP evaluation
        probe_runner: Optional ProbeRunner for predictions
        run_n_step: Whether to run N-STEP evaluation
        n_step_steps: Number of steps for N-STEP evaluation

    Returns:
        Dict with all evaluation results and figures
    """
    import matplotlib.pyplot as plt
    from .visualization import (
        create_prediction_comparison_figure,
        create_pareto_frontier_figure,
    )

    rng = jax.random.PRNGKey(42)
    results = {}

    # N-ENV prediction evaluation (existing)
    print(f"Running {n_predictions}-environment prediction evaluation (N-ENV)...")
    rng, rng_n_env = jax.random.split(rng)
    prediction_results = evaluate_n_env_predictions(
        rng_n_env,
        env,
        env_params,
        train_state,
        probe_network,
        sample_random_level,
        n_envs=n_predictions,
        env_height=env_height,
        env_width=env_width,
    )
    results["n_env"] = prediction_results

    # N-STEP prediction evaluation (NEW)
    if run_n_step and level_sampler is not None and mutate_level is not None:
        print(f"Running {n_step_steps}-step sequential prediction evaluation (N-STEP)...")
        rng, rng_n_step = jax.random.split(rng)
        n_step_results = evaluate_n_step_prediction(
            rng_n_step,
            env,
            env_params,
            train_state,
            level_sampler,
            sample_random_level,
            mutate_level,
            probe_runner=probe_runner,
            n_steps=n_step_steps,
            env_height=env_height,
            env_width=env_width,
        )
        results["n_step"] = n_step_results

    # Create comparison figure
    comparison_fig = create_prediction_comparison_figure(
        prediction_results,
        n_display=min(5, n_predictions),
        env_height=env_height,
        env_width=env_width,
    )
    results["comparison_figure"] = comparison_fig

    # Log to wandb
    if log_to_wandb:
        try:
            import wandb
            log_dict = {
                "post_training/n_env_mean_wall_accuracy": prediction_results["mean_wall_accuracy"],
                "post_training/n_env_mean_goal_accuracy": prediction_results["mean_goal_accuracy"],
                "post_training/n_env_mean_agent_pos_accuracy": prediction_results["mean_agent_pos_accuracy"],
                "post_training/n_env_mean_agent_dir_accuracy": prediction_results["mean_agent_dir_accuracy"],
                "post_training/prediction_comparison": wandb.Image(comparison_fig),
            }

            # Add N-STEP metrics
            if "n_step" in results:
                n_step = results["n_step"]
                log_dict["post_training/n_step_mean_wall_accuracy"] = n_step.get("mean_wall_accuracy", 0)
                log_dict["post_training/n_step_mean_goal_accuracy"] = n_step.get("mean_goal_accuracy", 0)
                log_dict["post_training/n_step_improvement"] = n_step.get("prediction_improvement", 0)

            wandb.log(log_dict)
        except ImportError:
            print("Warning: wandb not available, skipping logging")

    # Pareto frontier visualization (if history available)
    if novelty_history and learnability_history and update_steps:
        pareto_fig = create_pareto_frontier_figure(
            novelty_history,
            learnability_history,
            update_steps,
        )
        results["pareto_figure"] = pareto_fig
        if log_to_wandb:
            try:
                import wandb
                wandb.log({"post_training/pareto_frontier": wandb.Image(pareto_fig)})
            except ImportError:
                pass

    plt.close('all')

    print("Post-training evaluation complete.")
    print(f"  N-ENV prediction accuracies over {n_predictions} environments:")
    print(f"    Wall: {prediction_results['mean_wall_accuracy']:.4f}")
    print(f"    Goal: {prediction_results['mean_goal_accuracy']:.4f}")
    print(f"    Agent Pos: {prediction_results['mean_agent_pos_accuracy']:.4f}")
    print(f"    Agent Dir: {prediction_results['mean_agent_dir_accuracy']:.4f}")

    if "n_step" in results:
        n_step = results["n_step"]
        print(f"  N-STEP sequential prediction over {n_step_steps} steps:")
        print(f"    Mean Wall Accuracy: {n_step.get('mean_wall_accuracy', 'N/A')}")
        print(f"    Mean Goal Accuracy: {n_step.get('mean_goal_accuracy', 'N/A')}")
        print(f"    Prediction Improvement: {n_step.get('prediction_improvement', 'N/A')}")

    return results
