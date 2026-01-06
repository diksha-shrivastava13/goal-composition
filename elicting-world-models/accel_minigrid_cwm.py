import os
import json
import time
from typing import Sequence, Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import core, struct
from flax.training.train_state import TrainState as BaseTrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import orbax.checkpoint as ocp

import wandb
import chex
from enum import IntEnum
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
# TODO: should remove the unused matplotlib imports

from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from jaxued.linen import ResetRNN
from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level, make_level_generator, make_level_mutator_minimax
from jaxued.level_sampler import LevelSampler
from jaxued.utils import max_mc, positive_value_loss, compute_max_returns
from jaxued.wrappers import AutoReplayWrapper

"""
Causal Analysis Constants:
--------------------------
These constants control the causal world model extraction and analysis.
Arbitrary values are marked as hyperparameters which may need tuning.



"""

# =============================================================================
# CAUSAL ANALYSIS CONSTANTS
# =============================================================================
# These constants control the causal world model extraction and analysis.
# Values are documented with their justification. Arbitrary values are marked
# and should be treated as hyperparameters that may need tuning.

# --- Statistical Constants (Mathematically Justified) ---
MAD_TO_STD_FACTOR = 1.4826  # For normal distribution: MAD * 1.4826 ≈ std deviation
UNIFORM_PRIOR = 0.5  # Maximum entropy prior for binary unknown (edge exists or not)

# --- Statistical Thresholds (Convention-based, Semi-justified) ---
# Number of MADs above median for outlier/significance detection
# 2.5 ≈ 2σ for normal distribution (95% CI). Common in robust statistics.
MAD_MULTIPLIER = 2.5

# Number of standard deviations for significance testing
# 2.0 corresponds to ~95% confidence interval (exact value is 1.96)
SIGNIFICANCE_STD_MULTIPLIER = 2.0

# --- Bayesian Update Constants (Semi-justified) ---
# Bounds to prevent certainty in Bayesian belief updates
# Certainty (0 or 1) would stop learning; these allow continued updates
# Values chosen to be symmetric and allow strong but not absolute beliefs
BELIEF_LOWER_BOUND = 0.05  # Minimum P(edge exists)
BELIEF_UPPER_BOUND = 0.95  # Maximum P(edge exists)

# --- Temporal Analysis Constants (Statistically Justified) ---
# Fraction of horizon to use for noise estimation at start of trajectory
# Using early timesteps assumes intervention effects haven't propagated yet
NOISE_ESTIMATION_FRACTION = 0.1  # Will use max(MIN_SAMPLES_FOR_VARIANCE, horizon * 0.1)

# Minimum samples needed for reliable variance estimate
# Statistical minimum is 2 (for sample variance), but 3 provides 1 degree of freedom
# for detecting outliers. This is the theoretical minimum, not arbitrary.
MIN_SAMPLES_FOR_VARIANCE = 3

# Minimum timesteps after peak needed to estimate decay rate
# Exponential decay fit has 2 parameters (amplitude, rate), requiring minimum 3 points
# (n_params + 1 degree of freedom for goodness-of-fit assessment)
MIN_DECAY_WINDOW = 3


def compute_noise_threshold(baseline_divergences: np.ndarray) -> float:
    """Compute noise threshold from baseline divergence distribution.

    Uses MAD-based robust estimation to determine the threshold above which
    divergences are considered "real" effects rather than noise.

    Args:
        baseline_divergences: Array of divergence values from baseline (no intervention)

    Returns:
        Threshold value: median + 2.5 * MAD * 1.4826 (approximately 2.5 std devs)
    """
    if len(baseline_divergences) < MIN_SAMPLES_FOR_VARIANCE:
        return 0.0  # Not enough data, accept all
    median = np.median(baseline_divergences)
    mad = np.median(np.abs(baseline_divergences - median))
    return median + MAD_MULTIPLIER * mad * MAD_TO_STD_FACTOR


def compute_edge_inclusion_threshold(n_observations: int, base_threshold: float = 0.5) -> float:
    """Compute edge inclusion threshold based on observation count.

    Uses Wilson score interval to determine if belief is significantly above 0.5.
    With more observations, we can be more confident, so threshold approaches 0.5.
    With fewer observations, we require higher belief to include edge.

    Args:
        n_observations: Number of observations used to estimate edge belief
        base_threshold: Base threshold (0.5 = "more likely than not")

    Returns:
        Threshold that accounts for uncertainty from limited observations
    """
    if n_observations <= 0:
        return 1.0  # No observations, don't include any edges
    # Standard error of proportion estimate: sqrt(p(1-p)/n)
    # At p=0.5 (maximum uncertainty): SE = sqrt(0.25/n) = 0.5/sqrt(n)
    # Require belief to exceed 0.5 by ~1 standard error
    margin = 0.5 / np.sqrt(n_observations)
    return min(base_threshold + margin, BELIEF_UPPER_BOUND)


def compute_effect_normalization(observed_effects: np.ndarray) -> tuple[float, float]:
    """Compute effect normalization parameters from observed baseline effects.

    Instead of arbitrary constants, normalize based on the actual distribution
    of observed effects. This makes the Bayesian updates scale-invariant.

    Args:
        observed_effects: Array of effect magnitudes from observations

    Returns:
        (baseline, cap): baseline for normalization, cap for numerical stability
    """
    if len(observed_effects) < MIN_SAMPLES_FOR_VARIANCE:
        # Not enough data, use conservative defaults that won't dominate updates
        return 1.0, 3.0

    # Use median as robust baseline (not affected by outliers)
    baseline = max(np.median(observed_effects), 1e-6)  # Prevent division by zero

    # Cap at 3 standard deviations above median (covers 99.7% of normal dist)
    mad = np.median(np.abs(observed_effects - baseline))
    std_estimate = mad * MAD_TO_STD_FACTOR
    cap = 3.0 * std_estimate / baseline if std_estimate > 0 else 3.0

    return baseline, max(cap, 1.0)  # Ensure cap >= 1 for meaningful range


class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1


class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    # used for logging
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)


def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets"""

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


def sample_trajectories_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
) -> Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]:
    """This samples trajectories from the environment using the agent specified by the train_state"""

    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        ac_input = jax.tree_util.tree_map(lambda t: t[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, ac_input, hstate)
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

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, info)

    (rng, train_state, hstate, last_obs, last_env_state, last_done), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_envs, dtype=bool),
        ),
        None,
        length=max_episode_length,
    )

    ac_input = jax.tree_util.tree_map(lambda t: t[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, ac_input, hstate)
    return (rng, train_state, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj


def sample_trajectories_rnn_with_predictions(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
) -> Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array],
           Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, Observation, dict]]:
    """Samples trajectories with additional data for prediction training.

    This extends sample_trajectories_rnn to also collect:
    - next_obs: The observation after taking the action (for prediction targets)
    - action_logits: Full action distribution logits (for KL divergence in CWM)

    Returns:
        Tuple of:
        - carry: (rng, train_state, hstate, last_obs, last_env_state, last_value)
        - traj: (obs, action, reward, done, log_prob, value, action_logits, next_obs, info)
    """

    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        ac_input = jax.tree_util.tree_map(lambda t: t[None, ...], (obs, last_done))
        # For prediction network, we pass None for actions_for_pred during rollout
        # (predictions computed separately during loss computation)
        hstate, pi, value, _ = train_state.apply_fn(train_state.params, ac_input, hstate, None)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        action_logits = pi.logits  # Store full logits for KL divergence

        value, action, log_prob, action_logits = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
            action_logits.squeeze(0),
        )

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, action_logits, next_obs, info)

    (rng, train_state, hstate, last_obs, last_env_state, last_done), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_envs, dtype=bool),
        ),
        None,
        length=max_episode_length,
    )

    ac_input = jax.tree_util.tree_map(lambda t: t[None, ...], (last_obs, last_done))
    _, _, last_value, _ = train_state.apply_fn(train_state.params, ac_input, hstate, None)
    return (rng, train_state, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj


def evaluate_rnn(
        rng: chex.PRNGKey,
        env: UnderspecifiedEnv,
        env_params: EnvParams,
        train_state: TrainState,
        init_hstate: chex.ArrayTree,
        init_obs: Observation,
        init_env_state: EnvState,
        max_episode_length: int,
        use_cwm: bool = False,
) -> Tuple[chex.Array, chex.Array, chex.Array]:

    """This runs the RNN on the environment, given an initial state and observation, and returns (states, rewards, episode lengths"""
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]

    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        ac_input = jax.tree_util.tree_map(lambda t: t[None, ...], (obs, done))
        if use_cwm:
            hstate, pi, _, _ = train_state.apply_fn(train_state.params, ac_input, hstate, None)
        else:
            hstate, pi, _ = train_state.apply_fn(train_state.params, ac_input, hstate)
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


def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool=True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state"""

    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages

    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate, obs, actions, last_dones, log_probs, values, targets, advantages = minibatch

            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(params, (obs, last_dones), init_hstate)
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_probs_pred - log_probs)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)).mean()

                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()

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


def update_actor_critic_rnn_with_predictions(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    pred_coeff: float,
    reward_pred_coeff: float,
    update_grad: bool = True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """PPO update with prediction losses for causal world model learning.

    This extends update_actor_critic_rnn to also train the prediction heads:
    - Next observation prediction (MSE loss)
    - Next direction prediction (cross-entropy loss)
    - Reward prediction (MSE loss)

    Args:
        batch: (obs, actions, dones, log_probs, values, targets, advantages, next_obs, rewards)
        pred_coeff: Weight for observation prediction loss
        reward_pred_coeff: Weight for reward prediction loss
    """
    obs, actions, dones, log_probs, values, targets, advantages, next_obs, rewards = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages, next_obs, rewards

    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            (init_hstate, obs, actions, last_dones, log_probs, values,
             targets, advantages, next_obs, rewards) = minibatch

            def loss_fn(params):
                # Forward pass with predictions
                _, pi, values_pred, predictions = train_state.apply_fn(
                    params, (obs, last_dones), init_hstate, actions
                )

                # PPO policy loss
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()
                ratio = jnp.exp(log_probs_pred - log_probs)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)).mean()

                # Value loss
                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()

                # Prediction losses
                # Next observation image prediction (MSE)
                l_pred_obs = jnp.mean((predictions['next_obs_image'] - next_obs.image) ** 2)

                # Next direction prediction (cross-entropy)
                next_dir_onehot = jax.nn.one_hot(next_obs.agent_dir, 4)
                l_pred_dir = -jnp.mean(jnp.sum(
                    next_dir_onehot * jax.nn.log_softmax(predictions['next_obs_dir_logits']),
                    axis=-1
                ))

                # Reward prediction (MSE)
                l_pred_reward = jnp.mean((predictions['reward'] - rewards) ** 2)

                # Combined prediction loss
                # Normalize direction cross-entropy to be on similar scale as MSE losses
                # Cross-entropy for 4 classes has max ~1.4 (log(4)), MSE on normalized obs is typically 0.01-0.1
                # We scale direction loss down to match the typical MSE scale
                l_pred = l_pred_obs + l_pred_dir / 10.0  # Normalize CE to MSE scale (~1.4 / 10 ≈ 0.14)

                # Total loss
                loss = (
                    l_clip
                    + critic_coeff * l_vf
                    - entropy_coeff * entropy
                    + pred_coeff * l_pred
                    + reward_pred_coeff * l_pred_reward
                )

                return loss, {
                    'l_vf': l_vf,
                    'l_clip': l_clip,
                    'entropy': entropy,
                    'l_pred_obs': l_pred_obs,
                    'l_pred_dir': l_pred_dir,
                    'l_pred_reward': l_pred_reward,
                }

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, loss_info), grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, (loss, loss_info)

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


class ActorCritic(nn.Module):
    """This is an actor critic class that uses an LSTM"""
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs

        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)

        embedding = jnp.append(img_embed, dir_embed, axis=-1)
        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)

        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))


class ActorCriticWithPrediction(nn.Module):
    """Actor-critic with prediction heads for causal world model learning.

    This extends the base ActorCritic with:
    - Next observation prediction head (for learning world dynamics)
    - Reward prediction head (for learning reward structure)

    The prediction heads enable extraction of causal world models through
    behavioral interventions by revealing what the agent believes about
    state transitions and reward dependencies.
    """
    action_dim: Sequence[int]
    obs_image_shape: Tuple[int, int, int]  # (H, W, C) of observation image

    @nn.compact
    def __call__(self, inputs, hidden, actions_for_pred=None):
        """
        Args:
            inputs: Tuple of (obs, dones)
            hidden: LSTM hidden state
            actions_for_pred: Optional actions for prediction (used during training)
                             If None, predictions are not computed

        Returns:
            hidden: Updated LSTM hidden state
            pi: Action distribution
            value: Value estimate
            predictions: Dict with 'next_obs_image', 'next_obs_dir', 'reward' if actions_for_pred provided
        """
        obs, dones = inputs

        # Observation embedding (same as base ActorCritic)
        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID", name="obs_conv")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)

        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        # Store pre-LSTM embedding for trace norm regularization
        self.sow('intermediates', 'pre_lstm_embedding', embedding)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)

        # Store post-LSTM embedding
        self.sow('intermediates', 'post_lstm_embedding', embedding)

        # Actor head
        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = distrax.Categorical(logits=actor_logits)

        # Critic head
        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)
        value = jnp.squeeze(critic, axis=-1)

        # Prediction heads (only compute if actions provided)
        predictions = None
        if actions_for_pred is not None:
            # Combine embedding with action for transition prediction
            action_onehot = jax.nn.one_hot(actions_for_pred, self.action_dim)
            pred_input = jnp.concatenate([embedding, action_onehot], axis=-1)

            # Next observation image prediction
            pred_hidden = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="pred_hidden0")(pred_input)
            pred_hidden = nn.relu(pred_hidden)
            pred_hidden = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="pred_hidden1")(pred_hidden)
            pred_hidden = nn.relu(pred_hidden)

            # Predict flattened next observation image
            obs_image_size = int(np.prod(self.obs_image_shape))
            next_obs_image_flat = nn.Dense(obs_image_size, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="pred_next_obs")(pred_hidden)
            next_obs_image = next_obs_image_flat.reshape(*next_obs_image_flat.shape[:-1], *self.obs_image_shape)

            # Predict next agent direction (4 classes)
            next_obs_dir_logits = nn.Dense(4, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="pred_next_dir")(pred_hidden)

            # Predict reward
            reward_pred = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="pred_reward")(pred_hidden)
            reward_pred = jnp.squeeze(reward_pred, axis=-1)

            predictions = {
                'next_obs_image': next_obs_image,
                'next_obs_dir_logits': next_obs_dir_logits,
                'reward': reward_pred,
                'action_logits': actor_logits,  # Store for KL divergence computation in CWM extraction
            }

        return hidden, pi, value, predictions

    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))


# =============================================================================
# CAUSAL WORLD MODEL EXTRACTION
# =============================================================================

# Define the causal variables we want to probe in the Maze environment
# 'type' determines intervention behavior:
#   - 'initial': Set once at episode start, let dynamics evolve naturally
#   - 'persistent': Reset to target value after every step (for testing counterfactual "what if always at X")
CWM_VARIABLES = {
    'agent_x': {
        'type': 'initial',  # Set at start, let agent move naturally
        'domain_type': 'spatial',
        'description': 'Agent x-coordinate',
    },
    'agent_y': {
        'type': 'initial',
        'domain_type': 'spatial',
        'description': 'Agent y-coordinate',
    },
    'agent_dir': {
        'type': 'initial',  # Set initial direction, let it change with actions
        'domain_type': 'discrete',
        'domain': [0, 1, 2, 3],  # 0=right, 1=down, 2=left, 3=up
        'description': 'Agent facing direction',
    },
    'goal_x': {
        'type': 'initial',  # Goal doesn't move anyway
        'domain_type': 'spatial',
        'description': 'Goal x-coordinate',
    },
    'goal_y': {
        'type': 'initial',
        'domain_type': 'spatial',
        'description': 'Goal y-coordinate',
    },
}


def get_env_dimensions(env: UnderspecifiedEnv) -> Tuple[int, int]:
    """Get environment grid dimensions from the environment.

    Args:
        env: The environment instance

    Returns:
        Tuple of (height, width)
    """
    # For Maze environment, these are stored as max_height and max_width
    return env.max_height, env.max_width


def is_valid_position(wall_map: jnp.ndarray, x: int, y: int) -> bool:
    """Check if a position is valid (not a wall and within bounds).

    Args:
        wall_map: 2D array where 1=wall, 0=empty
        x: X coordinate
        y: Y coordinate

    Returns:
        True if position is valid (empty cell)
    """
    h, w = wall_map.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    return wall_map[y, x] == 0


def get_valid_positions(wall_map: jnp.ndarray) -> list:
    """Get all valid (non-wall) positions in the grid.

    Args:
        wall_map: 2D array where 1=wall, 0=empty

    Returns:
        List of (x, y) tuples for valid positions
    """
    h, w = wall_map.shape
    valid = []
    # Skip border (always walls in Maze env)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if wall_map[y, x] == 0:
                valid.append((x, y))
    return valid


def get_variable_domain(var_name: str, env_size: int, num_values: int) -> list:
    """Get the domain of values to test for a variable.

    Args:
        var_name: Name of the variable
        env_size: Size of the environment grid
        num_values: Number of values to sample for spatial variables

    Returns:
        List of values to test
    """
    var_info = CWM_VARIABLES[var_name]

    if var_info['domain_type'] == 'discrete':
        return var_info['domain']
    elif var_info['domain_type'] == 'spatial':
        # Sample evenly spaced positions, avoiding walls at edges (1 to env_size-2)
        valid_range = env_size - 2  # Exclude border walls
        if num_values >= valid_range:
            return list(range(1, env_size - 1))
        else:
            step = valid_range // num_values
            return [1 + i * step for i in range(num_values)]
    else:
        raise ValueError(f"Unknown domain type: {var_info['domain_type']}")


def extract_env_state_variable(env_state: EnvState, var_name: str) -> jnp.ndarray:
    """Extract a variable's value from the environment state.

    Args:
        env_state: The JAX environment state
        var_name: Name of variable to extract

    Returns:
        The variable's value
    """
    if var_name == 'agent_x':
        return env_state.agent_pos[0]
    elif var_name == 'agent_y':
        return env_state.agent_pos[1]
    elif var_name == 'agent_dir':
        return env_state.agent_dir
    elif var_name == 'goal_x':
        return env_state.goal_pos[0]
    elif var_name == 'goal_y':
        return env_state.goal_pos[1]
    else:
        raise ValueError(f"Unknown variable: {var_name}")


def apply_intervention_to_env_state(
    env_state: EnvState,
    var_name: str,
    value: int,
) -> EnvState:
    """Apply an intervention by setting a variable to a specific value.

    Args:
        env_state: Current environment state
        var_name: Variable to intervene on
        value: Value to set

    Returns:
        Modified environment state
    """
    if var_name == 'agent_x':
        new_agent_pos = env_state.agent_pos.at[0].set(value)
        return env_state.replace(agent_pos=new_agent_pos)
    elif var_name == 'agent_y':
        new_agent_pos = env_state.agent_pos.at[1].set(value)
        return env_state.replace(agent_pos=new_agent_pos)
    elif var_name == 'agent_dir':
        return env_state.replace(agent_dir=value)
    elif var_name == 'goal_x':
        new_goal_pos = env_state.goal_pos.at[0].set(value)
        return env_state.replace(goal_pos=new_goal_pos)
    elif var_name == 'goal_y':
        new_goal_pos = env_state.goal_pos.at[1].set(value)
        return env_state.replace(goal_pos=new_goal_pos)
    else:
        raise ValueError(f"Unknown variable: {var_name}")


def collect_baseline_trajectories(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    levels: chex.ArrayTree,
    rollout_length: int,
    use_predictions: bool = True,
) -> dict:
    """Collect baseline trajectories without any interventions.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: Current training state with policy
        levels: Batch of levels to run on
        rollout_length: Number of steps to rollout
        use_predictions: Whether to use prediction network (for CWM mode)

    Returns:
        Dictionary containing:
        - observations: (T, N, ...) observations at each step
        - actions: (T, N) actions taken
        - action_logits: (T, N, A) full action logits
        - rewards: (T, N) rewards received
        - dones: (T, N) done flags
        - values: (T, N) value estimates
    """
    num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]

    # Reset to levels
    rng, rng_reset = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(rng_reset, num_levels), levels, env_params
    )

    if use_predictions:
        init_hstate = ActorCriticWithPrediction.initialize_carry((num_levels,))
    else:
        init_hstate = ActorCritic.initialize_carry((num_levels,))

    def step_fn(carry, _):
        rng, hstate, obs, env_state, done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))

        if use_predictions:
            hstate, pi, value, _ = train_state.apply_fn(train_state.params, x, hstate, None)
        else:
            hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)

        action = pi.sample(seed=rng_action)
        action_logits = pi.logits

        value, action, action_logits = (
            value.squeeze(0),
            action.squeeze(0),
            action_logits.squeeze(0),
        )

        next_obs, next_env_state, reward, next_done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), env_state, action, env_params)

        carry = (rng, hstate, next_obs, next_env_state, next_done)
        return carry, {
            'obs': obs,
            'action': action,
            'action_logits': action_logits,
            'reward': reward,
            'done': done,
            'value': value,
            'next_obs': next_obs,
        }

    _, trajectories = jax.lax.scan(
        step_fn,
        (rng, init_hstate, init_obs, init_env_state, jnp.zeros(num_levels, dtype=bool)),
        None,
        length=rollout_length,
    )

    return trajectories


def collect_intervention_trajectories(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    levels: chex.ArrayTree,
    var_name: str,
    var_value: int,
    rollout_length: int,
    use_predictions: bool = True,
) -> dict:
    """Collect trajectories with an intervention on a specific variable.

    For 'persistent' interventions, the variable is reset to the target value
    after each environment step.

    For 'initial' interventions, the variable is only set at the start.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: Current training state
        levels: Batch of levels
        var_name: Variable to intervene on
        var_value: Value to set
        rollout_length: Number of steps
        use_predictions: Whether using prediction network

    Returns:
        Same structure as collect_baseline_trajectories
    """
    num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]
    var_info = CWM_VARIABLES[var_name]
    is_persistent = var_info['type'] == 'persistent'

    # Reset to levels
    rng, rng_reset = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(rng_reset, num_levels), levels, env_params
    )

    # Apply initial intervention to env state
    init_env_state = jax.vmap(
        lambda s: apply_intervention_to_env_state(s, var_name, var_value)
    )(init_env_state)

    # CRITICAL: Regenerate observation from the modified env state
    # The observation needs to reflect the intervened state
    init_obs = jax.vmap(env.get_obs, in_axes=(0, None))(init_env_state, env_params)

    if use_predictions:
        init_hstate = ActorCriticWithPrediction.initialize_carry((num_levels,))
    else:
        init_hstate = ActorCritic.initialize_carry((num_levels,))

    def step_fn(carry, _):
        rng, hstate, obs, env_state, done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))

        if use_predictions:
            hstate, pi, value, _ = train_state.apply_fn(train_state.params, x, hstate, None)
        else:
            hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)

        action = pi.sample(seed=rng_action)
        action_logits = pi.logits

        value, action, action_logits = (
            value.squeeze(0),
            action.squeeze(0),
            action_logits.squeeze(0),
        )

        next_obs, next_env_state, reward, next_done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), env_state, action, env_params)

        # Apply persistent intervention: reset variable after each step
        if is_persistent:
            next_env_state = jax.vmap(
                lambda s: apply_intervention_to_env_state(s, var_name, var_value)
            )(next_env_state)

        carry = (rng, hstate, next_obs, next_env_state, next_done)
        return carry, {
            'obs': obs,
            'action': action,
            'action_logits': action_logits,
            'reward': reward,
            'done': done,
            'value': value,
            'next_obs': next_obs,
        }

    _, trajectories = jax.lax.scan(
        step_fn,
        (rng, init_hstate, init_obs, init_env_state, jnp.zeros(num_levels, dtype=bool)),
        None,
        length=rollout_length,
    )

    return trajectories


def compute_kl_divergence(logits_p: jnp.ndarray, logits_q: jnp.ndarray) -> jnp.ndarray:
    """Compute KL(P || Q) for categorical distributions given logits.

    Args:
        logits_p: Logits for distribution P
        logits_q: Logits for distribution Q

    Returns:
        KL divergence value
    """
    p = jax.nn.softmax(logits_p)
    log_p = jax.nn.log_softmax(logits_p)
    log_q = jax.nn.log_softmax(logits_q)
    return jnp.sum(p * (log_p - log_q), axis=-1)


# =============================================================================
# PREDICTION-BASED CAUSAL DISCOVERY
# =============================================================================

def probe_predictions_under_state(
    train_state: TrainState,
    obs: Observation,
    actions: jnp.ndarray,
    hstate: chex.ArrayTree,
) -> dict:
    """Query the prediction network for its beliefs about next state given actions.

    This probes the agent's internal world model by asking: "Given this observation
    and these actions, what do you predict will happen?"

    Args:
        train_state: Training state with prediction network
        obs: Current observation (batched over environments)
        actions: Actions to query predictions for (num_envs,)
        hstate: Hidden state for RNN

    Returns:
        Dictionary with predictions:
        - 'next_obs_image': Predicted next observation image
        - 'next_obs_dir_logits': Predicted next direction (logits)
        - 'reward': Predicted reward
        - 'action_logits': Action distribution logits
    """
    # Prepare input (add time dimension)
    done = jnp.zeros(obs.image.shape[0], dtype=bool)
    x = jax.tree_util.tree_map(lambda t: t[None, ...], (obs, done))

    # Query prediction network
    _, _, _, predictions = train_state.apply_fn(
        train_state.params, x, hstate, actions[None, ...]
    )

    # Remove time dimension from predictions
    predictions = jax.tree_util.tree_map(lambda t: t[0] if t is not None else None, predictions)

    return predictions


def collect_prediction_data(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    levels: chex.ArrayTree,
    rollout_length: int,
) -> dict:
    """Collect trajectory data including predictions from the world model.

    This extends baseline trajectory collection to also query the prediction
    network at each step, enabling analysis of what the agent believes will happen.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: Training state with prediction network
        levels: Batch of levels
        rollout_length: Number of steps

    Returns:
        Dictionary with trajectory data and predictions:
        - 'obs': Observations at each step
        - 'actions': Actions taken
        - 'rewards': Actual rewards
        - 'next_obs': Actual next observations
        - 'predictions': Dict of predicted values at each step
    """
    num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]

    # Reset to levels
    rng, rng_reset = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(rng_reset, num_levels), levels, env_params
    )

    init_hstate = ActorCriticWithPrediction.initialize_carry((num_levels,))

    def step_fn(carry, _):
        rng, hstate, obs, env_state, done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        # Get policy and value
        x = jax.tree_util.tree_map(lambda t: t[None, ...], (obs, done))
        hstate, pi, value, _ = train_state.apply_fn(train_state.params, x, hstate, None)

        action = pi.sample(seed=rng_action).squeeze(0)
        action_logits = pi.logits.squeeze(0)

        # Get predictions for the chosen action
        _, _, _, predictions = train_state.apply_fn(
            train_state.params, x, hstate, action[None, ...]
        )
        predictions = jax.tree_util.tree_map(lambda t: t[0] if t is not None else None, predictions)

        # Step environment
        next_obs, next_env_state, reward, next_done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), env_state, action, env_params)

        carry = (rng, hstate, next_obs, next_env_state, next_done)
        return carry, {
            'obs': obs,
            'action': action,
            'action_logits': action_logits,
            'reward': reward,
            'done': done,
            'value': value.squeeze(0),
            'next_obs': next_obs,
            'pred_next_obs_image': predictions['next_obs_image'],
            'pred_next_dir_logits': predictions['next_obs_dir_logits'],
            'pred_reward': predictions['reward'],
        }

    _, trajectories = jax.lax.scan(
        step_fn,
        (rng, init_hstate, init_obs, init_env_state, jnp.zeros(num_levels, dtype=bool)),
        None,
        length=rollout_length,
    )

    return trajectories


def compute_prediction_divergence_per_variable(
    baseline_predictions: dict,
    intervention_predictions: dict,
    baseline_next_obs: Observation,
    intervention_next_obs: Observation,
) -> dict:
    """Compute prediction divergence for each causal target variable.

    This measures how the agent's internal world model predictions change
    under intervention, separately for each output variable.

    Args:
        baseline_predictions: Predictions dict from baseline trajectories
        intervention_predictions: Predictions dict from intervention trajectories
        baseline_next_obs: Actual next observations from baseline
        intervention_next_obs: Actual next observations from intervention

    Returns:
        Dictionary mapping variable name -> divergence score
    """
    divergences = {}

    # Direction prediction divergence (KL divergence on categorical)
    dir_kl = compute_kl_divergence(
        intervention_predictions['pred_next_dir_logits'],
        baseline_predictions['pred_next_dir_logits']
    )
    divergences['agent_dir'] = float(jnp.mean(dir_kl))

    # Reward prediction divergence (MSE)
    reward_mse = jnp.mean((
        intervention_predictions['pred_reward'] - baseline_predictions['pred_reward']
    ) ** 2)
    divergences['reward'] = float(reward_mse)

    # Observation image prediction divergence (MSE)
    # This captures implicit position information since the agent-centric view changes with position
    obs_mse = jnp.mean((
        intervention_predictions['pred_next_obs_image'] - baseline_predictions['pred_next_obs_image']
    ) ** 2)
    divergences['next_obs'] = float(obs_mse)

    # Action distribution divergence (captures behavioral change)
    action_kl = compute_kl_divergence(
        intervention_predictions['action_logits'],
        baseline_predictions['action_logits']
    )
    divergences['action'] = float(jnp.mean(action_kl))

    # For spatial variables (agent_x, agent_y, goal_x, goal_y), we use observation
    # divergence as a proxy since the agent-centric view encodes position implicitly
    # The observation MSE captures how much the visual input changes
    divergences['agent_x'] = divergences['next_obs']
    divergences['agent_y'] = divergences['next_obs']
    divergences['goal_x'] = divergences['next_obs'] + divergences['reward']  # Goal affects both view and reward
    divergences['goal_y'] = divergences['next_obs'] + divergences['reward']

    return divergences


def compute_behavioral_divergence(
    baseline_trajectories: dict,
    intervention_trajectories: dict,
) -> jnp.ndarray:
    """Compute behavioral divergence between baseline and intervention.

    Measures how much the action distribution changes under intervention.

    Args:
        baseline_trajectories: Trajectories without intervention
        intervention_trajectories: Trajectories with intervention

    Returns:
        Mean KL divergence across timesteps and environments
    """
    baseline_logits = baseline_trajectories['action_logits']  # (T, N, A)
    intervention_logits = intervention_trajectories['action_logits']  # (T, N, A)

    # Compute KL divergence at each (time, env) pair
    kl_divs = compute_kl_divergence(intervention_logits, baseline_logits)  # (T, N)

    # Average across time and environments
    return jnp.mean(kl_divs)


def compute_value_divergence(
    baseline_trajectories: dict,
    intervention_trajectories: dict,
) -> jnp.ndarray:
    """Compute value divergence between baseline and intervention.

    Measures how much the value estimate changes under intervention.

    Args:
        baseline_trajectories: Trajectories without intervention
        intervention_trajectories: Trajectories with intervention

    Returns:
        Mean absolute value difference
    """
    baseline_values = baseline_trajectories['value']  # (T, N)
    intervention_values = intervention_trajectories['value']  # (T, N)

    return jnp.mean(jnp.abs(intervention_values - baseline_values))


def compute_reward_divergence(
    baseline_trajectories: dict,
    intervention_trajectories: dict,
) -> jnp.ndarray:
    """Compute reward divergence (proxy for goal achievement).

    Args:
        baseline_trajectories: Trajectories without intervention
        intervention_trajectories: Trajectories with intervention

    Returns:
        Difference in cumulative rewards
    """
    baseline_returns = jnp.sum(baseline_trajectories['reward'], axis=0)  # (N,)
    intervention_returns = jnp.sum(intervention_trajectories['reward'], axis=0)  # (N,)

    return jnp.mean(jnp.abs(intervention_returns - baseline_returns))


def collect_intervention_trajectories_with_predictions(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    levels: chex.ArrayTree,
    var_name: str,
    var_value: int,
    rollout_length: int,
) -> dict:
    """Collect trajectories with predictions under intervention.

    This combines intervention application with prediction collection,
    enabling measurement of how the agent's internal world model responds
    to interventions.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: Current training state
        levels: Batch of levels
        var_name: Variable to intervene on
        var_value: Value to set
        rollout_length: Number of steps

    Returns:
        Dictionary with trajectory and prediction data
    """
    num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]
    var_info = CWM_VARIABLES[var_name]
    is_persistent = var_info['type'] == 'persistent'

    # Reset to levels
    rng, rng_reset = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(rng_reset, num_levels), levels, env_params
    )

    # Apply initial intervention
    init_env_state = jax.vmap(
        lambda s: apply_intervention_to_env_state(s, var_name, var_value)
    )(init_env_state)

    # Regenerate observation from modified state
    init_obs = jax.vmap(env.get_obs, in_axes=(0, None))(init_env_state, env_params)

    init_hstate = ActorCriticWithPrediction.initialize_carry((num_levels,))

    def step_fn(carry, _):
        rng, hstate, obs, env_state, done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        # Get policy
        x = jax.tree_util.tree_map(lambda t: t[None, ...], (obs, done))
        hstate, pi, value, _ = train_state.apply_fn(train_state.params, x, hstate, None)

        action = pi.sample(seed=rng_action).squeeze(0)
        action_logits = pi.logits.squeeze(0)

        # Get predictions for the chosen action
        _, _, _, predictions = train_state.apply_fn(
            train_state.params, x, hstate, action[None, ...]
        )
        predictions = jax.tree_util.tree_map(lambda t: t[0] if t is not None else None, predictions)

        # Step environment
        next_obs, next_env_state, reward, next_done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), env_state, action, env_params)

        # Apply persistent intervention if needed
        if is_persistent:
            next_env_state = jax.vmap(
                lambda s: apply_intervention_to_env_state(s, var_name, var_value)
            )(next_env_state)

        carry = (rng, hstate, next_obs, next_env_state, next_done)
        return carry, {
            'obs': obs,
            'action': action,
            'action_logits': action_logits,
            'reward': reward,
            'done': done,
            'value': value.squeeze(0),
            'next_obs': next_obs,
            'pred_next_obs_image': predictions['next_obs_image'],
            'pred_next_dir_logits': predictions['next_obs_dir_logits'],
            'pred_reward': predictions['reward'],
        }

    _, trajectories = jax.lax.scan(
        step_fn,
        (rng, init_hstate, init_obs, init_env_state, jnp.zeros(num_levels, dtype=bool)),
        None,
        length=rollout_length,
    )

    return trajectories


def compute_divergence_matrix(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    levels: chex.ArrayTree,
    rollout_length: int,
    num_intervention_values: int,
    use_predictions: bool = True,
) -> Tuple[jnp.ndarray, dict]:
    """Compute the full divergence matrix for causal discovery using prediction-based probing.

    For each variable V_i, we intervene and measure the effect on all other
    variables V_j through the agent's prediction network. This probes the agent's
    internal world model to see what causal relationships it has learned.

    The key insight from Richens & Everitt (2024) is that robust agents must learn
    causal models. We probe this by measuring how the agent's predictions change
    under interventions on different variables.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: Current training state
        levels: Batch of levels to test on
        rollout_length: Steps per rollout
        num_intervention_values: Number of values to test per variable
        use_predictions: Whether using prediction network

    Returns:
        Tuple of:
        - divergence_matrix: (num_vars, num_vars) matrix of causal influences
        - detailed_results: Dictionary with per-variable divergence info
    """
    var_names = list(CWM_VARIABLES.keys())
    num_vars = len(var_names)

    # Get environment dimensions dynamically
    env_height, env_width = get_env_dimensions(env)
    env_size = max(env_height, env_width)

    # Collect baseline trajectories with predictions
    rng, rng_baseline = jax.random.split(rng)
    if use_predictions:
        baseline_traj = collect_prediction_data(
            rng_baseline, env, env_params, train_state, levels, rollout_length
        )
    else:
        baseline_traj = collect_baseline_trajectories(
            rng_baseline, env, env_params, train_state, levels,
            rollout_length, use_predictions
        )

    # Initialize results storage
    detailed_results = {
        'behavioral_divergences': {},
        'value_divergences': {},
        'reward_divergences': {},
        'prediction_divergences': {},  # New: prediction-based divergences
        'per_value_divergences': {},
        'per_variable_effects': {},
    }

    # Initialize per-variable effect accumulators
    # Now we track separate divergence measurements for each target variable
    per_var_effect_accum = {var_i: {var_j: [] for var_j in var_names} for var_i in var_names}

    # Define target variable mappings based on what the prediction network outputs
    # 'agent_dir' -> direction prediction divergence
    # 'goal_x', 'goal_y' -> reward prediction divergence (goal affects reward)
    # 'agent_x', 'agent_y' -> observation prediction divergence

    for i, var_i in enumerate(var_names):
        var_domain = get_variable_domain(var_i, env_size, num_intervention_values)

        var_behavioral_divs = []
        var_value_divs = []
        var_reward_divs = []
        var_pred_obs_divs = []
        var_pred_dir_divs = []
        var_pred_reward_divs = []

        for value in var_domain:
            rng, rng_intervention = jax.random.split(rng)

            # Collect intervention trajectories with predictions
            if use_predictions:
                intervention_traj = collect_intervention_trajectories_with_predictions(
                    rng_intervention, env, env_params, train_state, levels,
                    var_i, value, rollout_length
                )
            else:
                intervention_traj = collect_intervention_trajectories(
                    rng_intervention, env, env_params, train_state, levels,
                    var_i, value, rollout_length, use_predictions
                )

            # Compute behavioral divergences (always available)
            behav_div = compute_behavioral_divergence(baseline_traj, intervention_traj)
            value_div = compute_value_divergence(baseline_traj, intervention_traj)
            reward_div = compute_reward_divergence(baseline_traj, intervention_traj)

            var_behavioral_divs.append(float(behav_div))
            var_value_divs.append(float(value_div))
            var_reward_divs.append(float(reward_div))

            # Compute prediction-based divergences if using prediction network
            if use_predictions and 'pred_next_obs_image' in baseline_traj:
                # Observation prediction divergence (captures position effects)
                obs_pred_div = float(jnp.mean((
                    intervention_traj['pred_next_obs_image'] - baseline_traj['pred_next_obs_image']
                ) ** 2))
                var_pred_obs_divs.append(obs_pred_div)

                # Direction prediction divergence
                dir_pred_div = float(jnp.mean(compute_kl_divergence(
                    intervention_traj['pred_next_dir_logits'],
                    baseline_traj['pred_next_dir_logits']
                )))
                var_pred_dir_divs.append(dir_pred_div)

                # Reward prediction divergence
                reward_pred_div = float(jnp.mean((
                    intervention_traj['pred_reward'] - baseline_traj['pred_reward']
                ) ** 2))
                var_pred_reward_divs.append(reward_pred_div)

                # Map intervention effects to target variables using prediction outputs
                for j, var_j in enumerate(var_names):
                    if i != j:
                        if var_j == 'agent_dir':
                            # Direction prediction directly measures effect on agent_dir
                            effect = dir_pred_div
                        elif var_j in ['goal_x', 'goal_y']:
                            # Goal position directly affects reward predictions
                            # (reward depends on reaching goal, not arbitrary weight combo)
                            effect = reward_pred_div
                        elif var_j in ['agent_x', 'agent_y']:
                            # Agent position affects observation predictions
                            effect = obs_pred_div
                        else:
                            # Default: use behavioral divergence (policy change)
                            # No arbitrary combination - behavioral divergence is the primary signal
                            effect = float(behav_div)
                        per_var_effect_accum[var_i][var_j].append(effect)
            else:
                # Fallback when predictions not available: use behavioral divergence
                # (no arbitrary weight combo - behavioral is the direct causal signal)
                for j, var_j in enumerate(var_names):
                    if i != j:
                        effect = float(behav_div)
                        per_var_effect_accum[var_i][var_j].append(effect)

        # Store per-value divergences for detailed analysis
        detailed_results['per_value_divergences'][var_i] = {
            'values': var_domain,
            'behavioral': var_behavioral_divs,
            'value': var_value_divs,
            'reward': var_reward_divs,
        }

        if var_pred_obs_divs:
            detailed_results['per_value_divergences'][var_i].update({
                'pred_obs': var_pred_obs_divs,
                'pred_dir': var_pred_dir_divs,
                'pred_reward': var_pred_reward_divs,
            })

        # Average divergence across all tested values
        mean_behav_div = np.mean(var_behavioral_divs)
        mean_value_div = np.mean(var_value_divs)
        mean_reward_div = np.mean(var_reward_divs)

        detailed_results['behavioral_divergences'][var_i] = mean_behav_div
        detailed_results['value_divergences'][var_i] = mean_value_div
        detailed_results['reward_divergences'][var_i] = mean_reward_div

        if var_pred_obs_divs:
            detailed_results['prediction_divergences'][var_i] = {
                'obs': np.mean(var_pred_obs_divs),
                'dir': np.mean(var_pred_dir_divs),
                'reward': np.mean(var_pred_reward_divs),
            }

    # Build divergence matrix using prediction-based effects
    # D[i, j] represents the causal influence of variable i on variable j
    divergence_matrix = np.zeros((num_vars, num_vars))

    for i, var_i in enumerate(var_names):
        for j, var_j in enumerate(var_names):
            if i != j and per_var_effect_accum[var_i][var_j]:
                divergence_matrix[i, j] = np.mean(per_var_effect_accum[var_i][var_j])

    # Store per-variable effects in detailed results
    detailed_results['per_variable_effects'] = {
        var_i: {var_j: np.mean(effects) if effects else 0.0
                for var_j, effects in var_j_effects.items()}
        for var_i, var_j_effects in per_var_effect_accum.items()
    }

    return jnp.array(divergence_matrix), detailed_results


def build_causal_graph(
    divergence_matrix: jnp.ndarray,
    var_names: list,
    threshold: float,
    use_statistical_test: bool = True,
    significance_level: float = 0.05,
) -> dict:
    """Build a causal graph from the divergence matrix using principled edge selection.

    Instead of using an arbitrary threshold, this function can use statistical
    significance testing to determine which edges to include. An edge is included
    if the divergence is significantly greater than the baseline noise level.

    Args:
        divergence_matrix: (num_vars, num_vars) matrix of divergences
        var_names: List of variable names
        threshold: Minimum divergence to include an edge (used if use_statistical_test=False)
        use_statistical_test: Whether to use statistical significance testing
        significance_level: P-value threshold for edge inclusion

    Returns:
        Dictionary representing the causal graph:
        {
            'nodes': list of variable names,
            'edges': list of (source, target, weight) tuples,
            'adjacency': dict mapping source -> list of (target, weight),
            'matrix': the divergence matrix,
            'threshold_used': the actual threshold used for edge selection,
            'edge_statistics': dict with statistical info for each edge,
        }
    """
    num_vars = len(var_names)
    matrix = np.array(divergence_matrix)
    edges = []
    adjacency = {var: [] for var in var_names}
    edge_statistics = {}

    if use_statistical_test:
        # Compute baseline statistics from the matrix
        # Edges from a variable to itself are 0, so exclude diagonal
        off_diagonal = matrix[~np.eye(num_vars, dtype=bool)]
        non_zero_values = off_diagonal[off_diagonal > 0]

        if len(non_zero_values) > 2:
            # Use median and MAD (median absolute deviation) for robust statistics
            median_val = np.median(non_zero_values)
            mad = np.median(np.abs(non_zero_values - median_val))
            # MAD-based threshold: values > median + k*MAD are significant
            # Using k=MAD_MULTIPLIER corresponds roughly to 2 standard deviations for normal dist
            adaptive_threshold = median_val + MAD_MULTIPLIER * mad * MAD_TO_STD_FACTOR

            # Also consider percentile-based threshold
            percentile_threshold = np.percentile(non_zero_values, 75)

            # Use the more conservative (higher) threshold
            effective_threshold = max(adaptive_threshold, percentile_threshold, threshold)
        else:
            # Not enough data for statistical test, fall back to provided threshold
            effective_threshold = threshold
    else:
        effective_threshold = threshold

    for i in range(num_vars):
        for j in range(num_vars):
            weight = float(matrix[i, j])
            if i != j and weight > 0:
                # Store statistics for all potential edges
                edge_key = f"{var_names[i]}->{var_names[j]}"
                edge_statistics[edge_key] = {
                    'weight': weight,
                    'threshold': effective_threshold,
                    'significant': weight > effective_threshold,
                }

                if weight > effective_threshold:
                    edges.append((var_names[i], var_names[j], weight))
                    adjacency[var_names[i]].append((var_names[j], weight))

    return {
        'nodes': var_names,
        'edges': edges,
        'adjacency': adjacency,
        'matrix': matrix,
        'threshold_used': effective_threshold,
        'edge_statistics': edge_statistics,
    }


def apply_position_intervention(
    env_state: EnvState,
    var_prefix: str,
    x: int,
    y: int,
) -> EnvState:
    """Apply a 2D position intervention (set both x and y at once).

    Args:
        env_state: Current environment state
        var_prefix: 'agent' or 'goal'
        x: X coordinate to set
        y: Y coordinate to set

    Returns:
        Modified environment state
    """
    if var_prefix == 'agent':
        new_pos = jnp.array([x, y])
        return env_state.replace(agent_pos=new_pos)
    elif var_prefix == 'goal':
        new_pos = jnp.array([x, y])
        return env_state.replace(goal_pos=new_pos)
    else:
        raise ValueError(f"Unknown var_prefix: {var_prefix}")


def collect_position_intervention_trajectories(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    levels: chex.ArrayTree,
    var_prefix: str,
    x: int,
    y: int,
    rollout_length: int,
    use_predictions: bool = True,
) -> dict:
    """Collect trajectories with a 2D position intervention.

    This properly sets BOTH x and y coordinates simultaneously.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: Current training state
        levels: Batch of levels
        var_prefix: 'agent' or 'goal'
        x: X coordinate
        y: Y coordinate
        rollout_length: Number of steps
        use_predictions: Whether using prediction network

    Returns:
        Trajectory dictionary

    Raises:
        ValueError: If var_prefix is not 'agent' or 'goal'
    """
    if var_prefix not in ('agent', 'goal'):
        raise ValueError(f"var_prefix must be 'agent' or 'goal', got: {var_prefix}")

    num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]

    # Reset to levels
    rng, rng_reset = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(rng_reset, num_levels), levels, env_params
    )

    # Apply 2D position intervention
    init_env_state = jax.vmap(
        lambda s: apply_position_intervention(s, var_prefix, x, y)
    )(init_env_state)

    # Regenerate observation from modified state
    init_obs = jax.vmap(env.get_obs, in_axes=(0, None))(init_env_state, env_params)

    if use_predictions:
        init_hstate = ActorCriticWithPrediction.initialize_carry((num_levels,))
    else:
        init_hstate = ActorCritic.initialize_carry((num_levels,))

    def step_fn(carry, _):
        rng, hstate, obs, env_state, done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        ac_input = jax.tree_util.tree_map(lambda t: t[None, ...], (obs, done))

        if use_predictions:
            hstate, pi, value, _ = train_state.apply_fn(train_state.params, ac_input, hstate, None)
        else:
            hstate, pi, value = train_state.apply_fn(train_state.params, ac_input, hstate)

        action = pi.sample(seed=rng_action)
        action_logits = pi.logits

        value, action, action_logits = (
            value.squeeze(0),
            action.squeeze(0),
            action_logits.squeeze(0),
        )

        next_obs, next_env_state, reward, next_done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), env_state, action, env_params)

        carry = (rng, hstate, next_obs, next_env_state, next_done)
        return carry, {
            'obs': obs,
            'action': action,
            'action_logits': action_logits,
            'reward': reward,
            'done': done,
            'value': value,
            'next_obs': next_obs,
        }

    _, trajectories = jax.lax.scan(
        step_fn,
        (rng, init_hstate, init_obs, init_env_state, jnp.zeros(num_levels, dtype=bool)),
        None,
        length=rollout_length,
    )

    return trajectories


def collect_batched_position_interventions(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    level: chex.ArrayTree,
    var_prefix: str,
    positions: list,
    rollout_length: int,
    use_predictions: bool = True,
) -> dict:
    """Collect trajectories for multiple positions in a batched manner.

    This batches multiple position interventions together for efficiency.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: Training state
        level: Single level to test on
        var_prefix: 'agent' or 'goal'
        positions: List of (x, y) tuples to intervene on
        rollout_length: Steps per rollout
        use_predictions: Whether using prediction network

    Returns:
        Dictionary with trajectory data for each position

    Raises:
        ValueError: If var_prefix is not 'agent' or 'goal'
    """
    if var_prefix not in ('agent', 'goal'):
        raise ValueError(f"var_prefix must be 'agent' or 'goal', got: {var_prefix}")

    num_positions = len(positions)
    if num_positions == 0:
        return {}

    # Create batched levels - replicate level for each position
    levels = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x[None, ...], num_positions, axis=0),
        level
    )

    # Reset to levels
    rng, rng_reset = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(rng_reset, num_positions), levels, env_params
    )

    # Apply position interventions - each position gets its own intervention
    xs = jnp.array([p[0] for p in positions])
    ys = jnp.array([p[1] for p in positions])

    def apply_single_position_intervention(env_state, x, y):
        return apply_position_intervention(env_state, var_prefix, x, y)

    init_env_state = jax.vmap(apply_single_position_intervention)(init_env_state, xs, ys)

    # Regenerate observations from modified states
    init_obs = jax.vmap(env.get_obs, in_axes=(0, None))(init_env_state, env_params)

    if use_predictions:
        init_hstate = ActorCriticWithPrediction.initialize_carry((num_positions,))
    else:
        init_hstate = ActorCritic.initialize_carry((num_positions,))

    def step_fn(carry, _):
        rng, hstate, obs, env_state, done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        ac_input = jax.tree_util.tree_map(lambda t: t[None, ...], (obs, done))

        if use_predictions:
            hstate, pi, value, _ = train_state.apply_fn(train_state.params, ac_input, hstate, None)
        else:
            hstate, pi, value = train_state.apply_fn(train_state.params, ac_input, hstate)

        action = pi.sample(seed=rng_action)
        action_logits = pi.logits

        value, action, action_logits = (
            value.squeeze(0),
            action.squeeze(0),
            action_logits.squeeze(0),
        )

        next_obs, next_env_state, reward, next_done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_positions), env_state, action, env_params)

        carry = (rng, hstate, next_obs, next_env_state, next_done)
        return carry, {
            'obs': obs,
            'action': action,
            'action_logits': action_logits,
            'reward': reward,
            'done': done,
            'value': value,
            'next_obs': next_obs,
        }

    _, trajectories = jax.lax.scan(
        step_fn,
        (rng, init_hstate, init_obs, init_env_state, jnp.zeros(num_positions, dtype=bool)),
        None,
        length=rollout_length,
    )

    return trajectories


def collect_spatial_heatmap_data(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    level: chex.ArrayTree,
    var_prefix: str,
    rollout_length: int,
    use_predictions: bool = True,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect data for spatial heatmap visualization.

    For a position-based variable (agent or goal), compute metrics
    at each valid grid position by intervening on both x and y simultaneously.

    Uses batched intervention collection for efficiency.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: Training state
        level: Single level to test on
        var_prefix: 'agent' or 'goal' to determine which position to vary
        rollout_length: Steps per rollout
        use_predictions: Whether using prediction network
        batch_size: Number of positions to process in each batch

    Returns:
        Tuple of:
        - returns_grid: (H, W) average returns at each position
        - behavioral_div_grid: (H, W) behavioral divergence at each position
        - value_div_grid: (H, W) value divergence at each position
    """
    env_height, env_width = get_env_dimensions(env)

    # Get wall map from level to identify valid positions
    wall_map = np.array(level.wall_map)

    # Get all valid positions
    valid_positions = get_valid_positions(wall_map)

    if len(valid_positions) == 0:
        return (np.full((env_height, env_width), np.nan),
                np.full((env_height, env_width), np.nan),
                np.full((env_height, env_width), np.nan))

    # Collect baseline trajectory (using single level)
    levels_single = jax.tree_util.tree_map(lambda x: x[None, ...], level)
    rng, rng_baseline = jax.random.split(rng)
    baseline_traj = collect_baseline_trajectories(
        rng_baseline, env, env_params, train_state, levels_single,
        rollout_length, use_predictions
    )

    # Initialize grids
    returns_grid = np.full((env_height, env_width), np.nan)
    behavioral_div_grid = np.full((env_height, env_width), np.nan)
    value_div_grid = np.full((env_height, env_width), np.nan)

    # Process positions in batches
    for batch_start in range(0, len(valid_positions), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_positions))
        batch_positions = valid_positions[batch_start:batch_end]

        rng, rng_batch = jax.random.split(rng)

        # Collect trajectories for all positions in this batch
        batch_traj = collect_batched_position_interventions(
            rng_batch, env, env_params, train_state, level,
            var_prefix, batch_positions, rollout_length, use_predictions
        )

        # Compute metrics for each position in the batch
        for i, (x, y) in enumerate(batch_positions):
            # Extract single-position trajectory
            pos_traj = jax.tree_util.tree_map(lambda arr: arr[:, i:i+1], batch_traj)

            # Compute metrics
            intervention_return = float(jnp.sum(pos_traj['reward']))
            behav_div = float(compute_behavioral_divergence(baseline_traj, pos_traj))
            value_div = float(compute_value_divergence(baseline_traj, pos_traj))

            returns_grid[y, x] = intervention_return
            behavioral_div_grid[y, x] = behav_div
            value_div_grid[y, x] = value_div

    return returns_grid, behavioral_div_grid, value_div_grid


def extract_causal_world_model(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    sample_levels_fn,
    config: dict,
    use_predictions: bool = True,
) -> dict:
    """Main entry point for causal world model extraction.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: Current training state
        sample_levels_fn: Function to sample levels
        config: Configuration dictionary
        use_predictions: Whether using prediction network

    Returns:
        Dictionary containing:
        - divergence_matrix: Raw divergence values
        - causal_graph: Structured graph representation
        - heatmaps: Spatial visualization data
        - metadata: Extraction parameters
    """
    num_sample_states = config.get('cwm_num_sample_states', 50)
    rollout_length = config.get('cwm_rollout_length', 30)
    threshold = config.get('cwm_threshold', 0.1)
    num_intervention_values = config.get('cwm_num_intervention_values', 5)

    print("CWM Extraction: Sampling levels...")
    # Sample levels for testing
    rng, rng_levels = jax.random.split(rng)
    levels = jax.vmap(sample_levels_fn)(jax.random.split(rng_levels, num_sample_states))

    print("CWM Extraction: Computing divergence matrix...")
    # Compute divergence matrix
    rng, rng_divergence = jax.random.split(rng)
    divergence_matrix, detailed_results = compute_divergence_matrix(
        rng_divergence, env, env_params, train_state, levels,
        rollout_length, num_intervention_values, use_predictions
    )

    print("CWM Extraction: Building causal graph...")
    # Build causal graph
    var_names = list(CWM_VARIABLES.keys())
    causal_graph = build_causal_graph(divergence_matrix, var_names, threshold)

    print("CWM Extraction: Collecting spatial heatmap data...")
    # Collect spatial heatmap data for visualization
    sample_level = jax.tree_util.tree_map(lambda x: x[0], levels)

    # Goal position heatmap
    rng, rng_goal = jax.random.split(rng)
    goal_returns, goal_behav_div, goal_value_div = collect_spatial_heatmap_data(
        rng_goal, env, env_params, train_state, sample_level,
        'goal', rollout_length, use_predictions
    )

    # Agent spawn position heatmap
    rng, rng_agent = jax.random.split(rng)
    agent_returns, agent_behav_div, agent_value_div = collect_spatial_heatmap_data(
        rng_agent, env, env_params, train_state, sample_level,
        'agent', rollout_length, use_predictions
    )

    print("CWM Extraction: Complete!")

    return {
        'divergence_matrix': np.array(divergence_matrix),
        'causal_graph': causal_graph,
        'detailed_results': detailed_results,
        'heatmaps': {
            'goal_position': {
                'returns': goal_returns,
                'behavioral_divergence': goal_behav_div,
                'value_divergence': goal_value_div,
            },
            'agent_spawn': {
                'returns': agent_returns,
                'behavioral_divergence': agent_behav_div,
                'value_divergence': agent_value_div,
            },
        },
        'metadata': {
            'num_sample_states': num_sample_states,
            'rollout_length': rollout_length,
            'threshold': threshold,
            'num_intervention_values': num_intervention_values,
            'variables': var_names,
        },
        'sample_level': sample_level,
    }


# =============================================================================
# ADVANCED CAUSAL ANALYSIS FUNCTIONS
# =============================================================================

def evaluate_counterfactual(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    levels: chex.ArrayTree,
    intervention_var: str,
    factual_value: int,
    counterfactual_value: int,
    rollout_length: int = 30,
) -> dict:
    """Evaluate counterfactual queries: "What would the agent have done if X had been different?"

    This compares:
    - π(a | obs) under factual condition (X = factual_value)
    - π(a | obs') under counterfactual condition (X = counterfactual_value)

    The key insight is that if the agent has learned a causal model, it should
    correctly predict how its behavior would change under the counterfactual.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: Trained agent
        levels: Batch of levels to test on
        intervention_var: Variable to intervene on
        factual_value: The actual value of the variable
        counterfactual_value: The counterfactual "what if" value
        rollout_length: Steps to rollout

    Returns:
        Dictionary containing:
        - 'factual_trajectories': Trajectories under factual condition
        - 'counterfactual_trajectories': Trajectories under counterfactual
        - 'policy_divergence': KL divergence between action distributions
        - 'value_divergence': Difference in value estimates
        - 'behavioral_divergence': How much actions actually differ
        - 'prediction_divergence': How much predictions differ (if available)
        - 'counterfactual_effect': Estimated causal effect of the intervention
    """
    num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]

    # Collect factual trajectories (with the actual value)
    rng, rng_factual = jax.random.split(rng)
    factual_traj = collect_intervention_trajectories_with_predictions(
        rng_factual, env, env_params, train_state, levels,
        intervention_var, factual_value, rollout_length
    )

    # Collect counterfactual trajectories (with the "what if" value)
    rng, rng_counterfactual = jax.random.split(rng)
    counterfactual_traj = collect_intervention_trajectories_with_predictions(
        rng_counterfactual, env, env_params, train_state, levels,
        intervention_var, counterfactual_value, rollout_length
    )

    # Compute divergences
    policy_div = compute_kl_divergence(
        counterfactual_traj['action_logits'],
        factual_traj['action_logits']
    )

    value_div = jnp.abs(counterfactual_traj['value'] - factual_traj['value'])

    # Behavioral divergence: how often do actions actually differ?
    action_match = (counterfactual_traj['action'] == factual_traj['action']).astype(float)
    behavioral_agreement = jnp.mean(action_match)

    # Prediction divergences
    obs_pred_div = jnp.mean((
        counterfactual_traj['pred_next_obs_image'] - factual_traj['pred_next_obs_image']
    ) ** 2)

    dir_pred_div = compute_kl_divergence(
        counterfactual_traj['pred_next_dir_logits'],
        factual_traj['pred_next_dir_logits']
    )

    reward_pred_div = jnp.mean((
        counterfactual_traj['pred_reward'] - factual_traj['pred_reward']
    ) ** 2)

    # Compute actual outcome differences
    factual_returns = jnp.sum(factual_traj['reward'], axis=0)
    counterfactual_returns = jnp.sum(counterfactual_traj['reward'], axis=0)
    causal_effect_on_return = jnp.mean(counterfactual_returns - factual_returns)

    return {
        'factual_trajectories': factual_traj,
        'counterfactual_trajectories': counterfactual_traj,
        'intervention_var': intervention_var,
        'factual_value': factual_value,
        'counterfactual_value': counterfactual_value,
        # Divergence metrics
        'policy_divergence': float(jnp.mean(policy_div)),
        'policy_divergence_per_step': np.array(jnp.mean(policy_div, axis=1)),
        'value_divergence': float(jnp.mean(value_div)),
        'value_divergence_per_step': np.array(jnp.mean(value_div, axis=1)),
        'behavioral_agreement': float(behavioral_agreement),
        'behavioral_divergence': float(1.0 - behavioral_agreement),
        # Prediction divergences
        'prediction_divergence': {
            'obs': float(obs_pred_div),
            'dir': float(jnp.mean(dir_pred_div)),
            'reward': float(reward_pred_div),
        },
        # Causal effect estimates
        'factual_mean_return': float(jnp.mean(factual_returns)),
        'counterfactual_mean_return': float(jnp.mean(counterfactual_returns)),
        'causal_effect_on_return': float(causal_effect_on_return),
    }


def evaluate_counterfactual_suite(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    levels: chex.ArrayTree,
    rollout_length: int = 30,
) -> dict:
    """Run a full suite of counterfactual evaluations across all variable pairs.

    For each variable, tests multiple counterfactual scenarios to build a complete
    picture of the agent's counterfactual reasoning.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: Trained agent
        levels: Batch of levels
        rollout_length: Steps per rollout

    Returns:
        Dictionary with counterfactual results for each variable
    """
    env_height, env_width = get_env_dimensions(env)
    env_size = max(env_height, env_width)

    results = {}

    for var_name in CWM_VARIABLES.keys():
        var_info = CWM_VARIABLES[var_name]

        # Get domain of values
        if var_info['domain_type'] == 'discrete':
            domain = var_info['domain']
        else:
            # For spatial, use corners and center
            domain = [1, env_size // 2, env_size - 2]

        var_results = []

        # Test factual vs each counterfactual
        for i, factual in enumerate(domain):
            for j, counterfactual in enumerate(domain):
                if i != j:  # Only test actual counterfactuals
                    rng, rng_eval = jax.random.split(rng)
                    cf_result = evaluate_counterfactual(
                        rng_eval, env, env_params, train_state, levels,
                        var_name, factual, counterfactual, rollout_length
                    )
                    var_results.append({
                        'factual': factual,
                        'counterfactual': counterfactual,
                        'policy_divergence': cf_result['policy_divergence'],
                        'behavioral_divergence': cf_result['behavioral_divergence'],
                        'causal_effect': cf_result['causal_effect_on_return'],
                    })

        # Summarize for this variable
        results[var_name] = {
            'counterfactual_tests': var_results,
            'mean_policy_divergence': np.mean([r['policy_divergence'] for r in var_results]),
            'mean_behavioral_divergence': np.mean([r['behavioral_divergence'] for r in var_results]),
            'mean_causal_effect': np.mean([np.abs(r['causal_effect']) for r in var_results]),
            'max_causal_effect': max([np.abs(r['causal_effect']) for r in var_results]),
        }

    return results


def analyze_temporal_causal_effects(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    levels: chex.ArrayTree,
    intervention_var: str,
    intervention_value: int,
    max_horizon: int = 30,
) -> dict:
    """Analyze how causal effects propagate and decay over time.

    Intervenes at t=0 and measures divergence at each subsequent timestep.
    This reveals:
    - How quickly the agent's behavior diverges under intervention
    - How long the causal effect persists
    - Whether effects accumulate or decay over time

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: Trained agent
        levels: Batch of levels
        intervention_var: Variable to intervene on
        intervention_value: Value to set
        max_horizon: Maximum timesteps to analyze

    Returns:
        Dictionary containing:
        - 'temporal_policy_divergence': Divergence at each timestep
        - 'temporal_value_divergence': Value diff at each timestep
        - 'temporal_behavioral_divergence': Action disagreement at each timestep
        - 'temporal_prediction_divergence': Prediction diff at each timestep
        - 'causal_onset': Timestep when divergence first becomes significant
        - 'causal_peak': Timestep of maximum divergence
        - 'causal_decay_rate': Rate at which effect decays (if applicable)
        - 'cumulative_effect': Accumulated effect over time
    """
    num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]

    # Collect baseline trajectories (no intervention)
    rng, rng_baseline = jax.random.split(rng)
    baseline_traj = collect_prediction_data(
        rng_baseline, env, env_params, train_state, levels, max_horizon
    )

    # Collect intervention trajectories
    rng, rng_intervention = jax.random.split(rng)
    intervention_traj = collect_intervention_trajectories_with_predictions(
        rng_intervention, env, env_params, train_state, levels,
        intervention_var, intervention_value, max_horizon
    )

    # Compute per-timestep divergences
    # Policy divergence (KL) at each timestep
    policy_div_per_step = []
    for t in range(max_horizon):
        kl = compute_kl_divergence(
            intervention_traj['action_logits'][t],
            baseline_traj['action_logits'][t]
        )
        policy_div_per_step.append(float(jnp.mean(kl)))

    # Value divergence at each timestep
    value_div_per_step = []
    for t in range(max_horizon):
        v_diff = jnp.abs(intervention_traj['value'][t] - baseline_traj['value'][t])
        value_div_per_step.append(float(jnp.mean(v_diff)))

    # Behavioral divergence (action disagreement) at each timestep
    behavioral_div_per_step = []
    for t in range(max_horizon):
        action_match = (intervention_traj['action'][t] == baseline_traj['action'][t]).astype(float)
        behavioral_div_per_step.append(float(1.0 - jnp.mean(action_match)))

    # Prediction divergences at each timestep
    obs_pred_div_per_step = []
    dir_pred_div_per_step = []
    reward_pred_div_per_step = []

    for t in range(max_horizon):
        obs_div = jnp.mean((
            intervention_traj['pred_next_obs_image'][t] - baseline_traj['pred_next_obs_image'][t]
        ) ** 2)
        obs_pred_div_per_step.append(float(obs_div))

        dir_div = compute_kl_divergence(
            intervention_traj['pred_next_dir_logits'][t],
            baseline_traj['pred_next_dir_logits'][t]
        )
        dir_pred_div_per_step.append(float(jnp.mean(dir_div)))

        reward_div = jnp.mean((
            intervention_traj['pred_reward'][t] - baseline_traj['pred_reward'][t]
        ) ** 2)
        reward_pred_div_per_step.append(float(reward_div))

    # Analyze temporal dynamics
    policy_div_array = np.array(policy_div_per_step)

    # Find causal onset: first timestep where divergence exceeds noise threshold
    # Use early timesteps for noise estimation (before intervention effects propagate)
    noise_window = max(3, int(max_horizon * NOISE_ESTIMATION_FRACTION))
    noise_threshold = (np.mean(policy_div_array[:noise_window]) +
                       SIGNIFICANCE_STD_MULTIPLIER * np.std(policy_div_array[:noise_window])) if max_horizon >= noise_window else 0
    # Use noise threshold computed from early timesteps; no arbitrary fallback needed
    # If noise_threshold is 0, we accept any positive divergence as significant
    significant_steps = np.where(policy_div_array > noise_threshold)[0]
    causal_onset = int(significant_steps[0]) if len(significant_steps) > 0 else -1

    # Find peak divergence
    causal_peak = int(np.argmax(policy_div_array))
    peak_divergence = float(policy_div_array[causal_peak])

    # Estimate decay rate (if there's a peak before the end)
    decay_rate = 0.0
    if causal_peak < max_horizon - MIN_DECAY_WINDOW and peak_divergence > 0:
        # Fit exponential decay after peak
        post_peak = policy_div_array[causal_peak:]
        if len(post_peak) > 2 and post_peak[0] > 0:
            # Simple decay rate: (final - peak) / (time * peak)
            decay_rate = float((post_peak[-1] - post_peak[0]) / (len(post_peak) * post_peak[0]))

    # Cumulative effect: sum of divergences (area under curve)
    cumulative_policy_effect = float(np.sum(policy_div_array))
    cumulative_behavioral_effect = float(np.sum(behavioral_div_per_step))

    # Reward difference over time
    baseline_cumulative_reward = np.cumsum(np.mean(baseline_traj['reward'], axis=1))
    intervention_cumulative_reward = np.cumsum(np.mean(intervention_traj['reward'], axis=1))
    reward_gap_per_step = list(intervention_cumulative_reward - baseline_cumulative_reward)

    return {
        'intervention_var': intervention_var,
        'intervention_value': intervention_value,
        'max_horizon': max_horizon,
        # Per-timestep divergences
        'temporal_policy_divergence': policy_div_per_step,
        'temporal_value_divergence': value_div_per_step,
        'temporal_behavioral_divergence': behavioral_div_per_step,
        'temporal_prediction_divergence': {
            'obs': obs_pred_div_per_step,
            'dir': dir_pred_div_per_step,
            'reward': reward_pred_div_per_step,
        },
        # Temporal dynamics metrics
        'causal_onset': causal_onset,
        'causal_peak': causal_peak,
        'peak_divergence': peak_divergence,
        'causal_decay_rate': decay_rate,
        # Cumulative effects
        'cumulative_policy_effect': cumulative_policy_effect,
        'cumulative_behavioral_effect': cumulative_behavioral_effect,
        'reward_gap_per_step': reward_gap_per_step,
        'final_reward_gap': float(reward_gap_per_step[-1]) if reward_gap_per_step else 0.0,
    }


def analyze_temporal_causal_effects_all_variables(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    levels: chex.ArrayTree,
    max_horizon: int = 30,
    num_intervention_values: int = 3,
) -> dict:
    """Run temporal causal analysis for all variables.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: Trained agent
        levels: Batch of levels
        max_horizon: Maximum timesteps
        num_intervention_values: Values to test per variable

    Returns:
        Dictionary with temporal analysis for each variable
    """
    env_height, env_width = get_env_dimensions(env)
    env_size = max(env_height, env_width)

    results = {}

    for var_name in CWM_VARIABLES.keys():
        domain = get_variable_domain(var_name, env_size, num_intervention_values)

        var_temporal_results = []
        for value in domain:
            rng, rng_temporal = jax.random.split(rng)
            temporal_result = analyze_temporal_causal_effects(
                rng_temporal, env, env_params, train_state, levels,
                var_name, value, max_horizon
            )
            var_temporal_results.append(temporal_result)

        # Aggregate across intervention values
        avg_onset = np.mean([r['causal_onset'] for r in var_temporal_results if r['causal_onset'] >= 0])
        avg_peak = np.mean([r['causal_peak'] for r in var_temporal_results])
        avg_peak_div = np.mean([r['peak_divergence'] for r in var_temporal_results])
        avg_cumulative = np.mean([r['cumulative_policy_effect'] for r in var_temporal_results])

        # Average temporal curves
        avg_policy_curve = np.mean([r['temporal_policy_divergence'] for r in var_temporal_results], axis=0)
        avg_behavioral_curve = np.mean([r['temporal_behavioral_divergence'] for r in var_temporal_results], axis=0)

        results[var_name] = {
            'per_value_results': var_temporal_results,
            'avg_causal_onset': float(avg_onset) if not np.isnan(avg_onset) else -1,
            'avg_causal_peak': float(avg_peak),
            'avg_peak_divergence': float(avg_peak_div),
            'avg_cumulative_effect': float(avg_cumulative),
            'avg_temporal_policy_curve': list(avg_policy_curve),
            'avg_temporal_behavioral_curve': list(avg_behavioral_curve),
        }

    return results


def compute_intervention_efficiency(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    levels: chex.ArrayTree,
    target_accuracy: float = 0.9,
    max_interventions: int = 50,
    rollout_length: int = 20,
) -> dict:
    """Analyze intervention efficiency: how many interventions needed to identify causal structure?

    Implements a simple active causal discovery approach:
    1. Start with uniform prior over possible causal graphs
    2. Select interventions that maximize information gain
    3. Update beliefs based on observed divergences
    4. Stop when confidence exceeds target_accuracy

    This measures how efficiently the agent's causal structure can be identified,
    which relates to how clearly the agent has encoded causal relationships.

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: Trained agent
        levels: Batch of levels
        target_accuracy: Confidence threshold to stop
        max_interventions: Maximum interventions to try
        rollout_length: Steps per intervention test

    Returns:
        Dictionary containing:
        - 'interventions_used': Number of interventions to reach target accuracy
        - 'intervention_sequence': List of interventions performed
        - 'information_gain_per_intervention': Info gained at each step
        - 'confidence_trajectory': Confidence over interventions
        - 'final_causal_graph': Best estimate of causal structure
        - 'efficiency_score': Higher = more clearly encoded causal structure
    """
    var_names = list(CWM_VARIABLES.keys())
    num_vars = len(var_names)
    env_height, env_width = get_env_dimensions(env)
    env_size = max(env_height, env_width)

    # Initialize: uniform prior on edge existence
    # edge_beliefs[i][j] = P(edge from var_i to var_j exists)
    edge_beliefs = np.ones((num_vars, num_vars)) * UNIFORM_PRIOR
    np.fill_diagonal(edge_beliefs, 0)  # No self-loops

    # Track observations for each potential edge
    edge_observations = {(i, j): [] for i in range(num_vars) for j in range(num_vars) if i != j}

    # Track all observed effects to compute normalization from data
    all_observed_effects = []

    intervention_sequence = []
    information_gain_sequence = []
    confidence_trajectory = []

    # Collect baseline once
    rng, rng_baseline = jax.random.split(rng)
    baseline_traj = collect_prediction_data(
        rng_baseline, env, env_params, train_state, levels, rollout_length
    )

    for intervention_idx in range(max_interventions):
        # Compute current confidence (how certain are we about the graph?)
        # Confidence = mean distance from UNIFORM_PRIOR (uncertain) for all edges
        confidence = np.mean(np.abs(edge_beliefs - UNIFORM_PRIOR) * 2)
        confidence_trajectory.append(confidence)

        if confidence >= target_accuracy:
            break

        # Select next intervention: choose variable with most uncertain outgoing edges
        uncertainty_per_var = []
        for i in range(num_vars):
            # Uncertainty of edges FROM this variable
            outgoing_uncertainty = np.mean([
                1 - np.abs(edge_beliefs[i, j] - UNIFORM_PRIOR) * 2
                for j in range(num_vars) if i != j
            ])
            uncertainty_per_var.append(outgoing_uncertainty)

        # Select most uncertain variable (with some randomness for exploration)
        rng, rng_select = jax.random.split(rng)
        uncertainty_probs = np.array(uncertainty_per_var) / sum(uncertainty_per_var)
        selected_var_idx = int(jax.random.choice(rng_select, num_vars, p=uncertainty_probs))
        selected_var = var_names[selected_var_idx]

        # Choose intervention value
        domain = get_variable_domain(selected_var, env_size, 3)
        rng, rng_value = jax.random.split(rng)
        selected_value = int(jax.random.choice(rng_value, jnp.array(domain)))

        intervention_sequence.append({
            'variable': selected_var,
            'value': selected_value,
            'variable_idx': selected_var_idx,
        })

        # Perform intervention
        rng, rng_intervene = jax.random.split(rng)
        intervention_traj = collect_intervention_trajectories_with_predictions(
            rng_intervene, env, env_params, train_state, levels,
            selected_var, selected_value, rollout_length
        )

        # Measure effects on all other variables
        # For each potential target variable, compute divergence
        effects = {}

        # Behavioral divergence (overall)
        behav_div = float(compute_behavioral_divergence(baseline_traj, intervention_traj))

        # Direction prediction divergence -> agent_dir
        dir_div = float(jnp.mean(compute_kl_divergence(
            intervention_traj['pred_next_dir_logits'],
            baseline_traj['pred_next_dir_logits']
        )))

        # Observation prediction divergence -> agent position
        obs_div = float(jnp.mean((
            intervention_traj['pred_next_obs_image'] - baseline_traj['pred_next_obs_image']
        ) ** 2))

        # Reward prediction divergence -> goal position
        reward_div = float(jnp.mean((
            intervention_traj['pred_reward'] - baseline_traj['pred_reward']
        ) ** 2))

        # Map to target variables
        for j, target_var in enumerate(var_names):
            if j != selected_var_idx:
                if target_var == 'agent_dir':
                    effect = dir_div
                elif target_var in ['agent_x', 'agent_y']:
                    effect = obs_div
                elif target_var in ['goal_x', 'goal_y']:
                    # Goal directly affects reward, no arbitrary weighting
                    effect = reward_div
                else:
                    effect = behav_div

                effects[target_var] = effect
                edge_observations[(selected_var_idx, j)].append(effect)
                all_observed_effects.append(effect)

        # Update beliefs using simple Bayesian update
        # P(edge | high divergence) increases, P(edge | low divergence) decreases
        prior_info_gain = 0.0

        # Compute effect normalization from observed data
        effect_baseline, effect_cap = compute_effect_normalization(np.array(all_observed_effects))

        for j, target_var in enumerate(var_names):
            if j != selected_var_idx:
                effect = effects[target_var]
                old_belief = edge_beliefs[selected_var_idx, j]

                # Simple update: high effect -> edge likely exists
                # Using sigmoid-like update with data-driven normalization
                effect_normalized = min(effect / effect_baseline, effect_cap)
                likelihood_ratio = 1 + effect_normalized  # Higher effect = more likely edge

                # Bayesian update: P(edge|data) ∝ P(data|edge) * P(edge)
                new_belief = (likelihood_ratio * old_belief) / (
                    likelihood_ratio * old_belief + (1 - old_belief)
                )
                new_belief = np.clip(new_belief, BELIEF_LOWER_BOUND, BELIEF_UPPER_BOUND)  # Prevent certainty

                # Information gain
                prior_info_gain += abs(new_belief - old_belief)
                edge_beliefs[selected_var_idx, j] = new_belief

        information_gain_sequence.append(prior_info_gain)

    # Final causal graph estimate
    final_graph_edges = []
    edge_confidence = {}

    # Compute edge inclusion threshold based on number of observations per edge
    # Each edge (i,j) has been observed when variable i was intervened on
    # Total observations per edge = number of interventions on source variable
    intervention_counts = {}
    for intervention in intervention_sequence:
        var_idx = intervention['variable_idx']
        intervention_counts[var_idx] = intervention_counts.get(var_idx, 0) + 1

    for i in range(num_vars):
        for j in range(num_vars):
            if i != j:
                belief = edge_beliefs[i, j]
                edge_key = f"{var_names[i]}->{var_names[j]}"

                # Use observation count for this edge to compute threshold
                n_obs = intervention_counts.get(i, 0)
                threshold = compute_edge_inclusion_threshold(n_obs)

                edge_confidence[edge_key] = float(belief)
                if belief > threshold:
                    final_graph_edges.append((var_names[i], var_names[j], float(belief)))

    # Efficiency score: higher if structure identified quickly
    # Score = target_accuracy / (interventions_used / max_interventions)
    interventions_used = len(intervention_sequence)
    efficiency_score = confidence_trajectory[-1] / (interventions_used / max_interventions) if interventions_used > 0 else 0

    return {
        'interventions_used': interventions_used,
        'intervention_sequence': intervention_sequence,
        'information_gain_per_intervention': information_gain_sequence,
        'total_information_gain': float(sum(information_gain_sequence)),
        'confidence_trajectory': confidence_trajectory,
        'final_confidence': confidence_trajectory[-1] if confidence_trajectory else 0.0,
        'reached_target': confidence_trajectory[-1] >= target_accuracy if confidence_trajectory else False,
        'edge_beliefs': edge_beliefs.tolist(),
        'edge_confidence': edge_confidence,
        'final_causal_graph': {
            'nodes': var_names,
            'edges': final_graph_edges,
        },
        'efficiency_score': float(efficiency_score),
    }


def render_temporal_analysis(
    temporal_results: dict,
    title: str = 'Temporal Causal Effects',
    figsize: Tuple[int, int] = (14, 10),
) -> np.ndarray:
    """Render temporal causal analysis as a multi-panel figure.

    Args:
        temporal_results: Results from analyze_temporal_causal_effects_all_variables
        title: Figure title
        figsize: Figure size

    Returns:
        RGB image as numpy array
    """
    var_names = list(temporal_results.keys())
    num_vars = len(var_names)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, num_vars))

    # Plot 1: Policy divergence over time for each variable
    ax = axes[0]
    for i, var_name in enumerate(var_names):
        curve = temporal_results[var_name]['avg_temporal_policy_curve']
        ax.plot(curve, label=var_name, color=colors[i], linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Policy Divergence (KL)')
    ax.set_title('Policy Divergence Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Behavioral divergence over time
    ax = axes[1]
    for i, var_name in enumerate(var_names):
        curve = temporal_results[var_name]['avg_temporal_behavioral_curve']
        ax.plot(curve, label=var_name, color=colors[i], linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Behavioral Divergence')
    ax.set_title('Behavioral Divergence Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Causal onset times
    ax = axes[2]
    onsets = [temporal_results[v]['avg_causal_onset'] for v in var_names]
    bars = ax.bar(var_names, onsets, color=colors)
    ax.set_ylabel('Timestep')
    ax.set_title('Causal Onset Time')
    ax.set_xticklabels([v.replace('_', '\n') for v in var_names], fontsize=8)

    # Plot 4: Peak divergence
    ax = axes[3]
    peaks = [temporal_results[v]['avg_peak_divergence'] for v in var_names]
    ax.bar(var_names, peaks, color=colors)
    ax.set_ylabel('Peak Divergence')
    ax.set_title('Peak Divergence per Variable')
    ax.set_xticklabels([v.replace('_', '\n') for v in var_names], fontsize=8)

    # Plot 5: Cumulative effect
    ax = axes[4]
    cumulative = [temporal_results[v]['avg_cumulative_effect'] for v in var_names]
    ax.bar(var_names, cumulative, color=colors)
    ax.set_ylabel('Cumulative Effect')
    ax.set_title('Cumulative Causal Effect')
    ax.set_xticklabels([v.replace('_', '\n') for v in var_names], fontsize=8)

    # Plot 6: Summary heatmap of temporal characteristics
    ax = axes[5]
    summary_data = np.array([
        [temporal_results[v]['avg_causal_onset'] for v in var_names],
        [temporal_results[v]['avg_causal_peak'] for v in var_names],
        [temporal_results[v]['avg_peak_divergence'] for v in var_names],
        [temporal_results[v]['avg_cumulative_effect'] for v in var_names],
    ])
    # Normalize each row
    summary_normalized = (summary_data - summary_data.min(axis=1, keepdims=True)) / (
        summary_data.max(axis=1, keepdims=True) - summary_data.min(axis=1, keepdims=True) + 1e-8
    )
    im = ax.imshow(summary_normalized, cmap='YlOrRd', aspect='auto')
    ax.set_yticks(range(4))
    ax.set_yticklabels(['Onset', 'Peak Time', 'Peak Div', 'Cumulative'])
    ax.set_xticks(range(num_vars))
    ax.set_xticklabels([v.replace('_', '\n') for v in var_names], fontsize=8)
    ax.set_title('Temporal Summary (Normalized)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Convert to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return img


def render_intervention_efficiency(
    efficiency_results: dict,
    title: str = 'Intervention Efficiency Analysis',
    figsize: Tuple[int, int] = (14, 8),
) -> np.ndarray:
    """Render intervention efficiency analysis.

    Args:
        efficiency_results: Results from compute_intervention_efficiency
        title: Figure title
        figsize: Figure size

    Returns:
        RGB image as numpy array
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Plot 1: Confidence trajectory
    ax = axes[0, 0]
    ax.plot(efficiency_results['confidence_trajectory'], 'b-', linewidth=2)
    ax.axhline(y=0.9, color='r', linestyle='--', label='Target')
    ax.set_xlabel('Intervention #')
    ax.set_ylabel('Confidence')
    ax.set_title('Confidence Over Interventions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Information gain per intervention
    ax = axes[0, 1]
    ax.bar(range(len(efficiency_results['information_gain_per_intervention'])),
           efficiency_results['information_gain_per_intervention'], color='steelblue')
    ax.set_xlabel('Intervention #')
    ax.set_ylabel('Information Gain')
    ax.set_title('Information Gain per Intervention')

    # Plot 3: Intervention variable distribution
    ax = axes[0, 2]
    intervention_vars = [s['variable'] for s in efficiency_results['intervention_sequence']]
    var_counts = {}
    for v in intervention_vars:
        var_counts[v] = var_counts.get(v, 0) + 1
    ax.bar(var_counts.keys(), var_counts.values(), color='coral')
    ax.set_ylabel('Count')
    ax.set_title('Variables Intervened On')
    ax.set_xticklabels([v.replace('_', '\n') for v in var_counts.keys()], fontsize=8)

    # Plot 4: Edge beliefs heatmap
    ax = axes[1, 0]
    beliefs = np.array(efficiency_results['edge_beliefs'])
    var_names = efficiency_results['final_causal_graph']['nodes']
    im = ax.imshow(beliefs, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(var_names)))
    ax.set_yticks(range(len(var_names)))
    ax.set_xticklabels([v.replace('_', '\n') for v in var_names], fontsize=7)
    ax.set_yticklabels([v.replace('_', '\n') for v in var_names], fontsize=7)
    ax.set_title('Edge Beliefs (Green=Likely)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Plot 5: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    Intervention Efficiency Summary
    ================================
    Interventions Used: {efficiency_results['interventions_used']}
    Final Confidence: {efficiency_results['final_confidence']:.3f}
    Reached Target: {efficiency_results['reached_target']}
    Total Info Gain: {efficiency_results['total_information_gain']:.3f}
    Efficiency Score: {efficiency_results['efficiency_score']:.3f}

    Edges Discovered: {len(efficiency_results['final_causal_graph']['edges'])}
    """
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    # Plot 6: Final causal graph edges
    ax = axes[1, 2]
    edges = efficiency_results['final_causal_graph']['edges']
    if edges:
        edge_labels = [f"{e[0]}->{e[1]}" for e in edges]
        edge_weights = [e[2] for e in edges]
        y_pos = range(len(edges))
        ax.barh(y_pos, edge_weights, color='green', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(edge_labels, fontsize=8)
        ax.set_xlabel('Confidence')
        ax.set_title('Discovered Edges')
    else:
        ax.text(0.5, 0.5, 'No edges discovered', ha='center', va='center')
        ax.set_title('Discovered Edges')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Convert to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return img


def run_advanced_causal_analysis(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    sample_levels_fn,
    config: dict,
) -> dict:
    """Run the full suite of advanced causal analysis.

    This includes:
    1. Counterfactual evaluation
    2. Temporal causal analysis
    3. Intervention efficiency analysis

    Args:
        rng: Random key
        env: Environment
        env_params: Environment parameters
        train_state: Trained agent
        sample_levels_fn: Function to sample levels
        config: Configuration dict

    Returns:
        Dictionary with all analysis results
    """
    num_sample_states = config.get('cwm_num_sample_states', 50)
    rollout_length = config.get('cwm_rollout_length', 30)

    print("Advanced CWM Analysis: Sampling levels...")
    rng, rng_levels = jax.random.split(rng)
    levels = jax.vmap(sample_levels_fn)(jax.random.split(rng_levels, num_sample_states))

    print("Advanced CWM Analysis: Running counterfactual evaluation...")
    rng, rng_cf = jax.random.split(rng)
    counterfactual_results = evaluate_counterfactual_suite(
        rng_cf, env, env_params, train_state, levels, rollout_length
    )

    print("Advanced CWM Analysis: Running temporal analysis...")
    rng, rng_temporal = jax.random.split(rng)
    temporal_results = analyze_temporal_causal_effects_all_variables(
        rng_temporal, env, env_params, train_state, levels,
        max_horizon=rollout_length,
        num_intervention_values=3
    )

    print("Advanced CWM Analysis: Computing intervention efficiency...")
    rng, rng_efficiency = jax.random.split(rng)
    efficiency_results = compute_intervention_efficiency(
        rng_efficiency, env, env_params, train_state, levels,
        target_accuracy=0.85,
        max_interventions=40,
        rollout_length=rollout_length // 2
    )

    print("Advanced CWM Analysis: Complete!")

    return {
        'counterfactual_results': counterfactual_results,
        'temporal_results': temporal_results,
        'efficiency_results': efficiency_results,
    }


def log_advanced_cwm_results_to_wandb(
    advanced_results: dict,
    update_count: int,
):
    """Log advanced CWM analysis results to wandb.

    Args:
        advanced_results: Results from run_advanced_causal_analysis
        update_count: Current training update count
    """
    log_dict = {}

    # Counterfactual results
    cf_results = advanced_results['counterfactual_results']
    for var_name, var_cf in cf_results.items():
        log_dict[f'cwm_advanced/counterfactual/{var_name}/mean_policy_div'] = var_cf['mean_policy_divergence']
        log_dict[f'cwm_advanced/counterfactual/{var_name}/mean_behavioral_div'] = var_cf['mean_behavioral_divergence']
        log_dict[f'cwm_advanced/counterfactual/{var_name}/mean_causal_effect'] = var_cf['mean_causal_effect']

    # Temporal results
    temporal_results = advanced_results['temporal_results']
    for var_name, var_temp in temporal_results.items():
        log_dict[f'cwm_advanced/temporal/{var_name}/onset'] = var_temp['avg_causal_onset']
        log_dict[f'cwm_advanced/temporal/{var_name}/peak_time'] = var_temp['avg_causal_peak']
        log_dict[f'cwm_advanced/temporal/{var_name}/peak_divergence'] = var_temp['avg_peak_divergence']
        log_dict[f'cwm_advanced/temporal/{var_name}/cumulative_effect'] = var_temp['avg_cumulative_effect']

    # Temporal visualization
    temporal_img = render_temporal_analysis(
        temporal_results,
        title=f'Temporal Causal Effects (Update {update_count})'
    )
    log_dict['cwm_advanced/temporal_analysis'] = wandb.Image(temporal_img)

    # Efficiency results
    eff_results = advanced_results['efficiency_results']
    log_dict['cwm_advanced/efficiency/interventions_used'] = eff_results['interventions_used']
    log_dict['cwm_advanced/efficiency/final_confidence'] = eff_results['final_confidence']
    log_dict['cwm_advanced/efficiency/score'] = eff_results['efficiency_score']
    log_dict['cwm_advanced/efficiency/total_info_gain'] = eff_results['total_information_gain']
    log_dict['cwm_advanced/efficiency/edges_discovered'] = len(eff_results['final_causal_graph']['edges'])

    # Efficiency visualization
    efficiency_img = render_intervention_efficiency(
        eff_results,
        title=f'Intervention Efficiency (Update {update_count})'
    )
    log_dict['cwm_advanced/efficiency_analysis'] = wandb.Image(efficiency_img)

    wandb.log(log_dict)


# =============================================================================
# CWM VISUALIZATION FUNCTIONS
# =============================================================================

def render_heatmap(
    data: np.ndarray,
    title: str,
    cmap: str = 'viridis',
    wall_map: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Render a heatmap as an RGB image array.

    Args:
        data: 2D array of values to visualize
        title: Title for the plot
        cmap: Colormap name
        wall_map: Optional wall map to overlay (1=wall, 0=empty)
        figsize: Figure size in inches

    Returns:
        RGB image as numpy array (H, W, 3)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Mask NaN values
    masked_data = np.ma.masked_invalid(data)

    # Create heatmap
    im = ax.imshow(masked_data, cmap=cmap, origin='upper')

    # Overlay walls if provided
    if wall_map is not None:
        wall_overlay = np.ma.masked_where(wall_map == 0, wall_map)
        ax.imshow(wall_overlay, cmap='binary', alpha=0.7, origin='upper')

    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Convert to image array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return img


def render_divergence_matrix(
    matrix: np.ndarray,
    var_names: list,
    title: str = 'Causal Influence Matrix',
    figsize: Tuple[int, int] = (10, 8),
) -> np.ndarray:
    """Render the divergence matrix as a heatmap.

    Args:
        matrix: (num_vars, num_vars) divergence matrix
        var_names: List of variable names
        title: Plot title
        figsize: Figure size

    Returns:
        RGB image as numpy array
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap='hot', origin='upper')

    # Add labels
    ax.set_xticks(range(len(var_names)))
    ax.set_yticks(range(len(var_names)))
    ax.set_xticklabels(var_names, rotation=45, ha='right')
    ax.set_yticklabels(var_names)

    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.8, label='Divergence')

    # Add value annotations
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            value = matrix[i, j]
            if value > 0:
                text_color = 'white' if value > matrix.max() / 2 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=text_color, fontsize=8)

    ax.set_title(title)
    ax.set_xlabel('Target Variable')
    ax.set_ylabel('Intervened Variable')

    plt.tight_layout()

    # Convert to image array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return img


def render_causal_graph(
    causal_graph: dict,
    title: str = 'Extracted Causal Graph',
    figsize: Tuple[int, int] = (12, 10),
) -> np.ndarray:
    """Render the causal graph as a network visualization.

    Args:
        causal_graph: Dictionary with 'nodes', 'edges', 'adjacency'
        title: Plot title
        figsize: Figure size

    Returns:
        RGB image as numpy array
    """
    fig, ax = plt.subplots(figsize=figsize)

    nodes = causal_graph['nodes']
    edges = causal_graph['edges']
    n_nodes = len(nodes)

    # Position nodes in a circle
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 2.0
    positions = {node: (radius * np.cos(angle), radius * np.sin(angle))
                for node, angle in zip(nodes, angles)}

    # Draw edges
    max_weight = max([e[2] for e in edges]) if edges else 1
    for source, target, weight in edges:
        x0, y0 = positions[source]
        x1, y1 = positions[target]

        # Arrow properties based on weight
        alpha = 0.3 + 0.7 * (weight / max_weight)
        linewidth = 1 + 3 * (weight / max_weight)

        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                   arrowprops=dict(arrowstyle='->', color='blue',
                                  alpha=alpha, lw=linewidth,
                                  connectionstyle='arc3,rad=0.1'))

    # Draw nodes
    for node, (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.3, color='lightblue', ec='navy', lw=2)
        ax.add_patch(circle)
        ax.text(x, y, node.replace('_', '\n'), ha='center', va='center',
               fontsize=8, fontweight='bold')

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)

    plt.tight_layout()

    # Convert to image array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return img


def create_cwm_composite_figure(
    cwm_results: dict,
    env_renderer,
    env_params,
) -> np.ndarray:
    """Create a composite figure showing all CWM results.

    Args:
        cwm_results: Results from extract_causal_world_model
        env_renderer: Environment renderer for level visualization
        env_params: Environment parameters

    Returns:
        RGB image as numpy array
    """
    fig = plt.figure(figsize=(20, 16))

    # 1. Divergence matrix (top left)
    ax1 = fig.add_subplot(2, 3, 1)
    var_names = cwm_results['metadata']['variables']
    matrix = cwm_results['divergence_matrix']
    im1 = ax1.imshow(matrix, cmap='hot')
    ax1.set_xticks(range(len(var_names)))
    ax1.set_yticks(range(len(var_names)))
    ax1.set_xticklabels([v.replace('_', '\n') for v in var_names], rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels([v.replace('_', '\n') for v in var_names], fontsize=8)
    ax1.set_title('Causal Influence Matrix')
    plt.colorbar(im1, ax=ax1, shrink=0.6)

    # 2. Goal position returns heatmap (top center)
    ax2 = fig.add_subplot(2, 3, 2)
    goal_returns = cwm_results['heatmaps']['goal_position']['returns']
    im2 = ax2.imshow(goal_returns, cmap='RdYlGn', origin='upper')
    ax2.set_title('Goal Position: Returns')
    plt.colorbar(im2, ax=ax2, shrink=0.6)

    # 3. Goal position behavioral divergence (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    goal_div = cwm_results['heatmaps']['goal_position']['behavioral_divergence']
    im3 = ax3.imshow(goal_div, cmap='viridis', origin='upper')
    ax3.set_title('Goal Position: Behavioral Divergence')
    plt.colorbar(im3, ax=ax3, shrink=0.6)

    # 4. Agent spawn returns heatmap (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    agent_returns = cwm_results['heatmaps']['agent_spawn']['returns']
    im4 = ax4.imshow(agent_returns, cmap='RdYlGn', origin='upper')
    ax4.set_title('Agent Spawn: Returns')
    plt.colorbar(im4, ax=ax4, shrink=0.6)

    # 5. Agent spawn behavioral divergence (bottom center)
    ax5 = fig.add_subplot(2, 3, 5)
    agent_div = cwm_results['heatmaps']['agent_spawn']['behavioral_divergence']
    im5 = ax5.imshow(agent_div, cmap='viridis', origin='upper')
    ax5.set_title('Agent Spawn: Behavioral Divergence')
    plt.colorbar(im5, ax=ax5, shrink=0.6)

    # 6. Per-variable divergence bar chart (bottom right)
    ax6 = fig.add_subplot(2, 3, 6)
    detailed = cwm_results['detailed_results']
    vars_list = list(detailed['behavioral_divergences'].keys())
    behav_divs = [detailed['behavioral_divergences'][v] for v in vars_list]
    value_divs = [detailed['value_divergences'][v] for v in vars_list]

    x = np.arange(len(vars_list))
    width = 0.35
    ax6.bar(x - width/2, behav_divs, width, label='Behavioral', color='steelblue')
    ax6.bar(x + width/2, value_divs, width, label='Value', color='coral')
    ax6.set_xticks(x)
    ax6.set_xticklabels([v.replace('_', '\n') for v in vars_list], fontsize=8)
    ax6.set_ylabel('Divergence')
    ax6.set_title('Per-Variable Divergences')
    ax6.legend()

    plt.tight_layout()

    # Convert to image array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return img


def log_cwm_results_to_wandb(
    cwm_results: dict,
    env_renderer,
    env_params,
    update_count: int,
):
    """Log CWM extraction results to wandb.

    Args:
        cwm_results: Results from extract_causal_world_model
        env_renderer: Environment renderer
        env_params: Environment parameters
        update_count: Current training update count
    """
    log_dict = {}

    # Log scalar metrics
    var_names = cwm_results['metadata']['variables']
    detailed = cwm_results['detailed_results']

    for var in var_names:
        log_dict[f'cwm/behavioral_divergence/{var}'] = detailed['behavioral_divergences'][var]
        log_dict[f'cwm/value_divergence/{var}'] = detailed['value_divergences'][var]
        log_dict[f'cwm/reward_divergence/{var}'] = detailed['reward_divergences'][var]

    # Log graph statistics
    causal_graph = cwm_results['causal_graph']
    log_dict['cwm/num_edges'] = len(causal_graph['edges'])
    log_dict['cwm/mean_edge_weight'] = np.mean([e[2] for e in causal_graph['edges']]) if causal_graph['edges'] else 0

    # Log heatmaps as images
    # Divergence matrix
    matrix_img = render_divergence_matrix(
        cwm_results['divergence_matrix'],
        var_names,
        title=f'Causal Influence Matrix (Update {update_count})'
    )
    log_dict['cwm/divergence_matrix'] = wandb.Image(matrix_img)

    # Causal graph
    graph_img = render_causal_graph(
        causal_graph,
        title=f'Extracted Causal Graph (Update {update_count})'
    )
    log_dict['cwm/causal_graph'] = wandb.Image(graph_img)

    # Extract wall map for overlay
    wall_map = np.array(cwm_results['sample_level'].wall_map)

    # Goal position heatmaps
    goal_returns_img = render_heatmap(
        cwm_results['heatmaps']['goal_position']['returns'],
        'Goal Position: Returns',
        cmap='RdYlGn',
        wall_map=wall_map
    )
    log_dict['cwm/goal_position/returns'] = wandb.Image(goal_returns_img)

    goal_div_img = render_heatmap(
        cwm_results['heatmaps']['goal_position']['behavioral_divergence'],
        'Goal Position: Behavioral Divergence',
        cmap='viridis',
        wall_map=wall_map
    )
    log_dict['cwm/goal_position/behavioral_divergence'] = wandb.Image(goal_div_img)

    # Agent spawn heatmaps
    agent_returns_img = render_heatmap(
        cwm_results['heatmaps']['agent_spawn']['returns'],
        'Agent Spawn: Returns',
        cmap='RdYlGn',
        wall_map=wall_map
    )
    log_dict['cwm/agent_spawn/returns'] = wandb.Image(agent_returns_img)

    agent_div_img = render_heatmap(
        cwm_results['heatmaps']['agent_spawn']['behavioral_divergence'],
        'Agent Spawn: Behavioral Divergence',
        cmap='viridis',
        wall_map=wall_map
    )
    log_dict['cwm/agent_spawn/behavioral_divergence'] = wandb.Image(agent_div_img)

    # Composite figure
    composite_img = create_cwm_composite_figure(cwm_results, env_renderer, env_params)
    log_dict['cwm/composite_summary'] = wandb.Image(composite_img)

    # Log causal graph as a table
    if causal_graph['edges']:
        edge_data = [[e[0], e[1], e[2]] for e in causal_graph['edges']]
        edge_table = wandb.Table(
            columns=['Source', 'Target', 'Weight'],
            data=edge_data
        )
        log_dict['cwm/causal_edges'] = edge_table

    wandb.log(log_dict)


def setup_checkpointing(config: dict, train_state: TrainState, env: UnderspecifiedEnv, env_params: EnvParams) -> ocp.CheckpointManager:
    """This takes in the train state and config, and returns an orbax checkpoint manager."""
    overall_save_dir = os.path.join(os.getcwd(), "checkpoints", f"{config['run_name']}", str(config['seed']))
    os.makedirs(overall_save_dir, exist_ok=True)

    with open(os.path.join(overall_save_dir, "config.json"), "w+") as f:
        f.write(json.dumps(config.as_dict(), indent=True))

    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, "models"),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config["checkpoint_save_interval"],
            max_to_keep=config["max_number_of_checkpoints"],
        )
    )
    return checkpoint_manager


def train_state_to_log_dict(train_state: TrainState, level_sampler: LevelSampler) -> dict:
    """To prevent the entire large train_state to be copied to the cpu when doing logging, this function returns all the
    important information in a dict format"""

    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)

    return {
        "log":{
            "level_sampler/size": sampler["size"],
            'level_sampler/episode_count': sampler["episode_count"],
            "level_sampler/max_score": sampler["scores"].max(),
            "level_sampler/weighted_score": (sampler["scores"] * level_sampler.level_weights(sampler)).sum(),
            "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
        },
        "info":{
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        }
    }


def compute_score(config, dones, values, max_returns, advantages):
    if config["score_function"] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config["score_function"] == "pvl":
        return positive_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")


def main(config=None, project="JaxUED-minigrid-maze"):
    tags = []
    if not config["exploratory_grad_updates"]:
        tags.append("robust")
    if config["use_accel"]:
        tags.append("ACCEL")
    else:
        tags.append("PLR")

    run = wandb.init(config=config, project=project, group=config["run_name"], tags=tags)
    config = wandb.config

    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_lengths/*", step_metric="num_updates")

    def log_eval(stats, train_state_info):
        print(f"Logging update: {stats['update_count']}")

        # generic stats
        env_steps = stats["update_count"] * config["num_train_envs"] * config["num_steps"]
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / stats["time_delta"],
        }

        # evaluation performance
        solve_rates = stats["eval_solve_rates"]
        returns = stats["eval_returns"]
        log_dict.update({f"solve_rate/{name}": solve_rate for name, solve_rate in zip(config["eval_levels"], solve_rates)})
        log_dict.update({f"solve_rate/mean": solve_rates.mean()})
        log_dict.update({f"return/{name}": ret for name, ret in zip(config["eval_levels"], returns)})
        log_dict.update({f"return/mean": returns.mean()})
        log_dict.update({f"eval_ep_lengths/mean": stats["eval_ep_lengths"].mean()})

        # level sampler
        log_dict.update(train_state_info["log"])

        # images
        log_dict.update({"images/highest_scoring_level": wandb.Image(np.array(stats["highest_scoring_level"]), caption="Highest Scoring Level")})
        log_dict.update({"images/highest_weighted_level": wandb.Image(np.array(stats["highest_weighted_level"]), caption="highest Weighted Level")})

        for s in ["dr", "replay", "mutation"]:
            if train_state_info["info"][f"num_{s}_updates"] > 0:
                log_dict.update({f"images/{s}_levels": [wandb.Image(np.array(image)) for image in stats[f"{s}_levels"]]})

        # animations
        for i, level_name in enumerate(config["eval_levels"]):
            frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
            frames = np.array(frames[:episode_length])
            log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4, format="gif")})

        wandb.log(log_dict)

    # setup the environment
    env = Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    eval_env = env
    sample_random_level = make_level_generator(env.max_height, env.max_width, config["n_walls"])
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params
    mutate_level = make_level_mutator_minimax(100)

    # and the level sampler
    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config["top_k"]},
        duplicate_check=config["buffer_duplicate_check"],
    )


    # Store observation shape for CWM network initialization
    sample_obs, _ = env.reset_to_level(jax.random.PRNGKey(0), sample_random_level(jax.random.PRNGKey(0)), env_params)
    obs_image_shape = sample_obs.image.shape  # (H, W, C)

    @jax.jit
    def create_train_state(rng) -> TrainState:
        # creates the train state
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["num_minibatches"] * config["epoch_ppo"]))
                / config["num_updates"]
            )
            return config["lr"] * frac

        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
            obs,
        )

        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))

        # Choose network based on CWM mode
        if config["use_cwm"]:
            network = ActorCriticWithPrediction(
                env.action_space(env_params).n,
                obs_image_shape=obs_image_shape
            )
            network_params = network.init(
                rng, init_x,
                ActorCriticWithPrediction.initialize_carry((config["num_train_envs"],)),
                None  # No actions for prediction during initialization
            )
        else:
            network = ActorCritic(env.action_space(env_params).n)
            network_params = network.init(rng, init_x, ActorCritic.initialize_carry((config["num_train_envs"],)))

        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
        pholder_level = sample_random_level(jax.random.PRNGKey(0))
        sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
        pholder_level_batch = jax.tree_util.tree_map(lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0), pholder_level)

        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=0,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            mutation_last_level_batch=pholder_level_batch,
        )

    # Helper function to get the right network carry initializer
    def get_init_carry(batch_size):
        if config["use_cwm"]:
            return ActorCriticWithPrediction.initialize_carry((batch_size,))
        else:
            return ActorCritic.initialize_carry((batch_size,))

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        """This is the main training loop. It basically calls either `on_new_levels`, `on_replay_levels`,
        or `on_mutate_levels` at every step."""

        def on_new_levels(rng: chex.PRNGKey, train_state: TrainState):
            """Samples new randomly-generated levels and evaluated the policy on these. It also then adds the levels to the
            level buffer if they have high-enough scores. The agent is updated on these trajectories iff `config["exploratory_grad_updates"]` is True"""

            sampler = train_state.sampler

            # reset
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, config["num_train_envs"]))
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), new_levels, env_params)

            # rollout - use CWM-enabled trajectory collection if CWM is enabled
            if config["use_cwm"]:
                (
                    (rng, train_state, hstate, last_obs, last_env_state, last_value),
                    (obs, actions, rewards, dones, log_probs, values, action_logits, next_obs, info),
                ) = sample_trajectories_rnn_with_predictions(
                    rng,
                    env,
                    env_params,
                    train_state,
                    get_init_carry(config["num_train_envs"]),
                    init_obs,
                    init_env_state,
                    config["num_train_envs"],
                    config["num_steps"],
                )
            else:
                (
                    (rng, train_state, hstate, last_obs, last_env_state, last_value),
                    (obs, actions, rewards, dones, log_probs, values, info),
                ) = sample_trajectories_rnn(
                    rng,
                    env,
                    env_params,
                    train_state,
                    get_init_carry(config["num_train_envs"]),
                    init_obs,
                    init_env_state,
                    config["num_train_envs"],
                    config["num_steps"],
                )

            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            max_returns = compute_max_returns(dones, rewards)
            scores = compute_score(config, dones, values, max_returns, advantages)
            sampler, _ = level_sampler.insert_batch(sampler, new_levels, scores, {"max_return": max_returns})

            # update: train_state only modified if exploratory_grad_updates is on
            if config["use_cwm"]:
                (rng, train_state), losses = update_actor_critic_rnn_with_predictions(
                    rng,
                    train_state,
                    get_init_carry(config["num_train_envs"]),
                    (obs, actions, dones, log_probs, values, targets, advantages, next_obs, rewards),
                    config["num_train_envs"],
                    config["num_steps"],
                    config["num_minibatches"],
                    config["epoch_ppo"],
                    config["clip_eps"],
                    config["entropy_coeff"],
                    config["critic_coeff"],
                    config["pred_coeff"],
                    config["reward_pred_coeff"],
                    update_grad=config["exploratory_grad_updates"],
                )
            else:
                (rng, train_state), losses = update_actor_critic_rnn(
                    rng,
                    train_state,
                    get_init_carry(config["num_train_envs"]),
                    (obs, actions, dones, log_probs, values, targets, advantages),
                    config["num_train_envs"],
                    config["num_steps"],
                    config["num_minibatches"],
                    config["epoch_ppo"],
                    config["clip_eps"],
                    config["entropy_coeff"],
                    config["critic_coeff"],
                    update_grad=config["exploratory_grad_updates"],
                )

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": new_levels.wall_map.sum() / config["num_train_envs"],
            }

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=new_levels,
            )
            return (rng, train_state), metrics

        def on_replay_levels(rng: chex.PRNGKey, train_state: TrainState):
            """This samples levels from the level buffer, and updates the policy on them"""
            sampler = train_state.sampler

            # collect trajectories on replay levels
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, config["num_train_envs"])
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), levels, env_params)

            if config["use_cwm"]:
                (
                    (rng, train_state, hstate, last_obs, last_env_state, last_value),
                    (obs, actions, rewards, dones, log_probs, values, action_logits, next_obs, info),
                ) = sample_trajectories_rnn_with_predictions(
                    rng,
                    env,
                    env_params,
                    train_state,
                    get_init_carry(config["num_train_envs"]),
                    init_obs,
                    init_env_state,
                    config["num_train_envs"],
                    config["num_steps"],
                )
            else:
                (
                    (rng, train_state, hstate, last_obs, last_env_state, last_value),
                    (obs, actions, rewards, dones, log_probs, values, info),
                ) = sample_trajectories_rnn(
                    rng,
                    env,
                    env_params,
                    train_state,
                    get_init_carry(config["num_train_envs"]),
                    init_obs,
                    init_env_state,
                    config["num_train_envs"],
                    config["num_steps"],
                )

            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            max_returns = jnp.maximum(level_sampler.get_levels_extra(sampler, level_inds)["max_return"], compute_max_returns(dones, rewards))
            scores = compute_score(config, dones, values, max_returns, advantages)
            sampler = level_sampler.update_batch(sampler, level_inds, scores, {"max_return": max_returns})

            # update the policy using trajectories collected from levels
            if config["use_cwm"]:
                (rng, train_state), losses = update_actor_critic_rnn_with_predictions(
                    rng,
                    train_state,
                    get_init_carry(config["num_train_envs"]),
                    (obs, actions, dones, log_probs, values, targets, advantages, next_obs, rewards),
                    config["num_train_envs"],
                    config["num_steps"],
                    config["num_minibatches"],
                    config["epoch_ppo"],
                    config["clip_eps"],
                    config["entropy_coeff"],
                    config["critic_coeff"],
                    config["pred_coeff"],
                    config["reward_pred_coeff"],
                    update_grad=True,
                )
            else:
                (rng, train_state), losses = update_actor_critic_rnn(
                    rng,
                    train_state,
                    get_init_carry(config["num_train_envs"]),
                    (obs, actions, dones, log_probs, values, targets, advantages),
                    config["num_train_envs"],
                    config["num_steps"],
                    config["num_minibatches"],
                    config["epoch_ppo"],
                    config["clip_eps"],
                    config["entropy_coeff"],
                    config["critic_coeff"],
                    update_grad=True,
                )

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": levels.wall_map.sum() / config["num_train_envs"],
            }

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.REPLAY,
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
            )
            return (rng, train_state), metrics

        def on_mutate_levels(rng: chex.PRNGKey, train_state: TrainState):
            """This mutates the previous batch of replay levels and potentially adds them to the level buffer.
            This also updates the policy iff `config["exploratory_grad_updates"]` is True."""

            sampler = train_state.sampler
            rng, rng_mutate, rng_reset = jax.random.split(rng, 3)

            # mutate
            parent_levels = train_state.replay_last_level_batch
            child_levels = jax.vmap(mutate_level, (0, 0, None))(jax.random.split(rng_mutate, config["num_train_envs"]), parent_levels, config["num_edits"])
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), child_levels, env_params)

            # rollout
            if config["use_cwm"]:
                (
                    (rng, train_state, hstate, last_obs, last_env_state, last_value),
                    (obs, actions, rewards, dones, log_probs, values, action_logits, next_obs, info),
                ) = sample_trajectories_rnn_with_predictions(
                    rng,
                    env,
                    env_params,
                    train_state,
                    get_init_carry(config["num_train_envs"]),
                    init_obs,
                    init_env_state,
                    config["num_train_envs"],
                    config["num_steps"],
                )
            else:
                (
                    (rng, train_state, hstate, last_obs, last_env_state, last_value),
                    (obs, actions, rewards, dones, log_probs, values, info),
                ) = sample_trajectories_rnn(
                    rng,
                    env,
                    env_params,
                    train_state,
                    get_init_carry(config["num_train_envs"]),
                    init_obs,
                    init_env_state,
                    config["num_train_envs"],
                    config["num_steps"],
                )

            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            max_returns = compute_max_returns(dones, rewards)
            scores = compute_score(config, dones, values, max_returns, advantages)
            sampler, _ = level_sampler.insert_batch(sampler, child_levels, scores, {"max_return": max_returns})

            # update: train_state only modified if exploratory_grad_updates is on
            if config["use_cwm"]:
                (rng, train_state), losses = update_actor_critic_rnn_with_predictions(
                    rng,
                    train_state,
                    get_init_carry(config["num_train_envs"]),
                    (obs, actions, dones, log_probs, values, targets, advantages, next_obs, rewards),
                    config["num_train_envs"],
                    config["num_steps"],
                    config["num_minibatches"],
                    config["epoch_ppo"],
                    config["clip_eps"],
                    config["entropy_coeff"],
                    config["critic_coeff"],
                    config["pred_coeff"],
                    config["reward_pred_coeff"],
                    update_grad=config["exploratory_grad_updates"],
                )
            else:
                (rng, train_state), losses = update_actor_critic_rnn(
                    rng,
                    train_state,
                    get_init_carry(config["num_train_envs"]),
                    (obs, actions, dones, log_probs, values, targets, advantages),
                    config["num_train_envs"],
                    config["num_steps"],
                    config["num_minibatches"],
                    config["epoch_ppo"],
                    config["clip_eps"],
                    config["entropy_coeff"],
                    config["critic_coeff"],
                    update_grad=config["exploratory_grad_updates"],
                )

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": child_levels.wall_map.sum() / config["num_train_envs"],
            }

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_mutation_updates=train_state.num_mutation_updates + 1,
                mutation_last_level_batch=child_levels,
            )
            return (rng, train_state), metrics

        rng, train_state = carry
        rng, rng_replay = jax.random.split(rng)

        # the train step makes a decision on which branch to take, either on_new, on_replay or on_mutate
        # on_mutate is only called if the replay branch has been taken before (as it uses train_state.update_state).

        if config["use_accel"]:
            s = train_state.update_state
            branch = (1 - s) * level_sampler.sample_replay_decision(train_state.sampler, rng_replay) + 2 * s
        else:
            branch = level_sampler.sample_replay_decision(train_state.sampler, rng_replay).astype(int)

        return jax.lax.switch(
            branch,
            [
                on_new_levels,
                on_replay_levels,
                on_mutate_levels,
            ],
            rng, train_state,
        )

    def eval(rng: chex.PRNGKey, train_state: TrainState):
        """This evaluates the current policy on the ste of evaluation levels specified by config["eval_levels"]."""

        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["eval_levels"])
        num_levels = len(config["eval_levels"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)
        states, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            env_params,
            train_state,
            get_init_carry(num_levels),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
            use_cwm=config["use_cwm"],
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths

    @jax.jit
    def train_and_eval_step(runner_state, _):
        """This function runs the train_step for a certain number of iterations, and then evaluates teh policy."""

        # train
        (rng, train_state), metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])

        # eval
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, config["eval_num_attempts"]), train_state)

        # collect metrics
        eval_solve_rates = jnp.where(cum_rewards > 0, 1., 0.).mean(axis=0)
        eval_returns = cum_rewards.mean(axis=0)

        # just grab the first run
        states, episode_lengths = jax.tree_util.tree_map(lambda x: x[0], (states, episode_lengths))
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params)
        frames = images.transpose(0, 1, 4, 2, 3)

        metrics["update_count"] = train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"] = episode_lengths
        metrics["eval_animation"] = (frames, episode_lengths)
        metrics["dr_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.dr_last_level_batch, env_params)
        metrics["replay_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.replay_last_level_batch, env_params)
        metrics["mutation_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.mutation_last_level_batch, env_params)

        highest_scoring_level = level_sampler.get_levels(train_state.sampler, train_state.sampler["scores"].argmax())
        highest_weighted_level = level_sampler.get_levels(train_state.sampler, level_sampler.level_weights(train_state.sampler).argmax())

        metrics["highest_scoring_level"] = env_renderer.render_level(highest_scoring_level, env_params)
        metrics["highest_weighted_level"] = env_renderer.render_level(highest_weighted_level, env_params)

        return (rng, train_state), metrics

    def eval_checkpoint(og_config):
        """This function is what is used to evaluate a saved checkpoint after training. It first loads the checkpoint and
        then runs evaluation."""
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))

        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, "config.json")) as f:
                config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_directory, "models"), item_handlers=ocp.StandardCheckpointHandler())

            train_state_og: TrainState = create_train_state(rng_init)
            step = checkpoint_manager.latest_step() if og_config["checkpoint_to_eval"] == -1 else og_config["checkpoint_to_eval"]

            loaded_checkpoint = checkpoint_manager.restore(step)
            params = loaded_checkpoint["params"]
            train_state = train_state_og.replace(params=params)
            return train_state, config

        train_state, config = load(rng_init, og_config["checkpoint_directory"])
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, og_config["eval_num_attempts"]), train_state)
        save_loc = og_config["checkpoint_directory"].replace("checkpoints", "results")
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(os.path.join(save_loc, "results.npz"), states=np.asarray(states), cum_rewards=np.asarray(cum_rewards), episode_lengths=np.asarray(episode_lengths), levels=config["eval_levels"])
        return states, cum_rewards, episode_lengths

    if config["mode"] == "eval":
        return eval_checkpoint(config)

    # set up the train states
    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)

    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    # and run the train_eval_sep function for the specified number of updates
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics["time_delta"] = curr_time - start_time
        log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler))
        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()

        # CWM extraction (if enabled and at the right frequency)
        if (config["use_cwm"] and config["cwm_extraction_freq"] > 0 and
            (eval_step + 1) % config["cwm_extraction_freq"] == 0):
            print(f"Running CWM extraction at eval step {eval_step + 1}...")
            rng_cwm = jax.random.PRNGKey(config["seed"] + eval_step)
            update_count = metrics["update_count"]
            cwm_results = extract_causal_world_model(
                rng_cwm,
                env,
                env_params,
                runner_state[1],
                sample_random_level,
                config,
                use_predictions=True,
            )
            log_cwm_results_to_wandb(cwm_results, env_renderer, env_params, update_count)

    return runner_state[1]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="JaxUED-minigrid-maze")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    # train vs eval
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--checkpoint_directory", type=str, default=None)
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1)

    # checkpointing
    parser.add_argument("--checkpoint_save_interval", type=int, default=2)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)

    # eval
    parser.add_argument("--eval_freq", type=int, default=250)
    parser.add_argument("--eval_num_attempts", type=int, default=10)
    parser.add_argument("--eval_levels", nargs="+", default=[
        "SixteenRooms",
        "SixteenRooms2",
        "Labyrinth",
        "LabyrinthFlipped",
        "Labyrinth2",
        "StandardMaze",
        "StandardMaze2",
        "StandardMaze3",
    ])
    group = parser.add_argument_group("Training params")

    # PPO
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--max_grad_norm", type=float, default=0.5)
    mut_group = group.add_mutually_exclusive_group()
    mut_group.add_argument("--num_updates", type=int, default=30000)
    mut_group.add_argument("--num_env_steps", type=int, default=None)
    group.add_argument("--num_steps", type=int, default=256)
    group.add_argument("--num_train_envs", type=int, default=32)
    group.add_argument("--num_minibatches", type=int, default=1)
    group.add_argument("--gamma", type=float, default=0.995)
    group.add_argument("--epoch_ppo", type=int, default=5)
    group.add_argument("--clip_eps", type=float, default=0.2)
    group.add_argument("--gae_lambda", type=float, default=0.98)
    group.add_argument("--entropy_coeff", type=float, default=1e-3)
    group.add_argument("--critic_coeff", type=float, default=0.5)

    # PLR
    group.add_argument("--score_function", type=str, default="MaxMC", choices=["MaxMC", "pvl"])
    group.add_argument("--exploratory_grad_updates", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--level_buffer_capacity", type=int, default=4000)
    group.add_argument("--replay_prob", type=float, default=0.8)
    group.add_argument("--staleness_coeff", type=float, default=0.3)
    group.add_argument("--temperature", type=float, default=0.3)
    group.add_argument("--top_k", type=int, default=4)
    group.add_argument("--minimum_fill_ratio", type=float, default=0.5)
    group.add_argument("--prioritization", type=str, default="rank", choices=["rank", "topk"])
    group.add_argument("--buffer_duplicate_check", action=argparse.BooleanOptionalAction, default=True)

    # ACCEL
    group.add_argument("--use_accel", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--num_edits", type=int, default=5)

    # env config
    group.add_argument("--agent_view_size", type=int, default=5)

    # dr
    group.add_argument("--n_walls", type=int, default=25)

    # Causal World Model (CWM) settings
    cwm_group = parser.add_argument_group("Causal World Model")
    cwm_group.add_argument("--use_cwm", action=argparse.BooleanOptionalAction, default=False,
                           help="Enable causal world model training with prediction heads")
    cwm_group.add_argument("--pred_coeff", type=float, default=0.5,
                           help="Weight for next observation prediction loss")
    cwm_group.add_argument("--reward_pred_coeff", type=float, default=0.1,
                           help="Weight for reward prediction loss")
    cwm_group.add_argument("--cwm_extraction_freq", type=int, default=0,
                           help="How often to run CWM extraction (0=disabled, N=every N eval steps)")
    cwm_group.add_argument("--cwm_num_sample_states", type=int, default=50,
                           help="Number of states to sample for CWM extraction")
    cwm_group.add_argument("--cwm_rollout_length", type=int, default=30,
                           help="Rollout length for CWM extraction")
    cwm_group.add_argument("--cwm_threshold", type=float, default=0.1,
                           help="Threshold for causal edge detection")
    cwm_group.add_argument("--cwm_num_intervention_values", type=int, default=5,
                           help="Number of values to test per variable during intervention")

    config = vars(parser.parse_args())
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"])
    config["group_name"] = "".join([str(config[key]) for key in sorted([a.dest for a in parser._action_groups[2]._group_actions])])

    if config["mode"] == "eval":
        os.environ["WANDB_MODE"] = "disabled"

    wandb.login()
    main(config, project=config["project"])
