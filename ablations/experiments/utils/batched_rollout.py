"""Shared GPU-batched rollout utility using jax.lax.scan.

Replaces per-experiment serial for-loops with a single vectorized scan
over all environments in parallel. Supports optional per-step collection
of values, rewards, actions, entropies, and hidden-state snapshots.
"""

from typing import NamedTuple, Optional, List
import jax
import jax.numpy as jnp
import numpy as np
import chex
import time
import logging

logger = logging.getLogger(__name__)


class RolloutResult(NamedTuple):
    """Result from batched_rollout."""
    episode_returns: np.ndarray       # (n_envs,)
    episode_solved: np.ndarray        # (n_envs,) bool
    episode_lengths: np.ndarray       # (n_envs,) int
    # Optional per-step arrays — shape (n_envs, max_steps), filled only if requested
    values: Optional[np.ndarray]      # per-step V(s), NaN after done
    rewards: Optional[np.ndarray]     # per-step rewards, 0 after done
    actions: Optional[np.ndarray]     # per-step actions, 0 after done
    entropies: Optional[np.ndarray]   # per-step policy entropy, NaN after done
    logits: Optional[np.ndarray]      # per-step policy logits (n_envs, max_steps, n_actions), NaN after done
    # Optional hstate snapshots — Dict[step_key → (n_envs, hidden_dim)]
    hstates_by_step: Optional[dict]
    # Final hidden state — same pytree as input hstate, for cross-episode use
    final_hstate: Optional[object]


def batched_rollout(
    rng: chex.PRNGKey,
    levels,                          # Batched Level pytree, leading dim = n_envs
    max_steps: int,
    apply_fn,                        # Network apply function
    params,                          # Network parameters
    env,                             # Environment (vmappable)
    env_params,                      # Environment params (shared)
    init_hstate,                     # Initial hstate pytree, shape (n_envs, hidden_dim)
    *,
    collect_values: bool = False,
    collect_rewards: bool = False,
    collect_actions: bool = False,
    collect_entropies: bool = False,
    collect_logits: bool = False,
    collection_steps: Optional[List[int]] = None,  # hstate snapshot steps; -1 = terminal
    return_final_hstate: bool = False,
) -> RolloutResult:
    """GPU-batched rollout over all levels simultaneously using jax.lax.scan.

    Args:
        rng: PRNG key.
        levels: Batched Level pytree with leading dim n_envs.
        max_steps: Maximum episode length.
        apply_fn: Network apply function.
        params: Network parameters.
        env: Environment instance (vmappable).
        env_params: Environment parameters (shared across envs).
        init_hstate: Initial hidden state pytree, shape (n_envs, hidden_dim).
        collect_values: If True, collect per-step V(s) predictions.
        collect_rewards: If True, collect per-step rewards.
        collect_actions: If True, collect per-step actions.
        collect_entropies: If True, collect per-step policy entropy.
        collect_logits: If True, collect per-step policy logits.
        collection_steps: List of timesteps at which to snapshot hidden states.
                          -1 means terminal (first done or end of episode).
                          Positive integers are 1-indexed timesteps.
        return_final_hstate: If True, return the final hidden state pytree.

    Returns:
        RolloutResult with requested fields populated and others set to None.
    """
    n_envs = jax.tree_util.tree_leaves(levels)[0].shape[0]

    # Reset all environments in parallel
    rng, rng_reset = jax.random.split(rng)
    reset_rngs = jax.random.split(rng_reset, n_envs)
    obs, env_states = jax.vmap(
        env.reset_to_level, in_axes=(0, 0, None)
    )(reset_rngs, levels, env_params)

    hstate = init_hstate

    # --- Collection step setup ---
    if collection_steps is not None:
        positive_steps = sorted([s for s in collection_steps if s > 0])
        collect_terminal = -1 in collection_steps
    else:
        positive_steps = []
        collect_terminal = False

    n_positive = len(positive_steps)
    collected_positive = jax.tree_util.tree_map(
        lambda x: jnp.zeros((n_positive,) + x.shape), hstate
    ) if n_positive > 0 else None

    # Terminal hstate buffer
    terminal_hstate = jax.tree_util.tree_map(jnp.zeros_like, hstate)

    # Tracking arrays
    ep_done = jnp.zeros(n_envs, dtype=bool)
    total_return = jnp.zeros(n_envs)
    ep_reward_at_done = jnp.zeros(n_envs)
    terminal_captured = jnp.zeros(n_envs, dtype=bool)

    positive_steps_arr = jnp.array(positive_steps, dtype=jnp.int32) if n_positive > 0 else jnp.zeros((0,), dtype=jnp.int32)

    def scan_body(carry, step_idx):
        (hstate, obs, env_states, ep_done, total_return,
         ep_reward_at_done, terminal_captured,
         terminal_hstate, collected_positive, rng) = carry

        rng, rng_action = jax.random.split(rng)

        # Forward pass: obs (n_envs, *obs_shape) -> add time dim -> (1, n_envs, *obs_shape)
        obs_batch = jax.tree_util.tree_map(lambda x: x[None, ...], obs)
        done_batch = ep_done[None, :]  # (1, n_envs)
        hstate, pi, value = apply_fn(params, (obs_batch, done_batch), hstate)

        # --- Collect at positive steps ---
        current_step = step_idx + 1  # 1-indexed
        if n_positive > 0:
            for i, s in enumerate(positive_steps):
                should_collect = (current_step == s)
                collected_positive = jax.tree_util.tree_map(
                    lambda buf, h: buf.at[i].set(
                        jnp.where(should_collect, h, buf[i])
                    ),
                    collected_positive, hstate,
                )

        # --- Per-step outputs ---
        # value shape: (1, n_envs) -> (n_envs,)
        v = value[0]
        v_masked = jnp.where(ep_done, jnp.nan, v) if collect_values else jnp.float32(0.0)

        # Entropy
        if collect_entropies:
            try:
                entropy = pi.entropy()[0]  # (n_envs,)
            except (AttributeError, NotImplementedError):
                # Fallback: compute from log_prob over action space
                entropy = jnp.zeros(n_envs)
            e_masked = jnp.where(ep_done, jnp.nan, entropy)
        else:
            e_masked = jnp.float32(0.0)

        # Sample actions
        action = pi.sample(seed=rng_action)[0]  # (n_envs,)

        # Logits: pi.logits shape is (1, n_envs, n_actions) -> [0] gives (n_envs, n_actions)
        if collect_logits:
            l_masked = jnp.where(ep_done[:, None], jnp.nan, pi.logits[0])
        else:
            l_masked = jnp.zeros((n_envs, 1))

        # Step all environments in parallel
        step_rngs = jax.random.split(rng_action, n_envs)
        obs, env_states, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(step_rngs, env_states, action, env_params)

        # Masked reward
        r_masked = jnp.where(ep_done, 0.0, reward) if collect_rewards else jnp.float32(0.0)

        # Action (masked to 0 after done)
        a_masked = jnp.where(ep_done, 0, action) if collect_actions else jnp.int32(0)

        # Accumulate returns only for non-terminated episodes
        total_return = total_return + reward * (~ep_done).astype(jnp.float32)

        # Capture terminal hstate at first done
        newly_done = done & (~ep_done)
        terminal_hstate = jax.tree_util.tree_map(
            lambda th, h: jnp.where(newly_done[:, None], h, th),
            terminal_hstate, hstate,
        )
        ep_reward_at_done = jnp.where(newly_done, reward, ep_reward_at_done)
        terminal_captured = terminal_captured | newly_done

        # Update episode done tracking
        ep_done = ep_done | done

        carry = (hstate, obs, env_states, ep_done, total_return,
                 ep_reward_at_done, terminal_captured,
                 terminal_hstate, collected_positive, rng)
        return carry, (v_masked, r_masked, a_masked, e_masked, l_masked, done)

    # --- Run scan ---
    init_carry = (hstate, obs, env_states, ep_done, total_return,
                  ep_reward_at_done, terminal_captured,
                  terminal_hstate, collected_positive, rng)
    t_scan = time.time()
    final_carry, scan_outputs = jax.lax.scan(scan_body, init_carry, jnp.arange(max_steps))

    (hstate_final, _, _, ep_done_final, total_return_final,
     ep_reward_at_done_final, terminal_captured_final,
     terminal_hstate_final, collected_positive_final, _) = final_carry

    all_values, all_rewards, all_actions, all_entropies, all_logits, all_dones = scan_outputs

    jax.block_until_ready(final_carry)
    logger.info(f"[batched_rollout] scan ({max_steps} steps x {n_envs} envs): "
                f"{time.time() - t_scan:.2f}s (includes JIT on first call)")

    # --- For levels that never terminated, use final hstate as terminal ---
    terminal_hstate_final = jax.tree_util.tree_map(
        lambda th, h: jnp.where(
            (~terminal_captured_final)[:, None], h, th
        ),
        terminal_hstate_final, hstate_final,
    )

    # --- Build hstates_by_step ---
    hstates_by_step = None
    if collection_steps is not None:
        def flatten_hstate(hs):
            h_c, h_h = hs
            return jnp.concatenate([h_c, h_h], axis=-1)

        hstates_by_step = {}
        step_idx_map = {s: i for i, s in enumerate(positive_steps)}

        for s in collection_steps:
            if s == -1:
                hstates_by_step["-1"] = np.array(flatten_hstate(terminal_hstate_final))
            else:
                i = step_idx_map[s]
                hs = jax.tree_util.tree_map(lambda x: x[i], collected_positive_final)
                hstates_by_step[str(s)] = np.array(flatten_hstate(hs))

        # Fill uncollected positive steps with terminal for early-ending episodes
        terminal_flat = hstates_by_step.get("-1")
        if terminal_flat is not None:
            for s_key in [str(s) for s in positive_steps]:
                h = hstates_by_step[s_key]
                is_zero = np.all(h == 0, axis=-1)
                if np.any(is_zero):
                    hstates_by_step[s_key] = np.where(
                        is_zero[:, None], terminal_flat, h
                    )

    # --- Per-step arrays ---
    # all_* shapes: (max_steps, n_envs) -> transpose to (n_envs, max_steps)
    out_values = np.array(all_values.T) if collect_values else None
    out_rewards = np.array(all_rewards.T) if collect_rewards else None
    out_actions = np.array(all_actions.T) if collect_actions else None
    out_entropies = np.array(all_entropies.T) if collect_entropies else None
    # all_logits shape: (max_steps, n_envs, n_actions) -> (n_envs, max_steps, n_actions)
    out_logits = np.array(all_logits.transpose(1, 0, 2)) if collect_logits else None

    # --- Episode lengths ---
    all_dones_np = np.array(all_dones.T)  # (n_envs, max_steps)
    done_any = all_dones_np.any(axis=1)
    first_done = all_dones_np.argmax(axis=1)
    episode_lengths = np.where(done_any, first_done + 1, max_steps)

    # --- Episode returns & solved ---
    episode_returns = np.array(total_return_final)
    episode_solved = np.array(ep_reward_at_done_final > 0)

    # --- Final hstate ---
    out_final_hstate = hstate_final if return_final_hstate else None

    return RolloutResult(
        episode_returns=episode_returns,
        episode_solved=episode_solved,
        episode_lengths=episode_lengths,
        values=out_values,
        rewards=out_rewards,
        actions=out_actions,
        entropies=out_entropies,
        logits=out_logits,
        hstates_by_step=hstates_by_step,
        final_hstate=out_final_hstate,
    )
