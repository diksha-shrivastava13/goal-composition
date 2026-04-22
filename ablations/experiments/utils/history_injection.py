"""
History injection utilities for counterfactual experiments.

Contains functions for creating and injecting false histories into
agent memory mechanisms.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import chex


def create_failure_history(
    agent=None,
    train_state=None,
    rng: Optional[chex.PRNGKey] = None,
    n_episodes: int = 10,
    hidden_dim: int = 256,
    mean_return: float = -1.0,
    context_dim: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Create failure history for injection, preferring real rollout data.

    When agent and train_state are provided, runs actual rollouts and filters
    for failure episodes. Falls back to synthetic generation when no agent
    is available (e.g., for controlled baselines).

    Args:
        agent: Optional agent instance for real rollouts
        train_state: Optional train state for real rollouts
        rng: JAX PRNGKey for reproducible randomness
        n_episodes: Number of failure episodes to collect/simulate
        hidden_dim: Hidden state dimension (synthetic fallback only)
        mean_return: Mean return for failure episodes (synthetic fallback only)
        context_dim: Context vector dimension (synthetic fallback only)

    Returns:
        Dict with history components (hidden_states, returns, solved, lengths, context_vector)
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # --- Real rollout path ---
    if agent is not None and train_state is not None:
        from .paired_helpers import run_batched_rollout
        from .batched_rollout import generate_levels

        # Generate more levels than needed, then filter for failures
        n_candidates = n_episodes * 4
        rng, level_rng = jax.random.split(rng)
        levels = generate_levels(agent, level_rng, n_candidates)

        rng, roll_rng = jax.random.split(rng)
        result = run_batched_rollout(
            roll_rng, levels, train_state, agent,
            max_steps=256,
            return_final_hstate=True,
        )

        returns = np.array(result.episode_returns)
        solved = np.array(result.episode_solved)
        lengths = np.array(result.episode_lengths)

        # Filter for failure episodes (unsolved or low return)
        failure_mask = ~solved
        if failure_mask.sum() < n_episodes:
            # Not enough unsolved — take lowest-return episodes
            sorted_idx = np.argsort(returns)
            failure_idx = sorted_idx[:n_episodes]
        else:
            failure_idx = np.where(failure_mask)[0][:n_episodes]

        hstates = np.array(result.final_hstate)
        if hstates.ndim == 3:
            # Shape: (2, batch, hidden_dim) — (c, h) tuple stacked
            hstates = np.concatenate([hstates[0], hstates[1]], axis=-1)
        selected_hstates = hstates[failure_idx]

        sel_returns = returns[failure_idx]
        sel_solved = solved[failure_idx]
        sel_lengths = lengths[failure_idx]

        # Build context vector from actual episode statistics
        context = np.zeros(context_dim)
        context[0] = float(np.mean(sel_returns))
        context[1] = float(np.mean(sel_solved))
        context[2] = float(np.mean(sel_lengths)) / 256.0

        return {
            "hidden_states": selected_hstates,
            "returns": sel_returns,
            "solved": sel_solved,
            "lengths": sel_lengths,
            "context_vector": context,
        }

    # --- Synthetic fallback (for controlled baselines) ---
    rng, hstate_rng = jax.random.split(rng)
    hstates = np.array(jax.random.normal(hstate_rng, (n_episodes, 2 * hidden_dim))) * 0.1

    rng, ret_rng, len_rng = jax.random.split(rng, 3)
    returns = np.array(jax.random.uniform(ret_rng, (n_episodes,), minval=mean_return - 0.5, maxval=mean_return + 0.5))
    solved = np.zeros(n_episodes, dtype=bool)
    lengths = np.array(jax.random.randint(len_rng, (n_episodes,), minval=200, maxval=256))

    context = np.zeros(context_dim)
    context[0] = mean_return
    context[1] = 0.0
    context[2] = 0.9

    return {
        "hidden_states": hstates,
        "returns": returns,
        "solved": solved,
        "lengths": lengths,
        "context_vector": context,
    }


def create_success_history(
    agent=None,
    train_state=None,
    rng: Optional[chex.PRNGKey] = None,
    n_episodes: int = 10,
    hidden_dim: int = 256,
    mean_return: float = 1.0,
    context_dim: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Create success history for injection, preferring real rollout data.

    When agent and train_state are provided, runs actual rollouts and filters
    for success episodes. Falls back to synthetic generation when no agent
    is available (e.g., for controlled baselines).

    Args:
        agent: Optional agent instance for real rollouts
        train_state: Optional train state for real rollouts
        rng: JAX PRNGKey for reproducible randomness
        n_episodes: Number of success episodes to collect/simulate
        hidden_dim: Hidden state dimension (synthetic fallback only)
        mean_return: Mean return for success episodes (synthetic fallback only)
        context_dim: Context vector dimension (synthetic fallback only)

    Returns:
        Dict with history components (hidden_states, returns, solved, lengths, context_vector)
    """
    if rng is None:
        rng = jax.random.PRNGKey(1)

    # --- Real rollout path ---
    if agent is not None and train_state is not None:
        from .paired_helpers import run_batched_rollout
        from .batched_rollout import generate_levels

        # Generate more levels than needed, then filter for successes
        n_candidates = n_episodes * 4
        rng, level_rng = jax.random.split(rng)
        levels = generate_levels(agent, level_rng, n_candidates)

        rng, roll_rng = jax.random.split(rng)
        result = run_batched_rollout(
            roll_rng, levels, train_state, agent,
            max_steps=256,
            return_final_hstate=True,
        )

        returns = np.array(result.episode_returns)
        solved = np.array(result.episode_solved)
        lengths = np.array(result.episode_lengths)

        # Filter for success episodes (solved or high return)
        success_mask = solved
        if success_mask.sum() < n_episodes:
            # Not enough solved — take highest-return episodes
            sorted_idx = np.argsort(returns)[::-1]
            success_idx = sorted_idx[:n_episodes]
        else:
            success_idx = np.where(success_mask)[0][:n_episodes]

        hstates = np.array(result.final_hstate)
        if hstates.ndim == 3:
            hstates = np.concatenate([hstates[0], hstates[1]], axis=-1)
        selected_hstates = hstates[success_idx]

        sel_returns = returns[success_idx]
        sel_solved = solved[success_idx]
        sel_lengths = lengths[success_idx]

        context = np.zeros(context_dim)
        context[0] = float(np.mean(sel_returns))
        context[1] = float(np.mean(sel_solved))
        context[2] = float(np.mean(sel_lengths)) / 256.0

        return {
            "hidden_states": selected_hstates,
            "returns": sel_returns,
            "solved": sel_solved,
            "lengths": sel_lengths,
            "context_vector": context,
        }

    # --- Synthetic fallback (for controlled baselines) ---
    rng, hstate_rng = jax.random.split(rng)
    hstates = np.array(jax.random.normal(hstate_rng, (n_episodes, 2 * hidden_dim))) * 0.5

    rng, ret_rng, len_rng = jax.random.split(rng, 3)
    returns = np.array(jax.random.uniform(ret_rng, (n_episodes,), minval=mean_return - 0.2, maxval=mean_return + 0.2))
    solved = np.ones(n_episodes, dtype=bool)
    lengths = np.array(jax.random.randint(len_rng, (n_episodes,), minval=10, maxval=100))

    context = np.zeros(context_dim)
    context[0] = mean_return
    context[1] = 1.0
    context[2] = 0.2

    return {
        "hidden_states": hstates,
        "returns": returns,
        "solved": solved,
        "lengths": lengths,
        "context_vector": context,
    }


def inject_hidden_state(
    current_hstate: chex.ArrayTree,
    target_pattern,
    scale: float = 1.0,
    hidden_dim: int = 256,
) -> chex.ArrayTree:
    """
    Inject pattern into hidden state.

    Args:
        current_hstate: Current (c, h) tuple
        target_pattern: Either a string ("failure", "success", "random") for synthetic
            perturbation, or a dict from create_*_history() with real hidden states.
        scale: Scale of injection
        hidden_dim: Hidden state dimension (synthetic mode only)

    Returns:
        Modified hidden state
    """
    h_c, h_h = current_hstate

    # --- Dict path: use real episode hidden states ---
    if isinstance(target_pattern, dict) and "hidden_states" in target_pattern:
        real_hstates = np.asarray(target_pattern["hidden_states"])
        # Average the real hidden states to get a representative delta
        mean_hstate = np.mean(real_hstates, axis=0)

        # Split into (c, h) components
        h_dim = h_c.shape[-1]
        if mean_hstate.shape[-1] == 2 * h_dim:
            delta_c_base = jnp.array(mean_hstate[..., :h_dim])
            delta_h_base = jnp.array(mean_hstate[..., h_dim:])
        else:
            # Dimensions don't match — use as-is for h, zero for c
            delta_c_base = jnp.zeros(h_c.shape[-1:])
            delta_h_base = jnp.array(mean_hstate[..., :h_h.shape[-1]])

        # Broadcast to batch dimension and apply scale
        if h_c.ndim > 1:
            delta_c = jnp.broadcast_to(delta_c_base, h_c.shape) * scale
            delta_h = jnp.broadcast_to(delta_h_base, h_h.shape) * scale
        else:
            delta_c = delta_c_base * scale
            delta_h = delta_h_base * scale

        return (h_c + delta_c, h_h + delta_h)

    # --- String path: synthetic perturbation ---
    if target_pattern == "failure":
        key = jax.random.PRNGKey(99)
        delta_c = jax.random.normal(key, h_c.shape) * scale * 0.8
        key, _ = jax.random.split(key)
        delta_h = jax.random.normal(key, h_h.shape) * scale * 0.8
        if h_c.ndim > 1:
            delta_c = delta_c.at[:, :10].set(-scale * 0.5)
            delta_h = delta_h.at[:, :10].set(-scale * 0.5)
        else:
            delta_c = delta_c.at[:10].set(-scale * 0.5)
            delta_h = delta_h.at[:10].set(-scale * 0.5)
    elif target_pattern == "success":
        delta_c = jnp.zeros_like(h_c)
        delta_h = jnp.zeros_like(h_h)
        if h_c.ndim > 1:
            delta_c = delta_c.at[:, :10].set(scale * 0.5)
            delta_h = delta_h.at[:, :10].set(scale * 0.5)
        else:
            delta_c = delta_c.at[:10].set(scale * 0.5)
            delta_h = delta_h.at[:10].set(scale * 0.5)
    elif target_pattern == "random":
        key = jax.random.PRNGKey(42)
        delta_c = jax.random.normal(key, h_c.shape) * scale * 0.1
        key, _ = jax.random.split(key)
        delta_h = jax.random.normal(key, h_h.shape) * scale * 0.1
    else:
        delta_c = jnp.zeros_like(h_c)
        delta_h = jnp.zeros_like(h_h)

    return (h_c + delta_c, h_h + delta_h)


def inject_context_vector(
    current_context: chex.Array,
    history: Dict[str, np.ndarray],
    decay: float = 0.9,
) -> chex.Array:
    """
    Inject history into context vector via EMA update.

    Args:
        current_context: Current context vector
        history: History dict from create_*_history
        decay: EMA decay factor

    Returns:
        Updated context vector
    """
    injected_context = jnp.array(history["context_vector"])

    # Blend with current context using decay
    new_context = decay * current_context + (1 - decay) * injected_context

    return new_context


def inject_episodic_memory(
    memory_buffer: Dict[str, chex.Array],
    history: Dict[str, np.ndarray],
    n_inject: int = 10,
    rng: Optional[chex.PRNGKey] = None,
) -> Dict[str, chex.Array]:
    """
    Inject false episodes into episodic memory buffer.

    Args:
        memory_buffer: Current episodic memory state
        history: History dict to inject
        n_inject: Number of episodes to inject
        rng: Optional JAX PRNGKey for reproducible randomness

    Returns:
        Modified memory buffer
    """
    buffer_size = memory_buffer["episode_embeddings"].shape[0]
    embed_dim = memory_buffer["episode_embeddings"].shape[1]

    # Create embeddings from history
    n_inject = min(n_inject, len(history["returns"]))

    # Random projection embedding: project [return, solved, length] into embed_dim
    if rng is not None:
        seed = int(jax.random.key_data(rng)[0])
    else:
        seed = int(np.sum(history["returns"][:n_inject] * np.arange(1, n_inject + 1)) % (2**31))
    rng_proj = np.random.default_rng(seed=seed)
    projection = rng_proj.standard_normal((3, embed_dim)) / np.sqrt(embed_dim)
    raw_features = np.stack([
        history["returns"][:n_inject],
        history["solved"][:n_inject].astype(float),
        history["lengths"][:n_inject] / 256.0,
    ], axis=1)  # (n_inject, 3)
    embeddings = raw_features @ projection  # (n_inject, embed_dim)

    # Insert into buffer (overwrite oldest entries)
    new_embeddings = memory_buffer["episode_embeddings"].at[:n_inject].set(embeddings)
    new_returns = memory_buffer["episode_returns"].at[:n_inject].set(history["returns"][:n_inject])
    new_lengths = memory_buffer["episode_lengths"].at[:n_inject].set(history["lengths"][:n_inject])
    new_solved = memory_buffer["episode_solved"].at[:n_inject].set(history["solved"][:n_inject])

    return {
        **memory_buffer,
        "episode_embeddings": new_embeddings,
        "episode_returns": new_returns,
        "episode_lengths": new_lengths,
        "episode_solved": new_solved,
    }


def measure_injection_effect(
    baseline_predictions: Dict[str, np.ndarray],
    injected_predictions: Dict[str, np.ndarray],
    baseline_behavior: Dict[str, np.ndarray],
    injected_behavior: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Measure effect of history injection on predictions and behavior.

    Args:
        baseline_predictions: Predictions without injection
        injected_predictions: Predictions with injection
        baseline_behavior: Policy outputs without injection
        injected_behavior: Policy outputs with injection

    Returns:
        Dict with effect metrics
    """
    results = {}

    # Prediction shift
    for key in baseline_predictions:
        if key in injected_predictions:
            baseline = np.asarray(baseline_predictions[key])
            injected = np.asarray(injected_predictions[key])

            # L2 distance
            pred_shift = np.linalg.norm(injected - baseline)
            results[f"prediction_shift_{key}"] = float(pred_shift)

    # Behavioral shift
    if "policy_logits" in baseline_behavior and "policy_logits" in injected_behavior:
        baseline_policy = np.asarray(baseline_behavior["policy_logits"])
        injected_policy = np.asarray(injected_behavior["policy_logits"])

        # Convert to probabilities
        def softmax(x):
            x_max = np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        baseline_probs = softmax(baseline_policy)
        injected_probs = softmax(injected_policy)

        # KL divergence
        eps = 1e-10
        kl = np.sum(injected_probs * np.log((injected_probs + eps) / (baseline_probs + eps)), axis=-1)
        results["behavioral_kl_divergence"] = float(np.mean(kl))

        # Action agreement
        baseline_actions = np.argmax(baseline_probs, axis=-1)
        injected_actions = np.argmax(injected_probs, axis=-1)
        action_agreement = np.mean(baseline_actions == injected_actions)
        results["action_agreement"] = float(action_agreement)

    if "values" in baseline_behavior and "values" in injected_behavior:
        baseline_v = np.asarray(baseline_behavior["values"])
        injected_v = np.asarray(injected_behavior["values"])

        value_shift = np.mean(injected_v - baseline_v)
        results["value_shift"] = float(value_shift)
        results["value_shift_abs"] = float(np.mean(np.abs(injected_v - baseline_v)))

    return results


def compute_intervention_magnitude(
    original_hstate: chex.ArrayTree,
    modified_hstate: chex.ArrayTree,
) -> Dict[str, float]:
    """
    Compute magnitude of hidden state intervention.

    Args:
        original_hstate: Original (c, h) tuple
        modified_hstate: Modified (c, h) tuple

    Returns:
        Dict with magnitude metrics
    """
    orig_c, orig_h = original_hstate
    mod_c, mod_h = modified_hstate

    orig_c = np.asarray(orig_c)
    orig_h = np.asarray(orig_h)
    mod_c = np.asarray(mod_c)
    mod_h = np.asarray(mod_h)

    # L2 distances
    c_diff = np.linalg.norm(mod_c - orig_c)
    h_diff = np.linalg.norm(mod_h - orig_h)
    total_diff = np.sqrt(c_diff ** 2 + h_diff ** 2)

    # Relative magnitude
    orig_norm = np.sqrt(np.linalg.norm(orig_c) ** 2 + np.linalg.norm(orig_h) ** 2)
    relative_magnitude = total_diff / (orig_norm + 1e-10)

    # Cosine similarity before/after
    orig_flat = np.concatenate([orig_c.flatten(), orig_h.flatten()])
    mod_flat = np.concatenate([mod_c.flatten(), mod_h.flatten()])
    cosine_sim = np.dot(orig_flat, mod_flat) / (np.linalg.norm(orig_flat) * np.linalg.norm(mod_flat) + 1e-10)

    return {
        "cell_state_diff": float(c_diff),
        "hidden_state_diff": float(h_diff),
        "total_diff": float(total_diff),
        "relative_magnitude": float(relative_magnitude),
        "cosine_similarity": float(cosine_sim),
    }
