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
    n_episodes: int = 10,
    hidden_dim: int = 256,
    mean_return: float = -1.0,
    context_dim: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Create synthetic failure history for injection.

    Args:
        n_episodes: Number of failure episodes to simulate
        hidden_dim: Hidden state dimension
        mean_return: Mean return for failure episodes
        context_dim: Context vector dimension

    Returns:
        Dict with synthetic history components
    """
    # Low-magnitude hidden states (encoding failure/confusion)
    hstates = np.random.randn(n_episodes, 2 * hidden_dim) * 0.1

    # Low returns and unsolved flags
    returns = np.random.uniform(mean_return - 0.5, mean_return + 0.5, n_episodes)
    solved = np.zeros(n_episodes, dtype=bool)
    lengths = np.random.randint(200, 256, n_episodes)  # Long episodes (timeouts)

    # Context vector encoding failure
    context = np.zeros(context_dim)
    context[0] = mean_return  # Return feature
    context[1] = 0.0  # Solved flag
    context[2] = 0.9  # High length (normalized)

    return {
        "hidden_states": hstates,
        "returns": returns,
        "solved": solved,
        "lengths": lengths,
        "context_vector": context,
    }


def create_success_history(
    n_episodes: int = 10,
    hidden_dim: int = 256,
    mean_return: float = 1.0,
    context_dim: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Create synthetic success history for injection.

    Args:
        n_episodes: Number of success episodes to simulate
        hidden_dim: Hidden state dimension
        mean_return: Mean return for success episodes
        context_dim: Context vector dimension

    Returns:
        Dict with synthetic history components
    """
    # Higher-magnitude hidden states (confident representations)
    hstates = np.random.randn(n_episodes, 2 * hidden_dim) * 0.5

    # High returns and solved flags
    returns = np.random.uniform(mean_return - 0.2, mean_return + 0.2, n_episodes)
    solved = np.ones(n_episodes, dtype=bool)
    lengths = np.random.randint(10, 100, n_episodes)  # Short episodes (efficient)

    # Context vector encoding success
    context = np.zeros(context_dim)
    context[0] = mean_return  # Return feature
    context[1] = 1.0  # Solved flag
    context[2] = 0.2  # Low length (normalized)

    return {
        "hidden_states": hstates,
        "returns": returns,
        "solved": solved,
        "lengths": lengths,
        "context_vector": context,
    }


def inject_hidden_state(
    current_hstate: chex.ArrayTree,
    target_pattern: str,
    scale: float = 1.0,
    hidden_dim: int = 256,
) -> chex.ArrayTree:
    """
    Inject pattern into hidden state.

    Args:
        current_hstate: Current (c, h) tuple
        target_pattern: "failure", "success", or "random"
        scale: Scale of injection
        hidden_dim: Hidden state dimension

    Returns:
        Modified hidden state
    """
    h_c, h_h = current_hstate
    batch_size = h_c.shape[0] if h_c.ndim > 1 else 1

    if target_pattern == "failure":
        # Low-magnitude, diffuse pattern
        delta_c = jnp.zeros_like(h_c) * scale
        delta_h = jnp.zeros_like(h_h) * scale
    elif target_pattern == "success":
        # Higher-magnitude, structured pattern
        # Use first few dimensions to encode "success signal"
        delta_c = jnp.zeros_like(h_c)
        delta_h = jnp.zeros_like(h_h)
        if h_c.ndim > 1:
            delta_c = delta_c.at[:, :10].set(scale * 0.5)
            delta_h = delta_h.at[:, :10].set(scale * 0.5)
        else:
            delta_c = delta_c.at[:10].set(scale * 0.5)
            delta_h = delta_h.at[:10].set(scale * 0.5)
    elif target_pattern == "random":
        # Random perturbation
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
) -> Dict[str, chex.Array]:
    """
    Inject false episodes into episodic memory buffer.

    Args:
        memory_buffer: Current episodic memory state
        history: History dict to inject
        n_inject: Number of episodes to inject

    Returns:
        Modified memory buffer
    """
    buffer_size = memory_buffer["episode_embeddings"].shape[0]
    embed_dim = memory_buffer["episode_embeddings"].shape[1]

    # Create embeddings from history
    n_inject = min(n_inject, len(history["returns"]))

    # Simple embedding: [return, solved, length_normalized]
    embeddings = np.zeros((n_inject, embed_dim))
    embeddings[:, 0] = history["returns"][:n_inject]
    embeddings[:, 1] = history["solved"][:n_inject].astype(float)
    embeddings[:, 2] = history["lengths"][:n_inject] / 256.0

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
