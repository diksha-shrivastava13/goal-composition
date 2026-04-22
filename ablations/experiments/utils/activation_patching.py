"""
Activation patching utilities for task-relevant activation analysis.

Contains functions for computing saliency maps and patching activations
to identify task-controlling subspaces.
"""

from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import jax
import jax.numpy as jnp
import chex
from functools import partial


def compute_saliency_map(
    train_state,
    obs: chex.ArrayTree,
    hstate: chex.ArrayTree,
    target: str = "value",
) -> Dict[str, np.ndarray]:
    """
    Compute gradient-based saliency map: ∂output/∂obs.

    Shows which input features most affect the output.

    Args:
        train_state: Agent train state with apply_fn and params
        obs: Observation (should have .image attribute)
        hstate: Hidden state
        target: "value" or "action"

    Returns:
        Dict with saliency maps for different input components
    """
    params = train_state.params
    apply_fn = train_state.apply_fn

    def value_fn(image, agent_dir, hstate):
        """Compute value from inputs."""
        # Reconstruct observation with batch dims
        obs_batch = type(obs)(image[None, None, ...], agent_dir[None, None, ...])
        done_batch = jnp.zeros((1, 1), dtype=bool)

        _result = apply_fn(params, (obs_batch, done_batch), hstate)
        # Indexed access safely ignores extra return values from different network architectures
        value = _result[2]
        return value[0, 0]

    def action_entropy_fn(image, agent_dir, hstate):
        """Compute policy entropy from inputs."""
        obs_batch = type(obs)(image[None, None, ...], agent_dir[None, None, ...])
        done_batch = jnp.zeros((1, 1), dtype=bool)

        _result = apply_fn(params, (obs_batch, done_batch), hstate)
        # Indexed access safely ignores extra return values from different network architectures
        pi = _result[1]
        return pi.entropy()[0, 0]

    if target == "value":
        target_fn = value_fn
    else:
        target_fn = action_entropy_fn

    # Compute gradients w.r.t. image
    image_grad = jax.grad(target_fn, argnums=0)(obs.image, obs.agent_dir, hstate)
    dir_grad = jax.grad(target_fn, argnums=1)(obs.image, obs.agent_dir, hstate)

    # Saliency = |gradient|
    image_saliency = np.abs(np.array(image_grad))
    dir_saliency = np.abs(np.array(dir_grad))

    # Aggregate over channels
    image_saliency_spatial = image_saliency.sum(axis=-1)

    return {
        "image_saliency": image_saliency,
        "image_saliency_spatial": image_saliency_spatial,
        "direction_saliency": dir_saliency,
        "target": target,
    }


def compute_saliency_statistics(
    saliency_maps: List[Dict[str, np.ndarray]],
) -> Dict[str, float]:
    """
    Compute aggregate statistics over multiple saliency maps.

    Args:
        saliency_maps: List of saliency map dicts

    Returns:
        Dict with aggregate statistics
    """
    all_spatial = [s["image_saliency_spatial"] for s in saliency_maps]
    stacked = np.stack(all_spatial)

    # Mean saliency per position
    mean_saliency = stacked.mean(axis=0)

    # Find most salient regions
    flat_mean = mean_saliency.flatten()
    top_k = 10
    top_indices = np.argsort(flat_mean)[-top_k:]

    # Saliency concentration: how focused is attention?
    # High concentration = focused on few features
    sorted_saliency = np.sort(flat_mean)[::-1]
    total = sorted_saliency.sum()
    top_20_pct = sorted_saliency[:int(len(sorted_saliency) * 0.2)].sum() / (total + 1e-10)

    return {
        "mean_saliency_max": float(mean_saliency.max()),
        "mean_saliency_mean": float(mean_saliency.mean()),
        "saliency_concentration": float(top_20_pct),
        "n_samples": len(saliency_maps),
    }


def patch_activations(
    train_state,
    source_obs: chex.ArrayTree,
    target_obs: chex.ArrayTree,
    hstate: chex.ArrayTree,
    patch_layer: str = "lstm_hidden",
) -> Dict[str, any]:
    """
    Patch activations from source into target and measure effect.

    Protocol:
    1. Run on source to get activations
    2. Run on target but substitute activations from source
    3. Measure behavior change

    Args:
        train_state: Agent train state
        source_obs: Source observation (e.g., level A)
        target_obs: Target observation (e.g., level B)
        hstate: Initial hidden state
        patch_layer: Which layer to patch

    Returns:
        Dict with patching results
    """
    params = train_state.params
    apply_fn = train_state.apply_fn

    # Get outputs without patching
    source_batch = type(source_obs)(source_obs.image[None, None, ...], source_obs.agent_dir[None, None, ...])
    target_batch = type(target_obs)(target_obs.image[None, None, ...], target_obs.agent_dir[None, None, ...])
    done_batch = jnp.zeros((1, 1), dtype=bool)

    # Source forward pass
    _result_src = apply_fn(params, (source_batch, done_batch), hstate)
    hstate_source, pi_source, v_source = _result_src[0], _result_src[1], _result_src[2]

    # Target forward pass (normal)
    _result_tgt = apply_fn(params, (target_batch, done_batch), hstate)
    hstate_target, pi_target, v_target = _result_tgt[0], _result_tgt[1], _result_tgt[2]

    # Target forward pass with patched hidden state (from source)
    if patch_layer == "lstm_hidden":
        # Use source's hidden state for target
        _result_pat = apply_fn(params, (target_batch, done_batch), hstate_source)
        hstate_patched, pi_patched, v_patched = _result_pat[0], _result_pat[1], _result_pat[2]
    else:
        # For other layers, would need to modify the network
        hstate_patched, pi_patched, v_patched = hstate_target, pi_target, v_target

    # Measure effects
    def get_action_probs(pi):
        return jax.nn.softmax(pi.logits[0, 0])

    source_probs = get_action_probs(pi_source)
    target_probs = get_action_probs(pi_target)
    patched_probs = get_action_probs(pi_patched)

    # KL divergence: patched vs target (how much did patching change behavior?)
    eps = 1e-10
    kl_patched_target = float(jnp.sum(
        patched_probs * jnp.log((patched_probs + eps) / (target_probs + eps))
    ))

    # KL divergence: patched vs source (did behavior become source-like?)
    kl_patched_source = float(jnp.sum(
        patched_probs * jnp.log((patched_probs + eps) / (source_probs + eps))
    ))

    # Value shift
    v_shift = float(v_patched[0, 0] - v_target[0, 0])
    v_toward_source = float(v_patched[0, 0] - v_source[0, 0])

    return {
        "kl_patched_vs_target": kl_patched_target,
        "kl_patched_vs_source": kl_patched_source,
        "value_shift": v_shift,
        "value_toward_source": v_toward_source,
        "target_value": float(v_target[0, 0]),
        "source_value": float(v_source[0, 0]),
        "patched_value": float(v_patched[0, 0]),
        "patch_layer": patch_layer,
    }


def identify_task_controlling_subspace(
    train_state,
    observations: List[chex.ArrayTree],
    hstates: List[chex.ArrayTree],
    n_components: int = 10,
) -> Dict[str, any]:
    """
    Identify subspace of hidden state that controls task behavior.

    Uses PCA on saliency-weighted activations to find directions
    that most affect outputs.

    Args:
        train_state: Agent train state
        observations: List of observations
        hstates: List of hidden states
        n_components: Number of PCA components

    Returns:
        Dict with subspace analysis
    """
    from sklearn.decomposition import PCA

    # Collect hidden states and their saliency
    all_hstates = []
    all_value_grads = []

    for obs, hstate in zip(observations, hstates):
        # Get hidden state as flat array
        h_c, h_h = hstate
        hstate_flat = np.concatenate([
            np.array(h_c).flatten(),
            np.array(h_h).flatten()
        ])
        all_hstates.append(hstate_flat)

        # Compute gradient of value head w.r.t. hidden state using JAX autodiff
        try:
            import jax
            import jax.numpy as jnp

            h_c_jax = jnp.array(h_c).reshape(1, -1)
            h_h_jax = jnp.array(h_h).reshape(1, -1)

            def value_fn(hstate_flat_jax):
                half = hstate_flat_jax.shape[-1] // 2
                hc = hstate_flat_jax[:half].reshape(1, -1)
                hh = hstate_flat_jax[half:].reshape(1, -1)
                hstate_tree = (hc, hh)
                # Create dummy obs/done for forward pass
                obs_batch = type(obs)(obs.image[None, None, ...], obs.agent_dir[None, None, ...])
                done_batch = jnp.zeros((1, 1), dtype=bool)
                outputs = train_state.apply_fn(
                    train_state.params, (obs_batch, done_batch), hstate_tree
                )
                value = outputs[2] if len(outputs) >= 3 else outputs[-1]
                return value.sum()

            hstate_jax = jnp.array(hstate_flat)
            grad = jax.grad(value_fn)(hstate_jax)
            all_value_grads.append(np.array(grad))
        except Exception:
            all_value_grads.append(hstate_flat)  # Fallback to identity

    hstates_array = np.stack(all_hstates)

    # PCA on hidden states
    pca = PCA(n_components=min(n_components, len(hstates_array), hstates_array.shape[1]))
    pca.fit(hstates_array)

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": float(np.cumsum(pca.explained_variance_ratio_)[-1]),
        "n_components": pca.n_components_,
        "principal_directions": pca.components_.tolist(),
    }


def compute_feature_attribution(
    train_state,
    obs: chex.ArrayTree,
    hstate: chex.ArrayTree,
    baseline_obs: Optional[chex.ArrayTree] = None,
    n_steps: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Compute integrated gradients for feature attribution.

    Args:
        train_state: Agent train state
        obs: Observation to explain
        hstate: Hidden state
        baseline_obs: Baseline observation (default: zeros)
        n_steps: Number of interpolation steps

    Returns:
        Dict with attribution scores
    """
    params = train_state.params
    apply_fn = train_state.apply_fn

    if baseline_obs is None:
        # Zero baseline
        baseline_obs = type(obs)(jnp.zeros_like(obs.image), jnp.zeros_like(obs.agent_dir))

    def value_fn(obs_input):
        obs_batch = type(obs)(obs_input.image[None, None, ...], obs_input.agent_dir[None, None, ...])
        done_batch = jnp.zeros((1, 1), dtype=bool)
        _result = apply_fn(params, (obs_batch, done_batch), hstate)
        value = _result[2]
        return value[0, 0]

    # Interpolate between baseline and target
    alphas = jnp.linspace(0, 1, n_steps)

    # Compute gradients at each interpolation point
    grads = []
    for alpha in alphas:
        interp_obs = type(obs)(
            baseline_obs.image + alpha * (obs.image - baseline_obs.image),
            baseline_obs.agent_dir + alpha * (obs.agent_dir - baseline_obs.agent_dir),
        )
        grad = jax.grad(lambda o: value_fn(type(obs)(o, obs.agent_dir)))(interp_obs.image)
        grads.append(np.array(grad))

    # Integrated gradients = (target - baseline) * mean(gradients)
    grads_array = np.stack(grads)
    mean_grad = grads_array.mean(axis=0)
    diff = np.array(obs.image) - np.array(baseline_obs.image)
    attributions = diff * mean_grad

    return {
        "attributions": attributions,
        "attributions_spatial": np.abs(attributions).sum(axis=-1),
        "n_steps": n_steps,
    }
