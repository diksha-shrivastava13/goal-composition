"""
Transfer metrics for mutation adaptation experiments.

Measures behavioral and representational transfer from replay to mutation levels.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import entropy


def compute_behavioral_transfer(
    mutation_steps_to_solve: np.ndarray,
    random_steps_to_solve: np.ndarray,
    mutation_success_rate: float,
    random_success_rate: float,
    mutation_first_actions: np.ndarray,
    replay_first_actions: np.ndarray,
) -> Dict[str, float]:
    """
    Compute behavioral transfer metrics from replay to mutation.

    Args:
        mutation_steps_to_solve: Steps to solve mutations
        random_steps_to_solve: Steps to solve random levels (same difficulty)
        mutation_success_rate: Success rate on mutations
        random_success_rate: Success rate on random levels
        mutation_first_actions: First K actions on mutations, shape (n, K)
        replay_first_actions: First K actions on corresponding replays, shape (n, K)

    Returns:
        Dict with transfer metrics
    """
    mutation_steps = np.asarray(mutation_steps_to_solve)
    random_steps = np.asarray(random_steps_to_solve)

    # Speed ratio: < 1 means faster on mutations (transfer helps)
    speed_ratio = np.mean(mutation_steps) / (np.mean(random_steps) + 1e-10)

    # Success rate advantage
    success_advantage = mutation_success_rate - random_success_rate

    # Action similarity: do first actions match between replay and mutation?
    mutation_first = np.asarray(mutation_first_actions)
    replay_first = np.asarray(replay_first_actions)

    if mutation_first.shape == replay_first.shape and len(mutation_first) > 0:
        action_match_rate = np.mean(mutation_first == replay_first)
    else:
        action_match_rate = 0.0

    return {
        "speed_ratio": float(speed_ratio),
        "transfer_helps": speed_ratio < 1.0,
        "success_advantage": float(success_advantage),
        "first_action_match_rate": float(action_match_rate),
    }


def compute_representational_transfer(
    mutation_hstates: np.ndarray,
    replay_hstates: np.ndarray,
    mutation_values: np.ndarray,
    replay_values: np.ndarray,
) -> Dict[str, float]:
    """
    Compute representational transfer metrics.

    Args:
        mutation_hstates: Hidden states at start of mutations, shape (n, hidden_dim)
        replay_hstates: Hidden states at start of corresponding replays, shape (n, hidden_dim)
        mutation_values: Value estimates at start of mutations, shape (n,)
        replay_values: Value estimates at start of corresponding replays, shape (n,)

    Returns:
        Dict with representational metrics
    """
    mutation_h = np.asarray(mutation_hstates)
    replay_h = np.asarray(replay_hstates)
    mutation_v = np.asarray(mutation_values)
    replay_v = np.asarray(replay_values)

    # Cosine similarity between hidden states
    def cosine_sim(a, b):
        norm_a = np.linalg.norm(a, axis=-1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=-1, keepdims=True)
        return np.sum(a * b, axis=-1) / (norm_a.squeeze() * norm_b.squeeze() + 1e-10)

    hstate_similarity = cosine_sim(mutation_h, replay_h).mean()

    # Value correlation
    if len(mutation_v) > 1:
        value_correlation = np.corrcoef(mutation_v, replay_v)[0, 1]
    else:
        value_correlation = 0.0

    # Value transfer: how well does replay value predict mutation value?
    value_mae = np.mean(np.abs(mutation_v - replay_v))

    # L2 distance in hidden space
    hstate_distance = np.linalg.norm(mutation_h - replay_h, axis=-1).mean()

    return {
        "hstate_cosine_similarity": float(hstate_similarity),
        "value_correlation": float(value_correlation),
        "value_mae": float(value_mae),
        "hstate_l2_distance": float(hstate_distance),
    }


def compute_td_error_surprise(
    values: np.ndarray,
    rewards: np.ndarray,
    gamma: float = 0.995,
) -> Dict[str, float]:
    """
    Compute TD error as a surprise signal.

    High TD error indicates unexpected outcomes (agent surprised by level).

    Args:
        values: Value estimates over episode, shape (T,) or (T, n_envs)
        rewards: Rewards over episode, shape (T,) or (T, n_envs)
        gamma: Discount factor

    Returns:
        Dict with TD error statistics
    """
    values = np.asarray(values)
    rewards = np.asarray(rewards)

    if values.ndim == 1:
        values = values[:, None]
        rewards = rewards[:, None]

    # Compute TD errors: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    T = values.shape[0]
    td_errors = np.zeros_like(values[:-1])

    for t in range(T - 1):
        td_errors[t] = rewards[t] + gamma * values[t + 1] - values[t]

    td_errors_flat = td_errors.flatten()

    return {
        "mean_td_error": float(np.mean(td_errors_flat)),
        "std_td_error": float(np.std(td_errors_flat)),
        "max_td_error": float(np.max(np.abs(td_errors_flat))),
        "td_error_series": td_errors.mean(axis=-1).tolist() if values.shape[1] > 1 else td_errors_flat.tolist(),
    }


def compute_policy_divergence(
    mutation_policy: np.ndarray,
    replay_policy: np.ndarray,
) -> Dict[str, float]:
    """
    Compute policy divergence between replay and mutation.

    Args:
        mutation_policy: Policy logits on mutations, shape (n, n_actions)
        replay_policy: Policy logits on replays, shape (n, n_actions)

    Returns:
        Dict with divergence metrics
    """
    mutation_p = np.asarray(mutation_policy)
    replay_p = np.asarray(replay_policy)

    # Convert to probabilities
    def softmax(x):
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    mutation_probs = softmax(mutation_p)
    replay_probs = softmax(replay_p)

    # KL divergence: D_KL(mutation || replay)
    # Clip to avoid log(0)
    eps = 1e-10
    kl_divs = np.sum(
        mutation_probs * np.log((mutation_probs + eps) / (replay_probs + eps)),
        axis=-1
    )
    mean_kl = np.mean(kl_divs)

    # JS divergence (symmetric)
    m = 0.5 * (mutation_probs + replay_probs)
    js_divs = 0.5 * np.sum(mutation_probs * np.log((mutation_probs + eps) / (m + eps)), axis=-1)
    js_divs += 0.5 * np.sum(replay_probs * np.log((replay_probs + eps) / (m + eps)), axis=-1)
    mean_js = np.mean(js_divs)

    # Total variation distance
    tv_dists = 0.5 * np.sum(np.abs(mutation_probs - replay_probs), axis=-1)
    mean_tv = np.mean(tv_dists)

    return {
        "kl_divergence": float(mean_kl),
        "js_divergence": float(mean_js),
        "total_variation": float(mean_tv),
    }


def compute_adaptation_dynamics(
    policy_over_time: np.ndarray,
    initial_replay_policy: np.ndarray,
    timesteps: Optional[List[int]] = None,
) -> Dict[str, any]:
    """
    Track how policy diverges from replay behavior over episode.

    Args:
        policy_over_time: Policy logits over episode, shape (T, n_envs, n_actions)
        initial_replay_policy: Policy from replay level, shape (n_envs, n_actions)
        timesteps: Timesteps to measure (default: [1, 5, 10, 20, 50])

    Returns:
        Dict with divergence over time
    """
    if timesteps is None:
        timesteps = [1, 5, 10, 20, 50]

    policy_t = np.asarray(policy_over_time)
    replay_p = np.asarray(initial_replay_policy)

    T = policy_t.shape[0]
    divergences = {}

    for t in timesteps:
        if t >= T:
            continue

        div_metrics = compute_policy_divergence(policy_t[t], replay_p)
        divergences[f"t={t}"] = div_metrics["kl_divergence"]

    # Compute when policy diverges significantly (KL > 0.5)
    all_kls = []
    for t in range(T):
        div = compute_policy_divergence(policy_t[t], replay_p)
        all_kls.append(div["kl_divergence"])

    all_kls = np.array(all_kls)
    divergence_point = np.argmax(all_kls > 0.5) if any(all_kls > 0.5) else T

    return {
        "divergence_by_timestep": divergences,
        "divergence_point": int(divergence_point),
        "max_divergence": float(np.max(all_kls)),
        "final_divergence": float(all_kls[-1]) if len(all_kls) > 0 else 0.0,
    }
