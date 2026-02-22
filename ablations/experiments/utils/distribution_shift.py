"""
Distribution shift utilities for causal intervention experiments.

Contains functions for creating curriculum ablations and measuring
distribution shifts in hidden state space.
"""

from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import jax
import jax.numpy as jnp
import chex


def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "rbf",
    sigma: Optional[float] = None,
) -> float:
    """
    Compute Maximum Mean Discrepancy between two distributions.

    MMD measures how different two sample distributions are in feature space.

    Args:
        X: Samples from distribution P, shape (n_x, d)
        Y: Samples from distribution Q, shape (n_y, d)
        kernel: "rbf" or "linear"
        sigma: RBF kernel bandwidth (None = median heuristic)

    Returns:
        MMD estimate (biased)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    n_x, d = X.shape
    n_y = Y.shape[0]

    if kernel == "linear":
        # Linear kernel: k(x,y) = x^T y
        K_xx = X @ X.T
        K_yy = Y @ Y.T
        K_xy = X @ Y.T
    elif kernel == "rbf":
        # RBF kernel with median heuristic
        if sigma is None:
            # Compute pairwise distances for bandwidth selection
            all_data = np.vstack([X, Y])
            dists = _pairwise_sq_dists(all_data, all_data)
            sigma = np.sqrt(np.median(dists[dists > 0]))
            if sigma == 0:
                sigma = 1.0

        K_xx = _rbf_kernel(X, X, sigma)
        K_yy = _rbf_kernel(Y, Y, sigma)
        K_xy = _rbf_kernel(X, Y, sigma)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Biased MMD^2 estimate
    mmd_sq = (K_xx.sum() / (n_x * n_x) +
              K_yy.sum() / (n_y * n_y) -
              2 * K_xy.sum() / (n_x * n_y))

    return float(np.sqrt(max(mmd_sq, 0)))


def _pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances."""
    X_sqnorms = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sqnorms = np.sum(Y ** 2, axis=1, keepdims=True)
    return X_sqnorms + Y_sqnorms.T - 2 * X @ Y.T


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Compute RBF kernel matrix."""
    sq_dists = _pairwise_sq_dists(X, Y)
    return np.exp(-sq_dists / (2 * sigma ** 2))


class BranchAblation:
    """
    Creates curriculum with specific branch ablations.

    Ablation types:
    - dr_only: Only DR branch (no replay, no mutation)
    - no_mutation: Only DR + Replay (no mutation)
    - all_replay: Only Replay branch
    - inverted: Swap Replay <-> Mutate schedule
    """

    def __init__(self, ablation_type: str):
        self.ablation_type = ablation_type
        self._validate_type()

    def _validate_type(self):
        valid_types = ["dr_only", "no_mutation", "all_replay", "inverted"]
        if self.ablation_type not in valid_types:
            raise ValueError(f"Unknown ablation type: {self.ablation_type}. "
                           f"Valid types: {valid_types}")

    def should_replay(self, original_replay_decision: bool) -> bool:
        """Transform replay decision based on ablation."""
        if self.ablation_type == "dr_only":
            return False
        elif self.ablation_type == "no_mutation":
            return original_replay_decision
        elif self.ablation_type == "all_replay":
            return True
        elif self.ablation_type == "inverted":
            return original_replay_decision
        return original_replay_decision

    def should_mutate(self, original_mutate_decision: bool) -> bool:
        """Transform mutate decision based on ablation."""
        if self.ablation_type == "dr_only":
            return False
        elif self.ablation_type == "no_mutation":
            return False
        elif self.ablation_type == "all_replay":
            return False
        elif self.ablation_type == "inverted":
            # Swap: mutate when would replay, vice versa
            return not original_mutate_decision
        return original_mutate_decision

    def get_description(self) -> str:
        """Get human-readable description of ablation."""
        descriptions = {
            "dr_only": "Domain Randomization only (no replay or mutation)",
            "no_mutation": "No mutation branch (DR + Replay only)",
            "all_replay": "Replay only (no DR or mutation)",
            "inverted": "Inverted schedule (swap Replay <-> Mutate timing)",
        }
        return descriptions[self.ablation_type]


def create_branch_ablation(ablation_type: str) -> BranchAblation:
    """Create a branch ablation configuration."""
    return BranchAblation(ablation_type)


class DifficultyManipulation:
    """
    Creates curriculum with difficulty manipulations.

    Manipulation types:
    - only_easy: Only top 20% highest-return levels
    - only_hard: Only bottom 20% return levels
    - uniform: Remove PLR/ACCEL prioritization
    - reversed: Present hard early, easy late
    """

    def __init__(self, manipulation_type: str, percentile: float = 20.0):
        self.manipulation_type = manipulation_type
        self.percentile = percentile
        self._validate_type()

    def _validate_type(self):
        valid_types = ["only_easy", "only_hard", "uniform", "reversed"]
        if self.manipulation_type not in valid_types:
            raise ValueError(f"Unknown manipulation type: {self.manipulation_type}")

    def filter_levels(
        self,
        levels,
        scores: np.ndarray,
    ) -> Tuple[any, np.ndarray]:
        """
        Filter levels based on difficulty manipulation.

        Args:
            levels: Level batch
            scores: Difficulty scores for each level

        Returns:
            (filtered_levels, filtered_scores)
        """
        scores = np.asarray(scores)

        if self.manipulation_type == "only_easy":
            threshold = np.percentile(scores, 100 - self.percentile)
            mask = scores >= threshold
        elif self.manipulation_type == "only_hard":
            threshold = np.percentile(scores, self.percentile)
            mask = scores <= threshold
        elif self.manipulation_type == "uniform":
            # No filtering, but return shuffled
            mask = np.ones(len(scores), dtype=bool)
        elif self.manipulation_type == "reversed":
            # Inverse scores for reversed difficulty
            mask = np.ones(len(scores), dtype=bool)
        else:
            mask = np.ones(len(scores), dtype=bool)

        indices = np.where(mask)[0]
        if len(indices) == 0:
            indices = np.arange(len(scores))

        # Filter levels (assumes levels is pytree-like)
        filtered_levels = jax.tree_util.tree_map(
            lambda x: x[indices] if hasattr(x, '__getitem__') else x,
            levels
        )

        return filtered_levels, scores[indices]

    def modify_scores(self, scores: np.ndarray, training_step: int) -> np.ndarray:
        """
        Modify scores for level sampling based on manipulation.

        Args:
            scores: Original level scores
            training_step: Current training step (for reversed manipulation)

        Returns:
            Modified scores
        """
        scores = np.asarray(scores)

        if self.manipulation_type == "uniform":
            # Uniform scores = uniform sampling
            return np.ones_like(scores)
        elif self.manipulation_type == "reversed":
            # Invert scores so hard levels are sampled early
            return -scores
        else:
            return scores

    def get_description(self) -> str:
        descriptions = {
            "only_easy": f"Only top {self.percentile}% easiest levels",
            "only_hard": f"Only bottom {self.percentile}% hardest levels",
            "uniform": "Uniform sampling (no curriculum)",
            "reversed": "Reversed difficulty (hard early, easy late)",
        }
        return descriptions[self.manipulation_type]


def create_difficulty_manipulation(
    manipulation_type: str,
    percentile: float = 20.0,
) -> DifficultyManipulation:
    """Create a difficulty manipulation configuration."""
    return DifficultyManipulation(manipulation_type, percentile)


def measure_distribution_shift(
    baseline_hstates: np.ndarray,
    shifted_hstates: np.ndarray,
) -> Dict[str, float]:
    """
    Measure distribution shift in hidden state space.

    Args:
        baseline_hstates: Hidden states under normal curriculum
        shifted_hstates: Hidden states under intervention

    Returns:
        Dict with shift metrics
    """
    baseline = np.asarray(baseline_hstates)
    shifted = np.asarray(shifted_hstates)

    # MMD
    mmd = compute_mmd(baseline, shifted)

    # Mean shift
    mean_baseline = baseline.mean(axis=0)
    mean_shifted = shifted.mean(axis=0)
    mean_shift = np.linalg.norm(mean_shifted - mean_baseline)

    # Variance ratio
    var_baseline = baseline.var()
    var_shifted = shifted.var()
    var_ratio = var_shifted / (var_baseline + 1e-10)

    # Cosine similarity of means
    norm_baseline = np.linalg.norm(mean_baseline)
    norm_shifted = np.linalg.norm(mean_shifted)
    cosine_sim = np.dot(mean_baseline, mean_shifted) / (norm_baseline * norm_shifted + 1e-10)

    return {
        "mmd": float(mmd),
        "mean_shift_l2": float(mean_shift),
        "variance_ratio": float(var_ratio),
        "mean_cosine_similarity": float(cosine_sim),
    }


def compute_intervention_effect(
    baseline_performance: np.ndarray,
    intervention_performance: np.ndarray,
) -> Dict[str, float]:
    """
    Compute effect of curriculum intervention on performance.

    Args:
        baseline_performance: Performance metrics under normal curriculum
        intervention_performance: Performance under intervention

    Returns:
        Dict with effect metrics
    """
    baseline = np.asarray(baseline_performance)
    intervention = np.asarray(intervention_performance)

    # Immediate drop (first few episodes)
    n_immediate = min(10, len(intervention))
    immediate_drop = baseline.mean() - intervention[:n_immediate].mean()

    # Asymptotic performance
    n_asymptotic = min(50, len(intervention))
    asymptotic_perf = intervention[-n_asymptotic:].mean()
    asymptotic_gap = baseline.mean() - asymptotic_perf

    # Adaptation rate: how fast does performance recover?
    if len(intervention) > 20:
        early = intervention[:10].mean()
        late = intervention[-10:].mean()
        recovery = late - early
    else:
        recovery = 0.0

    return {
        "immediate_drop": float(immediate_drop),
        "asymptotic_gap": float(asymptotic_gap),
        "recovery": float(recovery),
        "baseline_mean": float(baseline.mean()),
        "intervention_mean": float(intervention.mean()),
    }
