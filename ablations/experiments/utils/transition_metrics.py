"""
Transition metrics for training dynamics characterization.

Contains functions for estimating Fisher information, effective dimensionality,
and synergy measures. These are EXPLORATORY diagnostics, not phase detection.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import linalg


def estimate_fisher_information(
    gradients: np.ndarray,
) -> Dict[str, float]:
    """
    Estimate Fisher information from parameter gradients.

    Fisher information indicates how sensitive the loss is to parameter changes.
    High Fisher = parameters are important; Low Fisher = parameters don't matter.

    Note: This is an approximation. True Fisher requires expected value over data.

    Args:
        gradients: Gradient samples, shape (n_samples, n_params)

    Returns:
        Dict with Fisher information estimates
    """
    gradients = np.asarray(gradients)
    n_samples, n_params = gradients.shape

    # Empirical Fisher: E[g * g^T] - uses diagonal for efficiency
    fisher_diagonal = np.mean(gradients ** 2, axis=0)

    # Summary statistics
    mean_fisher = float(np.mean(fisher_diagonal))
    max_fisher = float(np.max(fisher_diagonal))
    min_fisher = float(np.min(fisher_diagonal))

    # Effective rank: how many parameters are "active"
    # Based on eigenvalue distribution
    normalized = fisher_diagonal / (fisher_diagonal.sum() + 1e-10)
    entropy = -np.sum(normalized * np.log(normalized + 1e-10))
    effective_rank = np.exp(entropy)

    return {
        "mean_fisher": mean_fisher,
        "max_fisher": max_fisher,
        "min_fisher": min_fisher,
        "fisher_std": float(np.std(fisher_diagonal)),
        "effective_rank": float(effective_rank),
        "n_params": n_params,
    }


def compute_effective_dimensionality(
    representations: np.ndarray,
    method: str = "participation_ratio",
) -> Dict[str, float]:
    """
    Compute effective dimensionality of representations.

    Measures how many dimensions are actually used by the representations.

    Args:
        representations: Shape (n_samples, n_features)
        method: "participation_ratio" or "intrinsic_dimension"

    Returns:
        Dict with dimensionality estimates
    """
    representations = np.asarray(representations)
    n_samples, n_features = representations.shape

    # Center data
    centered = representations - representations.mean(axis=0)

    # Compute covariance eigenvalues
    if n_samples < n_features:
        # Use Gram matrix for efficiency
        gram = centered @ centered.T / (n_samples - 1)
        eigenvalues = np.linalg.eigvalsh(gram)
    else:
        cov = centered.T @ centered / (n_samples - 1)
        eigenvalues = np.linalg.eigvalsh(cov)

    # Keep only positive eigenvalues
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    eigenvalues = np.sort(eigenvalues)[::-1]

    if method == "participation_ratio":
        # PR = (sum of eigenvalues)^2 / sum of squared eigenvalues
        pr = (eigenvalues.sum() ** 2) / (np.sum(eigenvalues ** 2) + 1e-10)
        effective_dim = float(pr)
    else:
        # Intrinsic dimension via MLE (simplified)
        # Based on nearest neighbor distances
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(20, n_samples - 1))
        nn.fit(representations)
        distances, _ = nn.kneighbors()
        # MLE estimator
        r1 = distances[:, 0]
        r2 = distances[:, -1]
        ratio = r2 / (r1 + 1e-10)
        effective_dim = float(1 / np.mean(np.log(ratio + 1e-10)))

    # Explained variance by top components
    total_var = eigenvalues.sum()
    cumsum = np.cumsum(eigenvalues)
    n_90 = np.searchsorted(cumsum, 0.9 * total_var) + 1
    n_95 = np.searchsorted(cumsum, 0.95 * total_var) + 1

    return {
        "effective_dimensionality": effective_dim,
        "n_features": n_features,
        "n_for_90_variance": int(n_90),
        "n_for_95_variance": int(n_95),
        "top_eigenvalue_ratio": float(eigenvalues[0] / (eigenvalues.sum() + 1e-10)),
        "method": method,
    }


def compute_synergy_measure(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute synergy/redundancy between variables.

    Synergy: information about Z that is only available when both X and Y are known.
    Redundancy: information about Z that is available in both X and Y separately.

    Args:
        x: First variable, shape (n_samples,)
        y: Second variable, shape (n_samples,)
        z: Target variable, shape (n_samples,)
        n_bins: Number of bins for discretization

    Returns:
        Dict with information-theoretic measures
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Discretize continuous variables
    x_binned = np.digitize(x, np.percentile(x, np.linspace(0, 100, n_bins + 1)[1:-1]))
    y_binned = np.digitize(y, np.percentile(y, np.linspace(0, 100, n_bins + 1)[1:-1]))
    z_binned = np.digitize(z, np.percentile(z, np.linspace(0, 100, n_bins + 1)[1:-1]))

    # Compute entropies
    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log(probs + 1e-10))

    def joint_entropy(*args):
        combined = np.column_stack(args)
        _, counts = np.unique(combined, axis=0, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log(probs + 1e-10))

    H_x = entropy(x_binned)
    H_y = entropy(y_binned)
    H_z = entropy(z_binned)
    H_xy = joint_entropy(x_binned, y_binned)
    H_xz = joint_entropy(x_binned, z_binned)
    H_yz = joint_entropy(y_binned, z_binned)
    H_xyz = joint_entropy(x_binned, y_binned, z_binned)

    # Mutual information
    I_xz = H_x + H_z - H_xz
    I_yz = H_y + H_z - H_yz
    I_xy = H_x + H_y - H_xy

    # Interaction information (can be negative)
    I_xyz = I_xz + I_yz + I_xy - H_x - H_y - H_z + H_xyz

    # Positive interaction = synergy, negative = redundancy
    is_synergistic = I_xyz > 0

    return {
        "interaction_information": float(I_xyz),
        "mi_xz": float(I_xz),
        "mi_yz": float(I_yz),
        "mi_xy": float(I_xy),
        "is_synergistic": bool(is_synergistic),
        "entropy_z": float(H_z),
    }


def track_training_dynamics(
    loss_history: np.ndarray,
    gradient_norms: np.ndarray,
    representation_samples: List[np.ndarray],
    sample_interval: int = 100,
) -> Dict[str, any]:
    """
    Track training dynamics over time for exploratory analysis.

    Args:
        loss_history: Loss values over training, shape (n_steps,)
        gradient_norms: Gradient norms over training, shape (n_steps,)
        representation_samples: List of representation snapshots
        sample_interval: Steps between snapshots

    Returns:
        Dict with dynamics metrics over time
    """
    loss_history = np.asarray(loss_history)
    gradient_norms = np.asarray(gradient_norms)

    # Loss smoothing
    window = 50
    if len(loss_history) > window:
        loss_smooth = np.convolve(loss_history, np.ones(window)/window, mode='valid')
    else:
        loss_smooth = loss_history

    # Gradient stability
    grad_smooth = np.convolve(gradient_norms, np.ones(window)/window, mode='valid') if len(gradient_norms) > window else gradient_norms

    # Compute effective dimensionality over time
    eff_dims = []
    for rep in representation_samples:
        if len(rep) > 10:
            dim_result = compute_effective_dimensionality(rep)
            eff_dims.append(dim_result["effective_dimensionality"])

    # Detect potential transitions (smoothed loss derivative changes sign)
    if len(loss_smooth) > 10:
        loss_deriv = np.gradient(loss_smooth)
        sign_changes = np.where(np.diff(np.sign(loss_deriv)))[0]
    else:
        sign_changes = []

    return {
        "loss_smooth": loss_smooth.tolist(),
        "gradient_norm_smooth": grad_smooth.tolist(),
        "effective_dims_over_time": eff_dims,
        "potential_transition_points": sign_changes.tolist() if len(sign_changes) > 0 else [],
        "caveats": [
            "These are EXPLORATORY metrics, not confirmed phase transitions",
            "Transition points are smoothing artifacts until validated",
            "Effective dimension depends on sample size",
        ],
    }


def compute_loss_curvature(
    loss_values: np.ndarray,
    steps: np.ndarray,
    window: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Compute local curvature of loss landscape.

    High curvature = rapid changes in loss behavior.

    Args:
        loss_values: Loss over training
        steps: Training steps
        window: Window for local fitting

    Returns:
        Dict with curvature estimates
    """
    loss_values = np.asarray(loss_values)
    steps = np.asarray(steps)

    # Compute second derivative (curvature proxy)
    if len(loss_values) < window + 2:
        return {"error": "Insufficient data"}

    curvatures = []
    for i in range(window, len(loss_values) - window):
        local_loss = loss_values[i-window:i+window]
        local_steps = steps[i-window:i+window]

        # Fit quadratic
        coeffs = np.polyfit(local_steps - local_steps.mean(), local_loss, 2)
        curvature = 2 * coeffs[0]  # Second derivative of quadratic
        curvatures.append(curvature)

    curvatures = np.array(curvatures)

    return {
        "curvatures": curvatures,
        "mean_curvature": float(np.mean(curvatures)),
        "max_curvature": float(np.max(np.abs(curvatures))),
        "curvature_variance": float(np.var(curvatures)),
    }
