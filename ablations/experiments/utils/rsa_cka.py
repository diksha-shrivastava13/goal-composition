"""
Representation Similarity Analysis (RSA) and Centered Kernel Alignment (CKA).

These methods compare representation geometry between:
- Training phases (how does geometry evolve?)
- Agents (which develop similar structures?)
- Layers (CNN vs LSTM vs output)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform


def compute_rdm(
    representations: np.ndarray,
    metric: str = "correlation",
) -> np.ndarray:
    """
    Compute Representational Dissimilarity Matrix (RDM).

    Args:
        representations: Shape (n_samples, n_features)
        metric: Distance metric ("correlation", "euclidean", "cosine")

    Returns:
        RDM: Shape (n_samples, n_samples) symmetric matrix
    """
    representations = np.asarray(representations)

    if metric == "correlation":
        # 1 - Pearson correlation
        distances = pdist(representations, metric="correlation")
    elif metric == "cosine":
        # Cosine distance
        distances = pdist(representations, metric="cosine")
    elif metric == "euclidean":
        # Euclidean distance
        distances = pdist(representations, metric="euclidean")
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return squareform(distances)


def compute_rsa(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    method: str = "spearman",
) -> Tuple[float, float]:
    """
    Compute Representation Similarity Analysis between two RDMs.

    Measures how similar the geometry of two representation spaces are.

    Args:
        rdm1: First RDM, shape (n, n)
        rdm2: Second RDM, shape (n, n)
        method: "spearman" (rank correlation) or "pearson"

    Returns:
        (correlation, p_value)
    """
    # Extract upper triangular (excluding diagonal)
    n = rdm1.shape[0]
    idx = np.triu_indices(n, k=1)
    vec1 = rdm1[idx]
    vec2 = rdm2[idx]

    if method == "spearman":
        corr, p = spearmanr(vec1, vec2)
    else:
        corr, p = pearsonr(vec1, vec2)

    return float(corr), float(p)


def compute_cka(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "linear",
) -> float:
    """
    Compute Centered Kernel Alignment (CKA).

    CKA measures similarity between representations in a way that's
    invariant to orthogonal transformations and isotropic scaling.

    Args:
        X: First representation, shape (n_samples, n_features_x)
        Y: Second representation, shape (n_samples, n_features_y)
        kernel: "linear" or "rbf"

    Returns:
        CKA similarity in [0, 1]
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Compute kernels
    if kernel == "linear":
        K = X @ X.T
        L = Y @ Y.T
    elif kernel == "rbf":
        # RBF kernel with median heuristic
        K = _rbf_kernel(X)
        L = _rbf_kernel(Y)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Center kernels
    K_centered = _center_kernel(K)
    L_centered = _center_kernel(L)

    # Compute HSIC
    hsic_kl = np.sum(K_centered * L_centered)
    hsic_kk = np.sum(K_centered * K_centered)
    hsic_ll = np.sum(L_centered * L_centered)

    # CKA
    cka = hsic_kl / (np.sqrt(hsic_kk * hsic_ll) + 1e-10)

    return float(cka)


def compute_layer_wise_cka(
    layer_representations: Dict[str, np.ndarray],
    reference_representations: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Compute CKA for each layer between two networks/checkpoints.

    Args:
        layer_representations: Dict mapping layer name to representations
        reference_representations: Reference dict with same keys

    Returns:
        Dict mapping layer name to CKA score
    """
    results = {}
    common_layers = set(layer_representations.keys()) & set(reference_representations.keys())

    for layer in common_layers:
        X = layer_representations[layer]
        Y = reference_representations[layer]

        # Ensure same number of samples
        n = min(X.shape[0], Y.shape[0])
        cka = compute_cka(X[:n], Y[:n])
        results[layer] = cka

    return results


def compute_rsa_over_training(
    representations_by_step: Dict[int, np.ndarray],
    reference_step: Optional[int] = None,
) -> Dict[str, any]:
    """
    Track how representation geometry changes over training.

    Args:
        representations_by_step: Dict mapping step to representations
        reference_step: Step to use as reference (default: final step)

    Returns:
        Dict with RSA scores over time
    """
    steps = sorted(representations_by_step.keys())
    if len(steps) < 2:
        return {"error": "Need at least 2 checkpoints"}

    if reference_step is None:
        reference_step = steps[-1]

    ref_rdm = compute_rdm(representations_by_step[reference_step])

    rsa_scores = []
    for step in steps:
        rdm = compute_rdm(representations_by_step[step])
        corr, _ = compute_rsa(rdm, ref_rdm)
        rsa_scores.append({"step": step, "rsa": corr})

    # Also compute consecutive RSA (how much geometry changes step-to-step)
    consecutive_rsa = []
    for i in range(1, len(steps)):
        rdm1 = compute_rdm(representations_by_step[steps[i-1]])
        rdm2 = compute_rdm(representations_by_step[steps[i]])
        corr, _ = compute_rsa(rdm1, rdm2)
        consecutive_rsa.append({
            "step_from": steps[i-1],
            "step_to": steps[i],
            "rsa": corr,
        })

    return {
        "rsa_vs_reference": rsa_scores,
        "consecutive_rsa": consecutive_rsa,
        "reference_step": reference_step,
    }


def compute_cross_agent_cka(
    agent_representations: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Compute pairwise CKA between different agents.

    Args:
        agent_representations: Dict mapping agent name to representations
            Representations must have same number of samples (matched levels)

    Returns:
        Dict with CKA matrix as nested dict
    """
    agents = list(agent_representations.keys())
    results = {}

    for i, agent1 in enumerate(agents):
        results[agent1] = {}
        for j, agent2 in enumerate(agents):
            X = agent_representations[agent1]
            Y = agent_representations[agent2]
            n = min(X.shape[0], Y.shape[0])
            cka = compute_cka(X[:n], Y[:n])
            results[agent1][agent2] = cka

    return results


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _rbf_kernel(X: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """Compute RBF kernel with median heuristic for sigma."""
    sq_dists = pdist(X, metric="sqeuclidean")
    if sigma is None:
        sigma = np.median(sq_dists) ** 0.5
        if sigma == 0:
            sigma = 1.0
    K = np.exp(-squareform(sq_dists) / (2 * sigma ** 2))
    return K


def compute_clustering_quality(
    representations: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute clustering quality metrics for representations.

    Args:
        representations: Shape (n_samples, n_features)
        labels: Cluster/group labels, shape (n_samples,)

    Returns:
        Dict with silhouette score and other metrics
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    representations = np.asarray(representations)
    labels = np.asarray(labels)

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {"error": "Need at least 2 clusters"}

    sil = silhouette_score(representations, labels)
    ch = calinski_harabasz_score(representations, labels)

    # Inter-cluster distances
    centroids = []
    for label in unique_labels:
        mask = labels == label
        centroids.append(representations[mask].mean(axis=0))
    centroids = np.array(centroids)
    inter_cluster_dist = pdist(centroids).mean() if len(centroids) > 1 else 0.0

    return {
        "silhouette_score": float(sil),
        "calinski_harabasz": float(ch),
        "inter_cluster_distance": float(inter_cluster_dist),
        "n_clusters": len(unique_labels),
    }
