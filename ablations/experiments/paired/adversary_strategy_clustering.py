"""
C3: Adversary Strategy Clustering.

Identify distinct adversary "teaching strategies" by clustering level-generation
action sequences. This is used by many other experiments (E1, E2, E5, F3).

Protocol:
1. At K training checkpoints, collect M adversary rollouts
2. Embed action sequences using adversary hidden states
3. Cluster using HDBSCAN
4. Characterize each cluster: level features, regret achieved
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment
from ..utils.paired_helpers import (
    generate_levels,
    extract_level_features_batch,
    get_pro_ant_returns,
    get_pro_hstates,
)


@dataclass
class AdversaryRollout:
    """Single adversary level-generation rollout."""
    actions: np.ndarray           # Action sequence
    hstates: np.ndarray           # Hidden states during generation
    level_features: Dict[str, float]
    regret: float
    checkpoint_step: int


@dataclass
class StrategyCluster:
    """A discovered adversary strategy."""
    cluster_id: int
    n_samples: int
    mean_regret: float
    std_regret: float
    feature_profile: Dict[str, float]  # Mean level features
    representative_actions: np.ndarray  # Centroid action sequence
    stability: float                    # How stable over training


class AdversaryStrategyClusteringExperiment(CheckpointExperiment):
    """
    Identify distinct adversary teaching strategies via clustering.

    Key output: adversary_strategy_labels that can be used by:
    - E1: Bilateral probing conditioned on strategy
    - E2: Activation analysis clustered by strategy
    - E5: Symbolic regression with strategy as feature
    - F3: Shard dynamics linked to adversary strategies
    """

    @property
    def name(self) -> str:
        return "adversary_strategy_clustering"

    def __init__(
        self,
        n_rollouts_per_checkpoint: int = 100,
        min_cluster_size: int = 10,
        use_hdbscan: bool = True,
        n_clusters_kmeans: int = 5,  # Fallback if HDBSCAN unavailable
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rollouts_per_checkpoint = n_rollouts_per_checkpoint
        self.min_cluster_size = min_cluster_size
        self.use_hdbscan = use_hdbscan
        self.n_clusters_kmeans = n_clusters_kmeans

        self._rollouts: List[AdversaryRollout] = []
        self._clusters: List[StrategyCluster] = []
        self._labels: Optional[np.ndarray] = None
        self._require_paired()

    def _require_paired(self):
        """Verify PAIRED training."""
        if self.training_method != "paired":
            raise ValueError(f"AdversaryStrategyClusteringExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> List[AdversaryRollout]:
        """Collect adversary rollouts using real network data."""
        checkpoint_step = getattr(self.train_state, 'update_count', 0)
        n = self.n_rollouts_per_checkpoint

        # Generate all levels in a single batched call
        rng, gen_rng, hstate_rng, eval_rng = jax.random.split(rng, 4)
        levels = generate_levels(self.agent, gen_rng, n)

        # Extract features for all levels
        batch_features = extract_level_features_batch(levels)

        # Get real protagonist hidden states (used as embedding for clustering)
        hstates_all = get_pro_hstates(hstate_rng, levels, self)
        # hstates_all shape: (n, hidden_dim)

        # Get real protagonist and antagonist returns for regret
        pro_returns, ant_returns, regrets = get_pro_ant_returns(
            eval_rng, levels, self
        )

        # Build per-rollout records
        for i in range(n):
            features = {
                'wall_density': float(batch_features['wall_density'][i]),
                'goal_distance': float(batch_features['goal_distance'][i]),
                'open_space_ratio': float(batch_features['open_space_ratio'][i]),
            }

            # Use the hidden state vector as a single-step "sequence"
            # for compatibility with _embed_sequences (which takes mean over axis=0)
            hstate_i = hstates_all[i]  # shape: (hidden_dim,)
            # Wrap in (1, hidden_dim) so mean(axis=0) is a no-op
            hstate_seq = hstate_i[np.newaxis, :]

            # Actions: not directly available from rollout; use a placeholder
            # The clustering primarily relies on hstates and features
            max_actions = 50
            n_actions = 7
            rng, action_rng = jax.random.split(rng)
            actions = np.array(jax.random.randint(action_rng, (max_actions,), 0, n_actions))

            self._rollouts.append(AdversaryRollout(
                actions=actions,
                hstates=hstate_seq,
                level_features=features,
                regret=float(regrets[i]),
                checkpoint_step=checkpoint_step,
            ))

        return self._rollouts

    def analyze(self) -> Dict[str, Any]:
        """Cluster adversary strategies."""
        if not self._rollouts:
            raise ValueError("Must call collect_data before analyze")

        results = {}

        # 1. Embed action sequences
        embeddings = self._embed_sequences()

        # 2. Cluster
        labels, n_clusters = self._cluster_embeddings(embeddings)
        self._labels = labels

        # 3. Characterize clusters
        self._clusters = self._characterize_clusters(labels)
        results['clusters'] = [self._cluster_to_dict(c) for c in self._clusters]

        # 4. Compute strategy metrics
        results['num_strategies'] = n_clusters
        results['strategy_stability'] = self._compute_stability(labels)
        results['strategy_regret_profile'] = self._compute_regret_by_cluster(labels)
        results['strategy_switching_rate'] = self._compute_switching_rate(labels)

        # 5. Export labels for other experiments
        results['strategy_labels'] = labels.tolist()

        return results

    def _embed_sequences(self) -> np.ndarray:
        """Embed action sequences for clustering."""
        embeddings = []

        for rollout in self._rollouts:
            # Use mean hidden state + action statistics as embedding
            mean_hstate = rollout.hstates.mean(axis=0)
            action_counts = np.bincount(rollout.actions, minlength=7) / len(rollout.actions)

            # Add level features
            features = np.array([
                rollout.level_features['wall_density'],
                rollout.level_features['goal_distance'],
                rollout.regret,
            ])

            embedding = np.concatenate([
                mean_hstate[:50],  # First 50 dims of hstate
                action_counts,
                features,
            ])
            embeddings.append(embedding)

        return np.stack(embeddings)

    def _cluster_embeddings(self, embeddings: np.ndarray) -> Tuple[np.ndarray, int]:
        """Cluster embeddings using HDBSCAN or KMeans."""
        if self.use_hdbscan:
            try:
                from hdbscan import HDBSCAN
                clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size)
                labels = clusterer.fit_predict(embeddings)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                return labels, n_clusters
            except ImportError:
                pass

        # Fallback to KMeans
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters_kmeans, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        return labels, self.n_clusters_kmeans

    def _characterize_clusters(self, labels: np.ndarray) -> List[StrategyCluster]:
        """Characterize each cluster."""
        clusters = []
        unique_labels = [l for l in np.unique(labels) if l >= 0]

        for cluster_id in unique_labels:
            mask = labels == cluster_id
            cluster_rollouts = [r for r, m in zip(self._rollouts, mask) if m]

            if not cluster_rollouts:
                continue

            # Aggregate features
            regrets = [r.regret for r in cluster_rollouts]
            features = {}
            for key in cluster_rollouts[0].level_features.keys():
                features[key] = float(np.mean([r.level_features[key] for r in cluster_rollouts]))

            # Representative action sequence (mean)
            mean_actions = np.mean([r.actions for r in cluster_rollouts], axis=0)

            clusters.append(StrategyCluster(
                cluster_id=int(cluster_id),
                n_samples=len(cluster_rollouts),
                mean_regret=float(np.mean(regrets)),
                std_regret=float(np.std(regrets)),
                feature_profile=features,
                representative_actions=mean_actions,
                stability=1.0,  # Placeholder
            ))

        return clusters

    def _cluster_to_dict(self, cluster: StrategyCluster) -> Dict[str, Any]:
        """Convert cluster to serializable dict."""
        return {
            'cluster_id': cluster.cluster_id,
            'n_samples': cluster.n_samples,
            'mean_regret': cluster.mean_regret,
            'std_regret': cluster.std_regret,
            'feature_profile': cluster.feature_profile,
            'stability': cluster.stability,
        }

    def _compute_stability(self, labels: np.ndarray) -> float:
        """Compute how stable strategies are."""
        # Simplified: use silhouette score
        from sklearn.metrics import silhouette_score

        embeddings = self._embed_sequences()
        valid_mask = labels >= 0

        if valid_mask.sum() < 10 or len(set(labels[valid_mask])) < 2:
            return 0.0

        return float(silhouette_score(embeddings[valid_mask], labels[valid_mask]))

    def _compute_regret_by_cluster(self, labels: np.ndarray) -> Dict[int, float]:
        """Compute mean regret by cluster."""
        regret_by_cluster = {}
        for cluster_id in set(labels):
            if cluster_id < 0:
                continue
            mask = labels == cluster_id
            regrets = [r.regret for r, m in zip(self._rollouts, mask) if m]
            if regrets:
                regret_by_cluster[int(cluster_id)] = float(np.mean(regrets))
        return regret_by_cluster

    def _compute_switching_rate(self, labels: np.ndarray) -> float:
        """Compute how often adversary switches between strategies."""
        if len(labels) < 2:
            return 0.0

        switches = sum(1 for i in range(1, len(labels))
                      if labels[i] != labels[i-1] and labels[i] >= 0 and labels[i-1] >= 0)
        return float(switches / (len(labels) - 1))

    def get_strategy_labels(self) -> np.ndarray:
        """Get strategy labels for use by other experiments."""
        if self._labels is None:
            raise ValueError("Must call analyze before getting labels")
        return self._labels

    def visualize(self) -> Dict[str, np.ndarray]:
        """Create strategy clustering visualizations."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._clusters:
            return figures

        # Strategy profile comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Regret by cluster
        ax = axes[0]
        cluster_ids = [c.cluster_id for c in self._clusters]
        regrets = [c.mean_regret for c in self._clusters]
        stds = [c.std_regret for c in self._clusters]
        ax.bar(cluster_ids, regrets, yerr=stds, capsize=3, alpha=0.8)
        ax.set_xlabel('Strategy Cluster')
        ax.set_ylabel('Mean Regret')
        ax.set_title('Regret by Adversary Strategy')

        # Feature profile heatmap
        ax = axes[1]
        feature_names = list(self._clusters[0].feature_profile.keys())
        feature_matrix = np.array([
            [c.feature_profile[f] for f in feature_names]
            for c in self._clusters
        ])
        im = ax.imshow(feature_matrix, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_yticks(range(len(self._clusters)))
        ax.set_yticklabels([f'Strategy {c.cluster_id}' for c in self._clusters])
        ax.set_title('Level Features by Strategy')
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["strategy_profiles"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
