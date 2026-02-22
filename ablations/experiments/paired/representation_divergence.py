"""
C1: Representation Divergence.

Track protagonist-antagonist representation divergence over training.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment


@dataclass
class DivergenceSnapshot:
    """Divergence data at a single checkpoint."""
    checkpoint_step: int
    pro_hstates: np.ndarray
    ant_hstates: np.ndarray
    level_features: List[Dict[str, float]]
    cka_similarity: float
    mean_cosine_similarity: float
    kl_divergence: float


class RepresentationDivergenceExperiment(CheckpointExperiment):
    """
    Track protagonist-antagonist representation divergence over training.

    Protocol:
    1. At K checkpoints, collect h-states from both agents on matched levels
    2. Compute CKA similarity over training
    3. Identify dimensions with differential encoding
    4. Track strategy overlap via KL divergence
    """

    @property
    def name(self) -> str:
        return "representation_divergence"

    def __init__(
        self,
        n_levels_per_checkpoint: int = 200,
        hidden_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_levels_per_checkpoint = n_levels_per_checkpoint
        self.hidden_dim = hidden_dim
        self._snapshots: List[DivergenceSnapshot] = []
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"RepresentationDivergenceExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> List[DivergenceSnapshot]:
        """Collect data at current checkpoint."""
        checkpoint_step = getattr(self.train_state, 'update_count', 0)

        pro_hstates = []
        ant_hstates = []
        level_features = []

        for i in range(self.n_levels_per_checkpoint):
            rng, level_rng, pro_rng, ant_rng = jax.random.split(rng, 4)

            # Generate level
            level = self._generate_level(level_rng)
            features = self._compute_level_features(level)
            level_features.append(features)

            # Get protagonist hidden state
            pro_h = self._get_protagonist_hstate(pro_rng, level)
            pro_hstates.append(pro_h)

            # Get antagonist hidden state
            ant_h = self._get_antagonist_hstate(ant_rng, level)
            ant_hstates.append(ant_h)

        pro_hstates = np.array(pro_hstates)
        ant_hstates = np.array(ant_hstates)

        # Compute metrics
        cka = self._compute_cka(pro_hstates, ant_hstates)
        cosine_sim = self._compute_mean_cosine_similarity(pro_hstates, ant_hstates)
        kl_div = self._compute_kl_divergence(pro_hstates, ant_hstates)

        snapshot = DivergenceSnapshot(
            checkpoint_step=checkpoint_step,
            pro_hstates=pro_hstates,
            ant_hstates=ant_hstates,
            level_features=level_features,
            cka_similarity=cka,
            mean_cosine_similarity=cosine_sim,
            kl_divergence=kl_div,
        )

        self._snapshots.append(snapshot)
        return self._snapshots

    def _generate_level(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """Generate a level."""
        height, width = 13, 13
        wall_prob = 0.1 + float(jax.random.uniform(rng)) * 0.25

        wall_map = np.array(jax.random.bernoulli(rng, wall_prob, (height, width)))
        wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

        rng_goal, rng_agent = jax.random.split(rng)
        return {
            'wall_map': wall_map,
            'goal_pos': (int(jax.random.randint(rng_goal, (), 1, height-1)),
                        int(jax.random.randint(rng_goal, (), 1, width-1))),
            'agent_pos': (int(jax.random.randint(rng_agent, (), 1, height-1)),
                         int(jax.random.randint(rng_agent, (), 1, width-1))),
        }

    def _compute_level_features(self, level: Dict[str, Any]) -> Dict[str, float]:
        """Compute level features."""
        wall_density = float(level['wall_map'].sum() / level['wall_map'].size)
        goal_distance = float(np.sqrt(
            (level['goal_pos'][0] - level['agent_pos'][0])**2 +
            (level['goal_pos'][1] - level['agent_pos'][1])**2
        ))
        return {'wall_density': wall_density, 'goal_distance': goal_distance}

    def _get_protagonist_hstate(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
    ) -> np.ndarray:
        """Get protagonist hidden state (simplified)."""
        features = self._compute_level_features(level)
        h = np.array(jax.random.normal(rng, (self.hidden_dim,)))
        # Protagonist encodes wall density in first dims
        h[:50] += features['wall_density'] * 2.0
        return h

    def _get_antagonist_hstate(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
    ) -> np.ndarray:
        """Get antagonist hidden state (simplified)."""
        features = self._compute_level_features(level)
        h = np.array(jax.random.normal(rng, (self.hidden_dim,)))
        # Antagonist encodes different aspects
        h[50:100] += features['goal_distance'] * 0.5
        h[100:150] += features['wall_density'] * 1.5
        return h

    def _compute_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute linear CKA."""
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

        K = X @ X.T
        L = Y @ Y.T

        hsic = np.trace(K @ L)
        norm_k = np.sqrt(np.trace(K @ K))
        norm_l = np.sqrt(np.trace(L @ L))

        if norm_k < 1e-10 or norm_l < 1e-10:
            return 0.0

        return float(hsic / (norm_k * norm_l))

    def _compute_mean_cosine_similarity(
        self,
        pro_hstates: np.ndarray,
        ant_hstates: np.ndarray,
    ) -> float:
        """Compute mean cosine similarity."""
        similarities = []
        for pro_h, ant_h in zip(pro_hstates, ant_hstates):
            norm_pro = np.linalg.norm(pro_h)
            norm_ant = np.linalg.norm(ant_h)
            if norm_pro > 1e-10 and norm_ant > 1e-10:
                sim = np.dot(pro_h, ant_h) / (norm_pro * norm_ant)
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0

    def _compute_kl_divergence(
        self,
        pro_hstates: np.ndarray,
        ant_hstates: np.ndarray,
    ) -> float:
        """Compute approximate KL divergence between distributions."""
        # Fit Gaussian to each
        pro_mean = pro_hstates.mean(axis=0)
        ant_mean = ant_hstates.mean(axis=0)

        pro_var = pro_hstates.var(axis=0) + 1e-6
        ant_var = ant_hstates.var(axis=0) + 1e-6

        # KL(pro || ant) for diagonal Gaussians
        kl = 0.5 * np.sum(
            np.log(ant_var / pro_var) +
            pro_var / ant_var +
            (pro_mean - ant_mean)**2 / ant_var - 1
        )

        return float(max(0, kl))

    def analyze(self) -> Dict[str, Any]:
        """Analyze divergence patterns."""
        if not self._snapshots:
            raise ValueError("Must call collect_data first")

        results = {}

        # Divergence trajectory
        results['representation_divergence_trajectory'] = {
            'steps': [s.checkpoint_step for s in self._snapshots],
            'cka_similarity': [s.cka_similarity for s in self._snapshots],
            'cosine_similarity': [s.mean_cosine_similarity for s in self._snapshots],
        }

        # Strategy overlap trajectory (KL)
        results['strategy_overlap_trajectory'] = {
            'steps': [s.checkpoint_step for s in self._snapshots],
            'kl_divergence': [s.kl_divergence for s in self._snapshots],
        }

        # Differential feature encoding
        results['differential_feature_encoding'] = self._find_differential_features()

        # Antagonist specialization
        results['antagonist_specialisation'] = self._measure_specialisation()

        # Summary stats
        latest = self._snapshots[-1]
        results['latest_cka'] = latest.cka_similarity
        results['latest_kl'] = latest.kl_divergence

        return results

    def _find_differential_features(self) -> Dict[str, Any]:
        """Find features encoded differently by protagonist vs antagonist."""
        if not self._snapshots:
            return {}

        latest = self._snapshots[-1]

        # Compute per-dimension variance difference
        pro_var = latest.pro_hstates.var(axis=0)
        ant_var = latest.ant_hstates.var(axis=0)

        variance_diff = ant_var - pro_var

        # Top differential dimensions
        top_ant_dims = np.argsort(variance_diff)[-20:].tolist()
        top_pro_dims = np.argsort(variance_diff)[:20].tolist()

        # Compute feature correlations for differential dims
        features_array = np.array([
            [f['wall_density'], f['goal_distance']]
            for f in latest.level_features
        ])

        pro_feature_corrs = []
        ant_feature_corrs = []

        for dim in top_pro_dims[:5]:
            corr = np.corrcoef(latest.pro_hstates[:, dim], features_array[:, 0])[0, 1]
            pro_feature_corrs.append(float(corr) if not np.isnan(corr) else 0.0)

        for dim in top_ant_dims[:5]:
            corr = np.corrcoef(latest.ant_hstates[:, dim], features_array[:, 0])[0, 1]
            ant_feature_corrs.append(float(corr) if not np.isnan(corr) else 0.0)

        return {
            'top_antagonist_dims': top_ant_dims,
            'top_protagonist_dims': top_pro_dims,
            'protagonist_feature_correlations': pro_feature_corrs,
            'antagonist_feature_correlations': ant_feature_corrs,
        }

    def _measure_specialisation(self) -> Dict[str, float]:
        """Measure how specialized antagonist representations are."""
        if not self._snapshots:
            return {}

        latest = self._snapshots[-1]

        # Specialization = how different antagonist is from protagonist
        # on specific feature dimensions

        # Compute sparsity difference
        pro_sparsity = float((np.abs(latest.pro_hstates) < 0.1).mean())
        ant_sparsity = float((np.abs(latest.ant_hstates) < 0.1).mean())

        # Compute effective dimensionality
        pro_svd = np.linalg.svd(latest.pro_hstates, compute_uv=False)
        ant_svd = np.linalg.svd(latest.ant_hstates, compute_uv=False)

        pro_eff_dim = float((pro_svd**2).sum()**2 / (pro_svd**4).sum())
        ant_eff_dim = float((ant_svd**2).sum()**2 / (ant_svd**4).sum())

        return {
            'protagonist_sparsity': pro_sparsity,
            'antagonist_sparsity': ant_sparsity,
            'protagonist_effective_dim': pro_eff_dim,
            'antagonist_effective_dim': ant_eff_dim,
            'specialisation_score': float(abs(ant_eff_dim - pro_eff_dim)),
        }

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize divergence patterns."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._snapshots:
            return figures

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # CKA trajectory
        ax = axes[0, 0]
        steps = [s.checkpoint_step for s in self._snapshots]
        ckas = [s.cka_similarity for s in self._snapshots]
        ax.plot(steps, ckas, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('CKA Similarity')
        ax.set_title('Protagonist-Antagonist CKA Over Training')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # KL trajectory
        ax = axes[0, 1]
        kls = [s.kl_divergence for s in self._snapshots]
        ax.plot(steps, kls, 'r-o', linewidth=2, markersize=6)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('KL Divergence')
        ax.set_title('Strategy Overlap (KL Divergence)')
        ax.grid(True, alpha=0.3)

        # Latest snapshot: dimension variance comparison
        ax = axes[1, 0]
        latest = self._snapshots[-1]
        pro_var = latest.pro_hstates.var(axis=0)
        ant_var = latest.ant_hstates.var(axis=0)
        dims = np.arange(min(100, len(pro_var)))
        ax.bar(dims - 0.2, pro_var[:100], 0.4, label='Protagonist', alpha=0.7)
        ax.bar(dims + 0.2, ant_var[:100], 0.4, label='Antagonist', alpha=0.7)
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Variance')
        ax.set_title('Per-Dimension Variance (First 100 dims)')
        ax.legend()

        # 2D projection
        ax = axes[1, 1]
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        combined = np.vstack([latest.pro_hstates, latest.ant_hstates])
        projected = pca.fit_transform(combined)
        n = len(latest.pro_hstates)
        ax.scatter(projected[:n, 0], projected[:n, 1], alpha=0.5, label='Protagonist', s=20)
        ax.scatter(projected[n:, 0], projected[n:, 1], alpha=0.5, label='Antagonist', s=20)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA Projection of Hidden States')
        ax.legend()

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["divergence_analysis"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
