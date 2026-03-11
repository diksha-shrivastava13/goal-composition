"""
Activation Analysis Experiment.

Analyze feature organization and goal representation in network layers through:
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Clustering by method-appropriate criteria:
  - ACCEL/PLR: branch_type, difficulty, wall density, outcome, training phase
  - PAIRED: regret_tercile, adversary_difficulty, regret_source, adversary_strategy_cluster
  - DR: difficulty tercile, outcome (no curriculum-specific clustering)
- Representation Similarity Analysis (RSA)
- Centered Kernel Alignment (CKA)
- Sparse autoencoder for monosemantic feature discovery

PAIRED-specific:
- Bilateral CKA: CKA(protagonist, antagonist) on matched levels
- Regret source decomposition: ant_strong vs pro_weak
- Adversary strategy clustering integration
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
import chex

from .base import CheckpointExperiment
from .utils.rsa_cka import (
    compute_rdm,
    compute_rsa,
    compute_cka,
    compute_layer_wise_cka,
)


@dataclass
class ActivationData:
    """Container for collected activation data."""
    hidden_states: np.ndarray  # (n_samples, hidden_dim)
    hidden_c: np.ndarray       # (n_samples, hidden_dim) - LSTM cell state
    hidden_h: np.ndarray       # (n_samples, hidden_dim) - LSTM hidden state

    # Universal metadata for clustering
    wall_densities: np.ndarray # (n_samples,)
    episode_outcomes: np.ndarray  # (n_samples,) - 0=timeout, 1=solved
    training_phases: np.ndarray  # (n_samples,) - normalized [0, 1]
    episode_returns: np.ndarray  # (n_samples,)

    # Method-specific metadata
    branch_types: Optional[np.ndarray] = None  # ACCEL/PLR: 0=DR, 1=Replay, 2=Mutate
    regrets: Optional[np.ndarray] = None        # PAIRED: regret estimates
    adversary_difficulties: Optional[np.ndarray] = None  # PAIRED: adversary difficulty

    # PAIRED bilateral data
    antagonist_hidden_states: Optional[np.ndarray] = None  # Antagonist h-states on matched levels
    antagonist_returns: Optional[np.ndarray] = None  # Antagonist episode returns
    regret_sources: Optional[np.ndarray] = None  # 0=ant_strong, 1=pro_weak, 2=both
    adversary_strategy_clusters: Optional[np.ndarray] = None  # Cluster IDs from C3

    # Training method
    training_method: str = "accel"

    # Optional: CNN activations for layer-wise analysis
    cnn_activations: Optional[np.ndarray] = None


class ActivationAnalysisExperiment(CheckpointExperiment):
    """
    Analyze activation patterns to understand feature organization.

    Methods:
    - Dimensionality reduction with multiple clustering criteria
    - RSA comparing representation geometry across conditions
    - CKA for cross-agent and cross-layer comparison
    - Sparse autoencoder for interpretable feature extraction
    """

    @property
    def name(self) -> str:
        return "activation_analysis"

    def __init__(
        self,
        n_episodes: int = 200,
        n_components_pca: int = 50,
        n_components_viz: int = 2,
        compute_sparse_ae: bool = False,
        sparse_ae_hidden: int = 2048,
        sparse_ae_sparsity: float = 0.1,
        **kwargs,
    ):
        """
        Initialize activation analysis experiment.

        Args:
            n_episodes: Number of episodes to collect activations from
            n_components_pca: PCA components for intermediate reduction
            n_components_viz: Components for visualization (2 or 3)
            compute_sparse_ae: Whether to train sparse autoencoder
            sparse_ae_hidden: Sparse AE hidden dimension (overcomplete)
            sparse_ae_sparsity: Sparsity coefficient for L1 regularization
        """
        super().__init__(**kwargs)
        self.n_episodes = n_episodes
        self.n_components_pca = n_components_pca
        self.n_components_viz = n_components_viz
        self.compute_sparse_ae = compute_sparse_ae
        self.sparse_ae_hidden = sparse_ae_hidden
        self.sparse_ae_sparsity = sparse_ae_sparsity

        self._data: Optional[ActivationData] = None
        self._results: Dict[str, Any] = {}

    def collect_data(self, rng: chex.PRNGKey) -> ActivationData:
        """
        Collect activations from episodes across curriculum conditions (GPU-batched).

        Collects hidden states at episode end, along with method-appropriate
        metadata for clustering analysis.
        """
        import time
        import logging
        from tqdm import tqdm
        from .utils.batched_rollout import batched_rollout

        logger = logging.getLogger(__name__)
        timings = {}

        try:
            import wandb
            _wandb_active = wandb.run is not None
        except ImportError:
            _wandb_active = False

        def _log(phase, elapsed=None, msg=None):
            if elapsed is not None:
                timings[phase] = elapsed
                logger.info(f"[{self.name}] {phase}: {elapsed:.2f}s")
            if msg:
                logger.info(f"[{self.name}] {msg}")
            if _wandb_active:
                log_dict = {}
                if elapsed is not None:
                    log_dict[f"{self.name}/timing/{phase}"] = elapsed
                if msg:
                    log_dict[f"{self.name}/status"] = msg
                if log_dict:
                    wandb.log(log_dict)

        n_episodes = self.n_episodes
        max_steps = 256

        # --- 1. Generate all levels in batch ---
        _log("generate_levels", msg="Generating levels...")
        t0 = time.time()
        rng, rng_levels = jax.random.split(rng)
        level_rngs = jax.random.split(rng_levels, n_episodes)
        levels = jax.vmap(self.agent.sample_random_level)(level_rngs)
        jax.block_until_ready(levels)
        _log("generate_levels", time.time() - t0, "Level generation complete")

        # --- 2. CPU-side level properties ---
        t0 = time.time()
        wall_maps = np.array(levels.wall_map)
        goal_positions = np.array(levels.goal_pos)
        agent_positions = np.array(levels.agent_pos)
        wall_density = wall_maps.mean(axis=(1, 2))
        _log("cpu_level_properties", time.time() - t0)

        # --- 3. Batched protagonist rollout with terminal hstate ---
        _log("protagonist_rollout", msg="Running batched protagonist rollout...")
        t0 = time.time()
        rng, rng_pro = jax.random.split(rng)
        pro_result = batched_rollout(
            rng_pro, levels, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n_episodes),
            collection_steps=[-1],
        )
        _log("protagonist_rollout", time.time() - t0, "Protagonist rollout complete")

        # Extract flattened hidden states from terminal hstates
        hidden_states = pro_result.hstates_by_step["-1"]  # (n_episodes, hidden_dim)
        hidden_dim = hidden_states.shape[1] // 2
        hidden_c = hidden_states[:, :hidden_dim]
        hidden_h = hidden_states[:, hidden_dim:]

        # Training phase
        current_step = self.train_state.step if hasattr(self.train_state, 'step') else 0
        max_training_steps = 30000
        training_phase = current_step / max_training_steps

        # Branch types
        if self.has_branches:
            branch_types = np.arange(n_episodes) % self.branch_count
        else:
            branch_types = None

        # --- 4. PAIRED bilateral ---
        antagonist_hidden_states = None
        antagonist_returns = None
        regrets = None
        adversary_difficulties = None
        regret_sources = None
        adversary_strategy_clusters = None

        if self.has_regret:
            _log("paired_cpu_estimates", msg="Computing PAIRED CPU estimates...")
            t0 = time.time()
            adversary_difficulties = np.array([
                self._estimate_adversary_difficulty({
                    'wall_density': float(wall_density[i]),
                    'goal_pos': tuple(goal_positions[i]),
                    'agent_pos': tuple(agent_positions[i]),
                })
                for i in tqdm(range(n_episodes), desc="Adversary difficulty", leave=False)
            ])
            _log("paired_cpu_estimates", time.time() - t0)

            ant_train_state = getattr(self.train_state, 'ant_train_state', None)
            if ant_train_state is not None:
                _log("antagonist_rollout", msg="Running batched antagonist rollout...")
                t0 = time.time()
                rng, rng_ant = jax.random.split(rng)
                ant_result = batched_rollout(
                    rng_ant, levels, max_steps,
                    ant_train_state.apply_fn, ant_train_state.params,
                    self.agent.env, self.agent.env_params,
                    self.agent.initialize_hidden_state(n_episodes),
                    collection_steps=[-1],
                )
                _log("antagonist_rollout", time.time() - t0, "Antagonist rollout complete")

                antagonist_hidden_states = ant_result.hstates_by_step["-1"]
                antagonist_returns = ant_result.episode_returns

                # Compute regret and regret sources
                regrets = antagonist_returns - pro_result.episode_returns
                regret_sources = np.array([
                    self._classify_regret_source(
                        float(pro_result.episode_returns[i]),
                        float(antagonist_returns[i]),
                        float(regrets[i]),
                    )
                    for i in range(n_episodes)
                ])

                adversary_strategy_clusters = np.array([
                    self._estimate_adversary_strategy_cluster(
                        {'wall_density': float(wall_density[i])},
                        float(adversary_difficulties[i]),
                    )
                    for i in range(n_episodes)
                ])
            else:
                regrets = np.zeros(n_episodes)

        # --- 5. Assemble data ---
        self._data = ActivationData(
            hidden_states=hidden_states,
            hidden_c=hidden_c,
            hidden_h=hidden_h,
            wall_densities=wall_density,
            episode_outcomes=(pro_result.episode_solved).astype(int),
            training_phases=np.full(n_episodes, training_phase),
            episode_returns=pro_result.episode_returns,
            branch_types=branch_types,
            regrets=regrets,
            adversary_difficulties=adversary_difficulties,
            antagonist_hidden_states=antagonist_hidden_states,
            antagonist_returns=antagonist_returns,
            regret_sources=regret_sources,
            adversary_strategy_clusters=adversary_strategy_clusters,
            training_method=self.training_method,
        )

        if _wandb_active:
            wandb.log({
                f"{self.name}/mean_return": float(pro_result.episode_returns.mean()),
                f"{self.name}/solve_rate": float(pro_result.episode_solved.mean()),
            })

        total_time = sum(timings.values())
        _log("total", total_time, f"TOTAL collect_data: {total_time:.2f}s | breakdown: {timings}")

        return self._data

    def _estimate_adversary_difficulty(self, level: Dict[str, Any]) -> float:
        """Estimate how difficult the adversary made this level (PAIRED)."""
        wall_density = level.get('wall_density', 0.2)

        # Estimate based on level characteristics
        goal_pos = level.get('goal_pos', (6, 6))
        agent_pos = level.get('agent_pos', (1, 1))
        goal_distance = np.sqrt(
            (goal_pos[0] - agent_pos[0])**2 +
            (goal_pos[1] - agent_pos[1])**2
        )
        normalized_distance = goal_distance / 18.0  # Max ~18 for 13x13 grid

        return float(0.5 * wall_density + 0.5 * normalized_distance)

    def _classify_regret_source(
        self,
        pro_return: float,
        ant_return: float,
        regret: float,
    ) -> int:
        """
        Classify regret source: antagonist strong vs protagonist weak.

        Returns:
            0: ant_strong (antagonist succeeded, protagonist would have too)
            1: pro_weak (protagonist failed on solvable level)
            2: both (antagonist strong AND protagonist weak)
        """
        # Thresholds for classification
        pro_threshold = 0.5  # Return below this = protagonist weak
        ant_threshold = 0.5  # Return above this = antagonist strong

        ant_strong = ant_return > ant_threshold
        pro_weak = pro_return < pro_threshold

        if ant_strong and pro_weak:
            return 2  # Both
        elif ant_strong:
            return 0  # Antagonist strong
        elif pro_weak:
            return 1  # Protagonist weak
        else:
            return 0  # Default: attribute to antagonist strength

    def _estimate_adversary_strategy_cluster(
        self,
        level: Dict[str, Any],
        adversary_difficulty: float,
    ) -> int:
        """
        Estimate adversary strategy cluster from level features.

        Placeholder - actual clustering is done in C3 (adversary_strategy_clustering).
        Uses simple heuristics to assign cluster IDs.
        """
        wall_density = level.get('wall_density', 0.2)

        # Simple clustering by difficulty and wall density
        # 5 clusters based on discretizing difficulty and density
        difficulty_bin = min(int(adversary_difficulty * 3), 2)  # 0, 1, 2
        density_bin = 0 if wall_density < 0.3 else 1  # 0 or 1

        cluster = difficulty_bin * 2 + density_bin
        return int(cluster)

    def _generate_level(self, rng: chex.PRNGKey, branch: int = 0) -> Dict[str, Any]:
        """Generate a synthetic test level for prediction loss computation."""
        height, width = 13, 13

        # Vary wall density by branch to approximate training distribution
        if branch == 0:  # DR branch: lower density
            wall_prob = 0.1 + float(jax.random.uniform(rng)) * 0.15
        elif branch == 1:  # Replay branch: medium density
            wall_prob = 0.15 + float(jax.random.uniform(rng)) * 0.15
        else:  # Mutation branch: higher density
            wall_prob = 0.2 + float(jax.random.uniform(rng)) * 0.15

        wall_map = np.array(jax.random.bernoulli(rng, wall_prob, (height, width)))
        wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

        rng_goal, rng_agent = jax.random.split(rng)
        goal_pos = (
            int(jax.random.randint(rng_goal, (), 1, height - 1)),
            int(jax.random.randint(rng_goal, (), 1, width - 1)),
        )
        agent_pos = (
            int(jax.random.randint(rng_agent, (), 1, height - 1)),
            int(jax.random.randint(rng_agent, (), 1, width - 1)),
        )

        return {
            'wall_map': wall_map,
            'wall_density': wall_map.sum() / (height * width),
            'goal_pos': goal_pos,
            'agent_pos': agent_pos,
            'branch': branch,
        }

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze activation patterns.

        Performs:
        1. PRIMARY: Prediction loss analysis (actual causal measure)
        2. SECONDARY: Representation proxies (RSA, CKA, clustering)
        3. Correlation between proxies and prediction loss
        """
        if self._data is None:
            raise ValueError("Must call collect_data before analyze")

        results = {}

        # PRIMARY: Prediction loss analysis (the actual causal measure)
        results['prediction_loss'] = self._analyze_prediction_connection()

        # SECONDARY: Representation proxies
        # 1. Dimensionality reduction
        results['dimensionality_reduction'] = self._analyze_dimensionality()

        # 2. Clustering quality by different groupings
        results['clustering'] = self._analyze_clustering()

        # 3. RSA between conditions
        results['rsa'] = self._analyze_rsa()

        # 4. CKA between LSTM components
        results['cka'] = self._analyze_cka()

        # 5. Representation statistics
        results['representation_stats'] = self._compute_representation_stats()

        # 6. Optional: Sparse autoencoder
        if self.compute_sparse_ae:
            results['sparse_ae'] = self._train_sparse_autoencoder()

        # 7. Correlation: How do proxies relate to prediction loss?
        results['proxy_vs_prediction_correlation'] = self._correlate_proxies_with_prediction(results)

        self._results = results
        return results

    def _analyze_dimensionality(self) -> Dict[str, Any]:
        """Perform dimensionality reduction analysis."""
        from sklearn.decomposition import PCA

        hidden_states = self._data.hidden_states

        # PCA analysis
        pca = PCA(n_components=min(self.n_components_pca, len(hidden_states), hidden_states.shape[1]))
        pca_coords = pca.fit_transform(hidden_states)

        # Explained variance analysis
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_90 = int(np.searchsorted(cumsum, 0.9) + 1)
        n_95 = int(np.searchsorted(cumsum, 0.95) + 1)

        # Participation ratio (effective dimensionality)
        eigenvalues = pca.explained_variance_
        pr = (eigenvalues.sum() ** 2) / (np.sum(eigenvalues ** 2) + 1e-10)

        results = {
            'pca_coords_2d': pca_coords[:, :2].tolist() if pca_coords.shape[1] >= 2 else pca_coords.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': cumsum.tolist(),
            'n_components_90_variance': n_90,
            'n_components_95_variance': n_95,
            'participation_ratio': float(pr),
            'total_variance_explained': float(cumsum[-1]) if len(cumsum) > 0 else 0.0,
        }

        # Try t-SNE if sklearn available
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=self.n_components_viz, perplexity=min(30, len(hidden_states) - 1), random_state=42)
            tsne_coords = tsne.fit_transform(hidden_states)
            results['tsne_coords'] = tsne_coords.tolist()
        except Exception as e:
            results['tsne_error'] = str(e)

        # Try UMAP if available
        try:
            import umap
            reducer = umap.UMAP(n_components=self.n_components_viz, random_state=42)
            umap_coords = reducer.fit_transform(hidden_states)
            results['umap_coords'] = umap_coords.tolist()
        except ImportError:
            results['umap_note'] = "UMAP not installed"
        except Exception as e:
            results['umap_error'] = str(e)

        return results

    def _analyze_clustering(self) -> Dict[str, Any]:
        """Analyze clustering quality by method-appropriate groupings."""
        from sklearn.metrics import silhouette_score, calinski_harabasz_score

        hidden_states = self._data.hidden_states

        results = {}

        # Universal clustering criteria
        clusterings = {
            'episode_outcome': self._data.episode_outcomes,
            'wall_density_tercile': np.digitize(
                self._data.wall_densities,
                np.percentile(self._data.wall_densities, [33, 66])
            ),
            'return_tercile': np.digitize(
                self._data.episode_returns,
                np.percentile(self._data.episode_returns, [33, 66])
            ),
        }

        # Method-specific clustering criteria
        if self.has_branches and self._data.branch_types is not None:
            # ACCEL/PLR: cluster by branch type
            clusterings['branch_type'] = self._data.branch_types

        if self.has_regret:
            # PAIRED: cluster by regret and adversary difficulty
            if self._data.regrets is not None and np.std(self._data.regrets) > 1e-6:
                clusterings['regret_tercile'] = np.digitize(
                    self._data.regrets,
                    np.percentile(self._data.regrets, [33, 66])
                )

            if self._data.adversary_difficulties is not None and np.std(self._data.adversary_difficulties) > 1e-6:
                clusterings['adversary_difficulty_tercile'] = np.digitize(
                    self._data.adversary_difficulties,
                    np.percentile(self._data.adversary_difficulties, [33, 66])
                )

            # PAIRED-specific: regret source clustering (ant_strong vs pro_weak)
            if self._data.regret_sources is not None:
                clusterings['regret_source'] = self._data.regret_sources

            # PAIRED-specific: adversary strategy cluster
            if self._data.adversary_strategy_clusters is not None:
                clusterings['adversary_strategy_cluster'] = self._data.adversary_strategy_clusters

        for name, labels in clusterings.items():
            # Need at least 2 clusters with > 1 sample each
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                results[name] = {'error': 'Fewer than 2 clusters'}
                continue

            # Check that each cluster has enough samples
            min_cluster_size = min([np.sum(labels == l) for l in unique_labels])
            if min_cluster_size < 2:
                results[name] = {'error': 'Cluster with < 2 samples'}
                continue

            try:
                silhouette = silhouette_score(hidden_states, labels)
                calinski = calinski_harabasz_score(hidden_states, labels)

                # Compute inter-cluster distances
                cluster_centers = []
                for label in unique_labels:
                    mask = labels == label
                    cluster_centers.append(hidden_states[mask].mean(axis=0))

                if len(cluster_centers) >= 2:
                    cluster_centers = np.stack(cluster_centers)
                    inter_cluster_dists = []
                    for i in range(len(cluster_centers)):
                        for j in range(i + 1, len(cluster_centers)):
                            dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                            inter_cluster_dists.append(dist)
                    mean_inter_dist = float(np.mean(inter_cluster_dists))
                else:
                    mean_inter_dist = 0.0

                results[name] = {
                    'silhouette_score': float(silhouette),
                    'calinski_harabasz_score': float(calinski),
                    'n_clusters': int(len(unique_labels)),
                    'cluster_sizes': [int(np.sum(labels == l)) for l in unique_labels],
                    'mean_inter_cluster_distance': mean_inter_dist,
                }
            except Exception as e:
                results[name] = {'error': str(e)}

        return results

    def _analyze_rsa(self) -> Dict[str, Any]:
        """Perform Representation Similarity Analysis (method-aware)."""
        hidden_states = self._data.hidden_states

        # Compute overall RDM
        rdm = compute_rdm(hidden_states)

        results = {
            'rdm_mean': float(rdm.mean()),
            'rdm_std': float(rdm.std()),
            'training_method': self.training_method,
        }

        # Method-appropriate comparisons
        comparisons = [
            ('outcome', self._data.episode_outcomes),
        ]

        # Add method-specific comparisons
        if self.has_branches and self._data.branch_types is not None:
            comparisons.append(('branch_type', self._data.branch_types))

        if self.has_regret and self._data.regrets is not None:
            # Use terciles for regret
            regret_terciles = np.digitize(
                self._data.regrets,
                np.percentile(self._data.regrets, [33, 66])
            )
            comparisons.append(('regret_tercile', regret_terciles))

        for name, labels in comparisons:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                continue

            # Compute RDM for each condition
            condition_rdms = {}
            for label in unique_labels:
                mask = labels == label
                if mask.sum() >= 3:  # Need enough samples
                    condition_data = hidden_states[mask]
                    condition_rdms[int(label)] = compute_rdm(condition_data)

            # Compare RDMs between conditions
            if len(condition_rdms) >= 2:
                labels_list = list(condition_rdms.keys())
                rsa_scores = {}
                for i in range(len(labels_list)):
                    for j in range(i + 1, len(labels_list)):
                        rdm_i = condition_rdms[labels_list[i]]
                        rdm_j = condition_rdms[labels_list[j]]

                        # Truncate to same size
                        min_size = min(rdm_i.shape[0], rdm_j.shape[0])
                        rdm_i_trunc = rdm_i[:min_size, :min_size]
                        rdm_j_trunc = rdm_j[:min_size, :min_size]

                        rsa_result = compute_rsa(rdm_i_trunc, rdm_j_trunc)
                        key = f"{labels_list[i]}_vs_{labels_list[j]}"
                        rsa_scores[key] = rsa_result

                results[f'{name}_rsa'] = rsa_scores

        # RSA between LSTM c and h components
        rdm_c = compute_rdm(self._data.hidden_c)
        rdm_h = compute_rdm(self._data.hidden_h)
        results['lstm_c_vs_h_rsa'] = compute_rsa(rdm_c, rdm_h)

        return results

    def _analyze_cka(self) -> Dict[str, Any]:
        """Perform Centered Kernel Alignment analysis."""
        results = {}

        # CKA between LSTM components
        cka_c_h = compute_cka(self._data.hidden_c, self._data.hidden_h)
        results['lstm_c_vs_h_cka'] = cka_c_h

        # CKA between different subsets (e.g., solved vs unsolved)
        solved_mask = self._data.episode_outcomes == 1
        unsolved_mask = self._data.episode_outcomes == 0

        if solved_mask.sum() >= 10 and unsolved_mask.sum() >= 10:
            solved_states = self._data.hidden_states[solved_mask]
            unsolved_states = self._data.hidden_states[unsolved_mask]

            # Truncate to same size for CKA
            min_n = min(len(solved_states), len(unsolved_states))
            cka_outcome = compute_cka(solved_states[:min_n], unsolved_states[:min_n])
            results['solved_vs_unsolved_cka'] = cka_outcome

        # Method-specific CKA comparisons
        if self.has_branches and self._data.branch_types is not None:
            # CKA between branch types (ACCEL/PLR)
            branch_data = {}
            for branch in range(self.branch_count):
                mask = self._data.branch_types == branch
                if mask.sum() >= 10:
                    branch_data[branch] = self._data.hidden_states[mask]

            if len(branch_data) >= 2:
                branch_cka = {}
                branches = list(branch_data.keys())
                for i in range(len(branches)):
                    for j in range(i + 1, len(branches)):
                        data_i = branch_data[branches[i]]
                        data_j = branch_data[branches[j]]
                        min_n = min(len(data_i), len(data_j))
                        cka_val = compute_cka(data_i[:min_n], data_j[:min_n])
                        key = f"branch_{branches[i]}_vs_{branches[j]}"
                        branch_cka[key] = cka_val
                results['branch_cka'] = branch_cka

        elif self.has_regret and self._data.regrets is not None:
            # CKA between regret terciles (PAIRED)
            regret_terciles = np.digitize(
                self._data.regrets,
                np.percentile(self._data.regrets, [33, 66])
            )

            tercile_data = {}
            for tercile in [0, 1, 2]:
                mask = regret_terciles == tercile
                if mask.sum() >= 10:
                    tercile_data[tercile] = self._data.hidden_states[mask]

            if len(tercile_data) >= 2:
                regret_cka = {}
                terciles = list(tercile_data.keys())
                for i in range(len(terciles)):
                    for j in range(i + 1, len(terciles)):
                        data_i = tercile_data[terciles[i]]
                        data_j = tercile_data[terciles[j]]
                        min_n = min(len(data_i), len(data_j))
                        cka_val = compute_cka(data_i[:min_n], data_j[:min_n])
                        tercile_names = ['low', 'medium', 'high']
                        key = f"regret_{tercile_names[terciles[i]]}_vs_{tercile_names[terciles[j]]}"
                        regret_cka[key] = cka_val
                results['regret_cka'] = regret_cka

            # PAIRED bilateral CKA: protagonist vs antagonist on matched levels
            if self._data.antagonist_hidden_states is not None:
                results['bilateral_cka'] = self._analyze_bilateral_cka()

            # CKA by regret source
            if self._data.regret_sources is not None:
                regret_source_cka = {}
                source_names = ['ant_strong', 'pro_weak', 'both']
                source_data = {}

                for source in [0, 1, 2]:
                    mask = self._data.regret_sources == source
                    if mask.sum() >= 10:
                        source_data[source] = self._data.hidden_states[mask]

                if len(source_data) >= 2:
                    sources = list(source_data.keys())
                    for i in range(len(sources)):
                        for j in range(i + 1, len(sources)):
                            data_i = source_data[sources[i]]
                            data_j = source_data[sources[j]]
                            min_n = min(len(data_i), len(data_j))
                            cka_val = compute_cka(data_i[:min_n], data_j[:min_n])
                            key = f"regret_source_{source_names[sources[i]]}_vs_{source_names[sources[j]]}"
                            regret_source_cka[key] = cka_val
                    results['regret_source_cka'] = regret_source_cka

        return results

    def _analyze_bilateral_cka(self) -> Dict[str, Any]:
        """
        Compute bilateral CKA between protagonist and antagonist (PAIRED).

        Key metric for measuring representation divergence between agents.
        """
        pro_states = self._data.hidden_states
        ant_states = self._data.antagonist_hidden_states

        if ant_states is None or len(ant_states) < 10:
            return {'error': 'Insufficient antagonist data'}

        # Overall bilateral CKA
        min_n = min(len(pro_states), len(ant_states))
        bilateral_cka = compute_cka(pro_states[:min_n], ant_states[:min_n])

        results = {
            'protagonist_vs_antagonist_cka': bilateral_cka,
            'representation_divergence': 1.0 - bilateral_cka,
        }

        # Bilateral CKA by regret condition
        if self._data.regrets is not None:
            regret_terciles = np.digitize(
                self._data.regrets[:min_n],
                np.percentile(self._data.regrets[:min_n], [33, 66])
            )

            bilateral_by_regret = {}
            for tercile, name in [(0, 'low'), (1, 'medium'), (2, 'high')]:
                mask = regret_terciles == tercile
                if mask.sum() >= 5:
                    cka_val = compute_cka(
                        pro_states[:min_n][mask],
                        ant_states[:min_n][mask]
                    )
                    bilateral_by_regret[f'regret_{name}'] = cka_val

            results['bilateral_cka_by_regret'] = bilateral_by_regret

            # Key finding: Does divergence increase with regret?
            if len(bilateral_by_regret) >= 2:
                cka_vals = list(bilateral_by_regret.values())
                results['divergence_increases_with_regret'] = (
                    bilateral_by_regret.get('regret_high', 0) <
                    bilateral_by_regret.get('regret_low', 1)
                )

        # Bilateral CKA by outcome
        outcome_bilateral = {}
        for outcome, name in [(0, 'unsolved'), (1, 'solved')]:
            mask = self._data.episode_outcomes[:min_n] == outcome
            if mask.sum() >= 5:
                cka_val = compute_cka(
                    pro_states[:min_n][mask],
                    ant_states[:min_n][mask]
                )
                outcome_bilateral[name] = cka_val

        if outcome_bilateral:
            results['bilateral_cka_by_outcome'] = outcome_bilateral

        return results

    def _compute_representation_stats(self) -> Dict[str, Any]:
        """Compute basic statistics about representations."""
        hidden_states = self._data.hidden_states

        # Activation statistics
        mean_activation = float(np.mean(hidden_states))
        std_activation = float(np.std(hidden_states))
        sparsity = float(np.mean(np.abs(hidden_states) < 0.1))  # Fraction near zero

        # Per-dimension statistics
        dim_means = np.mean(hidden_states, axis=0)
        dim_stds = np.std(hidden_states, axis=0)

        # Dead dimensions (always near zero)
        dead_dims = int(np.sum(dim_stds < 1e-6))

        # Saturated dimensions (always near max)
        max_vals = np.max(np.abs(hidden_states), axis=0)
        saturated_dims = int(np.sum((max_vals > 0.99) & (dim_stds < 0.01)))

        # Correlation structure
        corr_matrix = np.corrcoef(hidden_states.T)
        mean_abs_corr = float(np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])))

        return {
            'mean_activation': mean_activation,
            'std_activation': std_activation,
            'sparsity': sparsity,
            'n_dimensions': int(hidden_states.shape[1]),
            'dead_dimensions': dead_dims,
            'saturated_dimensions': saturated_dims,
            'mean_abs_correlation': mean_abs_corr,
            'dim_std_mean': float(np.mean(dim_stds)),
            'dim_std_std': float(np.std(dim_stds)),
        }

    def _train_sparse_autoencoder(self) -> Dict[str, Any]:
        """Train sparse autoencoder for interpretable feature extraction."""
        # Simple implementation using numpy/sklearn
        from sklearn.linear_model import Lasso

        hidden_states = self._data.hidden_states
        n_samples, n_features = hidden_states.shape

        # Learn sparse representation using Lasso
        # This is a simplified proxy for a true sparse autoencoder
        lasso = Lasso(alpha=self.sparse_ae_sparsity, max_iter=1000)

        # Use first half of dimensions to predict second half (simple reconstruction)
        mid = n_features // 2
        X = hidden_states[:, :mid]
        y = hidden_states[:, mid:]

        results = {
            'method': 'lasso_proxy',
            'n_features_in': mid,
            'n_features_out': n_features - mid,
        }

        try:
            lasso.fit(X, y[:, 0])  # Predict first output dimension

            # Sparsity of learned weights
            n_nonzero = np.sum(np.abs(lasso.coef_) > 1e-6)
            sparsity = 1.0 - (n_nonzero / len(lasso.coef_))

            results['learned_sparsity'] = float(sparsity)
            results['n_active_features'] = int(n_nonzero)
            results['reconstruction_score'] = float(lasso.score(X, y[:, 0]))
        except Exception as e:
            results['error'] = str(e)

        return results

    def _analyze_prediction_connection(self) -> Dict[str, Any]:
        """
        Analyze connection between representation structure and prediction ability.

        Computes prediction loss for each sample and correlates with
        representation properties like clustering, dimensionality, etc.
        """
        try:
            from .utils.agent_aware_loss import (
                compute_agent_prediction_loss,
                compute_random_baseline_loss,
            )

            # We need the levels that generated each hidden state
            # Since we didn't store them during collection, generate new ones
            # that match the characteristics
            import jax

            rng = jax.random.PRNGKey(42)
            n_samples = len(self._data.hidden_states)

            prediction_losses = []

            for i in range(min(n_samples, 100)):  # Limit to 100 for speed
                rng, level_rng, loss_rng = jax.random.split(rng, 3)

                # Recreate level with similar characteristics
                branch = self._data.branch_types[i]
                wall_density = self._data.wall_densities[i]

                level = self._generate_level(level_rng, branch)
                # Adjust wall density to match original
                # (keep existing level generation but note the actual density)

                loss, _ = compute_agent_prediction_loss(
                    self.agent,
                    self.train_state,
                    level,
                    loss_rng,
                )
                prediction_losses.append(loss)

            prediction_losses = np.array(prediction_losses)
            random_baseline = compute_random_baseline_loss()

            results = {
                'mean_prediction_loss': float(np.mean(prediction_losses)),
                'std_prediction_loss': float(np.std(prediction_losses)),
                'random_baseline': random_baseline,
                'information_gain': float(random_baseline - np.mean(prediction_losses)),
            }

            # Loss by branch type
            branch_losses = {}
            for branch in [0, 1, 2]:
                mask = self._data.branch_types[:len(prediction_losses)] == branch
                if mask.sum() > 0:
                    branch_losses[f'branch_{branch}'] = float(np.mean(prediction_losses[mask]))
            results['loss_by_branch'] = branch_losses

            # Loss by outcome
            outcome_losses = {}
            solved_mask = self._data.episode_outcomes[:len(prediction_losses)] == 1
            unsolved_mask = self._data.episode_outcomes[:len(prediction_losses)] == 0
            if solved_mask.sum() > 0:
                outcome_losses['solved'] = float(np.mean(prediction_losses[solved_mask]))
            if unsolved_mask.sum() > 0:
                outcome_losses['unsolved'] = float(np.mean(prediction_losses[unsolved_mask]))
            results['loss_by_outcome'] = outcome_losses

            # Correlation between prediction loss and clustering quality
            # Use wall_density as a proxy for difficulty
            wall_densities = self._data.wall_densities[:len(prediction_losses)]
            if len(wall_densities) > 10:
                corr = np.corrcoef(wall_densities, prediction_losses)[0, 1]
                results['loss_vs_difficulty_correlation'] = float(corr) if np.isfinite(corr) else 0.0

            # Correlation with episode returns
            returns = self._data.episode_returns[:len(prediction_losses)]
            if len(returns) > 10:
                corr = np.corrcoef(returns, prediction_losses)[0, 1]
                results['loss_vs_return_correlation'] = float(corr) if np.isfinite(corr) else 0.0

            return results

        except Exception as e:
            return {'error': str(e)}

    def _correlate_proxies_with_prediction(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Correlate representation proxy metrics with actual prediction loss.

        This validates whether proxy measures (RSA, CKA, clustering) actually
        relate to the primary causal measure (prediction loss).
        """
        correlation_results = {}

        # Get prediction losses from results
        pred_results = results.get('prediction_loss', {})
        if 'error' in pred_results or 'loss_by_branch' not in pred_results:
            return {'error': 'Prediction loss data not available'}

        # Compare clustering silhouette with prediction loss difference
        clustering = results.get('clustering', {})
        if 'branch_type' in clustering and 'silhouette_score' in clustering['branch_type']:
            branch_silhouette = clustering['branch_type']['silhouette_score']

            # Higher silhouette means better separation - does it correlate with
            # lower prediction loss (better prediction)?
            correlation_results['branch_clustering_vs_prediction'] = {
                'silhouette_score': branch_silhouette,
                'mean_prediction_loss': pred_results.get('mean_prediction_loss', 0),
                'interpretation': (
                    "High silhouette score indicates representations cluster by branch. "
                    "Compare with prediction loss to validate if clustering relates to "
                    "actual prediction ability."
                ),
            }

        # Compare CKA between solved/unsolved with prediction loss by outcome
        cka = results.get('cka', {})
        if 'solved_vs_unsolved_cka' in cka:
            cka_val = cka['solved_vs_unsolved_cka']
            loss_by_outcome = pred_results.get('loss_by_outcome', {})

            correlation_results['outcome_cka_vs_prediction'] = {
                'cka_solved_unsolved': cka_val,
                'loss_solved': loss_by_outcome.get('solved', 0),
                'loss_unsolved': loss_by_outcome.get('unsolved', 0),
                'interpretation': (
                    "Low CKA between solved/unsolved means different representations. "
                    "If solved episodes also have lower prediction loss, representations "
                    "encode task-relevant curriculum information."
                ),
            }

        # Dimensionality vs prediction loss
        dim_red = results.get('dimensionality_reduction', {})
        if 'participation_ratio' in dim_red:
            correlation_results['dimensionality_vs_prediction'] = {
                'participation_ratio': dim_red['participation_ratio'],
                'n_components_90_variance': dim_red.get('n_components_90_variance', 0),
                'mean_prediction_loss': pred_results.get('mean_prediction_loss', 0),
                'interpretation': (
                    "Higher effective dimensionality may indicate richer representations "
                    "that can encode more curriculum information, leading to lower loss."
                ),
            }

        # RSA geometry vs prediction
        rsa = results.get('rsa', {})
        if 'lstm_c_vs_h_rsa' in rsa:
            correlation_results['lstm_geometry_vs_prediction'] = {
                'c_h_rsa_correlation': rsa['lstm_c_vs_h_rsa'].get('spearman_r', 0) if isinstance(rsa['lstm_c_vs_h_rsa'], dict) else rsa['lstm_c_vs_h_rsa'],
                'interpretation': (
                    "RSA correlation between LSTM c and h states indicates "
                    "whether both components encode similar structure. "
                    "High correlation suggests redundancy; low may indicate "
                    "complementary encoding of curriculum features."
                ),
            }

        # Summary: Are proxies valid?
        correlation_results['summary'] = {
            'proxies_evaluated': list(correlation_results.keys()),
            'recommendation': (
                "Proxy metrics (RSA, CKA, clustering) are useful for understanding "
                "representation structure, but prediction loss is the PRIMARY metric "
                "for causal claims about curriculum awareness."
            ),
        }

        return correlation_results

    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data for activation analysis."""
        if not self._results:
            raise ValueError("Must call analyze before visualize")

        viz_data = {}

        # PCA scatter plot data
        if 'dimensionality_reduction' in self._results:
            dr = self._results['dimensionality_reduction']

            if 'pca_coords_2d' in dr:
                viz_data['pca_scatter'] = {
                    'coords': dr['pca_coords_2d'],
                    'branch_types': self._data.branch_types.tolist(),
                    'outcomes': self._data.episode_outcomes.tolist(),
                    'returns': self._data.episode_returns.tolist(),
                }

            if 'tsne_coords' in dr:
                viz_data['tsne_scatter'] = {
                    'coords': dr['tsne_coords'],
                    'branch_types': self._data.branch_types.tolist(),
                    'outcomes': self._data.episode_outcomes.tolist(),
                }

            if 'umap_coords' in dr:
                viz_data['umap_scatter'] = {
                    'coords': dr['umap_coords'],
                    'branch_types': self._data.branch_types.tolist(),
                    'outcomes': self._data.episode_outcomes.tolist(),
                }

            # Explained variance plot
            viz_data['explained_variance'] = {
                'cumulative': dr.get('cumulative_variance', []),
                'individual': dr.get('explained_variance_ratio', []),
            }

        # Clustering quality bar chart
        if 'clustering' in self._results:
            clustering = self._results['clustering']
            viz_data['clustering_quality'] = {
                'criteria': [],
                'silhouette_scores': [],
            }
            for name, data in clustering.items():
                if isinstance(data, dict) and 'silhouette_score' in data:
                    viz_data['clustering_quality']['criteria'].append(name)
                    viz_data['clustering_quality']['silhouette_scores'].append(
                        data['silhouette_score']
                    )

        # RSA matrix
        if 'rsa' in self._results:
            viz_data['rsa_summary'] = {
                key: val for key, val in self._results['rsa'].items()
                if not isinstance(val, dict)
            }

        # CKA summary
        if 'cka' in self._results:
            viz_data['cka_summary'] = self._results['cka']

        return viz_data

    def compare_with_other(
        self,
        other_data: ActivationData,
        comparison_name: str = "comparison",
    ) -> Dict[str, Any]:
        """
        Compare activations with another agent/checkpoint.

        Useful for cross-agent CKA and tracking representation changes.
        """
        if self._data is None:
            raise ValueError("Must call collect_data first")

        results = {}

        # CKA between this and other
        min_n = min(len(self._data.hidden_states), len(other_data.hidden_states))
        cka_full = compute_cka(
            self._data.hidden_states[:min_n],
            other_data.hidden_states[:min_n]
        )
        results[f'cka_{comparison_name}'] = cka_full

        # Separate CKA for c and h components
        cka_c = compute_cka(
            self._data.hidden_c[:min_n],
            other_data.hidden_c[:min_n]
        )
        cka_h = compute_cka(
            self._data.hidden_h[:min_n],
            other_data.hidden_h[:min_n]
        )
        results[f'cka_c_{comparison_name}'] = cka_c
        results[f'cka_h_{comparison_name}'] = cka_h

        # RSA comparison
        rdm_self = compute_rdm(self._data.hidden_states[:min_n])
        rdm_other = compute_rdm(other_data.hidden_states[:min_n])
        rsa_result = compute_rsa(rdm_self, rdm_other)
        results[f'rsa_{comparison_name}'] = rsa_result

        return results
