"""
Domain Randomization Coverage Experiment.

DR-specific experiment to analyze what portion of level space is explored.

Questions this experiment addresses:
1. What portion of the level feature space is explored by DR?
2. Is the difficulty distribution uniform or biased?
3. Are there blind spots (undersampled regions)?
4. How does DR coverage compare to curriculum methods?
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
import chex

from .base import CheckpointExperiment


@dataclass
class CoverageData:
    """Container for DR coverage analysis data."""
    # Level features
    wall_densities: List[float] = field(default_factory=list)
    goal_distances: List[float] = field(default_factory=list)
    path_lengths: List[float] = field(default_factory=list)
    is_solvable: List[bool] = field(default_factory=list)

    # Performance on sampled levels
    episode_returns: List[float] = field(default_factory=list)
    episode_solved: List[bool] = field(default_factory=list)

    # Feature vectors for coverage analysis
    feature_vectors: List[np.ndarray] = field(default_factory=list)

    # Prediction losses
    prediction_losses: List[float] = field(default_factory=list)


class DRCoverageExperiment(CheckpointExperiment):
    """
    DR-specific experiment to analyze level space coverage.

    Analyzes:
    - Coverage metric: What % of feature space is explored
    - Difficulty distribution: Is sampling uniform?
    - Blind spots: Undersampled regions
    - Comparison to curriculum: Coverage vs ACCEL/PAIRED

    Note: This experiment is most meaningful for DR training,
    but can be run on any method for comparison.
    """

    @property
    def name(self) -> str:
        return "dr_coverage"

    def __init__(self, **kwargs):
        """
        Initialize DR coverage experiment.
        """
        super().__init__(**kwargs)
        self.n_levels = self.exp_config("n_levels")
        self.n_grid_bins = self.exp_config("n_grid_bins")
        self.feature_names = self.exp_config("feature_names")

        self._data: Optional[CoverageData] = None
        self._results: Dict[str, Any] = {}

    def collect_data(self, rng: chex.PRNGKey) -> CoverageData:
        """
        Collect DR-sampled levels and their features (GPU-batched).
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

        self._data = CoverageData()

        # --- 1. Generate all levels in batch ---
        _log("generate_levels", msg="Generating levels...")
        t0 = time.time()
        rng, rng_levels = jax.random.split(rng)
        level_rngs = jax.random.split(rng_levels, self.n_levels)
        levels = jax.vmap(self.agent.sample_random_level)(level_rngs)
        jax.block_until_ready(levels)
        _log("generate_levels", time.time() - t0, "Level generation complete")

        # --- 2. Extract CPU-side level properties ---
        _log("cpu_level_properties", msg="Computing level properties...")
        t0 = time.time()
        wall_maps = np.array(levels.wall_map)
        goal_positions = np.array(levels.goal_pos)
        agent_positions = np.array(levels.agent_pos)

        wall_density = wall_maps.mean(axis=(1, 2))
        goal_distance = np.sqrt(np.sum((goal_positions - agent_positions) ** 2, axis=-1))

        path_lengths = np.array([
            self._compute_path_length({
                'wall_map': wall_maps[i],
                'agent_pos': tuple(agent_positions[i]),
                'goal_pos': tuple(goal_positions[i]),
            })
            for i in tqdm(range(self.n_levels), desc="BFS path lengths", leave=False)
        ])
        is_solvable = path_lengths > 0
        _log("cpu_level_properties", time.time() - t0, "Level properties complete")

        # --- 3. Batched rollout ---
        _log("batched_rollout", msg="Running batched rollout...")
        t0 = time.time()
        rng, rng_rollout = jax.random.split(rng)
        n_levels = self.n_levels
        max_steps = self.config.get("max_steps", 256)
        result = batched_rollout(
            rng_rollout, levels, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n_levels),
            collect_rewards=True,
        )
        _log("batched_rollout", time.time() - t0, "Batched rollout complete")

        # --- 4. Compute prediction losses ---
        _log("prediction_losses", msg="Computing prediction losses...")
        t0 = time.time()
        from .utils.agent_aware_loss import compute_agent_prediction_loss
        prediction_losses = []
        for i in tqdm(range(self.n_levels), desc="Prediction losses", leave=False):
            rng, loss_rng = jax.random.split(rng)
            level_i = jax.tree_util.tree_map(lambda x: x[i], levels)
            pred_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state, level_i, loss_rng
            )
            prediction_losses.append(pred_loss)
        _log("prediction_losses", time.time() - t0, "Prediction losses complete")

        # --- 5. Assemble data ---
        self._data.wall_densities = wall_density.tolist()
        self._data.goal_distances = goal_distance.tolist()
        self._data.path_lengths = path_lengths.tolist()
        self._data.is_solvable = is_solvable.tolist()
        self._data.episode_returns = result.episode_returns.tolist()
        self._data.episode_solved = result.episode_solved.tolist()
        self._data.prediction_losses = prediction_losses

        # Feature vectors for coverage analysis (normalized)
        norm_density = wall_density
        env_inner = getattr(self.agent.env, '_env', self.agent.env)
        env_h = getattr(env_inner, 'max_height', 13)
        env_w = getattr(env_inner, 'max_width', 13)
        max_diagonal = float(np.sqrt(env_h**2 + env_w**2))
        norm_distance = np.minimum(goal_distance / max_diagonal, 1.0)
        self._data.feature_vectors = [
            np.array([norm_density[i], norm_distance[i]])
            for i in range(self.n_levels)
        ]

        if _wandb_active:
            wandb.log({
                f"{self.name}/mean_return": float(result.episode_returns.mean()),
                f"{self.name}/solve_rate": float(result.episode_solved.mean()),
            })

        total_time = sum(timings.values())
        _log("total", total_time, f"TOTAL collect_data: {total_time:.2f}s | breakdown: {timings}")

        return self._data

    def _compute_path_length(self, level: Dict[str, Any]) -> int:
        """Compute BFS path length."""
        wall_map = np.array(level['wall_map'])
        agent_pos = tuple(level['agent_pos'])
        goal_pos = tuple(level['goal_pos'])

        if agent_pos == goal_pos:
            return 0

        from collections import deque
        queue = deque([(agent_pos, 0)])
        visited = {agent_pos}
        h, w = wall_map.shape

        while queue:
            (x, y), dist = queue.popleft()

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < h and 0 <= ny < w and
                    (nx, ny) not in visited and not wall_map[nx, ny]):
                    if (nx, ny) == goal_pos:
                        return dist + 1
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))

        return -1  # Unsolvable

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze DR coverage.

        Computes:
        - Coverage metric: Grid coverage percentage
        - Difficulty distribution: Histogram analysis
        - Blind spots: Undersampled regions
        - Performance by region: How agent performs in different areas
        - Chance baseline: Permutation-based baseline for prediction losses
        """
        if self._data is None or len(self._data.wall_densities) == 0:
            return {'error': 'No data collected'}

        results = {
            'training_method': self.training_method,
        }

        # 1. Coverage analysis
        results['coverage'] = self._analyze_coverage()

        # 2. Difficulty distribution
        results['difficulty_distribution'] = self._analyze_difficulty_distribution()

        # 3. Blind spots
        results['blind_spots'] = self._analyze_blind_spots()

        # 4. Performance by region
        results['performance_by_region'] = self._analyze_performance_by_region()

        # 5. Comparison metrics (for comparing with curriculum)
        results['comparison_metrics'] = self._compute_comparison_metrics()

        # 6. Chance baseline for prediction losses
        results['chance_baseline'] = self._compute_chance_baseline()

        # 7. Summary
        results['summary'] = self._compute_summary()

        self._results = results
        return results

    def _analyze_coverage(self) -> Dict[str, Any]:
        """Analyze coverage of feature space."""
        features = np.stack(self._data.feature_vectors)

        # Create grid
        n_dims = features.shape[1]
        grid = np.zeros([self.n_grid_bins] * n_dims)

        # Count samples per cell
        for f in features:
            # Convert to grid indices
            indices = tuple(
                min(int(f[d] * self.n_grid_bins), self.n_grid_bins - 1)
                for d in range(n_dims)
            )
            grid[indices] += 1

        # Coverage: fraction of cells with at least one sample
        cells_covered = np.sum(grid > 0)
        total_cells = grid.size
        coverage_ratio = cells_covered / total_cells

        # Evenness: how uniform is the distribution
        # Using coefficient of variation (lower = more uniform)
        non_empty = grid[grid > 0]
        if len(non_empty) > 1:
            cv = np.std(non_empty) / (np.mean(non_empty) + 1e-8)
        else:
            cv = 0.0

        return {
            'coverage_ratio': float(coverage_ratio),
            'cells_covered': int(cells_covered),
            'total_cells': int(total_cells),
            'n_grid_bins': self.n_grid_bins,
            'distribution_cv': float(cv),
            'uniformity_score': float(1.0 / (1.0 + cv)),  # Higher = more uniform
            'interpretation': (
                f'DR covers {coverage_ratio:.1%} of feature space '
                f'with uniformity score {1.0 / (1.0 + cv):.2f}'
            ),
        }

    def _analyze_difficulty_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of difficulty levels."""
        wall_densities = np.array(self._data.wall_densities)
        path_lengths = np.array(self._data.path_lengths)
        path_lengths = path_lengths[path_lengths > 0]  # Only solvable

        # Wall density histogram
        density_hist, density_edges = np.histogram(
            wall_densities, bins=10, range=(0, 0.4)
        )

        # Path length histogram
        if len(path_lengths) > 0:
            path_hist, path_edges = np.histogram(
                path_lengths, bins=10, range=(0, 30)
            )
        else:
            path_hist = np.zeros(10)
            path_edges = np.linspace(0, 30, 11)

        # Check for uniformity (chi-squared test)
        from scipy import stats
        expected = len(wall_densities) / 10

        chi2_density, p_density = stats.chisquare(density_hist)

        return {
            'mean_wall_density': float(np.mean(wall_densities)),
            'std_wall_density': float(np.std(wall_densities)),
            'mean_path_length': float(np.mean(path_lengths)) if len(path_lengths) > 0 else 0.0,
            'std_path_length': float(np.std(path_lengths)) if len(path_lengths) > 0 else 0.0,
            'density_histogram': density_hist.tolist(),
            'density_edges': density_edges.tolist(),
            'path_histogram': path_hist.tolist(),
            'path_edges': path_edges.tolist(),
            'density_chi2': float(chi2_density),
            'density_uniformity_p': float(p_density),
            'is_uniform_density': p_density > 0.05,
        }

    def _analyze_blind_spots(self) -> Dict[str, Any]:
        """Identify undersampled regions."""
        features = np.stack(self._data.feature_vectors)

        # Create grid
        n_dims = features.shape[1]
        grid = np.zeros([self.n_grid_bins] * n_dims)

        for f in features:
            indices = tuple(
                min(int(f[d] * self.n_grid_bins), self.n_grid_bins - 1)
                for d in range(n_dims)
            )
            grid[indices] += 1

        # Expected samples per cell (uniform)
        expected_per_cell = len(features) / grid.size

        # Find undersampled cells (< 20% of expected)
        threshold = expected_per_cell * 0.2
        undersampled = grid < threshold

        # Identify blind spot regions
        blind_spots = []
        for idx in np.ndindex(grid.shape):
            if undersampled[idx]:
                # Convert grid index to feature range
                feature_ranges = []
                for d, i in enumerate(idx):
                    low = i / self.n_grid_bins
                    high = (i + 1) / self.n_grid_bins
                    feature_ranges.append((low, high))

                blind_spots.append({
                    'grid_index': idx,
                    'density_range': feature_ranges[0] if len(feature_ranges) > 0 else None,
                    'distance_range': feature_ranges[1] if len(feature_ranges) > 1 else None,
                    'samples': int(grid[idx]),
                })

        # Categorize blind spots
        high_difficulty_blind_spots = [
            bs for bs in blind_spots
            if bs.get('density_range', (0, 0))[0] > 0.25
        ]
        far_distance_blind_spots = [
            bs for bs in blind_spots
            if bs.get('distance_range', (0, 0))[0] > 0.7
        ]

        return {
            'n_blind_spots': len(blind_spots),
            'fraction_undersampled': float(np.sum(undersampled) / grid.size),
            'n_high_difficulty_blind_spots': len(high_difficulty_blind_spots),
            'n_far_distance_blind_spots': len(far_distance_blind_spots),
            'blind_spot_regions': blind_spots[:10],  # Top 10 for brevity
            'interpretation': (
                f'Found {len(blind_spots)} undersampled regions '
                f'({len(high_difficulty_blind_spots)} in high difficulty, '
                f'{len(far_distance_blind_spots)} in far distance)'
            ),
        }

    def _analyze_performance_by_region(self) -> Dict[str, Any]:
        """Analyze how agent performs in different regions."""
        wall_densities = np.array(self._data.wall_densities)
        episode_returns = np.array(self._data.episode_returns)
        episode_solved = np.array(self._data.episode_solved)
        pred_losses = np.array(self._data.prediction_losses)

        # Performance by difficulty tercile
        density_terciles = np.percentile(wall_densities, [33, 66])
        regions = {}

        for label, (low, high) in [
            ('easy', (0, density_terciles[0])),
            ('medium', (density_terciles[0], density_terciles[1])),
            ('hard', (density_terciles[1], 1.0)),
        ]:
            mask = (wall_densities >= low) & (wall_densities < high)
            if high == 1.0:  # Include upper bound for last bin
                mask = (wall_densities >= low) & (wall_densities <= high)

            if mask.sum() > 0:
                regions[label] = {
                    'n_samples': int(mask.sum()),
                    'solve_rate': float(np.mean(episode_solved[mask])),
                    'mean_return': float(np.mean(episode_returns[mask])),
                    'mean_pred_loss': float(np.mean(pred_losses[mask])),
                    'density_range': f'{low:.2f}-{high:.2f}',
                }

        return regions

    def _compute_comparison_metrics(self) -> Dict[str, Any]:
        """Compute metrics for comparison with curriculum methods."""
        wall_densities = np.array(self._data.wall_densities)
        episode_solved = np.array(self._data.episode_solved)
        pred_losses = np.array(self._data.prediction_losses)

        # Metrics that can be compared across methods
        return {
            'overall_solve_rate': float(np.mean(episode_solved)),
            'mean_difficulty': float(np.mean(wall_densities)),
            'difficulty_std': float(np.std(wall_densities)),
            'mean_prediction_loss': float(np.mean(pred_losses)),
            'prediction_loss_std': float(np.std(pred_losses)),
            'solvable_rate': float(np.mean(self._data.is_solvable)),
            'n_levels_sampled': len(self._data.wall_densities),
        }

    @staticmethod
    def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
        """Correlation coefficient safe against zero-variance inputs."""
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if np.isfinite(c) else 0.0

    def _compute_chance_baseline(self, n_permutations: int = 100) -> Dict[str, Any]:
        """Compute chance baseline for prediction losses via permutation.

        Shuffles level-to-prediction-loss mappings to estimate what loss
        would look like if the agent's representations had no information
        about level features. Compares actual losses to this baseline to
        determine which prediction tiers are above chance.
        """
        pred_losses = np.array(self._data.prediction_losses)
        wall_densities = np.array(self._data.wall_densities)
        goal_distances = np.array(self._data.goal_distances)
        returns = np.array(self._data.episode_returns)

        actual_mean_loss = float(np.mean(pred_losses))

        # Permutation baseline: shuffle feature-loss correspondence
        rng_perm = np.random.default_rng(seed=42)
        permuted_correlations = []
        for _ in range(n_permutations):
            shuffled_losses = rng_perm.permutation(pred_losses)
            # Correlation between loss and features under permutation
            corr_density = self._safe_corrcoef(wall_densities, shuffled_losses)
            corr_distance = self._safe_corrcoef(goal_distances, shuffled_losses)
            corr_return = self._safe_corrcoef(returns, shuffled_losses)
            permuted_correlations.append({
                'density': corr_density,
                'distance': corr_distance,
                'return': corr_return,
            })

        # Actual correlations
        actual_corr_density = self._safe_corrcoef(wall_densities, pred_losses)
        actual_corr_distance = self._safe_corrcoef(goal_distances, pred_losses)
        actual_corr_return = self._safe_corrcoef(returns, pred_losses)

        # Compute p-values: fraction of permuted correlations >= actual
        perm_density = np.array([p['density'] for p in permuted_correlations])
        perm_distance = np.array([p['distance'] for p in permuted_correlations])
        perm_return = np.array([p['return'] for p in permuted_correlations])

        p_density = float(np.mean(np.abs(perm_density) >= np.abs(actual_corr_density)))
        p_distance = float(np.mean(np.abs(perm_distance) >= np.abs(actual_corr_distance)))
        p_return = float(np.mean(np.abs(perm_return) >= np.abs(actual_corr_return)))

        # Determine which tiers are above chance (p < 0.05)
        tier_results = {
            'tier0_env_vars': {
                'feature': 'wall_density',
                'actual_correlation': actual_corr_density,
                'permutation_mean': float(np.mean(np.abs(perm_density))),
                'permutation_std': float(np.std(np.abs(perm_density))),
                'p_value': p_density,
                'above_chance': p_density < 0.05,
            },
            'tier1_curriculum': {
                'feature': 'goal_distance',
                'actual_correlation': actual_corr_distance,
                'permutation_mean': float(np.mean(np.abs(perm_distance))),
                'permutation_std': float(np.std(np.abs(perm_distance))),
                'p_value': p_distance,
                'above_chance': p_distance < 0.05,
            },
            'tier2_agent_curriculum': {
                'feature': 'episode_return',
                'actual_correlation': actual_corr_return,
                'permutation_mean': float(np.mean(np.abs(perm_return))),
                'permutation_std': float(np.std(np.abs(perm_return))),
                'p_value': p_return,
                'above_chance': p_return < 0.05,
            },
        }

        return {
            'actual_mean_loss': actual_mean_loss,
            'n_permutations': n_permutations,
            'tier_results': tier_results,
            'interpretation': (
                f"Tiers above chance: "
                f"{'T0' if tier_results['tier0_env_vars']['above_chance'] else ''} "
                f"{'T1' if tier_results['tier1_curriculum']['above_chance'] else ''} "
                f"{'T2' if tier_results['tier2_agent_curriculum']['above_chance'] else ''}"
            ).strip(),
        }

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        return {
            'n_levels': len(self._data.wall_densities),
            'training_method': self.training_method,
            'is_dr_method': self.training_method == 'dr',
            'solvable_rate': float(np.mean(self._data.is_solvable)),
            'mean_wall_density': float(np.mean(self._data.wall_densities)),
            'solve_rate': float(np.mean(self._data.episode_solved)),
        }

    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data."""
        if not self._results or 'error' in self._results:
            return self._results

        viz_data = {
            'coverage': self._results.get('coverage', {}),
            'difficulty_distribution': self._results.get('difficulty_distribution', {}),
            'performance_by_region': self._results.get('performance_by_region', {}),
            'training_method': self.training_method,
        }

        # Scatter data for 2D coverage plot
        if self._data:
            viz_data['scatter_data'] = {
                'wall_densities': self._data.wall_densities,
                'goal_distances': self._data.goal_distances,
                'episode_solved': [int(s) for s in self._data.episode_solved],
            }

        return viz_data
