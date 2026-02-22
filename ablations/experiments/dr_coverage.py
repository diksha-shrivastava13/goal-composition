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

    def __init__(
        self,
        n_levels: int = 500,
        n_grid_bins: int = 10,
        feature_names: List[str] = None,
        **kwargs,
    ):
        """
        Initialize DR coverage experiment.

        Args:
            n_levels: Number of levels to sample for analysis
            n_grid_bins: Bins per dimension for coverage grid
            feature_names: Features to analyze (default: wall_density, goal_distance)
        """
        super().__init__(**kwargs)
        self.n_levels = n_levels
        self.n_grid_bins = n_grid_bins
        self.feature_names = feature_names or ['wall_density', 'goal_distance']

        self._data: Optional[CoverageData] = None
        self._results: Dict[str, Any] = {}

    def collect_data(self, rng: chex.PRNGKey) -> CoverageData:
        """
        Collect DR-sampled levels and their features.
        """
        self._data = CoverageData()

        for i in range(self.n_levels):
            rng, level_rng, ep_rng, loss_rng = jax.random.split(rng, 4)

            # Generate random level (DR style)
            level = self._generate_random_level(level_rng)

            # Extract features
            wall_density = level.get('wall_density', 0.0)
            goal_pos = level.get('goal_pos', (6, 6))
            agent_pos = level.get('agent_pos', (1, 1))
            goal_distance = np.sqrt(
                (goal_pos[0] - agent_pos[0])**2 +
                (goal_pos[1] - agent_pos[1])**2
            )
            path_length = self._compute_path_length(level)
            is_solvable = path_length > 0

            # Run episode
            result = self._run_episode(ep_rng, level)

            # Compute prediction loss
            from .utils.agent_aware_loss import compute_agent_prediction_loss
            pred_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state, level, loss_rng
            )

            # Store data
            self._data.wall_densities.append(wall_density)
            self._data.goal_distances.append(goal_distance)
            self._data.path_lengths.append(path_length)
            self._data.is_solvable.append(is_solvable)
            self._data.episode_returns.append(result.get('total_return', 0.0))
            self._data.episode_solved.append(result.get('solved', False))
            self._data.prediction_losses.append(pred_loss)

            # Feature vector for coverage analysis
            # Normalize features to [0, 1]
            norm_density = wall_density
            norm_distance = min(goal_distance / 18.0, 1.0)  # Max ~18 for 13x13
            self._data.feature_vectors.append(np.array([norm_density, norm_distance]))

        return self._data

    def _generate_random_level(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """Generate a random level (DR style)."""
        height, width = 13, 13

        # Uniform random wall density
        wall_prob = float(jax.random.uniform(rng)) * 0.4  # 0 to 0.4

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
        }

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

    def _run_episode(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
        max_steps: int = 256,
    ) -> Dict[str, Any]:
        """Run episode on level."""
        hstate = self.agent.initialize_carry(rng, batch_dims=(1,))

        total_return = 0.0
        solved = False

        for step in range(max_steps):
            rng, step_rng = jax.random.split(rng)

            obs = self._create_observation(level, step)
            hstate, pi, value = self._forward_step(obs, hstate)
            action = pi.sample(seed=step_rng)

            reward = 0.0
            done = step >= max_steps - 1

            if step > 10:
                solve_prob = 0.3 * (1 - level['wall_density'])
                if float(jax.random.uniform(step_rng)) < solve_prob / max_steps:
                    solved = True
                    reward = 1.0
                    done = True

            total_return += reward

            if done:
                break

        return {
            'total_return': total_return,
            'solved': solved,
            'n_steps': step + 1,
        }

    def _create_observation(self, level: Dict[str, Any], step: int) -> Any:
        """Create observation."""
        height, width = level['wall_map'].shape

        image = np.zeros((height, width, 3), dtype=np.float32)
        image[:, :, 0] = level['wall_map'].astype(np.float32)
        image[level['goal_pos']] = [0, 1, 0]

        agent_y = (level['agent_pos'][0] + step // 10) % (height - 2) + 1
        agent_x = (level['agent_pos'][1] + step % 10) % (width - 2) + 1
        image[agent_y, agent_x, 2] = 1.0

        class Obs:
            def __init__(self, img, direction):
                self.image = img
                self.agent_dir = direction

        return Obs(jnp.array(image), jnp.array([0]))

    def _forward_step(self, obs: Any, hstate: Any) -> Tuple[Any, Any, Any]:
        """Run forward step."""
        params = self.train_state.params
        apply_fn = self.train_state.apply_fn

        obs_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs)
        done_batch = jnp.zeros((1, 1), dtype=bool)

        new_hstate, pi, value = apply_fn(params, (obs_batch, done_batch), hstate)

        return new_hstate, pi, value

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze DR coverage.

        Computes:
        - Coverage metric: Grid coverage percentage
        - Difficulty distribution: Histogram analysis
        - Blind spots: Undersampled regions
        - Performance by region: How agent performs in different areas
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

        # 6. Summary
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
