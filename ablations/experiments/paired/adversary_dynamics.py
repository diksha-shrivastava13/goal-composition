"""
Adversary Dynamics Experiment.

PAIRED-specific experiment to analyze adversary level generation patterns.

Questions this experiment addresses:
1. Does adversary find exploits in protagonist over time?
2. Is there mode collapse in level generation?
3. How does regret trajectory evolve?
4. What level features does adversary learn to manipulate?
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import jax
import chex

from ..base import CheckpointExperiment
from ..utils.batched_rollout import batched_rollout


@dataclass
class AdversaryData:
    """Container for adversary dynamics data."""
    # Level characteristics over time
    wall_densities: List[float] = field(default_factory=list)
    goal_distances: List[float] = field(default_factory=list)
    path_lengths: List[float] = field(default_factory=list)

    # Performance metrics
    protagonist_returns: List[float] = field(default_factory=list)
    antagonist_returns: List[float] = field(default_factory=list)
    regrets: List[float] = field(default_factory=list)

    # Level diversity metrics
    level_features: List[np.ndarray] = field(default_factory=list)

    # Timestamps
    episode_indices: List[int] = field(default_factory=list)


class AdversaryDynamicsExperiment(CheckpointExperiment):
    """
    PAIRED-specific experiment to analyze adversary behavior evolution.

    Analyzes:
    - Difficulty trajectory: Does adversary generate increasingly hard levels?
    - Mode collapse: Does adversary get stuck generating similar levels?
    - Regret efficiency: How much regret does adversary extract per update?
    - Exploit discovery: Does adversary find specific weaknesses?

    Note: This experiment is only meaningful for PAIRED training.
    For other methods, it will return a warning.
    """

    @property
    def name(self) -> str:
        return "adversary_dynamics"

    def __init__(
        self,
        n_episodes: int = 200,
        window_size: int = 20,
        **kwargs,
    ):
        """
        Initialize adversary dynamics experiment.

        Args:
            n_episodes: Number of episodes to analyze
            window_size: Rolling window for trend analysis
        """
        super().__init__(**kwargs)
        self.n_episodes = n_episodes
        self.window_size = window_size

        self._data: Optional[AdversaryData] = None
        self._results: Dict[str, Any] = {}

    def collect_data(self, rng: chex.PRNGKey) -> AdversaryData:
        """
        Collect adversary-generated levels and performance data.

        For PAIRED: Generates levels in batch via vmap, then runs batched
        protagonist (and optionally antagonist) rollouts to compute regret.
        """
        import time, logging
        from tqdm import tqdm
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

        # Check if this is PAIRED
        if self.training_method != "paired":
            return AdversaryData()  # Return empty for non-PAIRED

        self._data = AdversaryData()
        n = self.n_episodes
        max_steps = 256

        # --- Generate all levels at once ---
        _log("generate_levels", msg="Generating levels via vmap...")
        t0 = time.time()
        rng, rng_levels = jax.random.split(rng)
        level_rngs = jax.random.split(rng_levels, n)
        levels = jax.vmap(self.agent.sample_random_level)(level_rngs)
        jax.block_until_ready(levels)
        _log("generate_levels", time.time() - t0, "Level generation complete")

        # --- Extract level features from Level pytrees ---
        _log("extract_features", msg="Extracting level features...")
        t0 = time.time()
        wall_maps = np.array(levels.wall_map)
        goal_positions = np.array(levels.goal_pos)
        agent_positions = np.array(levels.agent_pos)
        wall_density = wall_maps.mean(axis=(1, 2))

        goal_distances = np.sqrt(
            np.sum((goal_positions - agent_positions) ** 2, axis=-1)
        ) if goal_positions.ndim > 1 else np.sqrt(
            (goal_positions - agent_positions) ** 2
        )
        _log("extract_features", time.time() - t0)

        # --- Batched protagonist rollout ---
        _log("protagonist_rollout", msg="Running batched protagonist rollout...")
        t0 = time.time()
        rng, rng_pro = jax.random.split(rng)
        pro_result = batched_rollout(
            rng_pro, levels, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n),
            collection_steps=[-1],
        )
        jax.block_until_ready(pro_result.episode_returns)
        _log("protagonist_rollout", time.time() - t0)

        # --- Batched antagonist rollout (PAIRED bilateral) ---
        ant_train_state = getattr(self.train_state, 'ant_train_state', None)
        if ant_train_state is not None:
            _log("antagonist_rollout", msg="Running batched antagonist rollout...")
            t0 = time.time()
            rng, rng_ant = jax.random.split(rng)
            ant_result = batched_rollout(
                rng_ant, levels, max_steps,
                ant_train_state.apply_fn, ant_train_state.params,
                self.agent.env, self.agent.env_params,
                self.agent.initialize_hidden_state(n),
                collection_steps=[-1],
            )
            jax.block_until_ready(ant_result.episode_returns)
            _log("antagonist_rollout", time.time() - t0)
            ant_returns = np.array(ant_result.episode_returns)
        else:
            # No separate antagonist; use protagonist returns as stand-in
            ant_returns = np.array(pro_result.episode_returns)

        pro_returns = np.array(pro_result.episode_returns)
        regrets = ant_returns - pro_returns

        # --- CPU-side path length computation ---
        _log("path_lengths", msg="Computing BFS path lengths (CPU)...")
        t0 = time.time()
        path_lengths = np.empty(n, dtype=np.float64)
        for i in tqdm(range(n), desc="BFS path lengths", leave=False):
            level_dict = {
                'wall_map': wall_maps[i],
                'goal_pos': tuple(int(x) for x in goal_positions[i])
                    if goal_positions.ndim > 1
                    else (int(goal_positions[i]),),
                'agent_pos': tuple(int(x) for x in agent_positions[i])
                    if agent_positions.ndim > 1
                    else (int(agent_positions[i]),),
            }
            pl = self._compute_path_length(level_dict)
            path_lengths[i] = pl if pl > 0 else -1
        _log("path_lengths", time.time() - t0)

        # --- Populate AdversaryData ---
        self._data.wall_densities = wall_density.tolist()
        self._data.goal_distances = goal_distances.tolist() if hasattr(goal_distances, 'tolist') else [float(goal_distances)]
        self._data.path_lengths = path_lengths.tolist()
        self._data.protagonist_returns = pro_returns.tolist()
        self._data.antagonist_returns = ant_returns.tolist()
        self._data.regrets = regrets.tolist()
        self._data.episode_indices = list(range(n))

        # Level features for diversity analysis
        safe_path = np.where(path_lengths > 0, path_lengths / 30.0, 0.0)
        goal_dist_arr = np.array(self._data.goal_distances)
        features = np.stack([
            wall_density,
            goal_dist_arr / 18.0,
            safe_path,
        ], axis=-1)
        self._data.level_features = [features[i] for i in range(n)]

        _log("collect_data_done", msg=f"Data collection complete ({n} episodes)")
        return self._data

    def _compute_path_length(self, level: Dict[str, Any]) -> int:
        """Compute BFS path length from agent to goal."""
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
        Analyze adversary dynamics.

        Computes:
        - Difficulty trajectory: trend in level difficulty over time
        - Mode collapse metric: diversity of generated levels
        - Regret efficiency: regret per level over time
        - Exploit discovery: correlation between level features and regret
        """
        if self.training_method != "paired":
            return {
                'error': f'AdversaryDynamicsExperiment only applies to PAIRED training. '
                         f'Current method: {self.training_method}',
                'training_method': self.training_method,
            }

        if self._data is None or len(self._data.episode_indices) == 0:
            return {'error': 'No data collected'}

        results = {
            'training_method': self.training_method,
        }

        # 1. Difficulty trajectory
        results['difficulty_trajectory'] = self._analyze_difficulty_trajectory()

        # 2. Mode collapse
        results['mode_collapse'] = self._analyze_mode_collapse()

        # 3. Regret efficiency
        results['regret_efficiency'] = self._analyze_regret_efficiency()

        # 4. Exploit discovery
        results['exploit_discovery'] = self._analyze_exploit_discovery()

        # 5. Summary statistics
        results['summary'] = self._compute_summary()

        self._results = results
        return results

    def _analyze_difficulty_trajectory(self) -> Dict[str, Any]:
        """Analyze how difficulty evolves over training."""
        wall_densities = np.array(self._data.wall_densities)
        goal_distances = np.array(self._data.goal_distances)

        # Compute rolling statistics
        n = len(wall_densities)
        if n < self.window_size:
            return {'error': 'Insufficient data for trajectory analysis'}

        # Rolling mean difficulty
        rolling_density = np.convolve(
            wall_densities,
            np.ones(self.window_size) / self.window_size,
            mode='valid'
        )

        # Trend analysis (linear regression)
        x = np.arange(len(rolling_density))
        slope, intercept = np.polyfit(x, rolling_density, 1)

        return {
            'initial_difficulty': float(np.mean(wall_densities[:self.window_size])),
            'final_difficulty': float(np.mean(wall_densities[-self.window_size:])),
            'difficulty_trend_slope': float(slope),
            'trend_direction': 'increasing' if slope > 0.001 else ('decreasing' if slope < -0.001 else 'stable'),
            'mean_goal_distance': float(np.mean(goal_distances)),
        }

    def _analyze_mode_collapse(self) -> Dict[str, Any]:
        """Analyze level diversity / mode collapse."""
        if len(self._data.level_features) < 10:
            return {'error': 'Insufficient data for mode collapse analysis'}

        features = np.stack(self._data.level_features)

        # Compute pairwise distances between generated levels
        from scipy.spatial.distance import pdist

        distances = pdist(features)
        mean_distance = float(np.mean(distances))
        std_distance = float(np.std(distances))

        # Early vs late diversity
        n = len(features)
        early_features = features[:n // 3]
        late_features = features[-n // 3:]

        early_diversity = float(np.mean(pdist(early_features))) if len(early_features) >= 2 else 0.0
        late_diversity = float(np.mean(pdist(late_features))) if len(late_features) >= 2 else 0.0

        # Mode collapse detection: if diversity decreases significantly
        diversity_ratio = late_diversity / (early_diversity + 1e-6)
        mode_collapse_detected = diversity_ratio < 0.5

        return {
            'mean_level_diversity': mean_distance,
            'std_level_diversity': std_distance,
            'early_diversity': early_diversity,
            'late_diversity': late_diversity,
            'diversity_ratio': float(diversity_ratio),
            'mode_collapse_detected': mode_collapse_detected,
            'interpretation': (
                'Mode collapse detected: adversary generates less diverse levels over time'
                if mode_collapse_detected else
                'No mode collapse: adversary maintains level diversity'
            ),
        }

    def _analyze_regret_efficiency(self) -> Dict[str, Any]:
        """Analyze how efficiently adversary extracts regret."""
        regrets = np.array(self._data.regrets)
        pro_returns = np.array(self._data.protagonist_returns)
        ant_returns = np.array(self._data.antagonist_returns)

        if len(regrets) < self.window_size:
            return {'error': 'Insufficient data for regret analysis'}

        # Rolling regret
        rolling_regret = np.convolve(
            regrets,
            np.ones(self.window_size) / self.window_size,
            mode='valid'
        )

        # Regret trend
        x = np.arange(len(rolling_regret))
        slope, intercept = np.polyfit(x, rolling_regret, 1)

        return {
            'mean_regret': float(np.mean(regrets)),
            'std_regret': float(np.std(regrets)),
            'initial_regret': float(np.mean(regrets[:self.window_size])),
            'final_regret': float(np.mean(regrets[-self.window_size:])),
            'regret_trend_slope': float(slope),
            'mean_protagonist_return': float(np.mean(pro_returns)),
            'mean_antagonist_return': float(np.mean(ant_returns)),
            'antagonist_advantage': float(np.mean(ant_returns) - np.mean(pro_returns)),
        }

    def _analyze_exploit_discovery(self) -> Dict[str, Any]:
        """Analyze if adversary discovers exploits (correlations)."""
        regrets = np.array(self._data.regrets)
        wall_densities = np.array(self._data.wall_densities)
        goal_distances = np.array(self._data.goal_distances)

        if len(regrets) < 10:
            return {'error': 'Insufficient data for exploit analysis'}

        # Correlations between level features and regret
        correlations = {}

        # Wall density vs regret
        corr = np.corrcoef(wall_densities, regrets)[0, 1]
        correlations['wall_density_vs_regret'] = float(corr) if np.isfinite(corr) else 0.0

        # Goal distance vs regret
        corr = np.corrcoef(goal_distances, regrets)[0, 1]
        correlations['goal_distance_vs_regret'] = float(corr) if np.isfinite(corr) else 0.0

        # Detect exploit: strong correlation means adversary found what hurts protagonist
        exploit_threshold = 0.3
        exploits_found = []
        for feature, corr_val in correlations.items():
            if abs(corr_val) > exploit_threshold:
                direction = 'high' if corr_val > 0 else 'low'
                feature_name = feature.split('_vs_')[0]
                exploits_found.append(f'{direction} {feature_name}')

        return {
            'correlations': correlations,
            'exploits_found': exploits_found,
            'n_exploits': len(exploits_found),
            'interpretation': (
                f'Adversary discovered exploits: {", ".join(exploits_found)}'
                if exploits_found else
                'No clear exploits discovered'
            ),
        }

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        regrets = np.array(self._data.regrets)
        wall_densities = np.array(self._data.wall_densities)

        return {
            'n_episodes_analyzed': len(self._data.episode_indices),
            'mean_regret': float(np.mean(regrets)),
            'max_regret': float(np.max(regrets)),
            'mean_wall_density': float(np.mean(wall_densities)),
            'positive_regret_rate': float(np.mean(regrets > 0)),
        }

    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data."""
        if not self._results or 'error' in self._results:
            return self._results

        viz_data = {
            'difficulty_trajectory': self._results.get('difficulty_trajectory', {}),
            'regret_efficiency': self._results.get('regret_efficiency', {}),
            'mode_collapse': self._results.get('mode_collapse', {}),
            'training_method': self.training_method,
        }

        # Time series data for plotting
        viz_data['time_series'] = {
            'episode_indices': self._data.episode_indices,
            'wall_densities': self._data.wall_densities,
            'regrets': self._data.regrets,
        }

        return viz_data
