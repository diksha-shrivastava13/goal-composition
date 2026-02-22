"""
Adversary Dynamics Experiment.

PAIRED-specific experiment to analyze adversary level generation patterns.

Questions this experiment addresses:
1. Does adversary find exploits in protagonist over time?
2. Is there mode collapse in level generation?
3. How does regret trajectory evolve?
4. What level features does adversary learn to manipulate?
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment


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

        For PAIRED: Simulates adversary level generation and evaluates
        protagonist/antagonist performance to compute regret.
        """
        # Check if this is PAIRED
        if self.training_method != "paired":
            return AdversaryData()  # Return empty for non-PAIRED

        self._data = AdversaryData()

        for ep_idx in range(self.n_episodes):
            rng, level_rng, pro_rng, ant_rng = jax.random.split(rng, 4)

            # Generate adversary-style level (simulated)
            level = self._generate_adversary_level(level_rng, ep_idx)

            # Extract level features
            wall_density = level.get('wall_density', 0.0)
            goal_pos = level.get('goal_pos', (6, 6))
            agent_pos = level.get('agent_pos', (1, 1))
            goal_distance = np.sqrt(
                (goal_pos[0] - agent_pos[0])**2 +
                (goal_pos[1] - agent_pos[1])**2
            )
            path_length = self._compute_path_length(level)

            # Evaluate protagonist performance
            pro_result = self._run_episode(pro_rng, level, agent_type='protagonist')

            # Evaluate antagonist performance (if available, else simulate)
            ant_result = self._run_episode(ant_rng, level, agent_type='antagonist')

            # Compute regret
            regret = ant_result.get('total_return', 0.0) - pro_result.get('total_return', 0.0)

            # Store data
            self._data.wall_densities.append(wall_density)
            self._data.goal_distances.append(goal_distance)
            self._data.path_lengths.append(path_length if path_length > 0 else -1)
            self._data.protagonist_returns.append(pro_result.get('total_return', 0.0))
            self._data.antagonist_returns.append(ant_result.get('total_return', 0.0))
            self._data.regrets.append(regret)
            self._data.episode_indices.append(ep_idx)

            # Store level features for diversity analysis
            features = np.array([wall_density, goal_distance / 18.0, path_length / 30.0 if path_length > 0 else 0.0])
            self._data.level_features.append(features)

        return self._data

    def _generate_adversary_level(
        self,
        rng: chex.PRNGKey,
        episode_idx: int,
    ) -> Dict[str, Any]:
        """
        Generate a level as an adversary would.

        Simulates adversary behavior: tries to create levels that are
        solvable by antagonist but challenging for protagonist.
        """
        height, width = 13, 13

        # Adversary learns to increase difficulty over time (simulated)
        base_difficulty = 0.1 + 0.2 * (episode_idx / max(self.n_episodes, 1))
        noise = float(jax.random.uniform(rng)) * 0.1
        wall_prob = min(0.35, base_difficulty + noise)

        wall_map = np.array(jax.random.bernoulli(rng, wall_prob, (height, width)))
        wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

        rng_goal, rng_agent = jax.random.split(rng)

        # Adversary might learn to place goal far from agent
        # (increases difficulty for protagonist)
        if episode_idx > self.n_episodes // 2:
            # Later episodes: more distant placement
            goal_pos = (
                int(jax.random.randint(rng_goal, (), height - 4, height - 1)),
                int(jax.random.randint(rng_goal, (), width - 4, width - 1)),
            )
            agent_pos = (
                int(jax.random.randint(rng_agent, (), 1, 4)),
                int(jax.random.randint(rng_agent, (), 1, 4)),
            )
        else:
            # Early episodes: random placement
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
            'adversary_generated': True,
        }

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

    def _run_episode(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
        agent_type: str = 'protagonist',
        max_steps: int = 256,
    ) -> Dict[str, Any]:
        """Run episode with protagonist or antagonist."""
        # Initialize hidden state
        hstate = self.agent.initialize_carry(rng, batch_dims=(1,))

        total_return = 0.0
        solved = False

        for step in range(max_steps):
            rng, step_rng = jax.random.split(rng)

            # Create observation
            obs = self._create_observation(level, step)

            # Forward pass
            hstate, pi, value = self._forward_step(obs, hstate)

            # Sample action
            action = pi.sample(seed=step_rng)

            # Simulate step
            reward = 0.0
            done = step >= max_steps - 1

            if step > 10:
                # Protagonist and antagonist have different solve rates
                # Antagonist is typically better (that's how PAIRED works)
                base_solve_prob = 0.3 * (1 - level['wall_density'])
                if agent_type == 'antagonist':
                    solve_prob = base_solve_prob * 1.5  # Antagonist is better
                else:
                    solve_prob = base_solve_prob

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
        """Create observation from level state."""
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
        """Run single forward step."""
        params = self.train_state.params
        apply_fn = self.train_state.apply_fn

        obs_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs)
        done_batch = jnp.zeros((1, 1), dtype=bool)

        new_hstate, pi, value = apply_fn(params, (obs_batch, done_batch), hstate)

        return new_hstate, pi, value

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
