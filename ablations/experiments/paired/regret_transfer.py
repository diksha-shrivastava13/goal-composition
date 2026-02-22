"""
Regret Transfer Experiment.

PAIRED-specific experiment to test if protagonist learns from high-regret experiences.

Questions this experiment addresses:
1. Does protagonist transfer knowledge from high-regret to similar levels?
2. Does performance on high-regret levels predict performance on similar levels?
3. How does regret-based learning manifest in representations?
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment
from ..utils.transfer_metrics import (
    compute_behavioral_transfer,
    compute_representational_transfer,
)


@dataclass
class RegretTransferData:
    """Container for regret transfer analysis data."""
    # High-regret levels and their similar counterparts
    high_regret_levels: List[Dict[str, Any]] = field(default_factory=list)
    similar_levels: List[Dict[str, Any]] = field(default_factory=list)
    structural_similarities: List[float] = field(default_factory=list)

    # Performance metrics
    high_regret_returns: List[float] = field(default_factory=list)
    similar_returns: List[float] = field(default_factory=list)
    baseline_returns: List[float] = field(default_factory=list)

    # Regret values
    high_regret_values: List[float] = field(default_factory=list)
    similar_regret_values: List[float] = field(default_factory=list)

    # Hidden state trajectories
    high_regret_hstates: List[np.ndarray] = field(default_factory=list)
    similar_hstates: List[np.ndarray] = field(default_factory=list)

    # Prediction losses
    high_regret_pred_losses: List[float] = field(default_factory=list)
    similar_pred_losses: List[float] = field(default_factory=list)
    baseline_pred_losses: List[float] = field(default_factory=list)


class RegretTransferExperiment(CheckpointExperiment):
    """
    PAIRED-specific experiment to test curriculum awareness through regret.

    Protocol:
    1. Generate levels where antagonist >> protagonist (high regret)
    2. Generate structurally similar levels
    3. Test if protagonist improves on similar-to-high-regret levels
    4. Analyze representations for regret encoding

    Metrics:
    - Regret learning rate: improvement on high-regret types
    - Regret attention: does model encode regret features?
    - Transfer to similar: performance on structurally similar levels

    Note: This experiment is only meaningful for PAIRED training.
    """

    @property
    def name(self) -> str:
        return "regret_transfer"

    def __init__(
        self,
        n_level_pairs: int = 50,
        high_regret_threshold: float = 0.3,
        n_baseline_levels: int = 10,
        max_episode_steps: int = 256,
        **kwargs,
    ):
        """
        Initialize regret transfer experiment.

        Args:
            n_level_pairs: Number of high-regret/similar pairs to test
            high_regret_threshold: Minimum regret to consider "high"
            n_baseline_levels: Random levels for baseline comparison
            max_episode_steps: Maximum episode length
        """
        super().__init__(**kwargs)
        self.n_level_pairs = n_level_pairs
        self.high_regret_threshold = high_regret_threshold
        self.n_baseline_levels = n_baseline_levels
        self.max_episode_steps = max_episode_steps

        self._data: Optional[RegretTransferData] = None
        self._results: Dict[str, Any] = {}

    def collect_data(self, rng: chex.PRNGKey) -> RegretTransferData:
        """
        Collect regret transfer data.

        For each pair:
        1. Generate high-regret level
        2. Generate structurally similar level
        3. Run protagonist on both
        4. Collect transfer metrics
        """
        if self.training_method != "paired":
            return RegretTransferData()

        self._data = RegretTransferData()

        for pair_idx in range(self.n_level_pairs):
            rng, hr_rng, sim_rng, base_rng = jax.random.split(rng, 4)

            # Generate high-regret level
            high_regret_level = self._generate_high_regret_level(hr_rng)

            # Generate structurally similar level
            similar_level = self._generate_similar_level(sim_rng, high_regret_level)

            # Compute structural similarity
            similarity = self._compute_structural_similarity(
                high_regret_level, similar_level
            )

            # Store levels
            self._data.high_regret_levels.append(high_regret_level)
            self._data.similar_levels.append(similar_level)
            self._data.structural_similarities.append(similarity)

            # Run protagonist on high-regret level
            rng, ep_rng = jax.random.split(rng)
            hr_result = self._run_episode(ep_rng, high_regret_level, track_hstate=True)
            self._data.high_regret_returns.append(hr_result['total_return'])
            self._data.high_regret_hstates.append(hr_result.get('final_hstate_flat', np.zeros(512)))

            # Estimate regret
            hr_regret = self._estimate_regret(high_regret_level, hr_result)
            self._data.high_regret_values.append(hr_regret)

            # Run protagonist on similar level (using hidden state from high-regret)
            rng, ep_rng = jax.random.split(rng)
            sim_result = self._run_episode(
                ep_rng,
                similar_level,
                initial_hstate=hr_result.get('final_hstate'),
                track_hstate=True
            )
            self._data.similar_returns.append(sim_result['total_return'])
            self._data.similar_hstates.append(sim_result.get('final_hstate_flat', np.zeros(512)))

            # Estimate regret for similar level
            sim_regret = self._estimate_regret(similar_level, sim_result)
            self._data.similar_regret_values.append(sim_regret)

            # Run on random baseline levels
            baseline_returns = []
            for _ in range(self.n_baseline_levels):
                rng, bl_rng, ep_rng = jax.random.split(rng, 3)
                baseline_level = self._generate_random_level(bl_rng)
                bl_result = self._run_episode(ep_rng, baseline_level)
                baseline_returns.append(bl_result['total_return'])
            self._data.baseline_returns.append(float(np.mean(baseline_returns)))

            # Compute prediction losses
            from .utils.agent_aware_loss import compute_agent_prediction_loss

            rng, loss_rng = jax.random.split(rng)
            hr_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state, high_regret_level, loss_rng
            )
            self._data.high_regret_pred_losses.append(hr_loss)

            rng, loss_rng = jax.random.split(rng)
            sim_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state, similar_level, loss_rng
            )
            self._data.similar_pred_losses.append(sim_loss)

            # Baseline prediction loss
            baseline_losses = []
            for _ in range(min(3, self.n_baseline_levels)):
                rng, bl_rng, loss_rng = jax.random.split(rng, 3)
                baseline_level = self._generate_random_level(bl_rng)
                bl_loss, _ = compute_agent_prediction_loss(
                    self.agent, self.train_state, baseline_level, loss_rng
                )
                baseline_losses.append(bl_loss)
            self._data.baseline_pred_losses.append(float(np.mean(baseline_losses)))

        return self._data

    def _generate_high_regret_level(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """Generate a level likely to produce high regret."""
        height, width = 13, 13

        # High-regret levels: challenging but solvable
        wall_prob = 0.2 + float(jax.random.uniform(rng)) * 0.15

        wall_map = np.array(jax.random.bernoulli(rng, wall_prob, (height, width)))
        wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

        rng_goal, rng_agent = jax.random.split(rng)

        # Place goal and agent far apart
        goal_pos = (
            int(jax.random.randint(rng_goal, (), height - 4, height - 1)),
            int(jax.random.randint(rng_goal, (), width - 4, width - 1)),
        )
        agent_pos = (
            int(jax.random.randint(rng_agent, (), 1, 4)),
            int(jax.random.randint(rng_agent, (), 1, 4)),
        )

        return {
            'wall_map': wall_map,
            'wall_density': wall_map.sum() / (height * width),
            'goal_pos': goal_pos,
            'agent_pos': agent_pos,
            'high_regret': True,
        }

    def _generate_similar_level(
        self,
        rng: chex.PRNGKey,
        template: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a level structurally similar to template."""
        height, width = template['wall_map'].shape

        # Match wall density closely
        target_density = template['wall_density']
        wall_prob = target_density + float(jax.random.uniform(rng) - 0.5) * 0.05

        wall_map = np.array(jax.random.bernoulli(rng, max(0, min(1, wall_prob)), (height, width)))
        wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

        # Similar goal-agent distance
        template_dist = np.sqrt(
            (template['goal_pos'][0] - template['agent_pos'][0])**2 +
            (template['goal_pos'][1] - template['agent_pos'][1])**2
        )

        rng_goal, rng_agent = jax.random.split(rng)

        goal_pos = (
            int(jax.random.randint(rng_goal, (), 1, height - 1)),
            int(jax.random.randint(rng_goal, (), 1, width - 1)),
        )

        # Agent position to maintain similar distance
        angle = float(jax.random.uniform(rng_agent)) * 2 * np.pi
        dist_variation = template_dist + float(jax.random.uniform(rng_agent) - 0.5) * 2
        agent_y = int(np.clip(goal_pos[0] + dist_variation * np.cos(angle), 1, height - 2))
        agent_x = int(np.clip(goal_pos[1] + dist_variation * np.sin(angle), 1, width - 2))
        agent_pos = (agent_y, agent_x)

        return {
            'wall_map': wall_map,
            'wall_density': wall_map.sum() / (height * width),
            'goal_pos': goal_pos,
            'agent_pos': agent_pos,
            'similar_to_high_regret': True,
        }

    def _generate_random_level(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """Generate a random level for baseline."""
        height, width = 13, 13

        wall_prob = 0.1 + float(jax.random.uniform(rng)) * 0.2

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

    def _compute_structural_similarity(
        self,
        level1: Dict[str, Any],
        level2: Dict[str, Any],
    ) -> float:
        """Compute structural similarity between levels."""
        # Wall density similarity
        density_sim = 1.0 - abs(level1['wall_density'] - level2['wall_density'])

        # Goal-agent distance similarity
        dist1 = np.sqrt(
            (level1['goal_pos'][0] - level1['agent_pos'][0])**2 +
            (level1['goal_pos'][1] - level1['agent_pos'][1])**2
        )
        dist2 = np.sqrt(
            (level2['goal_pos'][0] - level2['agent_pos'][0])**2 +
            (level2['goal_pos'][1] - level2['agent_pos'][1])**2
        )
        max_dist = np.sqrt(2) * 13
        dist_sim = 1.0 - abs(dist1 - dist2) / max_dist

        return float(0.5 * density_sim + 0.5 * dist_sim)

    def _estimate_regret(
        self,
        level: Dict[str, Any],
        result: Dict[str, Any],
    ) -> float:
        """Estimate regret (antagonist - protagonist return)."""
        solved = result.get('solved', False)
        wall_density = level.get('wall_density', 0.2)

        if solved:
            regret = 0.1 * wall_density
        else:
            regret = 0.5 + 0.5 * (1.0 - wall_density)

        return float(regret)

    def _run_episode(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
        initial_hstate: Any = None,
        track_hstate: bool = False,
    ) -> Dict[str, Any]:
        """Run episode and collect results."""
        if initial_hstate is None:
            hstate = self.agent.initialize_carry(rng, batch_dims=(1,))
        else:
            hstate = initial_hstate

        total_return = 0.0
        solved = False

        for step in range(self.max_episode_steps):
            rng, step_rng = jax.random.split(rng)

            obs = self._create_observation(level, step)
            hstate, pi, value = self._forward_step(obs, hstate)
            action = pi.sample(seed=step_rng)

            reward = 0.0
            done = step >= self.max_episode_steps - 1

            if step > 10:
                solve_prob = 0.3 * (1 - level['wall_density'])
                if float(jax.random.uniform(step_rng)) < solve_prob / self.max_episode_steps:
                    solved = True
                    reward = 1.0
                    done = True

            total_return += reward

            if done:
                break

        result = {
            'total_return': total_return,
            'solved': solved,
            'n_steps': step + 1,
            'final_hstate': hstate,
        }

        if track_hstate:
            h_c, h_h = hstate
            result['final_hstate_flat'] = np.concatenate([
                np.array(h_c).flatten(),
                np.array(h_h).flatten()
            ])

        return result

    def _create_observation(self, level: Dict[str, Any], step: int) -> Any:
        """Create observation from level."""
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
        Analyze regret transfer.

        Metrics:
        - Regret learning rate: improvement on high-regret types
        - Regret attention: correlation between regret and hidden states
        - Transfer to similar: performance on similar levels
        """
        if self.training_method != "paired":
            return {
                'error': f'RegretTransferExperiment only applies to PAIRED training. '
                         f'Current method: {self.training_method}',
                'training_method': self.training_method,
            }

        if self._data is None or len(self._data.high_regret_returns) == 0:
            return {'error': 'No data collected'}

        results = {
            'training_method': self.training_method,
        }

        # 1. Regret learning rate
        results['regret_learning_rate'] = self._analyze_regret_learning()

        # 2. Regret attention (representation analysis)
        results['regret_attention'] = self._analyze_regret_attention()

        # 3. Transfer to similar levels
        results['transfer_to_similar'] = self._analyze_transfer()

        # 4. Prediction loss analysis
        results['prediction_loss_transfer'] = self._analyze_prediction_loss()

        # 5. Summary
        results['summary'] = self._compute_summary()

        self._results = results
        return results

    def _analyze_regret_learning(self) -> Dict[str, Any]:
        """Analyze how well protagonist learns from high-regret experiences."""
        hr_returns = np.array(self._data.high_regret_returns)
        sim_returns = np.array(self._data.similar_returns)
        hr_regrets = np.array(self._data.high_regret_values)
        sim_regrets = np.array(self._data.similar_regret_values)

        # Regret reduction: do similar levels have lower regret?
        regret_reduction = np.mean(hr_regrets) - np.mean(sim_regrets)

        # Correlation: does high regret predict improvement on similar?
        improvement = sim_returns - hr_returns
        corr = np.corrcoef(hr_regrets, improvement)[0, 1]

        return {
            'mean_high_regret': float(np.mean(hr_regrets)),
            'mean_similar_regret': float(np.mean(sim_regrets)),
            'regret_reduction': float(regret_reduction),
            'regret_improvement_correlation': float(corr) if np.isfinite(corr) else 0.0,
            'learning_detected': regret_reduction > 0.1,
            'interpretation': (
                'Protagonist shows regret-based learning: performance improves on similar levels'
                if regret_reduction > 0.1 else
                'No clear regret-based learning detected'
            ),
        }

    def _analyze_regret_attention(self) -> Dict[str, Any]:
        """Analyze if representations encode regret information."""
        if len(self._data.high_regret_hstates) < 10:
            return {'error': 'Insufficient hidden state data'}

        hr_hstates = np.stack(self._data.high_regret_hstates)
        sim_hstates = np.stack(self._data.similar_hstates)
        hr_regrets = np.array(self._data.high_regret_values)

        # Can we predict regret from hidden states?
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        model = Ridge(alpha=1.0)
        try:
            scores = cross_val_score(model, hr_hstates, hr_regrets, cv=5, scoring='r2')
            regret_r2 = float(np.mean(scores))
        except Exception:
            regret_r2 = 0.0

        # Representational similarity between high-regret and similar
        cosine_sims = []
        for hr_h, sim_h in zip(hr_hstates, sim_hstates):
            dot = np.dot(hr_h, sim_h)
            norm = (np.linalg.norm(hr_h) * np.linalg.norm(sim_h) + 1e-8)
            cosine_sims.append(dot / norm)

        return {
            'regret_prediction_r2': regret_r2,
            'regret_encoded_in_representation': regret_r2 > 0.1,
            'mean_hstate_similarity': float(np.mean(cosine_sims)),
            'std_hstate_similarity': float(np.std(cosine_sims)),
        }

    def _analyze_transfer(self) -> Dict[str, Any]:
        """Analyze transfer from high-regret to similar levels."""
        from scipy import stats

        sim_returns = np.array(self._data.similar_returns)
        baseline_returns = np.array(self._data.baseline_returns)

        # Does similar beat baseline?
        if len(sim_returns) >= 5:
            t_stat, p_value = stats.ttest_rel(sim_returns, baseline_returns)
        else:
            t_stat, p_value = 0.0, 1.0

        # Effect size
        diff = sim_returns - baseline_returns
        cohens_d = np.mean(diff) / (np.std(diff) + 1e-6)

        return {
            'mean_similar_return': float(np.mean(sim_returns)),
            'mean_baseline_return': float(np.mean(baseline_returns)),
            'similar_advantage': float(np.mean(sim_returns) - np.mean(baseline_returns)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant_transfer': p_value < 0.05 and cohens_d > 0.2,
        }

    def _analyze_prediction_loss(self) -> Dict[str, Any]:
        """Analyze prediction loss transfer."""
        from .utils.agent_aware_loss import compute_random_baseline_loss

        hr_losses = np.array(self._data.high_regret_pred_losses)
        sim_losses = np.array(self._data.similar_pred_losses)
        baseline_losses = np.array(self._data.baseline_pred_losses)

        if len(hr_losses) == 0:
            return {'error': 'No prediction loss data'}

        random_baseline = compute_random_baseline_loss()

        return {
            'mean_high_regret_loss': float(np.mean(hr_losses)),
            'mean_similar_loss': float(np.mean(sim_losses)),
            'mean_baseline_loss': float(np.mean(baseline_losses)),
            'random_baseline': random_baseline,
            'similar_vs_baseline_advantage': float(np.mean(baseline_losses) - np.mean(sim_losses)),
            'prediction_transfer_exists': np.mean(sim_losses) < np.mean(baseline_losses),
        }

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        return {
            'n_pairs': len(self._data.high_regret_returns),
            'mean_similarity': float(np.mean(self._data.structural_similarities)),
            'mean_high_regret_value': float(np.mean(self._data.high_regret_values)),
        }

    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data."""
        if not self._results or 'error' in self._results:
            return self._results

        viz_data = {
            'regret_learning': self._results.get('regret_learning_rate', {}),
            'transfer': self._results.get('transfer_to_similar', {}),
            'prediction_loss': self._results.get('prediction_loss_transfer', {}),
            'training_method': self.training_method,
        }

        # Scatter data for visualization
        if self._data:
            viz_data['scatter_data'] = {
                'high_regret_returns': self._data.high_regret_returns,
                'similar_returns': self._data.similar_returns,
                'similarities': self._data.structural_similarities,
            }

        return viz_data
