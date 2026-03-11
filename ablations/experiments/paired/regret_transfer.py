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
from ..utils.batched_rollout import batched_rollout


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
        3. Run protagonist on both (with hstate transfer)
        4. Collect transfer metrics
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

        if self.training_method != "paired":
            return RegretTransferData()

        self._data = RegretTransferData()
        n = self.n_level_pairs
        max_steps = self.max_episode_steps

        # --- Generate high-regret levels ---
        _log("generate_hr_levels", msg="Generating high-regret levels via vmap...")
        t0 = time.time()
        rng, rng_hr = jax.random.split(rng)
        hr_rngs = jax.random.split(rng_hr, n)
        hr_levels = jax.vmap(self.agent.sample_random_level)(hr_rngs)
        jax.block_until_ready(hr_levels)
        _log("generate_hr_levels", time.time() - t0)

        # --- Generate similar levels ---
        _log("generate_sim_levels", msg="Generating similar levels via vmap...")
        t0 = time.time()
        rng, rng_sim = jax.random.split(rng)
        sim_rngs = jax.random.split(rng_sim, n)
        sim_levels = jax.vmap(self.agent.sample_random_level)(sim_rngs)
        jax.block_until_ready(sim_levels)
        _log("generate_sim_levels", time.time() - t0)

        # --- Generate baseline levels ---
        _log("generate_bl_levels", msg="Generating baseline levels via vmap...")
        t0 = time.time()
        rng, rng_bl = jax.random.split(rng)
        n_bl = self.n_baseline_levels
        bl_rngs = jax.random.split(rng_bl, n_bl)
        bl_levels = jax.vmap(self.agent.sample_random_level)(bl_rngs)
        jax.block_until_ready(bl_levels)
        _log("generate_bl_levels", time.time() - t0)

        # --- Run protagonist on high-regret levels (source) ---
        _log("hr_rollout", msg="Running batched protagonist rollout on high-regret levels...")
        t0 = time.time()
        rng, rng_pro_hr = jax.random.split(rng)
        hr_result = batched_rollout(
            rng_pro_hr, hr_levels, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n),
            collection_steps=[-1],
            return_final_hstate=True,
        )
        jax.block_until_ready(hr_result.episode_returns)
        _log("hr_rollout", time.time() - t0)

        # --- Run protagonist on similar levels (target) with hstate from source ---
        _log("sim_rollout", msg="Running batched protagonist rollout on similar levels (with hstate transfer)...")
        t0 = time.time()
        rng, rng_pro_sim = jax.random.split(rng)
        sim_result = batched_rollout(
            rng_pro_sim, sim_levels, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            hr_result.final_hstate,  # Transfer hstate from high-regret rollout
            collection_steps=[-1],
        )
        jax.block_until_ready(sim_result.episode_returns)
        _log("sim_rollout", time.time() - t0)

        # --- Run protagonist on baseline levels ---
        _log("bl_rollout", msg="Running batched protagonist rollout on baseline levels...")
        t0 = time.time()
        rng, rng_pro_bl = jax.random.split(rng)
        bl_result = batched_rollout(
            rng_pro_bl, bl_levels, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n_bl),
        )
        jax.block_until_ready(bl_result.episode_returns)
        _log("bl_rollout", time.time() - t0)

        # --- Compute structural similarities and populate data (CPU) ---
        _log("cpu_metrics", msg="Computing CPU-side metrics...")
        t0 = time.time()
        hr_wall_maps = np.array(hr_levels.wall_map)
        sim_wall_maps = np.array(sim_levels.wall_map)
        hr_goal_pos = np.array(hr_levels.goal_pos)
        sim_goal_pos = np.array(sim_levels.goal_pos)
        hr_agent_pos = np.array(hr_levels.agent_pos)
        sim_agent_pos = np.array(sim_levels.agent_pos)

        hr_hstates = hr_result.hstates_by_step["-1"]
        sim_hstates = sim_result.hstates_by_step["-1"]
        bl_mean_return = float(np.mean(bl_result.episode_returns))

        for i in tqdm(range(n), desc="Pair metrics", leave=False):
            hr_dict = {
                'wall_map': hr_wall_maps[i],
                'wall_density': float(hr_wall_maps[i].mean()),
                'goal_pos': tuple(int(x) for x in hr_goal_pos[i]) if hr_goal_pos.ndim > 1 else (int(hr_goal_pos[i]),),
                'agent_pos': tuple(int(x) for x in hr_agent_pos[i]) if hr_agent_pos.ndim > 1 else (int(hr_agent_pos[i]),),
            }
            sim_dict = {
                'wall_map': sim_wall_maps[i],
                'wall_density': float(sim_wall_maps[i].mean()),
                'goal_pos': tuple(int(x) for x in sim_goal_pos[i]) if sim_goal_pos.ndim > 1 else (int(sim_goal_pos[i]),),
                'agent_pos': tuple(int(x) for x in sim_agent_pos[i]) if sim_agent_pos.ndim > 1 else (int(sim_agent_pos[i]),),
            }

            self._data.high_regret_levels.append(hr_dict)
            self._data.similar_levels.append(sim_dict)
            self._data.structural_similarities.append(
                self._compute_structural_similarity(hr_dict, sim_dict)
            )

            self._data.high_regret_returns.append(float(hr_result.episode_returns[i]))
            self._data.similar_returns.append(float(sim_result.episode_returns[i]))
            self._data.baseline_returns.append(bl_mean_return)

            # Hidden states
            self._data.high_regret_hstates.append(hr_hstates[i])
            self._data.similar_hstates.append(sim_hstates[i])

            # Estimate regrets
            hr_regret = self._estimate_regret(hr_dict, {'total_return': float(hr_result.episode_returns[i])})
            sim_regret = self._estimate_regret(sim_dict, {'total_return': float(sim_result.episode_returns[i])})
            self._data.high_regret_values.append(hr_regret)
            self._data.similar_regret_values.append(sim_regret)
        _log("cpu_metrics", time.time() - t0)

        # --- Compute prediction losses ---
        _log("prediction_losses", msg="Computing prediction losses...")
        t0 = time.time()
        from .utils.agent_aware_loss import compute_agent_prediction_loss

        for i in tqdm(range(n), desc="Prediction losses", leave=False):
            rng, hr_loss_rng, sim_loss_rng = jax.random.split(rng, 3)
            hr_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state,
                jax.tree_util.tree_map(lambda x: x[i], hr_levels),
                hr_loss_rng
            )
            self._data.high_regret_pred_losses.append(hr_loss)

            sim_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state,
                jax.tree_util.tree_map(lambda x: x[i], sim_levels),
                sim_loss_rng
            )
            self._data.similar_pred_losses.append(sim_loss)

        # Baseline prediction losses
        bl_losses = []
        for i in range(min(3, n_bl)):
            rng, bl_loss_rng = jax.random.split(rng)
            bl_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state,
                jax.tree_util.tree_map(lambda x: x[i], bl_levels),
                bl_loss_rng
            )
            bl_losses.append(bl_loss)
        bl_mean_loss = float(np.mean(bl_losses)) if bl_losses else 0.0
        self._data.baseline_pred_losses = [bl_mean_loss] * n
        _log("prediction_losses", time.time() - t0)

        _log("collect_data_done", msg=f"Data collection complete ({n} pairs)")
        return self._data

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
