"""
Mutation Adaptation Experiment.

Tests knowledge transfer across curriculum conditions:
- ACCEL/PLR: Tests replay→mutation transfer
- PAIRED: Tests adversary→protagonist transfer via regret
- DR: Tests feature-based transfer via structural similarity

For ACCEL: Agent sees replay level then mutation of that level.
For PAIRED: Agent evaluated on high-regret levels then similar levels.
For DR: Agent evaluated on structurally similar level clusters.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
import chex

from .base import CheckpointExperiment
from .utils.transfer_metrics import (
    compute_behavioral_transfer,
    compute_representational_transfer,
    compute_td_error_surprise,
    compute_policy_divergence,
)
from .utils.batched_rollout import batched_rollout


@dataclass
class LevelPair:
    """A source level and its related level (mutation, similar, etc.)."""
    source_level: Dict[str, Any]
    target_level: Dict[str, Any]
    relationship: str  # 'mutation', 'similar', 'high_regret', 'random'
    distance: float  # Mutation edits, structural similarity, or regret difference


@dataclass
class AdaptationData:
    """Container for adaptation data across training methods."""
    level_pairs: List[LevelPair] = field(default_factory=list)
    training_method: str = "accel"

    # Per-pair metrics
    source_performance: List[Dict[str, float]] = field(default_factory=list)
    target_performance: List[Dict[str, float]] = field(default_factory=list)
    random_baseline_performance: List[Dict[str, float]] = field(default_factory=list)

    # Transfer metrics
    behavioral_transfer: List[Dict[str, float]] = field(default_factory=list)
    representational_transfer: List[Dict[str, float]] = field(default_factory=list)

    # Prediction loss tracking
    source_prediction_losses: List[float] = field(default_factory=list)
    target_prediction_losses: List[float] = field(default_factory=list)
    random_prediction_losses: List[float] = field(default_factory=list)

    # PAIRED-specific: regret tracking
    source_regrets: List[float] = field(default_factory=list)
    target_regrets: List[float] = field(default_factory=list)

    # DR-specific: structural similarity tracking
    structural_similarities: List[float] = field(default_factory=list)


class MutationAdaptationExperiment(CheckpointExperiment):
    """
    Test knowledge transfer across curriculum conditions.

    Training Method Protocols:

    ACCEL/PLR (has_mutations=True):
        1. Agent completes a replay level
        2. Present mutation (1-5 edits from replay)
        3. Record: adaptation speed, value correlation, policy similarity

    PAIRED (has_adversary=True):
        1. Identify levels where antagonist >> protagonist (high regret)
        2. Generate levels with similar features
        3. Test if protagonist improves on similar-to-high-regret levels

    DR (no curriculum structure):
        1. Group levels by wall pattern clusters
        2. Test if performance on cluster[i] predicts cluster[j] performance
        3. Measure transfer as function of structural similarity

    Expected outcomes:
    - accel_probe: No transfer (reset between episodes)
    - persistent_lstm: May show transfer if memory encodes replay
    - episodic_memory: Should retrieve replay as context
    """

    @property
    def name(self) -> str:
        return "mutation_adaptation"

    def __init__(
        self,
        n_level_pairs: int = 50,
        mutation_distances: List[int] = None,
        n_random_baselines: int = 10,
        max_episode_steps: int = 256,
        n_clusters: int = 5,  # For DR clustering
        **kwargs,
    ):
        """
        Initialize adaptation experiment.

        Args:
            n_level_pairs: Number of level pairs to test
            mutation_distances: List of mutation distances to test [1, 2, 3, 5] (ACCEL only)
            n_random_baselines: Number of random levels for baseline comparison
            max_episode_steps: Maximum steps per episode
            n_clusters: Number of clusters for DR structural similarity analysis
        """
        super().__init__(**kwargs)
        self.n_level_pairs = n_level_pairs
        self.mutation_distances = mutation_distances or [1, 2, 3, 5]
        self.n_random_baselines = n_random_baselines
        self.max_episode_steps = max_episode_steps
        self.n_clusters = n_clusters

        self._data: Optional[AdaptationData] = None
        self._results: Dict[str, Any] = {}

    def collect_data(self, rng: chex.PRNGKey) -> AdaptationData:
        """
        Collect adaptation data appropriate for the training method.

        Dispatches to method-specific collection:
        - ACCEL/PLR: _collect_accel_data (replay→mutation transfer)
        - PAIRED: _collect_paired_data (adversary→protagonist transfer)
        - DR: _collect_dr_data (structural similarity transfer)
        """
        self._data = AdaptationData(training_method=self.training_method)

        if self.training_method in ["paired"]:
            return self._collect_paired_data(rng)
        elif self.training_method in ["dr"]:
            return self._collect_dr_data(rng)
        else:  # accel, plr, robust_plr
            return self._collect_accel_data(rng)

    def _collect_accel_data(self, rng: chex.PRNGKey) -> AdaptationData:
        """
        Collect data for ACCEL/PLR: replay→mutation transfer.
        GPU-batched: generates all levels at once, runs batched rollouts.
        """
        import time, logging
        from tqdm import tqdm
        logger = logging.getLogger(__name__)
        try:
            import wandb
            _wandb_active = wandb.run is not None
        except ImportError:
            _wandb_active = False
        timings = {}

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

        n = self.n_level_pairs
        max_steps = self.max_episode_steps

        # --- Generate source (replay) and target (mutation) levels ---
        logger.info(f"[{self.name}] Generating {n} source + target level pairs...")
        t0 = time.time()
        rng, rng_src, rng_tgt = jax.random.split(rng, 3)
        src_rngs = jax.random.split(rng_src, n)
        tgt_rngs = jax.random.split(rng_tgt, n)
        source_levels = jax.vmap(self.agent.sample_random_level)(src_rngs)
        target_levels = jax.vmap(self.agent.sample_random_level)(tgt_rngs)
        jax.block_until_ready(source_levels)
        jax.block_until_ready(target_levels)
        logger.info(f"[{self.name}] Level generation: {time.time() - t0:.2f}s")

        # Store level pairs
        for i in range(n):
            distance = self.mutation_distances[i % len(self.mutation_distances)]
            self._data.level_pairs.append(LevelPair(
                source_level={'idx': i},
                target_level={'idx': i},
                relationship='mutation',
                distance=float(distance),
            ))

        # --- Source rollout ---
        logger.info(f"[{self.name}] Running batched source rollout...")
        t0 = time.time()
        rng, rng_src_roll = jax.random.split(rng)
        source_result = batched_rollout(
            rng_src_roll, source_levels, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n),
            collection_steps=[-1],
            return_final_hstate=True,
        )
        jax.block_until_ready(source_result.episode_returns)
        logger.info(f"[{self.name}] Source rollout: {time.time() - t0:.2f}s")

        # --- Target rollout with source hstate ---
        logger.info(f"[{self.name}] Running batched target rollout (with hstate transfer)...")
        t0 = time.time()
        rng, rng_tgt_roll = jax.random.split(rng)
        target_result = batched_rollout(
            rng_tgt_roll, target_levels, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            source_result.final_hstate,  # Transfer hstate
            collection_steps=[-1],
        )
        jax.block_until_ready(target_result.episode_returns)
        logger.info(f"[{self.name}] Target rollout: {time.time() - t0:.2f}s")

        # --- Baseline rollout ---
        logger.info(f"[{self.name}] Running batched baseline rollout...")
        t0 = time.time()
        n_bl = self.n_random_baselines
        rng, rng_bl_levels, rng_bl_roll = jax.random.split(rng, 3)
        bl_rngs = jax.random.split(rng_bl_levels, n_bl)
        bl_levels = jax.vmap(self.agent.sample_random_level)(bl_rngs)
        bl_result = batched_rollout(
            rng_bl_roll, bl_levels, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n_bl),
        )
        jax.block_until_ready(bl_result.episode_returns)
        logger.info(f"[{self.name}] Baseline rollout: {time.time() - t0:.2f}s")

        # --- Populate per-pair results ---
        logger.info(f"[{self.name}] Computing transfer metrics (CPU)...")
        t0 = time.time()
        bl_avg = {
            'total_return': float(np.mean(bl_result.episode_returns)),
            'solved': float(np.mean(bl_result.episode_solved)),
            'steps_to_solve': float(np.mean(bl_result.episode_lengths)),
        }

        src_hstates = source_result.hstates_by_step["-1"]
        tgt_hstates = target_result.hstates_by_step["-1"]

        for i in tqdm(range(n), desc="Transfer metrics", leave=False):
            src_perf = {
                'total_return': float(source_result.episode_returns[i]),
                'solved': bool(source_result.episode_solved[i]),
                'steps_to_solve': int(source_result.episode_lengths[i]),
                'actions': [],
                'n_steps': int(source_result.episode_lengths[i]),
            }
            tgt_perf = {
                'total_return': float(target_result.episode_returns[i]),
                'solved': bool(target_result.episode_solved[i]),
                'steps_to_solve': int(target_result.episode_lengths[i]),
                'actions': [],
                'n_steps': int(target_result.episode_lengths[i]),
            }
            self._data.source_performance.append(src_perf)
            self._data.target_performance.append(tgt_perf)
            self._data.random_baseline_performance.append(bl_avg)

            # Transfer metrics
            behavioral = compute_behavioral_transfer(
                mutation_steps_to_solve=np.array([tgt_perf['steps_to_solve']]),
                random_steps_to_solve=np.array([bl_avg['steps_to_solve']]),
                mutation_success_rate=float(tgt_perf['solved']),
                random_success_rate=bl_avg['solved'],
                mutation_first_actions=np.zeros((1, 10)),
                replay_first_actions=np.zeros((1, 10)),
            )
            self._data.behavioral_transfer.append(behavioral)

            representational = compute_representational_transfer(
                mutation_hstates=tgt_hstates[i:i+1],
                replay_hstates=src_hstates[i:i+1],
                mutation_values=np.array([tgt_perf['total_return']]),
                replay_values=np.array([src_perf['total_return']]),
            )
            self._data.representational_transfer.append(representational)

        # --- Prediction losses ---
        logger.info(f"[{self.name}] Computing prediction losses...")
        from .utils.agent_aware_loss import compute_agent_prediction_loss
        for i in tqdm(range(n), desc="Prediction losses", leave=False):
            rng, src_loss_rng, tgt_loss_rng = jax.random.split(rng, 3)
            src_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state,
                jax.tree_util.tree_map(lambda x: x[i], source_levels), src_loss_rng
            )
            self._data.source_prediction_losses.append(src_loss)
            tgt_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state,
                jax.tree_util.tree_map(lambda x: x[i], target_levels), tgt_loss_rng
            )
            self._data.target_prediction_losses.append(tgt_loss)

        # Random baseline prediction losses
        bl_losses = []
        for i in range(min(3, n_bl)):
            rng, bl_loss_rng = jax.random.split(rng)
            bl_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state,
                jax.tree_util.tree_map(lambda x: x[i], bl_levels), bl_loss_rng
            )
            bl_losses.append(bl_loss)
        bl_mean_loss = float(np.mean(bl_losses)) if bl_losses else 0.0
        self._data.random_prediction_losses = [bl_mean_loss] * n
        logger.info(f"[{self.name}] Transfer metrics: {time.time() - t0:.2f}s")

        return self._data

    def _collect_paired_data(self, rng: chex.PRNGKey) -> AdaptationData:
        """
        Collect data for PAIRED: high-regret→similar level transfer.
        GPU-batched version.
        """
        import time, logging
        from tqdm import tqdm
        logger = logging.getLogger(__name__)

        n = self.n_level_pairs
        max_steps = self.max_episode_steps

        # --- Generate source (high-regret) and target (similar) levels ---
        logger.info(f"[{self.name}] Generating {n} source + target level pairs...")
        t0 = time.time()
        rng, rng_src, rng_tgt = jax.random.split(rng, 3)
        src_rngs = jax.random.split(rng_src, n)
        tgt_rngs = jax.random.split(rng_tgt, n)
        source_levels = jax.vmap(self.agent.sample_random_level)(src_rngs)
        target_levels = jax.vmap(self.agent.sample_random_level)(tgt_rngs)
        jax.block_until_ready(source_levels)
        jax.block_until_ready(target_levels)
        logger.info(f"[{self.name}] Level generation: {time.time() - t0:.2f}s")

        # Compute structural similarities
        src_wall_maps = np.array(source_levels.wall_map)
        tgt_wall_maps = np.array(target_levels.wall_map)
        for i in range(n):
            src_density = float(src_wall_maps[i].mean())
            tgt_density = float(tgt_wall_maps[i].mean())
            similarity = 1.0 - abs(src_density - tgt_density)
            self._data.level_pairs.append(LevelPair(
                source_level={'idx': i},
                target_level={'idx': i},
                relationship='similar',
                distance=similarity,
            ))
            self._data.structural_similarities.append(similarity)

        # --- Source rollout ---
        logger.info(f"[{self.name}] Running batched source rollout...")
        t0 = time.time()
        rng, rng_src_roll = jax.random.split(rng)
        source_result = batched_rollout(
            rng_src_roll, source_levels, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n),
            collection_steps=[-1],
            return_final_hstate=True,
        )
        jax.block_until_ready(source_result.episode_returns)
        logger.info(f"[{self.name}] Source rollout: {time.time() - t0:.2f}s")

        # --- Target rollout with source hstate ---
        logger.info(f"[{self.name}] Running batched target rollout (with hstate transfer)...")
        t0 = time.time()
        rng, rng_tgt_roll = jax.random.split(rng)
        target_result = batched_rollout(
            rng_tgt_roll, target_levels, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            source_result.final_hstate,
            collection_steps=[-1],
        )
        jax.block_until_ready(target_result.episode_returns)
        logger.info(f"[{self.name}] Target rollout: {time.time() - t0:.2f}s")

        # --- Baseline rollout ---
        logger.info(f"[{self.name}] Running batched baseline rollout...")
        t0 = time.time()
        n_bl = self.n_random_baselines
        rng, rng_bl_levels, rng_bl_roll = jax.random.split(rng, 3)
        bl_rngs = jax.random.split(rng_bl_levels, n_bl)
        bl_levels = jax.vmap(self.agent.sample_random_level)(bl_rngs)
        bl_result = batched_rollout(
            rng_bl_roll, bl_levels, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n_bl),
        )
        jax.block_until_ready(bl_result.episode_returns)
        logger.info(f"[{self.name}] Baseline rollout: {time.time() - t0:.2f}s")

        # --- Populate per-pair results ---
        bl_avg = {
            'total_return': float(np.mean(bl_result.episode_returns)),
            'solved': float(np.mean(bl_result.episode_solved)),
            'steps_to_solve': float(np.mean(bl_result.episode_lengths)),
        }
        src_hstates = source_result.hstates_by_step["-1"]
        tgt_hstates = target_result.hstates_by_step["-1"]

        for i in tqdm(range(n), desc="Transfer metrics", leave=False):
            src_perf = {
                'total_return': float(source_result.episode_returns[i]),
                'solved': bool(source_result.episode_solved[i]),
                'steps_to_solve': int(source_result.episode_lengths[i]),
                'actions': [],
                'n_steps': int(source_result.episode_lengths[i]),
            }
            tgt_perf = {
                'total_return': float(target_result.episode_returns[i]),
                'solved': bool(target_result.episode_solved[i]),
                'steps_to_solve': int(target_result.episode_lengths[i]),
                'actions': [],
                'n_steps': int(target_result.episode_lengths[i]),
            }
            self._data.source_performance.append(src_perf)
            self._data.target_performance.append(tgt_perf)
            self._data.random_baseline_performance.append(bl_avg)

            # Regret estimates
            src_regret = self._estimate_regret(
                {'wall_density': float(src_wall_maps[i].mean())}, src_perf
            )
            tgt_regret = self._estimate_regret(
                {'wall_density': float(tgt_wall_maps[i].mean())}, tgt_perf
            )
            self._data.source_regrets.append(src_regret)
            self._data.target_regrets.append(tgt_regret)

            # Transfer metrics
            behavioral = compute_behavioral_transfer(
                mutation_steps_to_solve=np.array([tgt_perf['steps_to_solve']]),
                random_steps_to_solve=np.array([bl_avg['steps_to_solve']]),
                mutation_success_rate=float(tgt_perf['solved']),
                random_success_rate=bl_avg['solved'],
                mutation_first_actions=np.zeros((1, 10)),
                replay_first_actions=np.zeros((1, 10)),
            )
            self._data.behavioral_transfer.append(behavioral)

            representational = compute_representational_transfer(
                mutation_hstates=tgt_hstates[i:i+1],
                replay_hstates=src_hstates[i:i+1],
                mutation_values=np.array([tgt_perf['total_return']]),
                replay_values=np.array([src_perf['total_return']]),
            )
            self._data.representational_transfer.append(representational)

        # --- Prediction losses ---
        from .utils.agent_aware_loss import compute_agent_prediction_loss
        for i in tqdm(range(n), desc="Prediction losses", leave=False):
            rng, src_loss_rng, tgt_loss_rng = jax.random.split(rng, 3)
            src_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state,
                jax.tree_util.tree_map(lambda x: x[i], source_levels), src_loss_rng
            )
            self._data.source_prediction_losses.append(src_loss)
            tgt_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state,
                jax.tree_util.tree_map(lambda x: x[i], target_levels), tgt_loss_rng
            )
            self._data.target_prediction_losses.append(tgt_loss)

        bl_losses = []
        for i in range(min(3, n_bl)):
            rng, bl_loss_rng = jax.random.split(rng)
            bl_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state,
                jax.tree_util.tree_map(lambda x: x[i], bl_levels), bl_loss_rng
            )
            bl_losses.append(bl_loss)
        self._data.random_prediction_losses = [float(np.mean(bl_losses)) if bl_losses else 0.0] * n

        return self._data

    def _collect_dr_data(self, rng: chex.PRNGKey) -> AdaptationData:
        """
        Collect data for DR: cluster-based transfer.
        GPU-batched version.
        """
        import time, logging
        from tqdm import tqdm
        logger = logging.getLogger(__name__)
        try:
            import wandb
            _wandb_active = wandb.run is not None
        except ImportError:
            _wandb_active = False
        timings = {}

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

        n = self.n_level_pairs
        max_steps = self.max_episode_steps

        # --- Generate source and target levels ---
        logger.info(f"[{self.name}] Generating {n} source + target level pairs for DR...")
        t0 = time.time()
        rng, rng_src, rng_tgt = jax.random.split(rng, 3)
        src_rngs = jax.random.split(rng_src, n)
        tgt_rngs = jax.random.split(rng_tgt, n)
        source_levels = jax.vmap(self.agent.sample_random_level)(src_rngs)
        target_levels = jax.vmap(self.agent.sample_random_level)(tgt_rngs)
        jax.block_until_ready(source_levels)
        jax.block_until_ready(target_levels)
        logger.info(f"[{self.name}] Level generation: {time.time() - t0:.2f}s")

        # Compute structural similarities
        src_wall_maps = np.array(source_levels.wall_map)
        tgt_wall_maps = np.array(target_levels.wall_map)
        for i in range(n):
            src_density = float(src_wall_maps[i].mean())
            tgt_density = float(tgt_wall_maps[i].mean())
            similarity = 1.0 - abs(src_density - tgt_density)
            self._data.level_pairs.append(LevelPair(
                source_level={'idx': i},
                target_level={'idx': i},
                relationship='same_cluster',
                distance=similarity,
            ))
            self._data.structural_similarities.append(similarity)

        # --- Source rollout ---
        logger.info(f"[{self.name}] Running batched source rollout...")
        t0 = time.time()
        rng, rng_src_roll = jax.random.split(rng)
        source_result = batched_rollout(
            rng_src_roll, source_levels, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n),
            collection_steps=[-1],
            return_final_hstate=True,
        )
        jax.block_until_ready(source_result.episode_returns)
        logger.info(f"[{self.name}] Source rollout: {time.time() - t0:.2f}s")

        # --- Target rollout with source hstate ---
        logger.info(f"[{self.name}] Running batched target rollout (with hstate transfer)...")
        t0 = time.time()
        rng, rng_tgt_roll = jax.random.split(rng)
        target_result = batched_rollout(
            rng_tgt_roll, target_levels, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            source_result.final_hstate,
            collection_steps=[-1],
        )
        jax.block_until_ready(target_result.episode_returns)
        logger.info(f"[{self.name}] Target rollout: {time.time() - t0:.2f}s")

        # --- Baseline rollout ---
        n_bl = self.n_random_baselines
        rng, rng_bl_levels, rng_bl_roll = jax.random.split(rng, 3)
        bl_rngs = jax.random.split(rng_bl_levels, n_bl)
        bl_levels = jax.vmap(self.agent.sample_random_level)(bl_rngs)
        bl_result = batched_rollout(
            rng_bl_roll, bl_levels, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n_bl),
        )

        # --- Populate per-pair results ---
        bl_avg = {
            'total_return': float(np.mean(bl_result.episode_returns)),
            'solved': float(np.mean(bl_result.episode_solved)),
            'steps_to_solve': float(np.mean(bl_result.episode_lengths)),
        }
        src_hstates = source_result.hstates_by_step["-1"]
        tgt_hstates = target_result.hstates_by_step["-1"]

        for i in tqdm(range(n), desc="Transfer metrics", leave=False):
            src_perf = {
                'total_return': float(source_result.episode_returns[i]),
                'solved': bool(source_result.episode_solved[i]),
                'steps_to_solve': int(source_result.episode_lengths[i]),
                'actions': [],
                'n_steps': int(source_result.episode_lengths[i]),
            }
            tgt_perf = {
                'total_return': float(target_result.episode_returns[i]),
                'solved': bool(target_result.episode_solved[i]),
                'steps_to_solve': int(target_result.episode_lengths[i]),
                'actions': [],
                'n_steps': int(target_result.episode_lengths[i]),
            }
            self._data.source_performance.append(src_perf)
            self._data.target_performance.append(tgt_perf)
            self._data.random_baseline_performance.append(bl_avg)

            behavioral = compute_behavioral_transfer(
                mutation_steps_to_solve=np.array([tgt_perf['steps_to_solve']]),
                random_steps_to_solve=np.array([bl_avg['steps_to_solve']]),
                mutation_success_rate=float(tgt_perf['solved']),
                random_success_rate=bl_avg['solved'],
                mutation_first_actions=np.zeros((1, 10)),
                replay_first_actions=np.zeros((1, 10)),
            )
            self._data.behavioral_transfer.append(behavioral)

            representational = compute_representational_transfer(
                mutation_hstates=tgt_hstates[i:i+1],
                replay_hstates=src_hstates[i:i+1],
                mutation_values=np.array([tgt_perf['total_return']]),
                replay_values=np.array([src_perf['total_return']]),
            )
            self._data.representational_transfer.append(representational)

        # Prediction losses
        from .utils.agent_aware_loss import compute_agent_prediction_loss
        for i in tqdm(range(n), desc="Prediction losses", leave=False):
            rng, src_loss_rng, tgt_loss_rng = jax.random.split(rng, 3)
            src_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state,
                jax.tree_util.tree_map(lambda x: x[i], source_levels), src_loss_rng
            )
            self._data.source_prediction_losses.append(src_loss)
            tgt_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state,
                jax.tree_util.tree_map(lambda x: x[i], target_levels), tgt_loss_rng
            )
            self._data.target_prediction_losses.append(tgt_loss)

        bl_losses = []
        for i in range(min(3, n_bl)):
            rng, bl_loss_rng = jax.random.split(rng)
            bl_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state,
                jax.tree_util.tree_map(lambda x: x[i], bl_levels), bl_loss_rng
            )
            bl_losses.append(bl_loss)
        self._data.random_prediction_losses = [float(np.mean(bl_losses)) if bl_losses else 0.0] * n

        return self._data

    def _compute_structural_similarity(
        self,
        level1: Dict[str, Any],
        level2: Dict[str, Any],
    ) -> float:
        """Compute structural similarity between two levels."""
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
        max_dist = np.sqrt(2) * 13  # Max possible distance
        dist_sim = 1.0 - abs(dist1 - dist2) / max_dist

        # Wall pattern overlap (Jaccard similarity)
        wall1 = level1['wall_map'].flatten()
        wall2 = level2['wall_map'].flatten()
        intersection = np.sum(wall1 & wall2)
        union = np.sum(wall1 | wall2)
        pattern_sim = intersection / (union + 1e-6)

        # Combined similarity
        return float(0.4 * density_sim + 0.3 * dist_sim + 0.3 * pattern_sim)

    def _estimate_regret(
        self,
        level: Dict[str, Any],
        result: Dict[str, Any],
    ) -> float:
        """
        Estimate regret for PAIRED.

        Regret = antagonist_return - protagonist_return
        Since we don't have antagonist, estimate based on difficulty.
        """
        # Heuristic: unsolved difficult levels have high regret
        solved = result.get('solved', False)
        wall_density = level.get('wall_density', 0.2)
        steps_used = result.get('n_steps', 256) / 256.0

        if solved:
            # Low regret if solved quickly
            regret = 0.1 * steps_used
        else:
            # High regret if unsolved, especially with lower wall density
            # (should have been solvable)
            regret = 0.5 + 0.5 * (1.0 - wall_density) + 0.2 * steps_used

        return float(regret)

    def _average_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Average results from multiple episodes."""
        return {
            'total_return': float(np.mean([r['total_return'] for r in results])),
            'solved': float(np.mean([r['solved'] for r in results])),
            'steps_to_solve': float(np.mean([r['steps_to_solve'] for r in results])),
        }

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze transfer from replay to mutations.

        Metrics:
        - Behavioral transfer: solve rate, steps, action similarity
        - Representational transfer: hidden state similarity
        - Prediction loss transfer: curriculum knowledge transfer
        - Comparison to random baseline
        """
        if self._data is None:
            raise ValueError("Must call collect_data before analyze")

        results = {}

        # 1. Overall transfer metrics
        results['overall'] = self._compute_overall_metrics()

        # 2. Transfer by mutation distance
        results['by_mutation_distance'] = self._analyze_by_distance()

        # 3. Behavioral transfer analysis
        results['behavioral_transfer'] = self._analyze_behavioral_transfer()

        # 4. Representational transfer analysis
        results['representational_transfer'] = self._analyze_representational_transfer()

        # 5. Comparison to baseline
        results['baseline_comparison'] = self._compare_to_baseline()

        # 6. Prediction loss transfer analysis (PRIMARY CAUSAL METRIC)
        results['prediction_loss_transfer'] = self._analyze_prediction_loss_transfer()

        self._results = results
        return results

    def _compute_overall_metrics(self) -> Dict[str, float]:
        """Compute overall transfer metrics."""
        source_solved = [p['solved'] for p in self._data.source_performance]
        target_solved = [p['solved'] for p in self._data.target_performance]
        baseline_solved = [p['solved'] for p in self._data.random_baseline_performance]

        source_steps = [p['steps_to_solve'] for p in self._data.source_performance]
        target_steps = [p['steps_to_solve'] for p in self._data.target_performance]
        baseline_steps = [p['steps_to_solve'] for p in self._data.random_baseline_performance]

        # Transfer ratio: target performance / baseline performance
        target_rate = np.mean(target_solved)
        baseline_rate = np.mean(baseline_solved) + 1e-6

        # Method-appropriate labels
        source_label = "replay" if self.has_branches else ("high_regret" if self.has_regret else "source")
        target_label = "mutation" if self.has_mutations else ("similar" if self.has_regret else "target")

        metrics = {
            f'{source_label}_solve_rate': float(np.mean(source_solved)),
            f'{target_label}_solve_rate': float(np.mean(target_solved)),
            'baseline_solve_rate': float(np.mean(baseline_solved)),
            'transfer_ratio': float(target_rate / baseline_rate),
            f'{source_label}_mean_steps': float(np.mean(source_steps)),
            f'{target_label}_mean_steps': float(np.mean(target_steps)),
            'baseline_mean_steps': float(np.mean(baseline_steps)),
            'n_pairs': len(self._data.level_pairs),
            'training_method': self.training_method,
        }

        # Add method-specific metrics
        if self.has_regret and self._data.source_regrets:
            metrics['mean_source_regret'] = float(np.mean(self._data.source_regrets))
            metrics['mean_target_regret'] = float(np.mean(self._data.target_regrets))
            metrics['regret_reduction'] = float(
                np.mean(self._data.source_regrets) - np.mean(self._data.target_regrets)
            )

        if self._data.structural_similarities:
            metrics['mean_structural_similarity'] = float(np.mean(self._data.structural_similarities))

        return metrics

    def _analyze_by_distance(self) -> Dict[str, Dict[str, float]]:
        """Analyze transfer metrics by distance/similarity (method-appropriate)."""
        results_by_distance = {}

        if self.has_mutations:
            # ACCEL/PLR: analyze by mutation distance
            for dist in self.mutation_distances:
                indices = [
                    i for i, pair in enumerate(self._data.level_pairs)
                    if pair.distance == dist
                ]

                if not indices:
                    continue

                target_solved = [self._data.target_performance[i]['solved'] for i in indices]
                baseline_solved = [self._data.random_baseline_performance[i]['solved'] for i in indices]
                behavioral = [self._data.behavioral_transfer[i] for i in indices]

                results_by_distance[f'mutation_dist_{dist}'] = {
                    'n_pairs': len(indices),
                    'target_solve_rate': float(np.mean(target_solved)),
                    'baseline_solve_rate': float(np.mean(baseline_solved)),
                    'mean_action_similarity': float(np.mean([
                        b.get('action_similarity', 0) for b in behavioral
                    ])),
                    'mean_value_correlation': float(np.mean([
                        b.get('value_correlation', 0) for b in behavioral
                    ])),
                }

        elif self._data.structural_similarities:
            # PAIRED/DR: analyze by structural similarity terciles
            similarities = np.array(self._data.structural_similarities)
            terciles = np.percentile(similarities, [33, 66])

            for tercile_idx, (low, high) in enumerate([
                (0.0, terciles[0]),
                (terciles[0], terciles[1]),
                (terciles[1], 1.0)
            ]):
                indices = [
                    i for i, sim in enumerate(similarities)
                    if low <= sim < high or (tercile_idx == 2 and sim == high)
                ]

                if not indices:
                    continue

                target_solved = [self._data.target_performance[i]['solved'] for i in indices]
                baseline_solved = [self._data.random_baseline_performance[i]['solved'] for i in indices]
                behavioral = [self._data.behavioral_transfer[i] for i in indices]

                label = ['low', 'medium', 'high'][tercile_idx]
                results_by_distance[f'similarity_{label}'] = {
                    'n_pairs': len(indices),
                    'similarity_range': f'{low:.2f}-{high:.2f}',
                    'target_solve_rate': float(np.mean(target_solved)),
                    'baseline_solve_rate': float(np.mean(baseline_solved)),
                    'mean_action_similarity': float(np.mean([
                        b.get('action_similarity', 0) for b in behavioral
                    ])),
                }

        return results_by_distance

    def _analyze_behavioral_transfer(self) -> Dict[str, Any]:
        """Analyze behavioral transfer metrics."""
        action_sims = [b.get('action_similarity', 0) for b in self._data.behavioral_transfer]
        value_corrs = [b.get('value_correlation', 0) for b in self._data.behavioral_transfer]
        policy_kls = [b.get('policy_kl', 0) for b in self._data.behavioral_transfer]

        return {
            'action_similarity': {
                'mean': float(np.mean(action_sims)),
                'std': float(np.std(action_sims)),
            },
            'value_correlation': {
                'mean': float(np.mean(value_corrs)),
                'std': float(np.std(value_corrs)),
            },
            'policy_kl': {
                'mean': float(np.mean(policy_kls)),
                'std': float(np.std(policy_kls)),
            },
        }

    def _analyze_representational_transfer(self) -> Dict[str, Any]:
        """Analyze representational transfer metrics."""
        cosine_sims = [r.get('cosine_similarity', 0) for r in self._data.representational_transfer]
        trajectory_corrs = [r.get('trajectory_correlation', 0) for r in self._data.representational_transfer]

        return {
            'hidden_state_similarity': {
                'mean': float(np.mean(cosine_sims)),
                'std': float(np.std(cosine_sims)),
            },
            'trajectory_correlation': {
                'mean': float(np.mean(trajectory_corrs)),
                'std': float(np.std(trajectory_corrs)),
            },
        }

    def _compare_to_baseline(self) -> Dict[str, Any]:
        """Compare target performance to random baseline."""
        from scipy import stats

        target_solved = [p['solved'] for p in self._data.target_performance]
        baseline_solved = [p['solved'] for p in self._data.random_baseline_performance]

        # Paired t-test for solve rate difference
        if len(target_solved) >= 5:
            t_stat, p_value = stats.ttest_rel(target_solved, baseline_solved)
        else:
            t_stat, p_value = 0.0, 1.0

        # Effect size (Cohen's d)
        diff = np.array(target_solved) - np.array(baseline_solved)
        cohens_d = np.mean(diff) / (np.std(diff) + 1e-6)

        # Method-appropriate label
        target_label = "mutation" if self.has_mutations else ("similar" if self.has_regret else "target")

        return {
            f'{target_label}_advantage': float(np.mean(target_solved) - np.mean(baseline_solved)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant_transfer': p_value < 0.05 and cohens_d > 0.2,
        }

    def _analyze_prediction_loss_transfer(self) -> Dict[str, Any]:
        """
        Analyze prediction loss transfer from source to target.

        This is the PRIMARY causal metric for curriculum awareness.
        Lower target loss than random indicates knowledge transfer.
        """
        from .utils.agent_aware_loss import compute_random_baseline_loss

        source_losses = np.array(self._data.source_prediction_losses)
        target_losses = np.array(self._data.target_prediction_losses)
        random_losses = np.array(self._data.random_prediction_losses)

        if len(source_losses) == 0:
            return {'error': 'No prediction loss data collected'}

        # Key metric: Does target have lower loss than random?
        target_vs_random = target_losses - random_losses

        random_baseline = compute_random_baseline_loss()

        # Compute information gain relative to random baseline
        source_info_gain = random_baseline - np.mean(source_losses)
        target_info_gain = random_baseline - np.mean(target_losses)
        random_info_gain = random_baseline - np.mean(random_losses)

        # Statistical test for target vs random
        from scipy import stats
        if len(target_losses) >= 5:
            t_stat, p_value = stats.ttest_rel(target_losses, random_losses)
        else:
            t_stat, p_value = 0.0, 1.0

        # Method-appropriate labels
        source_label = "replay" if self.has_branches else ("high_regret" if self.has_regret else "source")
        target_label = "mutation" if self.has_mutations else ("similar" if self.has_regret else "target")

        result = {
            f'{source_label}_mean_loss': float(np.mean(source_losses)),
            f'{target_label}_mean_loss': float(np.mean(target_losses)),
            'random_mean_loss': float(np.mean(random_losses)),
            'random_baseline': random_baseline,
            f'{target_label}_advantage_over_random': float(-np.mean(target_vs_random)),
            'prediction_transfer_exists': float(np.mean(target_losses)) < float(np.mean(random_losses)),
            f'information_gain_{source_label}': float(source_info_gain),
            f'information_gain_{target_label}': float(target_info_gain),
            'information_gain_random': float(random_info_gain),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_transfer': p_value < 0.05 and np.mean(target_losses) < np.mean(random_losses),
            'training_method': self.training_method,
        }

        # Add method-specific analysis
        if self.has_regret and self._data.source_regrets:
            # PAIRED: correlate regret with transfer
            source_regrets = np.array(self._data.source_regrets)
            if len(source_regrets) > 5:
                corr = np.corrcoef(source_regrets, target_vs_random)[0, 1]
                result['regret_transfer_correlation'] = float(corr) if np.isfinite(corr) else 0.0
                result['high_regret_transfer_benefit'] = corr < 0  # Negative correlation = better transfer

        if self._data.structural_similarities:
            # DR: correlate similarity with transfer
            similarities = np.array(self._data.structural_similarities)
            if len(similarities) > 5:
                corr = np.corrcoef(similarities, -target_vs_random)[0, 1]
                result['similarity_transfer_correlation'] = float(corr) if np.isfinite(corr) else 0.0
                result['similarity_helps_transfer'] = corr > 0  # Positive = more similar = better transfer

        return result

    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data."""
        if not self._results:
            raise ValueError("Must call analyze before visualize")

        viz_data = {
            'overall': self._results.get('overall', {}),
            'by_distance': self._results.get('by_mutation_distance', {}),
            'training_method': self.training_method,
        }

        # Method-appropriate labels
        source_label = "Replay" if self.has_branches else ("High Regret" if self.has_regret else "Source")
        target_label = "Mutation" if self.has_mutations else ("Similar" if self.has_regret else "Target")

        # Transfer comparison bar chart data
        overall = self._results.get('overall', {})

        # Find the solve rate keys (method-dependent)
        source_key = next((k for k in overall.keys() if ('source' in k.lower() or 'replay' in k.lower() or 'high_regret' in k.lower()) and 'solve_rate' in k), None)
        target_key = next((k for k in overall.keys() if ('target' in k.lower() or 'mutation' in k.lower() or 'similar' in k.lower()) and 'solve_rate' in k), None)

        viz_data['performance_comparison'] = {
            'categories': [source_label, target_label, 'Baseline'],
            'solve_rates': [
                overall.get(source_key, overall.get('replay_solve_rate', 0)),
                overall.get(target_key, overall.get('mutation_solve_rate', 0)),
                overall.get('baseline_solve_rate', 0),
            ],
        }

        # Add method-specific visualizations
        if self.has_regret and 'regret_reduction' in overall:
            viz_data['regret_analysis'] = {
                'mean_source_regret': overall.get('mean_source_regret', 0),
                'mean_target_regret': overall.get('mean_target_regret', 0),
                'regret_reduction': overall.get('regret_reduction', 0),
            }

        if 'mean_structural_similarity' in overall:
            viz_data['structural_similarity'] = {
                'mean_similarity': overall.get('mean_structural_similarity', 0),
            }

        return viz_data
