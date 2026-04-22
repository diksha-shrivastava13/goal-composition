"""
Mutation Adaptation Experiment.

Tests adaptation speed when levels are mutated (walls toggled).
For each mutation distance k in {1, 2, 3, 5}:
  1. Generate source levels, run agent → baseline returns
  2. Create mutations via mutate_level_walls(level, rng, k)
  3. Run agent on mutated levels for multiple episodes → measure recovery
  4. Adaptation speed: episodes to recover performance (multi-episode)

Training method dispatch:
- ACCEL/PLR/RobustPLR: mutation-based (primary protocol)
- PAIRED: mutation-based with antagonist regret tracking
- DR: mutation-based (protagonist only)
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
)
from .utils.batched_rollout import batched_rollout
from .utils.paired_helpers import (
    generate_levels,
    mutate_level_walls,
)


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

    # Mutation-specific: adaptation speed tracking
    adaptation_speeds: List[float] = field(default_factory=list)
    mutation_distances: List[int] = field(default_factory=list)

    # Multi-episode recovery tracking
    recovery_curves: List[List[float]] = field(default_factory=list)
    episodes_to_recover: List[int] = field(default_factory=list)


class MutationAdaptationExperiment(CheckpointExperiment):
    """
    Test adaptation speed under actual level mutations.

    Protocol:
    1. Generate source levels
    2. For each k in {1, 2, 3, 5}: create mutations via mutate_level_walls
    3. Run agent on source → baseline returns
    4. Run agent on mutated levels for n_recovery_episodes
    5. Adaptation speed = 1 / episodes_to_recover(threshold)
    """

    @property
    def name(self) -> str:
        return "mutation_adaptation"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_level_pairs = self.exp_config("n_level_pairs")
        self.mutation_distances = self.exp_config("mutation_distances")
        self.n_random_baselines = self.exp_config("n_random_baselines")
        self.max_episode_steps = self.exp_config("max_episode_steps")
        self.n_clusters = self.exp_config("n_clusters")
        self.n_recovery_episodes = self.exp_config("n_recovery_episodes")
        self.recovery_threshold = self.exp_config("recovery_threshold")

        self._data: Optional[AdaptationData] = None
        self._results: Dict[str, Any] = {}

    def collect_data(self, rng: chex.PRNGKey) -> AdaptationData:
        """Dispatch to method-specific collection."""
        self._data = AdaptationData(training_method=self.training_method)

        if self.training_method in ["paired"]:
            return self._collect_paired_data(rng)
        elif self.training_method in ["dr"]:
            return self._collect_dr_data(rng)
        else:  # accel, plr, robust_plr
            return self._collect_accel_data(rng)

    def _run_recovery_episodes(self, rng, mutated_levels, init_hstate, pro_ts, n_per_k):
        """Run multiple episodes on mutated levels, carrying h-state forward.

        Returns:
            recovery_matrix: (n_recovery_episodes, n_per_k) array of returns
            final_hstate: h-state after last episode
        """
        max_steps = self.max_episode_steps
        hstate = init_hstate
        episode_returns_per_ep = []

        for ep in range(self.n_recovery_episodes):
            rng, ep_rng = jax.random.split(rng)
            ep_result = batched_rollout(
                ep_rng, mutated_levels, max_steps,
                pro_ts.apply_fn, pro_ts.params,
                self.agent.env, self.agent.env_params,
                hstate,
                return_final_hstate=True,
            )
            episode_returns_per_ep.append(np.array(ep_result.episode_returns))
            hstate = ep_result.final_hstate

        return np.stack(episode_returns_per_ep, axis=0), hstate

    def _compute_recovery_metrics(self, recovery_matrix, source_returns, idx, n_per_k):
        """Compute per-level recovery metrics from multi-episode results.

        Args:
            recovery_matrix: (n_recovery_episodes, n_per_k) returns
            source_returns: full source returns array
            idx: starting index into source_returns
            n_per_k: number of levels in this batch
        """
        for i in range(n_per_k):
            src_ret = float(source_returns[idx + i])
            threshold = src_ret * self.recovery_threshold

            episodes_to_recover = self.n_recovery_episodes
            for ep in range(self.n_recovery_episodes):
                if recovery_matrix[ep, i] >= threshold:
                    episodes_to_recover = ep + 1
                    break

            adaptation_speed = 1.0 / float(episodes_to_recover)
            self._data.adaptation_speeds.append(adaptation_speed)
            self._data.episodes_to_recover.append(episodes_to_recover)
            self._data.recovery_curves.append(recovery_matrix[:, i].tolist())

    def _collect_accel_data(self, rng: chex.PRNGKey) -> AdaptationData:
        """
        Collect data for ACCEL/PLR: actual mutation-based adaptation with multi-episode recovery.
        """
        import time
        import logging
        logger = logging.getLogger(__name__)

        n_per_k = self.n_level_pairs // len(self.mutation_distances)
        max_steps = self.max_episode_steps

        # --- Generate source levels ---
        logger.info(f"[{self.name}] Generating {n_per_k * len(self.mutation_distances)} source levels...")
        t0 = time.time()
        total_n = n_per_k * len(self.mutation_distances)
        rng, gen_rng = jax.random.split(rng)
        source_levels = generate_levels(self.agent, gen_rng, total_n)
        jax.block_until_ready(source_levels)
        logger.info(f"[{self.name}] Level generation: {time.time() - t0:.2f}s")

        # --- Source rollout ---
        logger.info(f"[{self.name}] Running source rollout...")
        t0 = time.time()
        rng, src_rng = jax.random.split(rng)
        pro_ts = getattr(self.train_state, 'pro_train_state', self.train_state)
        source_result = batched_rollout(
            src_rng, source_levels, max_steps,
            pro_ts.apply_fn, pro_ts.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(total_n),
            collection_steps=[-1],
            return_final_hstate=True,
        )
        jax.block_until_ready(source_result.episode_returns)
        logger.info(f"[{self.name}] Source rollout: {time.time() - t0:.2f}s")

        source_returns = np.array(source_result.episode_returns)

        # --- For each k, create mutations and run multi-episode recovery ---
        idx = 0
        for k in self.mutation_distances:
            logger.info(f"[{self.name}] Mutation distance k={k}...")
            t0 = time.time()

            # Create mutated levels for this k
            mutated_list = []
            for i in range(n_per_k):
                level_i = jax.tree_util.tree_map(lambda x: x[idx + i], source_levels)
                rng, mut_rng = jax.random.split(rng)
                mutated = mutate_level_walls(level_i, mut_rng, k)
                mutated_list.append(mutated)

            # Stack mutated levels into a batch
            mutated_levels = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0), *mutated_list
            )

            # Use source final hstate for transfer
            src_hstate_batch = jax.tree_util.tree_map(
                lambda x: x[idx:idx + n_per_k], source_result.final_hstate
            )

            # Multi-episode recovery
            rng, rec_rng = jax.random.split(rng)
            recovery_matrix, _ = self._run_recovery_episodes(
                rec_rng, mutated_levels, src_hstate_batch, pro_ts, n_per_k
            )

            # Compute recovery metrics
            self._compute_recovery_metrics(recovery_matrix, source_returns, idx, n_per_k)

            # Record per-pair data
            for i in range(n_per_k):
                self._data.mutation_distances.append(k)
                self._data.level_pairs.append(LevelPair(
                    source_level={'idx': idx + i},
                    target_level={'idx': i, 'k': k},
                    relationship='mutation',
                    distance=float(k),
                ))
                self._data.source_performance.append({
                    'total_return': float(source_returns[idx + i]),
                    'solved': bool(source_result.episode_solved[idx + i]),
                    'steps_to_solve': int(source_result.episode_lengths[idx + i]),
                })
                # Target performance = first episode on mutated level
                self._data.target_performance.append({
                    'total_return': float(recovery_matrix[0, i]),
                    'recovery_curve': recovery_matrix[:, i].tolist(),
                })

            logger.info(f"[{self.name}] k={k}: {time.time() - t0:.2f}s")
            idx += n_per_k

        # --- Baseline rollout ---
        logger.info(f"[{self.name}] Running baseline rollout...")
        n_bl = self.n_random_baselines
        rng, bl_gen_rng, bl_roll_rng = jax.random.split(rng, 3)
        bl_levels = generate_levels(self.agent, bl_gen_rng, n_bl)
        bl_result = batched_rollout(
            bl_roll_rng, bl_levels, max_steps,
            pro_ts.apply_fn, pro_ts.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n_bl),
        )
        bl_avg = {
            'total_return': float(np.mean(bl_result.episode_returns)),
            'solved': float(np.mean(bl_result.episode_solved)),
            'steps_to_solve': float(np.mean(bl_result.episode_lengths)),
        }
        self._data.random_baseline_performance = [bl_avg] * len(self._data.level_pairs)

        return self._data

    def _collect_paired_data(self, rng: chex.PRNGKey) -> AdaptationData:
        """Collect data for PAIRED: mutation-based with antagonist regret tracking."""
        import time
        import logging
        logger = logging.getLogger(__name__)

        n_per_k = self.n_level_pairs // len(self.mutation_distances)
        max_steps = self.max_episode_steps

        total_n = n_per_k * len(self.mutation_distances)
        rng, gen_rng = jax.random.split(rng)
        source_levels = generate_levels(self.agent, gen_rng, total_n)

        pro_ts = getattr(self.train_state, 'pro_train_state', self.train_state)
        ant_train_state = getattr(self.train_state, 'ant_train_state', None)

        # Source protagonist rollout
        rng, src_rng = jax.random.split(rng)
        source_result = batched_rollout(
            src_rng, source_levels, max_steps,
            pro_ts.apply_fn, pro_ts.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(total_n),
            collection_steps=[-1], return_final_hstate=True,
        )
        source_returns = np.array(source_result.episode_returns)

        # Source antagonist rollout for regret
        src_ant_returns = None
        if ant_train_state is not None:
            rng, ant_rng = jax.random.split(rng)
            src_ant_result = batched_rollout(
                ant_rng, source_levels, max_steps,
                ant_train_state.apply_fn, ant_train_state.params,
                self.agent.env, self.agent.env_params,
                self.agent.initialize_hidden_state(total_n),
            )
            src_ant_returns = np.array(src_ant_result.episode_returns)

        # For each k, create mutations and run multi-episode recovery
        idx = 0
        for k in self.mutation_distances:
            logger.info(f"[{self.name}] PAIRED mutation k={k}...")

            mutated_list = []
            for i in range(n_per_k):
                level_i = jax.tree_util.tree_map(lambda x: x[idx + i], source_levels)
                rng, mut_rng = jax.random.split(rng)
                mutated = mutate_level_walls(level_i, mut_rng, k)
                mutated_list.append(mutated)

            mutated_levels = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0), *mutated_list
            )

            src_hstate_batch = jax.tree_util.tree_map(
                lambda x: x[idx:idx + n_per_k], source_result.final_hstate
            )

            # Multi-episode recovery
            rng, rec_rng = jax.random.split(rng)
            recovery_matrix, _ = self._run_recovery_episodes(
                rec_rng, mutated_levels, src_hstate_batch, pro_ts, n_per_k
            )

            self._compute_recovery_metrics(recovery_matrix, source_returns, idx, n_per_k)

            # Antagonist on mutated levels for regret
            tgt_ant_returns = None
            if ant_train_state is not None:
                rng, ant_mut_rng = jax.random.split(rng)
                tgt_ant_result = batched_rollout(
                    ant_mut_rng, mutated_levels, max_steps,
                    ant_train_state.apply_fn, ant_train_state.params,
                    self.agent.env, self.agent.env_params,
                    self.agent.initialize_hidden_state(n_per_k),
                )
                tgt_ant_returns = np.array(tgt_ant_result.episode_returns)

            for i in range(n_per_k):
                self._data.mutation_distances.append(k)
                self._data.level_pairs.append(LevelPair(
                    source_level={'idx': idx + i},
                    target_level={'idx': i, 'k': k},
                    relationship='mutation',
                    distance=float(k),
                ))
                self._data.source_performance.append({
                    'total_return': float(source_returns[idx + i]),
                })
                self._data.target_performance.append({
                    'total_return': float(recovery_matrix[0, i]),
                    'recovery_curve': recovery_matrix[:, i].tolist(),
                })

                if src_ant_returns is not None and tgt_ant_returns is not None:
                    self._data.source_regrets.append(
                        float(src_ant_returns[idx + i] - source_returns[idx + i]))
                    self._data.target_regrets.append(
                        float(tgt_ant_returns[i] - recovery_matrix[0, i]))

            idx += n_per_k

        # Baseline
        n_bl = self.n_random_baselines
        rng, bl_gen_rng, bl_roll_rng = jax.random.split(rng, 3)
        bl_levels = generate_levels(self.agent, bl_gen_rng, n_bl)
        bl_result = batched_rollout(
            bl_roll_rng, bl_levels, max_steps,
            pro_ts.apply_fn, pro_ts.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n_bl),
        )
        bl_avg = {
            'total_return': float(np.mean(bl_result.episode_returns)),
            'solved': float(np.mean(bl_result.episode_solved)),
            'steps_to_solve': float(np.mean(bl_result.episode_lengths)),
        }
        self._data.random_baseline_performance = [bl_avg] * len(self._data.level_pairs)

        return self._data

    def _collect_dr_data(self, rng: chex.PRNGKey) -> AdaptationData:
        """Collect data for DR: mutation-based (protagonist only, no antagonist)."""
        import time
        import logging
        logger = logging.getLogger(__name__)

        n_per_k = self.n_level_pairs // len(self.mutation_distances)
        max_steps = self.max_episode_steps

        total_n = n_per_k * len(self.mutation_distances)
        rng, gen_rng = jax.random.split(rng)
        source_levels = generate_levels(self.agent, gen_rng, total_n)

        pro_ts = getattr(self.train_state, 'pro_train_state', self.train_state)

        # Source rollout
        rng, src_rng = jax.random.split(rng)
        source_result = batched_rollout(
            src_rng, source_levels, max_steps,
            pro_ts.apply_fn, pro_ts.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(total_n),
            collection_steps=[-1], return_final_hstate=True,
        )
        source_returns = np.array(source_result.episode_returns)

        # For each k, create mutations and run multi-episode recovery
        idx = 0
        for k in self.mutation_distances:
            logger.info(f"[{self.name}] DR mutation k={k}...")

            mutated_list = []
            for i in range(n_per_k):
                level_i = jax.tree_util.tree_map(lambda x: x[idx + i], source_levels)
                rng, mut_rng = jax.random.split(rng)
                mutated = mutate_level_walls(level_i, mut_rng, k)
                mutated_list.append(mutated)

            mutated_levels = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0), *mutated_list
            )

            src_hstate_batch = jax.tree_util.tree_map(
                lambda x: x[idx:idx + n_per_k], source_result.final_hstate
            )

            # Multi-episode recovery
            rng, rec_rng = jax.random.split(rng)
            recovery_matrix, _ = self._run_recovery_episodes(
                rec_rng, mutated_levels, src_hstate_batch, pro_ts, n_per_k
            )

            self._compute_recovery_metrics(recovery_matrix, source_returns, idx, n_per_k)

            for i in range(n_per_k):
                self._data.mutation_distances.append(k)
                self._data.level_pairs.append(LevelPair(
                    source_level={'idx': idx + i},
                    target_level={'idx': i, 'k': k},
                    relationship='mutation',
                    distance=float(k),
                ))
                self._data.source_performance.append({
                    'total_return': float(source_returns[idx + i]),
                })
                self._data.target_performance.append({
                    'total_return': float(recovery_matrix[0, i]),
                    'recovery_curve': recovery_matrix[:, i].tolist(),
                })

            idx += n_per_k

        # Baseline
        n_bl = self.n_random_baselines
        rng, bl_gen_rng, bl_roll_rng = jax.random.split(rng, 3)
        bl_levels = generate_levels(self.agent, bl_gen_rng, n_bl)
        bl_result = batched_rollout(
            bl_roll_rng, bl_levels, max_steps,
            pro_ts.apply_fn, pro_ts.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n_bl),
        )
        bl_avg = {
            'total_return': float(np.mean(bl_result.episode_returns)),
            'solved': float(np.mean(bl_result.episode_solved)),
            'steps_to_solve': float(np.mean(bl_result.episode_lengths)),
        }
        self._data.random_baseline_performance = [bl_avg] * len(self._data.level_pairs)

        return self._data

    def analyze(self) -> Dict[str, Any]:
        """Analyze adaptation results."""
        if self._data is None:
            raise ValueError("Must call collect_data before analyze")

        results = {}

        # 1. Overall transfer metrics
        results['overall'] = self._compute_overall_metrics()

        # 2. Transfer by distance
        results['by_mutation_distance'] = self._analyze_by_distance()

        # 3. Behavioral transfer
        results['behavioral_transfer'] = self._analyze_behavioral_transfer()

        # 4. Representational transfer
        results['representational_transfer'] = self._analyze_representational_transfer()

        # 5. Baseline comparison
        results['baseline_comparison'] = self._compare_to_baseline()

        # 6. Adaptation speed analysis (all methods now)
        if self._data.adaptation_speeds:
            results['adaptation_speed'] = self._analyze_adaptation_speed()

        self._results = results
        return results

    def _analyze_adaptation_speed(self) -> Dict[str, Any]:
        """Analyze adaptation speed grouped by mutation distance k."""
        from sklearn.linear_model import LinearRegression

        speeds = np.array(self._data.adaptation_speeds)
        dists = np.array(self._data.mutation_distances)
        eps_to_recover = np.array(self._data.episodes_to_recover)

        by_k = {}
        recovery_curves_by_k = {}
        for k in sorted(set(dists)):
            mask = dists == k
            k_speeds = speeds[mask]
            k_eps = eps_to_recover[mask]
            k_curves = [self._data.recovery_curves[i] for i, m in enumerate(mask) if m]

            by_k[f'k={k}'] = {
                'mean_speed': float(np.mean(k_speeds)),
                'std_speed': float(np.std(k_speeds)),
                'mean_episodes_to_recover': float(np.mean(k_eps)),
                'std_episodes_to_recover': float(np.std(k_eps)),
                'n': int(mask.sum()),
            }

            # Aggregate recovery curve for this k
            if k_curves:
                curves_arr = np.array(k_curves)
                recovery_curves_by_k[f'k={k}'] = {
                    'mean': curves_arr.mean(axis=0).tolist(),
                    'std': curves_arr.std(axis=0).tolist(),
                }

        # Linear regression: speed ~ k
        model = LinearRegression()
        model.fit(dists.reshape(-1, 1), speeds)
        from sklearn.metrics import r2_score
        r2 = r2_score(speeds, model.predict(dists.reshape(-1, 1)))

        return {
            'by_k': by_k,
            'recovery_curves_by_k': recovery_curves_by_k,
            'speed_vs_k_r2': float(r2),
            'speed_vs_k_slope': float(model.coef_[0]),
            'smaller_k_faster': float(model.coef_[0]) < 0,
            'overall_mean_speed': float(np.mean(speeds)),
            'overall_mean_episodes_to_recover': float(np.mean(eps_to_recover)),
            'n_recovery_episodes': self.n_recovery_episodes,
            'recovery_threshold': self.recovery_threshold,
        }

    def _compute_overall_metrics(self) -> Dict[str, float]:
        """Compute overall transfer metrics."""
        source_returns = [p.get('total_return', 0) for p in self._data.source_performance]
        target_returns = [p.get('total_return', 0) for p in self._data.target_performance]
        baseline_returns = [p.get('total_return', 0) for p in self._data.random_baseline_performance]

        metrics = {
            'source_mean_return': float(np.mean(source_returns)),
            'target_mean_return': float(np.mean(target_returns)),
            'baseline_mean_return': float(np.mean(baseline_returns)),
            'n_pairs': len(self._data.level_pairs),
            'training_method': self.training_method,
        }

        if self._data.source_regrets:
            metrics['mean_source_regret'] = float(np.mean(self._data.source_regrets))
            metrics['mean_target_regret'] = float(np.mean(self._data.target_regrets))

        if self._data.episodes_to_recover:
            metrics['mean_episodes_to_recover'] = float(np.mean(self._data.episodes_to_recover))

        return metrics

    def _analyze_by_distance(self) -> Dict[str, Dict[str, float]]:
        """Analyze transfer metrics by mutation distance."""
        results_by_distance = {}

        if self._data.adaptation_speeds:
            dists = np.array(self._data.mutation_distances)
            for dist in self.mutation_distances:
                indices = [
                    i for i, pair in enumerate(self._data.level_pairs)
                    if pair.distance == dist
                ]
                if not indices:
                    continue
                target_returns = [self._data.target_performance[i].get('total_return', 0) for i in indices]
                results_by_distance[f'mutation_dist_{dist}'] = {
                    'n_pairs': len(indices),
                    'mean_target_return': float(np.mean(target_returns)),
                    'mean_adaptation_speed': float(np.mean([
                        self._data.adaptation_speeds[i] for i in indices
                        if i < len(self._data.adaptation_speeds)
                    ])),
                    'mean_episodes_to_recover': float(np.mean([
                        self._data.episodes_to_recover[i] for i in indices
                        if i < len(self._data.episodes_to_recover)
                    ])),
                }

        return results_by_distance

    def _analyze_behavioral_transfer(self) -> Dict[str, Any]:
        if not self._data.behavioral_transfer:
            return {}
        action_sims = [b.get('action_similarity', 0) for b in self._data.behavioral_transfer]
        value_corrs = [b.get('value_correlation', 0) for b in self._data.behavioral_transfer]
        return {
            'action_similarity': {'mean': float(np.mean(action_sims)), 'std': float(np.std(action_sims))},
            'value_correlation': {'mean': float(np.mean(value_corrs)), 'std': float(np.std(value_corrs))},
        }

    def _analyze_representational_transfer(self) -> Dict[str, Any]:
        if not self._data.representational_transfer:
            return {}
        cosine_sims = [r.get('cosine_similarity', 0) for r in self._data.representational_transfer]
        return {
            'hidden_state_similarity': {'mean': float(np.mean(cosine_sims)), 'std': float(np.std(cosine_sims))},
        }

    def _compare_to_baseline(self) -> Dict[str, Any]:
        from scipy import stats
        target_returns = [p.get('total_return', 0) for p in self._data.target_performance]
        baseline_returns = [p.get('total_return', 0) for p in self._data.random_baseline_performance]

        if len(target_returns) >= 5 and len(baseline_returns) >= 5:
            # Ensure same length for paired test
            min_len = min(len(target_returns), len(baseline_returns))
            t_stat, p_value = stats.ttest_rel(target_returns[:min_len], baseline_returns[:min_len])
        else:
            t_stat, p_value = 0.0, 1.0

        min_len = min(len(target_returns), len(baseline_returns))
        diff = np.array(target_returns[:min_len]) - np.array(baseline_returns[:min_len])
        cohens_d = float(np.mean(diff) / (np.std(diff) + 1e-6))

        return {
            'target_vs_baseline_diff': float(np.mean(target_returns) - np.mean(baseline_returns)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': cohens_d,
            'significant_transfer': p_value < 0.05 and cohens_d > 0.2,
        }

    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data including recovery curves."""
        if not self._results:
            raise ValueError("Must call analyze before visualize")

        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        viz_data = {
            'overall': self._results.get('overall', {}),
            'by_distance': self._results.get('by_mutation_distance', {}),
            'training_method': self.training_method,
        }

        figures = {}

        # Recovery curve plot: mean return vs episode number, one line per k
        adaptation = self._results.get('adaptation_speed', {})
        curves_by_k = adaptation.get('recovery_curves_by_k', {})
        if curves_by_k:
            try:
                fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                colors = ['blue', 'green', 'orange', 'red', 'purple']
                for i, (k_label, curve_data) in enumerate(sorted(curves_by_k.items())):
                    means = np.array(curve_data['mean'])
                    stds = np.array(curve_data['std'])
                    episodes = np.arange(1, len(means) + 1)
                    color = colors[i % len(colors)]
                    ax.plot(episodes, means, f'-o', color=color, label=k_label, markersize=4)
                    ax.fill_between(episodes, means - stds, means + stds, alpha=0.2, color=color)

                ax.set_xlabel('Episode')
                ax.set_ylabel('Mean Return')
                ax.set_title(f'Recovery Curves by Mutation Distance ({self.training_method.upper()})')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                figures['recovery_curves'] = fig
                plt.close(fig)
            except Exception:
                pass

        # Adaptation speed by k
        by_k = adaptation.get('by_k', {})
        if by_k:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                k_labels = sorted(by_k.keys())
                k_speeds = [by_k[k]['mean_speed'] for k in k_labels]
                k_eps = [by_k[k]['mean_episodes_to_recover'] for k in k_labels]

                ax = axes[0]
                ax.bar(range(len(k_labels)), k_speeds, tick_label=k_labels, alpha=0.8)
                ax.set_ylabel('Adaptation Speed (1/episodes)')
                ax.set_title('Speed by Mutation Distance')
                ax.grid(True, alpha=0.3, axis='y')

                ax = axes[1]
                ax.bar(range(len(k_labels)), k_eps, tick_label=k_labels, alpha=0.8, color='orange')
                ax.set_ylabel('Episodes to Recover')
                ax.set_title('Recovery Time by Mutation Distance')
                ax.grid(True, alpha=0.3, axis='y')

                plt.tight_layout()
                figures['adaptation_by_k'] = fig
                plt.close(fig)
            except Exception:
                pass

        viz_data['figures'] = figures
        if 'adaptation_speed' in self._results:
            viz_data['adaptation_speed'] = self._results['adaptation_speed']

        return viz_data
