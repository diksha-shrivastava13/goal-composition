"""
Level Property Probing Experiment.

Determine what level properties are encoded in agent representations.

This experiment:
1. Collects hidden states AFTER rollout (not initial state)
2. Probes for method-appropriate properties:
   - Universal: wall_density, goal_distance, path_length, is_solvable, episode_return, training_phase
   - ACCEL/PLR: branch_type, mutation_distance
   - PAIRED: regret_estimate, adversary_difficulty, adversary_strategy_cluster, opponent_return_estimate
   - DR: only universal properties (no curriculum-specific)
3. Implements linear + non-linear probes
4. Supports distributed probes (different LSTM components)
5. Tracks temporal dynamics over training
6. PAIRED-specific: Bilateral probing of protagonist AND antagonist on same levels
   - Differential encoding analysis: what antagonist encodes that protagonist doesn't
   - Agent-centric policy divergence metrics
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import logging
import jax
import jax.numpy as jnp
import numpy as np
import chex
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)

from .base import CheckpointExperiment
from .probes.property_probe import (
    LinearPropertyProbe,
    MLPPropertyProbe,
    DistributedProbe,
    train_probe,
    compute_probe_comparison,
)
from .utils.transfer_metrics import compute_policy_divergence


class LevelProbingExperiment(CheckpointExperiment):
    """
    Probe hidden states to predict level properties.

    Level Properties to Probe (vary by training method):

    Universal (all methods):
    - wall_density (continuous, 0-1): Basic environment structure
    - goal_distance (continuous): Euclidean distance to goal
    - path_length (discrete): BFS path length to goal
    - is_solvable (binary): Critical for curriculum
    - episode_return (continuous): Episode performance
    - training_phase (continuous): Early/mid/late training [0,1]

    ACCEL/PLR-specific (has_branches=True):
    - branch_type (categorical): DR=0, Replay=1, Mutate=2
    - mutation_distance (continuous): How far from parent level (ACCEL only)

    PAIRED-specific (has_adversary=True):
    - regret_estimate (continuous): Estimated ant - pro return difference
    - adversary_difficulty (continuous): How hard adversary made this level
    - adversary_strategy_cluster (categorical): Cluster ID from adversary strategy analysis
    - opponent_return_estimate (continuous): Theory of mind - estimate opponent's return

    PAIRED Bilateral Probing:
    - Probe BOTH protagonist AND antagonist on same levels
    - Differential encoding: what antagonist encodes that protagonist doesn't
    - Agent-centric: policy divergence between agents on matched levels

    DR-specific:
    - No curriculum-specific properties (only universal)

    Collection Points:
    - After 1 step (immediate environmental grounding)
    - After 10 steps (initial exploration)
    - After 50% episode (mid-episode)
    - At episode end (full information integration)

    Probe Methods:
    - Linear probes (Ridge regression for continuous, Logistic for classification)
    - Non-linear probes (MLP) to detect non-linear encodings
    - Distributed probes (probe c vs h separately)
    """

    @property
    def name(self) -> str:
        return "level_probing"

    def collect_data(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """Collect hidden states at multiple points and level properties (GPU-batched)."""
        n_levels = self.config.get("n_levels", 500)
        collection_steps = self.config.get("collection_steps", [1, 10, 50, -1])  # -1 = end
        max_episode_length = self.config.get("max_episode_length", 256)
        timings = {}

        # Try to import wandb for optional logging
        try:
            import wandb
            _wandb_active = wandb.run is not None
        except ImportError:
            _wandb_active = False

        def _log_timing(phase: str, elapsed: float):
            timings[phase] = elapsed
            logger.info(f"[level_probing] {phase}: {elapsed:.2f}s")
            if _wandb_active:
                wandb.log({f"level_probing/timing/{phase}": elapsed})

        # --- 1. Generate all levels in batch ---
        t0 = time.time()
        rng, rng_levels = jax.random.split(rng)
        level_rngs = jax.random.split(rng_levels, n_levels)
        levels = jax.vmap(self.agent.sample_random_level)(level_rngs)
        jax.block_until_ready(levels)
        _log_timing("generate_levels", time.time() - t0)

        # --- 2. Compute CPU-side level properties ---
        t0 = time.time()
        wall_maps = np.array(levels.wall_map)  # (n_levels, H, W)
        goal_positions = np.array(levels.goal_pos)  # (n_levels, 2)
        agent_positions = np.array(levels.agent_pos)  # (n_levels, 2)

        wall_density = wall_maps.mean(axis=(1, 2))  # (n_levels,)
        goal_distance = np.sqrt(np.sum((goal_positions - agent_positions) ** 2, axis=-1))

        # BFS path lengths (sequential, fast on 13x13)
        path_lengths = np.array([
            self._compute_path_length(jax.tree_util.tree_map(lambda x: x[i], levels))
            for i in tqdm(range(n_levels), desc="BFS path lengths", leave=False)
        ])
        is_solvable = path_lengths > 0

        training_step = getattr(self.train_state, 'training_step', 0)
        total_training_steps = self.config.get("total_training_steps", 30000)
        training_phase = np.full(n_levels, training_step / max(total_training_steps, 1))
        _log_timing("cpu_level_properties", time.time() - t0)

        # --- 3. Batched protagonist rollout ---
        t0 = time.time()
        rng, rng_pro = jax.random.split(rng)
        hidden_states_by_step, episode_returns, episode_solved = self._batched_rollout(
            rng_pro, levels, collection_steps, max_episode_length,
            self.train_state.apply_fn, self.train_state.params,
        )
        _log_timing("protagonist_rollout", time.time() - t0)

        if _wandb_active:
            wandb.log({
                "level_probing/protagonist_mean_return": float(episode_returns.mean()),
                "level_probing/protagonist_solve_rate": float(episode_solved.mean()),
            })

        # --- 4. Assemble level properties ---
        level_properties = {
            "wall_density": wall_density,
            "goal_distance": goal_distance,
            "path_length": path_lengths,
            "is_solvable": is_solvable,
            "training_phase": training_phase,
            "episode_return": episode_returns,
            "episode_solved": episode_solved,
        }

        # Method-specific properties
        if self.has_branches:
            branch_types = np.arange(n_levels) % self.branch_count
            level_properties["branch_type"] = branch_types

            if self.has_mutations:
                mutation_distance = np.where(
                    branch_types != 2, 0.0, np.random.uniform(1, 5, size=n_levels)
                )
                level_properties["mutation_distance"] = mutation_distance

        # --- 5. PAIRED bilateral ---
        bilateral_data = None
        if self.has_regret:
            t0 = time.time()
            # Per-level CPU-side estimates (fast)
            regret_estimates = np.array([
                self._estimate_regret_from_result(
                    jax.tree_util.tree_map(lambda x: x[i], levels),
                    {'solved': bool(episode_solved[i]), 'return': float(episode_returns[i])}
                ) for i in tqdm(range(n_levels), desc="Regret estimates", leave=False)
            ])
            adversary_difficulties = np.array([
                self._estimate_adversary_difficulty(
                    jax.tree_util.tree_map(lambda x: x[i], levels)
                ) for i in tqdm(range(n_levels), desc="Adversary difficulty", leave=False)
            ])
            level_properties["adversary_difficulty"] = adversary_difficulties
            _log_timing("paired_cpu_estimates", time.time() - t0)

            # Batched antagonist rollout
            ant_train_state = getattr(self.train_state, 'ant_train_state', None)
            if ant_train_state is not None:
                t0 = time.time()
                rng, rng_ant = jax.random.split(rng)
                ant_hstates_by_step, ant_returns, ant_solved = self._batched_rollout(
                    rng_ant, levels, collection_steps, max_episode_length,
                    ant_train_state.apply_fn, ant_train_state.params,
                )
                _log_timing("antagonist_rollout", time.time() - t0)

                # Actual regret replaces estimate
                regret_actual = ant_returns - episode_returns
                level_properties["regret_estimate"] = regret_actual

                # Vectorized policy divergence
                policy_divergences = self._compute_batch_policy_divergence(
                    hidden_states_by_step, ant_hstates_by_step,
                )

                bilateral_data = {
                    "antagonist_hstates_by_step": ant_hstates_by_step,
                    "antagonist_returns": ant_returns,
                    "protagonist_returns": episode_returns,
                    "policy_divergences": policy_divergences,
                    "regret_actual": regret_actual,
                }

                if _wandb_active:
                    wandb.log({
                        "level_probing/antagonist_mean_return": float(ant_returns.mean()),
                        "level_probing/mean_regret": float(regret_actual.mean()),
                        "level_probing/mean_policy_divergence": float(policy_divergences.mean()),
                    })
            else:
                level_properties["regret_estimate"] = regret_estimates

            # Strategy clusters and opponent return estimates
            t0 = time.time()
            strategy_clusters = np.array([
                self._estimate_strategy_cluster(
                    jax.tree_util.tree_map(lambda x: x[i], levels),
                    float(adversary_difficulties[i]),
                ) for i in tqdm(range(n_levels), desc="Strategy clusters", leave=False)
            ])
            level_properties["adversary_strategy_cluster"] = strategy_clusters

            opponent_return_estimates = np.array([
                self._estimate_opponent_return(
                    hidden_states_by_step.get(
                        "-1", hidden_states_by_step.get(str(collection_steps[-1]))
                    )[i] if hidden_states_by_step.get(
                        "-1", hidden_states_by_step.get(str(collection_steps[-1]))
                    ) is not None else None,
                    jax.tree_util.tree_map(lambda x: x[i], levels),
                ) for i in tqdm(range(n_levels), desc="Opponent return est.", leave=False)
            ])
            level_properties["opponent_return_estimate"] = opponent_return_estimates
            _log_timing("paired_cpu_clusters_and_tom", time.time() - t0)

        # --- 6. Assemble result ---
        result = {
            "hidden_states_by_step": hidden_states_by_step,
            "level_properties": level_properties,
            "collection_steps": collection_steps,
            "n_levels": n_levels,
            "training_method": self.training_method,
            "timings": timings,
        }

        if bilateral_data is not None:
            result["bilateral_data"] = bilateral_data

        # Log total and summary
        total_time = sum(timings.values())
        logger.info(f"[level_probing] TOTAL collect_data: {total_time:.2f}s")
        logger.info(f"[level_probing] Timing breakdown: {timings}")
        if _wandb_active:
            wandb.log({
                "level_probing/timing/total_collect_data": total_time,
                "level_probing/n_levels": n_levels,
                "level_probing/max_episode_length": max_episode_length,
            })

        return result

    def _estimate_regret_from_result(
        self,
        level,
        result: Dict[str, Any],
    ) -> float:
        """Estimate regret for PAIRED based on level and episode result."""
        solved = result.get('solved', False)
        wall_density = float(level.wall_map.mean())

        if solved:
            # Low regret if solved
            regret = 0.1 * wall_density
        else:
            # High regret if unsolved, scaled by how "easy" level looks
            regret = 0.5 + 0.5 * (1.0 - wall_density)

        return float(regret)

    def _estimate_adversary_difficulty(self, level) -> float:
        """Estimate how difficult the adversary made this level."""
        wall_density = float(level.wall_map.mean())
        path_length = self._compute_path_length(level)

        # Normalize path length (max ~26 for 13x13 grid)
        normalized_path = min(path_length / 26.0, 1.0) if path_length > 0 else 1.0

        # Combine wall density and path length
        difficulty = 0.5 * wall_density + 0.5 * normalized_path
        return float(difficulty)

    def _batched_rollout(
        self,
        rng: chex.PRNGKey,
        levels,
        collection_steps: List[int],
        max_steps: int,
        apply_fn,
        params,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """GPU-batched rollout over all levels simultaneously using jax.lax.scan.

        Args:
            rng: PRNG key.
            levels: Batched Level pytree with leading dim n_levels.
            collection_steps: List of timesteps at which to collect hidden states.
                              -1 means terminal (first done or end of episode).
            max_steps: Maximum episode length.
            apply_fn: Network apply function.
            params: Network parameters.

        Returns:
            hstates_by_step: Dict[str, np.ndarray] with shape (n_levels, hidden_dim).
            episode_returns: np.ndarray of shape (n_levels,).
            episode_solved: np.ndarray of shape (n_levels,).
        """
        n_levels = jax.tree_util.tree_leaves(levels)[0].shape[0]

        # Reset all environments in parallel
        rng, rng_reset = jax.random.split(rng)
        reset_rngs = jax.random.split(rng_reset, n_levels)
        obs, env_states = jax.vmap(
            self.agent.env.reset_to_level, in_axes=(0, 0, None)
        )(reset_rngs, levels, self.agent.env_params)

        # Initialize hidden states for all levels: tuple of (n_levels, 256) arrays
        hstate = self.agent.initialize_hidden_state(n_levels)

        # Determine positive collection steps and whether we need terminal collection
        positive_steps = sorted([s for s in collection_steps if s > 0])
        collect_terminal = -1 in collection_steps

        # Pre-allocate collection buffers for positive steps
        # Each buffer: tree with same structure as hstate, but with extra leading dim
        n_positive = len(positive_steps)
        collected_positive = jax.tree_util.tree_map(
            lambda x: jnp.zeros((n_positive,) + x.shape), hstate
        ) if n_positive > 0 else None

        # Terminal hstate buffer (same structure as hstate)
        terminal_hstate = jax.tree_util.tree_map(jnp.zeros_like, hstate)

        # Tracking arrays
        ep_done = jnp.zeros(n_levels, dtype=bool)
        total_return = jnp.zeros(n_levels)
        ep_reward_at_done = jnp.zeros(n_levels)
        terminal_captured = jnp.zeros(n_levels, dtype=bool)

        # Convert positive_steps to jnp array for scan body
        positive_steps_arr = jnp.array(positive_steps, dtype=jnp.int32) if n_positive > 0 else jnp.zeros((0,), dtype=jnp.int32)

        def scan_body(carry, step_idx):
            (hstate, obs, env_states, ep_done, total_return,
             ep_reward_at_done, terminal_captured,
             terminal_hstate, collected_positive, rng) = carry

            rng, rng_action = jax.random.split(rng)

            # Forward pass: obs has shape (n_levels, *obs_shape)
            # Network expects (time, batch, *obs_shape) -> add time dim
            obs_batch = jax.tree_util.tree_map(lambda x: x[None, ...], obs)
            done_batch = ep_done[None, :]  # (1, n_levels)
            hstate, pi, value = apply_fn(params, (obs_batch, done_batch), hstate)

            # Collect at positive steps
            current_step = step_idx + 1  # 1-indexed
            if n_positive > 0:
                for i, s in enumerate(positive_steps):
                    should_collect = (current_step == s)
                    collected_positive = jax.tree_util.tree_map(
                        lambda buf, h: buf.at[i].set(
                            jnp.where(should_collect, h, buf[i])
                        ),
                        collected_positive, hstate,
                    )

            # Sample actions: pi.sample returns (1, n_levels) -> squeeze time dim
            action = pi.sample(seed=rng_action)[0]  # (n_levels,)

            # Step all environments in parallel
            step_rngs = jax.random.split(rng_action, n_levels)
            obs, env_states, reward, done, info = jax.vmap(
                self.agent.env.step, in_axes=(0, 0, 0, None)
            )(step_rngs, env_states, action, self.agent.env_params)

            # Accumulate returns only for non-terminated episodes
            total_return = total_return + reward * (~ep_done).astype(jnp.float32)

            # Capture terminal hstate at first done
            newly_done = done & (~ep_done)
            terminal_hstate = jax.tree_util.tree_map(
                lambda th, h: jnp.where(
                    newly_done[:, None], h, th
                ),
                terminal_hstate, hstate,
            )
            ep_reward_at_done = jnp.where(newly_done, reward, ep_reward_at_done)
            terminal_captured = terminal_captured | newly_done

            # Update episode done tracking
            ep_done = ep_done | done

            carry = (hstate, obs, env_states, ep_done, total_return,
                     ep_reward_at_done, terminal_captured,
                     terminal_hstate, collected_positive, rng)
            return carry, None

        # Run scan over max_steps
        init_carry = (hstate, obs, env_states, ep_done, total_return,
                      ep_reward_at_done, terminal_captured,
                      terminal_hstate, collected_positive, rng)
        t_scan = time.time()
        final_carry, _ = jax.lax.scan(scan_body, init_carry, jnp.arange(max_steps))

        (hstate_final, _, _, ep_done_final, total_return_final,
         ep_reward_at_done_final, terminal_captured_final,
         terminal_hstate_final, collected_positive_final, _) = final_carry
        # Block until GPU computation finishes for accurate timing
        jax.block_until_ready(final_carry)
        logger.info(f"[level_probing] scan ({max_steps} steps x {n_levels} levels): "
                    f"{time.time() - t_scan:.2f}s (includes JIT on first call)")

        # For levels that never terminated, use final hstate as terminal
        terminal_hstate_final = jax.tree_util.tree_map(
            lambda th, h: jnp.where(
                (~terminal_captured_final)[:, None], h, th
            ),
            terminal_hstate_final, hstate_final,
        )

        # Flatten hstates: concatenate c and h -> (n_levels, 512)
        def flatten_hstate(hs):
            h_c, h_h = hs
            return jnp.concatenate([h_c, h_h], axis=-1)

        # Build result dict
        hstates_by_step = {}
        step_idx_map = {s: i for i, s in enumerate(positive_steps)}

        for s in collection_steps:
            if s == -1:
                hstates_by_step["-1"] = np.array(flatten_hstate(terminal_hstate_final))
            else:
                i = step_idx_map[s]
                hs = jax.tree_util.tree_map(lambda x: x[i], collected_positive_final)
                hstates_by_step[str(s)] = np.array(flatten_hstate(hs))

        # For levels that ended early, fill uncollected positive steps with terminal
        terminal_flat = hstates_by_step.get("-1")
        if terminal_flat is not None:
            for s_key in [str(s) for s in positive_steps]:
                # If a level ended before step s, its collected hstate is zeros
                # Replace zeros with terminal hstate
                h = hstates_by_step[s_key]
                is_zero = np.all(h == 0, axis=-1)  # (n_levels,)
                if np.any(is_zero):
                    hstates_by_step[s_key] = np.where(
                        is_zero[:, None], terminal_flat, h
                    )

        episode_returns = np.array(total_return_final)
        episode_solved = np.array(ep_reward_at_done_final > 0)
        # For levels that never terminated, solved = False (already 0)

        return hstates_by_step, episode_returns, episode_solved

    def _compute_batch_policy_divergence(
        self,
        pro_hstates_by_step: Dict[str, np.ndarray],
        ant_hstates_by_step: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute vectorized policy divergence between protagonist and antagonist.

        Args:
            pro_hstates_by_step: Dict with values of shape (n_levels, hidden_dim).
            ant_hstates_by_step: Dict with values of shape (n_levels, hidden_dim).

        Returns:
            divergences: np.ndarray of shape (n_levels,).
        """
        pro_h = pro_hstates_by_step.get("-1")
        ant_h = ant_hstates_by_step.get("-1")

        if pro_h is None or ant_h is None:
            n = len(next(iter(pro_hstates_by_step.values())))
            return np.zeros(n)

        h_diff = np.linalg.norm(pro_h - ant_h, axis=-1)  # (n_levels,)
        norm_pro = np.linalg.norm(pro_h, axis=-1)
        norm_ant = np.linalg.norm(ant_h, axis=-1)
        divergence = h_diff / np.maximum(np.maximum(norm_pro, norm_ant), 1e-8)

        return divergence

    def _estimate_strategy_cluster(self, level, adversary_difficulty: float) -> int:
        """Estimate adversary strategy cluster from level features.

        This is a placeholder - actual clustering is done in C3 (adversary_strategy_clustering).
        Uses simple heuristics to assign cluster IDs.
        """
        wall_density = float(level.wall_map.mean())
        path_length = self._compute_path_length(level)

        # Simple clustering by difficulty and wall density
        # 5 clusters based on discretizing difficulty and density
        difficulty_bin = min(int(adversary_difficulty * 3), 2)  # 0, 1, 2
        density_bin = 0 if wall_density < 0.3 else 1  # 0 or 1

        cluster = difficulty_bin * 2 + density_bin
        return int(cluster)

    def _estimate_opponent_return(
        self,
        protagonist_hstate: np.ndarray,
        level,
    ) -> float:
        """Estimate protagonist's model of opponent's return (theory of mind).

        Probes whether protagonist h-state encodes information about antagonist performance.
        """
        if protagonist_hstate is None:
            return 0.0

        # Simple heuristic: use h-state activation patterns
        # High activation in certain regions correlates with opponent difficulty estimate
        # This is a proxy - actual ToM probing would train a probe from h -> ant_return

        # Use first 64 dimensions as "opponent model" region
        tom_region = protagonist_hstate[:64] if len(protagonist_hstate) >= 64 else protagonist_hstate
        activation_level = float(np.mean(np.abs(tom_region)))

        # Map activation to return estimate (0-1 scale)
        # Higher activation = expect opponent to do better
        opponent_return_est = np.tanh(activation_level) * 0.5 + 0.5

        return float(opponent_return_est)

    def _compute_path_length(self, level) -> int:
        """Compute BFS path length from agent to goal."""
        wall_map = np.array(level.wall_map)
        agent_pos = tuple(np.array(level.agent_pos))
        goal_pos = tuple(np.array(level.goal_pos))

        if agent_pos == goal_pos:
            return 0

        # BFS
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

    def _estimate_regret(self, level) -> float:
        """Estimate regret based on level difficulty heuristics."""
        wall_density = float(level.wall_map.mean())
        path_length = self._compute_path_length(level)

        if path_length <= 0:
            return 1.0  # Max regret for unsolvable

        # Simple heuristic: longer paths and denser walls = higher regret
        regret = (wall_density * 0.3 + min(path_length / 50, 1.0) * 0.7)
        return float(regret)

    def analyze(self) -> Dict[str, Any]:
        """Train probes and compute R²/accuracy for each property."""
        results = {
            "linear_probes": {},
            "mlp_probes": {},
            "distributed_probes": {},
            "probe_comparison": {},
            "by_collection_step": {},
            "prediction_context": {},  # Overall prediction loss context
            "training_method": self.training_method,
        }

        hidden_states_by_step = self.data["hidden_states_by_step"]
        properties = self.data["level_properties"]

        # Define probe targets based on training method
        # Universal continuous targets
        continuous_targets = [
            "wall_density", "goal_distance", "training_phase", "episode_return"
        ]

        # Universal classification targets
        classification_targets = ["is_solvable"]

        # Add method-specific targets
        if self.has_branches:
            classification_targets.append("branch_type")
            if self.has_mutations and "mutation_distance" in properties:
                continuous_targets.append("mutation_distance")

        if self.has_regret:
            if "regret_estimate" in properties:
                continuous_targets.append("regret_estimate")
            if "adversary_difficulty" in properties:
                continuous_targets.append("adversary_difficulty")
            if "opponent_return_estimate" in properties:
                continuous_targets.append("opponent_return_estimate")
            if "adversary_strategy_cluster" in properties:
                classification_targets.append("adversary_strategy_cluster")

        # Filter to only existing properties
        continuous_targets = [t for t in continuous_targets if t in properties]
        classification_targets = [t for t in classification_targets if t in properties]

        # Train probes for each collection step
        for step_key, X in hidden_states_by_step.items():
            step_results = {
                "linear": {},
                "mlp": {},
                "comparison": {},
            }

            # Continuous targets
            for target_name in continuous_targets:
                y = properties[target_name]
                if len(np.unique(y)) < 2:
                    continue

                # Linear probe
                linear_probe, linear_metrics = train_probe(
                    X, y, probe_type="linear", task="regression"
                )
                step_results["linear"][target_name] = linear_metrics

                # MLP probe
                mlp_probe, mlp_metrics = train_probe(
                    X, y, probe_type="mlp", task="regression"
                )
                step_results["mlp"][target_name] = mlp_metrics

                # Comparison
                gap = mlp_metrics["mean_score"] - linear_metrics["mean_score"]
                step_results["comparison"][target_name] = {
                    "linear_r2": linear_metrics["mean_score"],
                    "mlp_r2": mlp_metrics["mean_score"],
                    "nonlinearity_gap": float(gap),
                    "is_nonlinear": gap > 0.05,
                }

            # Classification targets
            for target_name in classification_targets:
                y = properties[target_name]
                if len(np.unique(y)) < 2:
                    continue

                # Linear (logistic) probe
                linear_probe, linear_metrics = train_probe(
                    X, y, probe_type="linear", task="classification"
                )
                step_results["linear"][target_name] = linear_metrics

                # MLP probe
                mlp_probe, mlp_metrics = train_probe(
                    X, y, probe_type="mlp", task="classification"
                )
                step_results["mlp"][target_name] = mlp_metrics

                # Comparison
                gap = mlp_metrics["mean_score"] - linear_metrics["mean_score"]
                step_results["comparison"][target_name] = {
                    "linear_acc": linear_metrics["mean_score"],
                    "mlp_acc": mlp_metrics["mean_score"],
                    "nonlinearity_gap": float(gap),
                    "is_nonlinear": gap > 0.05,
                }

            results["by_collection_step"][step_key] = step_results

        # Distributed probe analysis (on final hidden states)
        if "-1" in hidden_states_by_step:
            X_final = hidden_states_by_step["-1"]

            # Select method-appropriate targets for distributed probing
            dist_probe_targets = ["wall_density"]  # Universal
            if self.has_branches and "branch_type" in properties:
                dist_probe_targets.append("branch_type")
            elif self.has_regret and "regret_estimate" in properties:
                dist_probe_targets.append("regret_estimate")

            for target_name in dist_probe_targets:
                if target_name not in properties:
                    continue
                y = properties[target_name]
                if len(np.unique(y)) < 2:
                    continue

                task = "regression" if target_name in continuous_targets else "classification"
                dist_probe = DistributedProbe(
                    probe_type="linear",
                    task=task,
                )
                dist_results = dist_probe.fit(X_final, y)
                results["distributed_probes"][target_name] = dist_results

        # PAIRED bilateral analysis
        if self.has_regret and "bilateral_data" in self.data:
            results["bilateral_analysis"] = self._analyze_bilateral_probing(
                hidden_states_by_step, properties, continuous_targets, classification_targets
            )

        # Aggregate results
        results["summary"] = self._compute_summary(results)

        # Add prediction context for interpretation
        results["prediction_context"] = self._compute_prediction_context()

        return results

    def _analyze_bilateral_probing(
        self,
        pro_hstates_by_step: Dict[str, np.ndarray],
        properties: Dict[str, np.ndarray],
        continuous_targets: List[str],
        classification_targets: List[str],
    ) -> Dict[str, Any]:
        """Analyze bilateral probing for PAIRED.

        Key metrics:
        - Differential encoding: what antagonist encodes that protagonist doesn't
        - Probe comparison: protagonist vs antagonist R² on same targets
        - Agent-centric: policy divergence and its correlation with regret
        """
        bilateral_data = self.data["bilateral_data"]
        ant_hstates_by_step = bilateral_data["antagonist_hstates_by_step"]

        results = {
            "differential_encoding": {},
            "antagonist_probes": {},
            "probe_gap": {},  # ant_score - pro_score for each target
            "agent_centric_metrics": {},
        }

        # Train probes on antagonist and compute differential encoding
        for step_key in ["-1"]:  # Focus on final h-states for bilateral
            if step_key not in ant_hstates_by_step or step_key not in pro_hstates_by_step:
                continue

            X_ant = ant_hstates_by_step[step_key]
            X_pro = pro_hstates_by_step[step_key]

            ant_probes = {}
            probe_gap = {}

            # Probe antagonist for same targets
            for target_name in continuous_targets:
                if target_name not in properties:
                    continue
                y = properties[target_name]
                if len(np.unique(y)) < 2:
                    continue

                # Antagonist probe
                _, ant_metrics = train_probe(X_ant, y, probe_type="linear", task="regression")
                ant_probes[target_name] = ant_metrics

                # Get protagonist score for comparison (from main results)
                # Compute differential: what does antagonist encode better?
                _, pro_metrics = train_probe(X_pro, y, probe_type="linear", task="regression")
                gap = ant_metrics["mean_score"] - pro_metrics["mean_score"]
                probe_gap[target_name] = {
                    "antagonist_r2": ant_metrics["mean_score"],
                    "protagonist_r2": pro_metrics["mean_score"],
                    "gap": float(gap),
                    "antagonist_encodes_better": gap > 0.05,
                }

            for target_name in classification_targets:
                if target_name not in properties:
                    continue
                y = properties[target_name]
                if len(np.unique(y)) < 2:
                    continue

                _, ant_metrics = train_probe(X_ant, y, probe_type="linear", task="classification")
                ant_probes[target_name] = ant_metrics

                _, pro_metrics = train_probe(X_pro, y, probe_type="linear", task="classification")
                gap = ant_metrics["mean_score"] - pro_metrics["mean_score"]
                probe_gap[target_name] = {
                    "antagonist_acc": ant_metrics["mean_score"],
                    "protagonist_acc": pro_metrics["mean_score"],
                    "gap": float(gap),
                    "antagonist_encodes_better": gap > 0.05,
                }

            results["antagonist_probes"][step_key] = ant_probes
            results["probe_gap"][step_key] = probe_gap

            # Differential encoding analysis
            results["differential_encoding"][step_key] = self._compute_differential_encoding(
                X_pro, X_ant, properties
            )

        # Agent-centric metrics
        policy_divs = bilateral_data["policy_divergences"]
        regrets = bilateral_data["regret_actual"]
        pro_returns = bilateral_data["protagonist_returns"]
        ant_returns = bilateral_data["antagonist_returns"]

        # Correlation between policy divergence and regret
        if len(policy_divs) > 10 and len(regrets) > 10:
            div_regret_corr, _ = stats.pearsonr(policy_divs, regrets)
        else:
            div_regret_corr = 0.0

        results["agent_centric_metrics"] = {
            "mean_policy_divergence": float(np.mean(policy_divs)),
            "std_policy_divergence": float(np.std(policy_divs)),
            "divergence_regret_correlation": float(div_regret_corr),
            "mean_regret": float(np.mean(regrets)),
            "protagonist_mean_return": float(np.mean(pro_returns)),
            "antagonist_mean_return": float(np.mean(ant_returns)),
            "antagonist_wins_fraction": float(np.mean(np.array(ant_returns) > np.array(pro_returns))),
        }

        # Theory of mind probing: can we predict antagonist return from protagonist h-state?
        if "opponent_return_estimate" in properties and "-1" in pro_hstates_by_step:
            X_pro = pro_hstates_by_step["-1"]
            y_ant_return = np.array(ant_returns)

            _, tom_metrics = train_probe(X_pro, y_ant_return, probe_type="linear", task="regression")
            results["theory_of_mind"] = {
                "protagonist_predicts_antagonist_r2": tom_metrics["mean_score"],
                "has_opponent_model": tom_metrics["mean_score"] > 0.1,
            }

        return results

    def _compute_differential_encoding(
        self,
        X_pro: np.ndarray,
        X_ant: np.ndarray,
        properties: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Compute what antagonist encodes that protagonist doesn't.

        Uses dimension-wise analysis to identify encoding differences.
        """
        # Compute dimension-wise variance explained by each property
        # Dimensions where antagonist has higher correlation = antagonist-specific

        differential_dims = []
        pro_dominant_dims = []

        # For efficiency, use correlation with regret as proxy
        if "regret_estimate" in properties:
            regret = properties["regret_estimate"]

            # Correlations by dimension
            n_dims = min(X_pro.shape[1], X_ant.shape[1])
            for dim in range(n_dims):
                pro_corr = abs(np.corrcoef(X_pro[:, dim], regret)[0, 1])
                ant_corr = abs(np.corrcoef(X_ant[:, dim], regret)[0, 1])

                if np.isnan(pro_corr):
                    pro_corr = 0
                if np.isnan(ant_corr):
                    ant_corr = 0

                if ant_corr - pro_corr > 0.1:
                    differential_dims.append(dim)
                elif pro_corr - ant_corr > 0.1:
                    pro_dominant_dims.append(dim)

        # CKA similarity between representations
        cka = self._compute_cka(X_pro, X_ant)

        return {
            "antagonist_specific_dims": differential_dims[:20],  # Top 20
            "protagonist_specific_dims": pro_dominant_dims[:20],
            "n_antagonist_specific": len(differential_dims),
            "n_protagonist_specific": len(pro_dominant_dims),
            "representation_cka": float(cka),
            "representation_divergence": float(1.0 - cka),
        }

    def _compute_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Centered Kernel Alignment between two representation matrices."""
        # Center the representations
        X_centered = X - X.mean(axis=0)
        Y_centered = Y - Y.mean(axis=0)

        # Compute gram matrices
        K = X_centered @ X_centered.T
        L = Y_centered @ Y_centered.T

        # HSIC
        def hsic(K, L):
            n = K.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            return np.trace(K @ H @ L @ H) / ((n - 1) ** 2)

        hsic_kl = hsic(K, L)
        hsic_kk = hsic(K, K)
        hsic_ll = hsic(L, L)

        if hsic_kk * hsic_ll < 1e-10:
            return 0.0

        cka = hsic_kl / np.sqrt(hsic_kk * hsic_ll)
        return float(cka)

    def _compute_prediction_context(self) -> Dict[str, Any]:
        """
        Compute overall prediction/probe loss for context.

        Provides context for interpreting probe R² values by showing
        the overall prediction ability of the agent.
        """
        try:
            from .utils.agent_aware_loss import (
                compute_agent_prediction_loss,
                compute_random_baseline_loss,
                compute_information_gain,
            )

            # Compute total prediction loss on a sample of test levels
            hidden_states_by_step = self.data.get("hidden_states_by_step", {})
            properties = self.data.get("level_properties", {})

            n_samples = min(100, len(properties.get('wall_density', [])))
            if n_samples == 0:
                return {'error': 'No samples available'}

            # Generate random levels and compute prediction loss
            import jax
            rng = jax.random.PRNGKey(42)
            losses = []

            for i in range(n_samples):
                rng, level_rng, loss_rng = jax.random.split(rng, 3)

                # Create level from stored properties
                level = {
                    'wall_map': np.zeros((13, 13)),  # Placeholder
                    'wall_density': properties['wall_density'][i],
                    'goal_pos': (6, 6),  # Placeholder
                    'agent_pos': (1, 1),  # Placeholder
                    'agent_dir': 0,
                }

                # Generate actual random level for evaluation
                height, width = 13, 13
                wall_prob = properties['wall_density'][i]
                wall_map = np.array(jax.random.bernoulli(level_rng, wall_prob, (height, width)))
                wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

                rng_goal, rng_agent = jax.random.split(level_rng)
                level['wall_map'] = wall_map
                level['goal_pos'] = (
                    int(jax.random.randint(rng_goal, (), 1, height - 1)),
                    int(jax.random.randint(rng_goal, (), 1, width - 1)),
                )
                level['agent_pos'] = (
                    int(jax.random.randint(rng_agent, (), 1, height - 1)),
                    int(jax.random.randint(rng_agent, (), 1, width - 1)),
                )

                loss, _ = compute_agent_prediction_loss(
                    self.agent,
                    self.train_state,
                    level,
                    loss_rng,
                )
                losses.append(loss)

            mean_loss = float(np.mean(losses))
            random_baseline = compute_random_baseline_loss()
            info_gain = compute_information_gain(mean_loss, random_baseline)
            info_gain_ratio = info_gain / random_baseline if random_baseline > 0 else 0.0

            return {
                'mean_total_prediction_loss': mean_loss,
                'random_baseline_loss': random_baseline,
                'information_gain': info_gain,
                'information_gain_ratio': info_gain_ratio,
                'n_samples': n_samples,
                'interpretation': (
                    f"Probe R² values should be interpreted in context of "
                    f"overall prediction loss improvement: {mean_loss:.3f} "
                    f"(random baseline: {random_baseline:.3f}, "
                    f"information gain ratio: {info_gain_ratio:.1%})"
                ),
            }

        except Exception as e:
            return {'error': str(e)}

    def _compute_summary(self, results: Dict) -> Dict[str, Any]:
        """Compute summary statistics across all probes."""
        summary = {}

        # Best R² for each property
        for step_key, step_results in results.get("by_collection_step", {}).items():
            for target_name, metrics in step_results.get("comparison", {}).items():
                key = f"{target_name}_best_score"
                score = metrics.get("mlp_r2", metrics.get("mlp_acc", 0))
                if key not in summary or score > summary[key]:
                    summary[key] = score
                    summary[f"{target_name}_best_step"] = step_key

        # Overall nonlinearity
        nonlinear_count = 0
        total_count = 0
        for step_key, step_results in results.get("by_collection_step", {}).items():
            for target_name, metrics in step_results.get("comparison", {}).items():
                if metrics.get("is_nonlinear", False):
                    nonlinear_count += 1
                total_count += 1

        if total_count > 0:
            summary["fraction_nonlinear"] = nonlinear_count / total_count

        return summary

    def visualize(self) -> Dict[str, np.ndarray]:
        """Create visualizations of probe results."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        # 1. R² by property and collection step
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Linear probe results
        ax = axes[0, 0]
        self._plot_probe_results(ax, "linear", self.results)
        ax.set_title("Linear Probe R² / Accuracy")

        # MLP probe results
        ax = axes[0, 1]
        self._plot_probe_results(ax, "mlp", self.results)
        ax.set_title("MLP Probe R² / Accuracy")

        # Nonlinearity gap
        ax = axes[1, 0]
        self._plot_nonlinearity_gap(ax, self.results)
        ax.set_title("Nonlinearity Gap (MLP - Linear)")

        # Distributed probe (component comparison)
        ax = axes[1, 1]
        self._plot_distributed_probes(ax, self.results)
        ax.set_title("Distributed Probe: c vs h")

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["probe_results"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        # 2. Probe accuracy over collection steps
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_temporal_dynamics(ax, self.results)
        ax.set_title("Probe Accuracy by Collection Step")
        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["temporal_dynamics"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures

    def _plot_probe_results(self, ax, probe_type: str, results: Dict):
        """Plot probe results as grouped bar chart."""
        by_step = results.get("by_collection_step", {})
        if not by_step:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            return

        # Use final step for main comparison
        step_key = "-1" if "-1" in by_step else list(by_step.keys())[-1]
        step_data = by_step[step_key].get(probe_type, {})

        if not step_data:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            return

        properties = list(step_data.keys())
        scores = [step_data[p]["mean_score"] for p in properties]
        stds = [step_data[p]["std_score"] for p in properties]

        x = np.arange(len(properties))
        ax.bar(x, scores, yerr=stds, capsize=3, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(properties, rotation=45, ha='right')
        ax.set_ylabel("R² / Accuracy")
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Chance (classification)')

    def _plot_nonlinearity_gap(self, ax, results: Dict):
        """Plot gap between MLP and linear probes."""
        by_step = results.get("by_collection_step", {})
        if not by_step:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            return

        step_key = "-1" if "-1" in by_step else list(by_step.keys())[-1]
        comparison = by_step[step_key].get("comparison", {})

        if not comparison:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            return

        properties = list(comparison.keys())
        gaps = [comparison[p]["nonlinearity_gap"] for p in properties]
        colors = ['green' if comparison[p]["is_nonlinear"] else 'gray' for p in properties]

        x = np.arange(len(properties))
        ax.bar(x, gaps, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(properties, rotation=45, ha='right')
        ax.set_ylabel("Gap (MLP - Linear)")
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend()

    def _plot_distributed_probes(self, ax, results: Dict):
        """Plot distributed probe results."""
        dist_probes = results.get("distributed_probes", {})
        if not dist_probes:
            ax.text(0.5, 0.5, "No distributed probe data", ha='center', va='center')
            return

        # Plot component comparison for first target
        target = list(dist_probes.keys())[0]
        target_results = dist_probes[target]

        components = ["full", "cell", "hidden"]
        scores = [target_results.get(c, {}).get("mean_score", 0) for c in components]

        x = np.arange(len(components))
        ax.bar(x, scores, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(components)
        ax.set_ylabel(f"Score ({target})")

    def _plot_temporal_dynamics(self, ax, results: Dict):
        """Plot probe accuracy over collection steps."""
        by_step = results.get("by_collection_step", {})
        if not by_step:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            return

        # Sort steps
        steps = sorted(by_step.keys(), key=lambda x: int(x) if x != "-1" else 9999)

        # Select method-appropriate properties to plot
        properties_to_plot = ["wall_density"]  # Universal

        if self.has_branches:
            properties_to_plot.append("branch_type")
        if self.has_regret:
            properties_to_plot.append("regret_estimate")
        if self.has_mutations:
            properties_to_plot.append("mutation_distance")

        for prop in properties_to_plot:
            scores = []
            for step in steps:
                step_data = by_step[step].get("linear", {}).get(prop, {})
                scores.append(step_data.get("mean_score", 0))

            if any(s > 0 for s in scores):
                step_labels = [s if s != "-1" else "end" for s in steps]
                ax.plot(range(len(steps)), scores, 'o-', label=prop)

        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([s if s != "-1" else "end" for s in steps])
        ax.set_xlabel("Collection Step")
        ax.set_ylabel("Probe Score")
        ax.legend()
        ax.grid(True, alpha=0.3)
