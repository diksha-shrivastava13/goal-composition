"""
Value Calibration Experiment.

Assess whether V(s) accurately predicts episode outcomes.

Enhanced version with:
1. Multi-point V(s_t) collection at t ∈ {1, 10, 50, 100, 200}
2. Branch-conditioned calibration (separate ECE per branch)
3. ∂V/∂(goal_distance) gradient analysis
4. Temporal consistency checks

PAIRED-specific:
5. Regret-conditioned calibration (high/medium/low regret levels)
6. Bilateral comparison: protagonist vs antagonist calibration
7. Key question: Does protagonist underestimate V(s) on high-regret levels?
"""

from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import chex
from scipy.stats import pearsonr

from .base import CheckpointExperiment
from .utils.calibration_utils import (
    compute_multi_point_calibration,
    compute_branch_conditioned_ece,
    compute_temporal_consistency,
    compute_calibration_by_difficulty,
)


class ValueCalibrationExperiment(CheckpointExperiment):
    """
    Test value function calibration.

    IMPORTANT DISTINCTION:
    This experiment analyzes V(s) prediction calibration, which is DISTINCT from
    curriculum prediction loss. These measure different aspects of agent behavior:

    - V(s) calibration (this experiment): How accurately does the value head
      predict episode outcomes (expected discounted returns)? This measures
      the agent's ability to evaluate states for decision-making.

    - Curriculum prediction loss (cross_agent_comparison.py, etc.): How well
      does the agent encode level information (wall_map, goal_pos, etc.) in
      its hidden state? This measures curriculum awareness.

    Both are valid and important metrics:
    - V(s) calibration: Quality of learned value function for RL
    - Curriculum prediction: Information encoding about level structure

    For causal claims about CURRICULUM AWARENESS emergence, use curriculum
    prediction loss from experiments like cross_agent_comparison.py, which
    tracks actual prediction/probe loss.

    Method:
    1. At multiple timesteps, record V(s_t)
    2. After episode, compute actual returns G_t from each point
    3. Compute calibration metrics per timestep and per branch

    Metrics:
    - Pearson correlation(V(s_t), G_t)
    - Mean absolute error
    - Expected Calibration Error (ECE) per timestep
    - ECE per curriculum branch
    - Temporal consistency (V should decrease with discounting)
    - Overconfidence detection: |V - G| when V >> G
    """

    @property
    def name(self) -> str:
        return "value_calibration"

    def collect_data(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """Collect value predictions and actual returns at multiple timesteps."""
        n_episodes = self.config.get("n_episodes", 500)
        timesteps = self.config.get("calibration_timesteps", [1, 10, 50, 100, 200])
        max_episode_length = self.config.get("max_episode_length", 256)
        gamma = self.config.get("gamma", 0.995)

        # Data containers
        values_by_timestep = {t: [] for t in timesteps}
        returns_by_timestep = {t: [] for t in timesteps}
        episode_data = {
            "initial_values": [],
            "final_returns": [],
            "branches": [],
            "difficulties": [],
            "solved": [],
            "episode_lengths": [],
            "values_over_time": [],
            "rewards_over_time": [],
            "goal_distances": [],
        }

        # PAIRED-specific data containers
        paired_data = None
        if self.has_regret:
            paired_data = {
                "regrets": [],
                "antagonist_values": [],
                "antagonist_returns": [],
                "protagonist_returns": [],
            }

        for i in range(n_episodes):
            rng, rng_level, rng_reset, rng_rollout = jax.random.split(rng, 4)

            # Generate level
            level = self.agent.sample_random_level(rng_level)

            # Simulated branch assignment
            branch = i % 3  # Cycle through DR=0, Replay=1, Mutate=2

            # Difficulty proxy
            wall_density = float(level.wall_map.mean())

            # Goal distance
            goal_pos = np.array(level.goal_pos)
            agent_pos = np.array(level.agent_pos)
            goal_distance = float(np.sqrt(np.sum((goal_pos - agent_pos) ** 2)))

            # Reset environment
            obs, env_state = self.agent.env.reset_to_level(
                rng_reset, level, self.agent.env_params
            )

            # Initialize hidden state
            hstate = self.agent.initialize_hidden_state(1)

            # Run episode and collect data
            values = []
            rewards = []
            done = False

            for step in range(max_episode_length):
                rng_rollout, rng_action = jax.random.split(rng_rollout)

                # Get value prediction
                obs_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs)
                done_batch = jnp.array([[done]])
                hstate, pi, value = self.train_state.apply_fn(
                    self.train_state.params, (obs_batch, done_batch), hstate
                )
                v = float(value[0, 0])
                values.append(v)

                # Step environment
                action = pi.sample(seed=rng_action)[0, 0]
                obs, env_state, reward, done, _ = self.agent.env.step(
                    rng_action, env_state, action, self.agent.env_params
                )
                rewards.append(float(reward))

                if done:
                    break

            # Pad to max_length for consistent arrays
            actual_length = len(values)
            values = values + [np.nan] * (max_episode_length - len(values))
            rewards = rewards + [0.0] * (max_episode_length - len(rewards))

            values = np.array(values)
            rewards = np.array(rewards)

            # Compute returns from each timestep
            returns_from_t = self._compute_returns_from_timestep(
                rewards, gamma, max_episode_length
            )

            # Collect at specified timesteps
            for t in timesteps:
                if t < actual_length:
                    values_by_timestep[t].append(values[t])
                    returns_by_timestep[t].append(returns_from_t[t])

            # Episode-level data
            episode_data["initial_values"].append(values[0])
            episode_data["final_returns"].append(returns_from_t[0])
            episode_data["branches"].append(branch)
            episode_data["difficulties"].append(wall_density)
            episode_data["solved"].append(float(rewards[actual_length - 1]) > 0 if actual_length > 0 else False)
            episode_data["episode_lengths"].append(actual_length)
            episode_data["values_over_time"].append(values)
            episode_data["rewards_over_time"].append(rewards)
            episode_data["goal_distances"].append(goal_distance)

            # PAIRED-specific: collect antagonist data and compute regret
            if paired_data is not None:
                rng, ant_rng = jax.random.split(rng)
                ant_result = self._run_antagonist_episode(ant_rng, level, gamma, max_episode_length)
                paired_data["antagonist_values"].append(ant_result["initial_value"])
                paired_data["antagonist_returns"].append(ant_result["final_return"])
                paired_data["protagonist_returns"].append(returns_from_t[0])
                # Regret = antagonist return - protagonist return
                regret = ant_result["final_return"] - returns_from_t[0]
                paired_data["regrets"].append(regret)

        # Convert to arrays
        values_by_timestep = {t: np.array(v) for t, v in values_by_timestep.items()}
        returns_by_timestep = {t: np.array(r) for t, r in returns_by_timestep.items()}
        episode_data = {k: np.array(v) for k, v in episode_data.items()}

        result = {
            "values_by_timestep": values_by_timestep,
            "returns_by_timestep": returns_by_timestep,
            "episode_data": episode_data,
            "timesteps": timesteps,
            "n_episodes": n_episodes,
            "gamma": gamma,
        }

        # Add PAIRED-specific data
        if paired_data is not None:
            result["paired_data"] = {k: np.array(v) for k, v in paired_data.items()}

        return result

    def _compute_returns_from_timestep(
        self,
        rewards: np.ndarray,
        gamma: float,
        max_length: int,
    ) -> np.ndarray:
        """Compute discounted returns from each timestep."""
        returns = np.zeros(max_length)
        running_return = 0.0

        for t in range(len(rewards) - 1, -1, -1):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return

        return returns

    def _run_antagonist_episode(
        self,
        rng: chex.PRNGKey,
        level,
        gamma: float,
        max_episode_length: int,
    ) -> Dict[str, float]:
        """Run antagonist episode and collect value/return data (PAIRED bilateral)."""
        ant_train_state = getattr(self.train_state, 'ant_train_state', None)

        if ant_train_state is None:
            return {
                "initial_value": 0.0,
                "final_return": 0.0,
            }

        rng, rng_reset = jax.random.split(rng)

        # Reset environment
        obs, env_state = self.agent.env.reset_to_level(
            rng_reset, level, self.agent.env_params
        )

        # Initialize antagonist hidden state
        hstate = self.agent.initialize_hidden_state(1)

        values = []
        rewards = []

        for step in range(max_episode_length):
            rng, rng_action = jax.random.split(rng)

            # Get value prediction from antagonist
            obs_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs)
            done_batch = jnp.array([[False]])
            hstate, pi, value = ant_train_state.apply_fn(
                ant_train_state.params, (obs_batch, done_batch), hstate
            )
            v = float(value[0, 0])
            values.append(v)

            # Step environment
            action = pi.sample(seed=rng_action)[0, 0]
            obs, env_state, reward, done, _ = self.agent.env.step(
                rng_action, env_state, action, self.agent.env_params
            )
            rewards.append(float(reward))

            if done:
                break

        # Compute return
        rewards = np.array(rewards)
        final_return = 0.0
        for r in reversed(rewards):
            final_return = r + gamma * final_return

        return {
            "initial_value": values[0] if values else 0.0,
            "final_return": final_return,
        }

    def analyze(self) -> Dict[str, Any]:
        """Compute calibration metrics."""
        results = {}

        values_by_t = self.data["values_by_timestep"]
        returns_by_t = self.data["returns_by_timestep"]
        episode_data = self.data["episode_data"]
        gamma = self.data["gamma"]

        # 1. Multi-point calibration
        results["multi_point"] = {}
        for t in self.data["timesteps"]:
            if t not in values_by_t or len(values_by_t[t]) < 10:
                continue

            v = values_by_t[t]
            r = returns_by_t[t]

            # Basic metrics
            corr, p_value = pearsonr(v, r)
            mae = np.mean(np.abs(v - r))
            ece = self._compute_ece(v, r)

            # Overconfidence
            overconfident_mask = (v - r) > 0.1
            overconfidence_rate = float(overconfident_mask.mean())

            results["multi_point"][f"t={t}"] = {
                "correlation": float(corr),
                "p_value": float(p_value),
                "mae": float(mae),
                "ece": float(ece),
                "overconfidence_rate": overconfidence_rate,
                "n_samples": len(v),
            }

        # 2. Branch-conditioned calibration
        results["by_branch"] = compute_branch_conditioned_ece(
            episode_data["initial_values"],
            episode_data["final_returns"],
            episode_data["branches"],
        )

        # 3. Calibration by difficulty
        results["by_difficulty"] = compute_calibration_by_difficulty(
            episode_data["initial_values"],
            episode_data["final_returns"],
            episode_data["difficulties"],
        )

        # 4. Temporal consistency
        results["temporal_consistency"] = compute_temporal_consistency(
            episode_data["values_over_time"],
            gamma,
        )

        # 5. Value-goal distance correlation
        results["value_goal_correlation"] = self._analyze_value_goal_correlation(
            episode_data["initial_values"],
            episode_data["goal_distances"],
            episode_data["final_returns"],
        )

        # 6. PAIRED-specific: regret-conditioned calibration
        if "paired_data" in self.data:
            results["regret_conditioned"] = self._analyze_regret_conditioned_calibration()
            results["bilateral_comparison"] = self._analyze_bilateral_calibration()

        # 7. Overall summary
        results["summary"] = self._compute_summary(results)

        return results

    def _analyze_regret_conditioned_calibration(self) -> Dict[str, Any]:
        """
        Analyze calibration conditioned on regret level (PAIRED).

        Key question: Does protagonist underestimate V(s) on high-regret levels?
        """
        paired_data = self.data["paired_data"]
        episode_data = self.data["episode_data"]

        regrets = paired_data["regrets"]
        pro_values = episode_data["initial_values"]
        pro_returns = episode_data["final_returns"]

        # Split into regret terciles
        regret_33 = np.percentile(regrets, 33)
        regret_66 = np.percentile(regrets, 66)

        results = {}

        conditions = [
            ("low", regrets <= regret_33),
            ("medium", (regrets > regret_33) & (regrets <= regret_66)),
            ("high", regrets > regret_66),
        ]

        for name, mask in conditions:
            if mask.sum() < 10:
                results[name] = {"error": "Insufficient samples"}
                continue

            v = pro_values[mask]
            r = pro_returns[mask]

            # Calibration metrics
            corr, _ = pearsonr(v, r)
            mae = np.mean(np.abs(v - r))
            ece = self._compute_ece(v, r)

            # Value bias: positive = overestimation, negative = underestimation
            value_bias = np.mean(v - r)

            results[name] = {
                "correlation": float(corr),
                "mae": float(mae),
                "ece": float(ece),
                "value_bias": float(value_bias),
                "mean_value": float(np.mean(v)),
                "mean_return": float(np.mean(r)),
                "mean_regret": float(np.mean(regrets[mask])),
                "n_samples": int(mask.sum()),
            }

        # Key finding: Does protagonist underestimate on high-regret?
        if "high" in results and "low" in results:
            high_bias = results["high"].get("value_bias", 0)
            low_bias = results["low"].get("value_bias", 0)
            results["protagonist_underestimates_high_regret"] = high_bias < low_bias
            results["underestimation_gap"] = float(low_bias - high_bias)

        return results

    def _analyze_bilateral_calibration(self) -> Dict[str, Any]:
        """
        Compare protagonist vs antagonist calibration (PAIRED bilateral).
        """
        paired_data = self.data["paired_data"]
        episode_data = self.data["episode_data"]

        pro_values = episode_data["initial_values"]
        pro_returns = episode_data["final_returns"]
        ant_values = paired_data["antagonist_values"]
        ant_returns = paired_data["antagonist_returns"]
        regrets = paired_data["regrets"]

        results = {}

        # Overall calibration comparison
        if len(pro_values) >= 10:
            pro_corr, _ = pearsonr(pro_values, pro_returns)
            pro_ece = self._compute_ece(pro_values, pro_returns)
            results["protagonist"] = {
                "correlation": float(pro_corr),
                "ece": float(pro_ece),
                "mean_value": float(np.mean(pro_values)),
                "mean_return": float(np.mean(pro_returns)),
            }

        if len(ant_values) >= 10:
            ant_corr, _ = pearsonr(ant_values, ant_returns)
            ant_ece = self._compute_ece(ant_values, ant_returns)
            results["antagonist"] = {
                "correlation": float(ant_corr),
                "ece": float(ant_ece),
                "mean_value": float(np.mean(ant_values)),
                "mean_return": float(np.mean(ant_returns)),
            }

        # Who is better calibrated?
        if "protagonist" in results and "antagonist" in results:
            results["antagonist_better_calibrated"] = (
                results["antagonist"]["ece"] < results["protagonist"]["ece"]
            )
            results["calibration_gap"] = (
                results["protagonist"]["ece"] - results["antagonist"]["ece"]
            )

        # Regret-conditioned bilateral comparison
        regret_66 = np.percentile(regrets, 66)
        high_regret_mask = regrets > regret_66

        if high_regret_mask.sum() >= 10:
            # On high-regret levels, who predicts better?
            pro_high = pro_values[high_regret_mask]
            pro_high_r = pro_returns[high_regret_mask]
            ant_high = ant_values[high_regret_mask]
            ant_high_r = ant_returns[high_regret_mask]

            pro_high_ece = self._compute_ece(pro_high, pro_high_r)
            ant_high_ece = self._compute_ece(ant_high, ant_high_r)

            results["high_regret_comparison"] = {
                "protagonist_ece": float(pro_high_ece),
                "antagonist_ece": float(ant_high_ece),
                "protagonist_worse_on_high_regret": pro_high_ece > ant_high_ece,
            }

        return results

    def _compute_ece(self, values: np.ndarray, returns: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        v_min, v_max = values.min(), values.max()
        if v_max - v_min < 1e-6:
            return float(np.abs(values.mean() - returns.mean()))

        bin_edges = np.linspace(v_min - 1e-5, v_max + 1e-5, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (values >= bin_edges[i]) & (values < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_conf = values[mask].mean()
                bin_acc = returns[mask].mean()
                bin_size = mask.sum()
                ece += bin_size * np.abs(bin_conf - bin_acc)

        return float(ece / len(values))

    def _analyze_value_goal_correlation(
        self,
        values: np.ndarray,
        goal_distances: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        Analyze relationship between value and goal distance.

        Tests: ∂V/∂(goal_distance) - should be negative for goal-directed behavior.
        """
        if len(values) < 10:
            return {"error": "Insufficient data"}

        # Correlation between value and goal distance
        corr_vg, _ = pearsonr(values, goal_distances)

        # Expected: negative correlation (closer to goal = higher value)
        goal_directed = corr_vg < -0.1

        # Value for near vs far goals
        median_dist = np.median(goal_distances)
        near_mask = goal_distances < median_dist
        far_mask = goal_distances >= median_dist

        v_near = values[near_mask].mean() if near_mask.sum() > 0 else 0
        v_far = values[far_mask].mean() if far_mask.sum() > 0 else 0

        return {
            "value_goal_correlation": float(corr_vg),
            "goal_directed": bool(goal_directed),
            "value_near_goal": float(v_near),
            "value_far_goal": float(v_far),
            "value_difference": float(v_near - v_far),
        }

    def _compute_summary(self, results: Dict) -> Dict[str, Any]:
        """Compute summary statistics."""
        summary = {}

        # Best calibration timestep
        best_ece = float('inf')
        best_timestep = None
        for t, metrics in results.get("multi_point", {}).items():
            if metrics["ece"] < best_ece:
                best_ece = metrics["ece"]
                best_timestep = t

        summary["best_ece"] = best_ece
        summary["best_timestep"] = best_timestep

        # Initial value calibration
        if "t=1" in results.get("multi_point", {}):
            t1 = results["multi_point"]["t=1"]
            summary["initial_correlation"] = t1["correlation"]
            summary["initial_ece"] = t1["ece"]

        # Branch calibration comparison
        by_branch = results.get("by_branch", {})
        if "Replay" in by_branch and "DR" in by_branch:
            summary["replay_better_calibrated"] = by_branch.get("replay_better_calibrated", False)
            summary["replay_vs_dr_ece_diff"] = by_branch.get("replay_vs_dr_ece_diff", 0)

        # Goal-directedness
        vg = results.get("value_goal_correlation", {})
        summary["goal_directed"] = vg.get("goal_directed", False)

        return summary

    def visualize(self) -> Dict[str, np.ndarray]:
        """Create calibration plots."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        # 1. Multi-panel calibration plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Scatter: V(s₀) vs Return
        ax = axes[0, 0]
        v_pred = self.data["episode_data"]["initial_values"]
        v_actual = self.data["episode_data"]["final_returns"]
        ax.scatter(v_pred, v_actual, alpha=0.3, s=10)
        ax.plot([v_pred.min(), v_pred.max()], [v_pred.min(), v_pred.max()],
                'r--', label='Perfect calibration')
        ax.set_xlabel('Predicted V(s₀)')
        ax.set_ylabel('Actual Return')
        corr = self.results.get("summary", {}).get("initial_correlation", 0)
        ax.set_title(f'Initial Value Calibration (r={corr:.3f})')
        ax.legend()

        # ECE by timestep
        ax = axes[0, 1]
        multi_point = self.results.get("multi_point", {})
        if multi_point:
            timesteps = [t for t in multi_point.keys()]
            eces = [multi_point[t]["ece"] for t in timesteps]
            ax.bar(timesteps, eces, alpha=0.8)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('ECE')
            ax.set_title('Calibration Error by Timestep')

        # Calibration by branch
        ax = axes[0, 2]
        by_branch = self.results.get("by_branch", {})
        branch_names = ["DR", "Replay", "Mutate"]
        branch_eces = [by_branch.get(b, {}).get("ece", 0) for b in branch_names]
        colors = ['blue', 'green', 'orange']
        ax.bar(branch_names, branch_eces, color=colors, alpha=0.8)
        ax.set_ylabel('ECE')
        ax.set_title('Calibration by Branch')

        # Calibration curve
        ax = axes[1, 0]
        self._plot_calibration_curve(ax, v_pred, v_actual)
        ax.set_title('Calibration Curve')

        # Value vs goal distance
        ax = axes[1, 1]
        goal_dists = self.data["episode_data"]["goal_distances"]
        ax.scatter(goal_dists, v_pred, alpha=0.3, s=10)
        ax.set_xlabel('Goal Distance')
        ax.set_ylabel('V(s₀)')
        vg_corr = self.results.get("value_goal_correlation", {}).get("value_goal_correlation", 0)
        ax.set_title(f'Value vs Goal Distance (r={vg_corr:.3f})')

        # Temporal consistency
        ax = axes[1, 2]
        temp = self.results.get("temporal_consistency", {})
        if "td_violation_rate" in temp:
            metrics = ["td_violation_rate", "mean_actual_decrease"]
            values = [temp.get(m, 0) for m in metrics]
            ax.bar(["TD Violations", "Mean V Decrease"], values, alpha=0.8)
            ax.set_title('Temporal Consistency')

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["calibration"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        # 2. Calibration by difficulty
        fig, ax = plt.subplots(figsize=(10, 6))
        by_diff = self.results.get("by_difficulty", {})
        if by_diff:
            diff_bins = list(by_diff.keys())
            eces = [by_diff[b].get("ece", 0) for b in diff_bins]
            ax.bar(diff_bins, eces, alpha=0.8)
            ax.set_xlabel('Difficulty Bin')
            ax.set_ylabel('ECE')
            ax.set_title('Calibration by Level Difficulty')
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["calibration_by_difficulty"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures

    def _plot_calibration_curve(self, ax, v_pred: np.ndarray, v_actual: np.ndarray, n_bins: int = 10):
        """Plot calibration curve."""
        bin_edges = np.linspace(v_pred.min(), v_pred.max(), n_bins + 1)
        bin_centers = []
        bin_accuracies = []

        for i in range(n_bins):
            mask = (v_pred >= bin_edges[i]) & (v_pred < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_centers.append(v_pred[mask].mean())
                bin_accuracies.append(v_actual[mask].mean())

        if bin_centers:
            ax.plot(bin_centers, bin_accuracies, 'bo-', label='Model')
            min_val = min(min(bin_centers), min(bin_accuracies))
            max_val = max(max(bin_centers), max(bin_accuracies))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
            ax.set_xlabel('Mean Predicted Value')
            ax.set_ylabel('Mean Actual Return')
            ax.legend()
