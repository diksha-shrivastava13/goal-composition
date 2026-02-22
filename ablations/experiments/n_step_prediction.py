"""
N-STEP Prediction Experiment.

Evaluates probe prediction accuracy sequentially through curriculum steps.
Tests how long predicted losses continue to be correct as the agent steps
through the curriculum without further training.

Different from N-ENV:
- N-ENV: predict N independent random environments (aggregate robustness)
- N-STEP: predict step 1->2->3->...->N sequentially through curriculum
  (sequential prediction decay / theory-of-mind of environment generator)

This wraps evaluate_n_step_prediction() from training.py as a proper experiment.
"""

import jax
import numpy as np
from typing import Dict, Any, Optional

from .base import CheckpointExperiment


class NStepPredictionExperiment(CheckpointExperiment):
    """Sequential N-step prediction through curriculum.

    Tests curriculum prediction (theory-of-mind of environment generator).
    Follows the actual ACCEL training loop: DR -> Replay -> Mutate cycle.
    Measures how prediction accuracy changes over sequential curriculum steps
    when the agent is NOT being trained.
    """

    def __init__(
        self,
        agent,
        train_state,
        config: dict,
        output_dir: Optional[str] = None,
        training_method: str = "accel",
        n_steps: int = 20,
        num_envs: int = 32,
    ):
        super().__init__(agent, train_state, config, output_dir, training_method)
        self.n_steps = config.get("n_steps", n_steps)
        self.num_envs = config.get("num_envs", num_envs)

    @property
    def name(self) -> str:
        return "n_step_prediction"

    def collect_data(self, rng) -> Dict[str, Any]:
        """Run N-step sequential prediction evaluation."""
        from ..common.training import evaluate_n_step_prediction

        # N-STEP requires level_sampler and mutate_level
        level_sampler = getattr(self.agent, 'level_sampler', None)
        mutate_level = getattr(self.agent, 'mutate_level', None)
        probe_runner = getattr(self.agent, 'probe_runner', None)

        results = evaluate_n_step_prediction(
            rng=rng,
            env=self.agent.env,
            env_params=self.agent.env_params,
            train_state=self.train_state,
            level_sampler=level_sampler,
            sample_random_level=self.agent.sample_random_level,
            mutate_level=mutate_level if mutate_level is not None else self._default_mutate,
            probe_runner=probe_runner,
            n_steps=self.n_steps,
            num_envs=self.num_envs,
            env_height=self.config.get("env_height", 13),
            env_width=self.config.get("env_width", 13),
        )

        self.data = results
        return results

    def _default_mutate(self, rng, level, num_edits):
        """Fallback mutation: return level unchanged if no mutator available."""
        return level

    def analyze(self) -> Dict[str, Any]:
        """Analyze N-step prediction results."""
        results = self.data

        analysis = {
            "mean_wall_accuracy": results.get("mean_wall_accuracy", float('nan')),
            "mean_goal_accuracy": results.get("mean_goal_accuracy", float('nan')),
            "prediction_improvement": results.get("prediction_improvement", 0.0),
            "n_steps": self.n_steps,
            "per_step_wall_accuracy": results.get("per_step_wall_accuracy", []),
            "per_step_goal_accuracy": results.get("per_step_goal_accuracy", []),
            "per_step_branch": results.get("per_step_branch", []),
            "per_step_returns": results.get("per_step_returns", []),
            "per_branch_wall_accuracy": results.get("per_branch_wall_accuracy", {}),
            "per_branch_returns": results.get("per_branch_returns", {}),
        }

        # Compute prediction decay: how fast does accuracy drop over steps?
        wall_accs = results.get("per_step_wall_accuracy", [])
        valid_accs = [a for a in wall_accs if not np.isnan(a)]
        if len(valid_accs) >= 4:
            quarter = len(valid_accs) // 4
            analysis["first_quarter_accuracy"] = float(np.mean(valid_accs[:quarter]))
            analysis["last_quarter_accuracy"] = float(np.mean(valid_accs[-quarter:]))
            analysis["accuracy_decay"] = analysis["first_quarter_accuracy"] - analysis["last_quarter_accuracy"]

            # Steps until accuracy drops below threshold
            initial_acc = valid_accs[0] if valid_accs else 0
            threshold = initial_acc * 0.8  # 80% of initial
            steps_above = sum(1 for a in valid_accs if a >= threshold)
            analysis["steps_above_80pct"] = steps_above

        self.results = analysis
        return analysis

    def visualize(self) -> Dict[str, Any]:
        """Create N-step prediction visualizations."""
        import matplotlib.pyplot as plt

        viz = {}
        results = self.data

        # Per-step accuracy plot
        try:
            wall_accs = results.get("per_step_wall_accuracy", [])
            goal_accs = results.get("per_step_goal_accuracy", [])
            branches = results.get("per_step_branch", [])
            returns = results.get("per_step_returns", [])

            if wall_accs:
                fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

                steps = range(len(wall_accs))

                # Accuracy over steps
                ax = axes[0]
                ax.plot(steps, wall_accs, 'b-o', label="Wall Accuracy", markersize=4)
                ax.plot(steps, goal_accs, 'r-s', label="Goal Accuracy", markersize=4)
                ax.set_ylabel("Accuracy")
                ax.set_title("N-STEP: Prediction Accuracy Over Sequential Curriculum Steps")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Color by branch
                branch_colors = {0: 'green', 1: 'blue', 2: 'orange'}
                branch_labels = {0: 'DR', 1: 'Replay', 2: 'Mutate'}
                ax = axes[1]
                for s, b in zip(steps, branches):
                    ax.axvspan(s - 0.4, s + 0.4, alpha=0.3,
                               color=branch_colors.get(b, 'gray'))
                ax.plot(steps, wall_accs, 'k-o', markersize=4)
                ax.set_ylabel("Wall Accuracy")
                ax.set_title("Accuracy by Branch (green=DR, blue=Replay, orange=Mutate)")
                ax.grid(True, alpha=0.3)

                # Returns over steps
                ax = axes[2]
                ax.plot(steps, returns, 'g-^', label="Mean Returns", markersize=4)
                ax.set_xlabel("Curriculum Step")
                ax.set_ylabel("Returns")
                ax.set_title("Agent Returns Over Sequential Steps")
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                viz["n_step_trajectory"] = fig
                plt.close(fig)
        except Exception:
            pass

        # Per-branch comparison
        try:
            per_branch_wall = results.get("per_branch_wall_accuracy", {})
            per_branch_ret = results.get("per_branch_returns", {})

            if per_branch_wall:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                branch_names = {0: "DR", 1: "Replay", 2: "Mutate"}

                branches_present = sorted(per_branch_wall.keys())
                names = [branch_names.get(b, f"Branch {b}") for b in branches_present]

                ax = axes[0]
                vals = [per_branch_wall[b] for b in branches_present]
                ax.bar(names, vals, color=['green', 'blue', 'orange'][:len(branches_present)])
                ax.set_ylabel("Wall Accuracy")
                ax.set_title("Prediction Accuracy by Branch")

                ax = axes[1]
                if per_branch_ret:
                    vals = [per_branch_ret.get(b, 0) for b in branches_present]
                    ax.bar(names, vals, color=['green', 'blue', 'orange'][:len(branches_present)])
                    ax.set_ylabel("Mean Returns")
                    ax.set_title("Agent Returns by Branch")

                plt.tight_layout()
                viz["per_branch_comparison"] = fig
                plt.close(fig)
        except Exception:
            pass

        self.figures = viz
        return viz
