"""
N-ENV Prediction Experiment.

Evaluates probe prediction accuracy across N independent random environments.
Tests the agent's general ability to predict level features from hidden state.

This wraps evaluate_n_env_predictions() from training.py as a proper experiment.
"""

import jax
import numpy as np
from typing import Dict, Any, Optional

from .base import CheckpointExperiment


class NEnvPredictionExperiment(CheckpointExperiment):
    """Predict level features for N independent random environments.

    Tests aggregate prediction ability: given a random level, how well
    can the probe predict wall layout, goal position, agent position,
    and agent direction from the agent's hidden state?
    """

    def __init__(
        self,
        agent,
        train_state,
        config: dict,
        output_dir: Optional[str] = None,
        training_method: str = "accel",
        n_envs: int = 100,
        num_steps: int = 256,
    ):
        super().__init__(agent, train_state, config, output_dir, training_method)
        self.n_envs = config.get("n_envs", n_envs)
        self.num_steps = config.get("num_steps", num_steps)

    @property
    def name(self) -> str:
        return "n_env_prediction"

    def collect_data(self, rng) -> Dict[str, Any]:
        """Run N-env prediction evaluation."""
        from ..common.training import evaluate_n_env_predictions
        from ..common.networks import CurriculumProbe

        probe = CurriculumProbe(
            env_height=self.config.get("env_height", 13),
            env_width=self.config.get("env_width", 13),
            use_episode_context=False,
        )

        results = evaluate_n_env_predictions(
            rng=rng,
            env=self.agent.env,
            env_params=self.agent.env_params,
            train_state=self.train_state,
            probe_network=probe,
            sample_random_level=self.agent.sample_random_level,
            n_envs=self.n_envs,
            num_steps=self.num_steps,
            env_height=self.config.get("env_height", 13),
            env_width=self.config.get("env_width", 13),
        )

        self.data = results
        return results

    def analyze(self) -> Dict[str, Any]:
        """Analyze N-env prediction results."""
        results = self.data

        analysis = {
            "mean_wall_accuracy": results["mean_wall_accuracy"],
            "std_wall_accuracy": results["std_wall_accuracy"],
            "mean_goal_accuracy": results["mean_goal_accuracy"],
            "std_goal_accuracy": results["std_goal_accuracy"],
            "mean_agent_pos_accuracy": results["mean_agent_pos_accuracy"],
            "std_agent_pos_accuracy": results["std_agent_pos_accuracy"],
            "mean_agent_dir_accuracy": results["mean_agent_dir_accuracy"],
            "std_agent_dir_accuracy": results["std_agent_dir_accuracy"],
            "n_envs": results["n_envs"],
        }

        # Per-environment accuracy distribution
        raw = results.get("raw_results", {})
        if raw:
            analysis["wall_accuracy_distribution"] = raw.get("wall_accuracy", [])
            analysis["goal_accuracy_distribution"] = raw.get("goal_accuracy", [])

        self.results = analysis
        return analysis

    def visualize(self) -> Dict[str, Any]:
        """Create N-env prediction visualizations."""
        import matplotlib.pyplot as plt
        from ..common.visualization import create_n_env_prediction_summary

        viz = {}

        try:
            summary_img = create_n_env_prediction_summary(self.data)
            viz["n_env_summary"] = summary_img
        except Exception:
            pass

        # Accuracy histogram
        try:
            raw = self.data.get("raw_results", {})
            if raw and "wall_accuracy" in raw:
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                for idx, (key, title) in enumerate([
                    ("wall_accuracy", "Wall Accuracy"),
                    ("goal_accuracy", "Goal Accuracy"),
                    ("agent_pos_accuracy", "Agent Pos Accuracy"),
                    ("agent_dir_accuracy", "Agent Dir Accuracy"),
                ]):
                    ax = axes[idx // 2, idx % 2]
                    vals = raw.get(key, [])
                    if vals:
                        ax.hist(vals, bins=20, alpha=0.7)
                        ax.axvline(np.mean(vals), color='r', linestyle='--',
                                   label=f"Mean: {np.mean(vals):.3f}")
                        ax.set_title(title)
                        ax.legend()

                plt.tight_layout()
                viz["accuracy_histogram"] = fig
                plt.close(fig)
        except Exception:
            pass

        self.figures = viz
        return viz
