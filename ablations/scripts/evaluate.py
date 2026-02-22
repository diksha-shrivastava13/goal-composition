#!/usr/bin/env python3
"""
Evaluation entry point for curriculum awareness ablations.

Run interpretability experiments on trained checkpoints.

Usage:
    python -m ablations.scripts.evaluate --checkpoint_dir path/to/checkpoint --experiment level_probing
    python -m ablations.scripts.evaluate --checkpoint_dir path/to/checkpoint --experiment all
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
import numpy as np

from ablations.common.utils import load_checkpoint
from ablations.common.training import evaluate_n_env_predictions
from ablations.agents import (
    AccelProbeAgent,
    PersistentLSTMAgent,
    ContextVectorAgent,
    EpisodicMemoryAgent,
    NextEnvPredictionAgent,
)


AGENT_CLASSES = {
    "accel_probe": AccelProbeAgent,
    "persistent_lstm": PersistentLSTMAgent,
    "context_vector": ContextVectorAgent,
    "episodic_memory": EpisodicMemoryAgent,
    "next_env_prediction": NextEnvPredictionAgent,
}


EXPERIMENTS = [
    "level_probing",
    "value_calibration",
    "mutation_adaptation",
    "causal_intervention",
    "activation_analysis",
    "output_probing",
    "symbolic_regression",
    "behavioral_coupling",
    "counterfactual",
    "n_env_predictions",
]


def run_n_env_predictions(agent, train_state, config):
    """Run N-environment prediction evaluation."""
    from ablations.common.networks import CurriculumProbe
    from ablations.common.visualization import create_n_env_prediction_summary

    rng = jax.random.PRNGKey(42)

    probe = CurriculumProbe(env_height=13, env_width=13, use_episode_context=False)

    results = evaluate_n_env_predictions(
        rng=rng,
        env=agent.env,
        env_params=agent.env_params,
        train_state=train_state,
        probe_network=probe,
        sample_random_level=agent.sample_random_level,
        n_envs=config.get("n_env_predictions", 100),
    )

    # Create visualization
    img = create_n_env_prediction_summary(results)

    return {
        "summary": {
            "mean_wall_accuracy": results["mean_wall_accuracy"],
            "mean_goal_accuracy": results["mean_goal_accuracy"],
            "mean_agent_pos_accuracy": results["mean_agent_pos_accuracy"],
            "mean_agent_dir_accuracy": results["mean_agent_dir_accuracy"],
        },
        "visualization": img,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--experiment", type=str, default="all",
                       choices=EXPERIMENTS + ["all"],
                       help="Which experiment to run")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--n_env_predictions", type=int, default=100,
                       help="Number of environments for N-env predictions")

    args = parser.parse_args()

    # Load config from checkpoint
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Override with command line args
    config["n_env_predictions"] = args.n_env_predictions

    # Get agent type
    agent_type = config.get("agent_type", "accel_probe")
    agent_class = AGENT_CLASSES[agent_type]

    print(f"Loading checkpoint for {agent_type}...")

    # Create agent (for environment setup)
    agent = agent_class(config)

    # Create template train state
    rng = jax.random.PRNGKey(0)
    template_state = agent.create_train_state(rng)

    # Load checkpoint
    train_state, _ = load_checkpoint(args.checkpoint_dir, template_state)

    print(f"Checkpoint loaded. Running experiments...")

    # Setup output directory
    output_dir = args.output_dir or os.path.join(args.checkpoint_dir, "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    # Run experiments
    experiments_to_run = EXPERIMENTS if args.experiment == "all" else [args.experiment]

    results = {}
    for exp_name in experiments_to_run:
        print(f"Running: {exp_name}")

        if exp_name == "n_env_predictions":
            results[exp_name] = run_n_env_predictions(agent, train_state, config)

            # Save visualization
            if "visualization" in results[exp_name]:
                import matplotlib.pyplot as plt
                plt.imsave(
                    os.path.join(output_dir, f"{exp_name}.png"),
                    results[exp_name]["visualization"]
                )
        else:
            print(f"  (Not yet implemented)")
            results[exp_name] = {"status": "not_implemented"}

    # Save summary
    summary = {
        exp_name: result.get("summary", result)
        for exp_name, result in results.items()
    }

    with open(os.path.join(output_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Results saved to: {output_dir}")
    return results


if __name__ == "__main__":
    main()
