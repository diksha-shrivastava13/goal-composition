#!/usr/bin/env python3
"""
Training entry point for curriculum awareness ablations.

Usage:
    python -m ablations.scripts.train --agent_type accel_probe --seed 0
    python -m ablations.scripts.train --agent_type persistent_lstm --use_accel
    python -m ablations.scripts.train --agent_type context_vector --context_dim 64

See --help for full list of options.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import wandb

from ablations.common.utils import parse_args, get_default_config
from ablations.agents import (
    # Base agents (for accel, plr, robust_plr, dr)
    AccelProbeAgent,
    PersistentLSTMAgent,
    ContextVectorAgent,
    EpisodicMemoryAgent,
    NextEnvPredictionAgent,
    # PAIRED agents (for paired training method)
    PAIREDAccelProbeAgent,
    PAIREDPersistentLSTMAgent,
    PAIREDContextVectorAgent,
    PAIREDEpisodicMemoryAgent,
    PAIREDNextEnvPredictionAgent,
)


# Base agents (5) - for accel, plr, robust_plr, dr
BASE_AGENT_CLASSES = {
    "accel_probe": AccelProbeAgent,
    "persistent_lstm": PersistentLSTMAgent,
    "context_vector": ContextVectorAgent,
    "episodic_memory": EpisodicMemoryAgent,
    "next_env_prediction": NextEnvPredictionAgent,
}

# PAIRED agents (5) - for paired training method
PAIRED_AGENT_CLASSES = {
    "paired_accel_probe": PAIREDAccelProbeAgent,
    "paired_persistent_lstm": PAIREDPersistentLSTMAgent,
    "paired_context_vector": PAIREDContextVectorAgent,
    "paired_episodic_memory": PAIREDEpisodicMemoryAgent,
    "paired_next_env_prediction": PAIREDNextEnvPredictionAgent,
}

# All agents (10 classes, but 25 configurations with training methods)
AGENT_CLASSES = {**BASE_AGENT_CLASSES, **PAIRED_AGENT_CLASSES}


def main():
    """Main training entry point."""
    # Parse command line arguments
    config = parse_args()

    # Get agent type and training method
    agent_type = config["agent_type"]
    training_method = config.get("training_method", "accel")

    # Validate agent type
    if agent_type not in AGENT_CLASSES:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Available: {list(AGENT_CLASSES.keys())}"
        )

    # Validate training_method + agent_type combination
    if training_method == "paired" and not agent_type.startswith("paired_"):
        raise ValueError(
            f"PAIRED training method requires paired_* agent. "
            f"Got agent_type={agent_type}. Use --agent_type paired_{agent_type}"
        )
    if training_method != "paired" and agent_type.startswith("paired_"):
        raise ValueError(
            f"Non-PAIRED training method '{training_method}' cannot use paired_* agent. "
            f"Either use --training_method paired or use base agent type."
        )

    agent_class = AGENT_CLASSES[agent_type]

    # Set run name if not specified
    if config["run_name"] == "ablation":
        config["run_name"] = f"{training_method}_{agent_type}_seed{config['seed']}"

    print(f"=" * 60)
    print(f"Curriculum Awareness Ablation Study")
    print(f"=" * 60)
    print(f"Training method: {training_method}")
    print(f"Agent type: {agent_type}")
    print(f"Seed: {config['seed']}")
    if training_method != "paired":
        print(f"Use ACCEL: {config['use_accel']}")
        print(f"Replay prob: {config.get('replay_prob', 0.8)}")
    print(f"Use probe: {config.get('use_probe', True)}")
    print(f"=" * 60)

    # Initialize wandb
    wandb.login()

    # Create and train agent
    agent = agent_class(config)
    final_train_state = agent.train()

    print("Training complete!")
    return final_train_state


if __name__ == "__main__":
    main()
