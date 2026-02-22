"""
Configuration presets for the 25-agent ablation study.

5 Training Methods × 5 Memory Architectures = 25 configurations
"""

from typing import Dict, Any, List, Tuple
from itertools import product


# =============================================================================
# AGENT TYPES
# =============================================================================

BASE_AGENTS = [
    "accel_probe",
    "persistent_lstm",
    "context_vector",
    "episodic_memory",
    "next_env_prediction",
]

PAIRED_AGENTS = [f"paired_{agent}" for agent in BASE_AGENTS]

ALL_AGENTS = BASE_AGENTS + PAIRED_AGENTS


# =============================================================================
# TRAINING METHODS
# =============================================================================

ALL_TRAINING_METHODS = ["accel", "plr", "robust_plr", "dr", "paired"]

# Training method configurations
TRAINING_METHOD_CONFIGS = {
    "accel": {
        "use_accel": True,
        "exploratory_grad_updates": True,
        "replay_prob": 0.8,
    },
    "plr": {
        "use_accel": False,
        "exploratory_grad_updates": True,
        "replay_prob": 0.8,
    },
    "robust_plr": {
        "use_accel": False,
        "exploratory_grad_updates": False,
        "replay_prob": 0.8,
    },
    "dr": {
        "use_accel": False,
        "exploratory_grad_updates": True,
        "replay_prob": 0.0,  # No replay in DR
    },
    "paired": {
        "use_accel": False,
        "exploratory_grad_updates": True,
        # PAIRED-specific settings handled separately
    },
}


# =============================================================================
# AGENT-SPECIFIC CONFIGURATIONS
# =============================================================================

AGENT_CONFIGS = {
    "accel_probe": {
        "description": "Baseline: Reset LSTM + external probe",
        "memory_type": "reset",
        "use_probe": True,
    },
    "persistent_lstm": {
        "description": "Non-resetting LSTM + external probe",
        "memory_type": "persistent",
        "use_probe": True,
    },
    "context_vector": {
        "description": "EMA context compression + external probe",
        "memory_type": "context",
        "use_probe": True,
        "context_dim": 64,
        "context_decay": 0.9,
    },
    "episodic_memory": {
        "description": "Episodic buffer + attention + external probe",
        "memory_type": "episodic",
        "use_probe": True,
        "memory_buffer_size": 64,
        "memory_top_k": 8,
    },
    "next_env_prediction": {
        "description": "Integrated prediction head (upper bound)",
        "memory_type": "integrated",
        "use_probe": False,  # Uses integrated prediction instead
        "prediction_coeff": 0.1,
        "curriculum_history_length": 64,
    },
}

# PAIRED variants inherit from base + add PAIRED-specific settings
for base_agent in BASE_AGENTS:
    paired_agent = f"paired_{base_agent}"
    AGENT_CONFIGS[paired_agent] = {
        **AGENT_CONFIGS[base_agent],
        "description": f"PAIRED + {AGENT_CONFIGS[base_agent]['description']}",
        "training_method": "paired",
    }


# =============================================================================
# ALL 25 CONFIGURATIONS
# =============================================================================

def get_all_configurations() -> List[Tuple[str, str]]:
    """
    Get all 25 valid (training_method, agent_type) combinations.

    Note: PAIRED training method requires paired_* agents.
          Non-PAIRED methods use base agents.
    """
    configurations = []

    for method in ALL_TRAINING_METHODS:
        if method == "paired":
            # PAIRED uses paired_* agents
            for agent in PAIRED_AGENTS:
                configurations.append((method, agent))
        else:
            # Other methods use base agents
            for agent in BASE_AGENTS:
                configurations.append((method, agent))

    return configurations


ALL_CONFIGURATIONS = get_all_configurations()


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def get_training_method_config(training_method: str) -> Dict[str, Any]:
    """Get configuration for a training method."""
    if training_method not in TRAINING_METHOD_CONFIGS:
        raise ValueError(
            f"Unknown training method: {training_method}. "
            f"Available: {list(TRAINING_METHOD_CONFIGS.keys())}"
        )
    return TRAINING_METHOD_CONFIGS[training_method].copy()


def get_agent_config(agent_type: str) -> Dict[str, Any]:
    """Get configuration for an agent type."""
    if agent_type not in AGENT_CONFIGS:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Available: {list(AGENT_CONFIGS.keys())}"
        )
    return AGENT_CONFIGS[agent_type].copy()


def get_config(
    training_method: str,
    agent_type: str,
    **overrides,
) -> Dict[str, Any]:
    """
    Get full configuration for a (training_method, agent_type) pair.

    Args:
        training_method: One of accel, plr, robust_plr, dr, paired
        agent_type: One of the agent types (base or paired_*)
        **overrides: Override any config values

    Returns:
        Complete configuration dictionary
    """
    # Validate combination
    if training_method == "paired" and not agent_type.startswith("paired_"):
        raise ValueError(
            f"PAIRED training method requires paired_* agent, got {agent_type}"
        )
    if training_method != "paired" and agent_type.startswith("paired_"):
        raise ValueError(
            f"Non-PAIRED training method {training_method} cannot use paired_* agent"
        )

    # Start with base defaults
    from ..common.utils import get_default_config
    config = get_default_config()

    # Apply training method config
    config.update(get_training_method_config(training_method))
    config["training_method"] = training_method

    # Apply agent config
    config.update(get_agent_config(agent_type))
    config["agent_type"] = agent_type

    # Apply overrides
    config.update(overrides)

    return config


def print_all_configurations():
    """Print all 25 configurations for reference."""
    print("=" * 60)
    print("Curriculum Awareness Ablation Study: 25 Configurations")
    print("=" * 60)

    for method in ALL_TRAINING_METHODS:
        print(f"\n{method.upper()}:")
        if method == "paired":
            agents = PAIRED_AGENTS
        else:
            agents = BASE_AGENTS

        for agent in agents:
            desc = AGENT_CONFIGS[agent]["description"]
            print(f"  - {agent}: {desc}")

    print(f"\nTotal configurations: {len(ALL_CONFIGURATIONS)}")


if __name__ == "__main__":
    print_all_configurations()
