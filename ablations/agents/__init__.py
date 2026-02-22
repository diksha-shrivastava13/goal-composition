"""Agent implementations for curriculum awareness ablations."""

from .base import BaseAgent
from .accel_probe import AccelProbeAgent
from .persistent_lstm import PersistentLSTMAgent
from .context_vector import ContextVectorAgent
from .episodic_memory import EpisodicMemoryAgent
from .next_env_prediction import NextEnvPredictionAgent

# PAIRED agents
from .paired_base import PAIREDBaseAgent
from .paired_accel_probe import PAIREDAccelProbeAgent          # TODO: bad naming
from .paired_persistent_lstm import PAIREDPersistentLSTMAgent
from .paired_context_vector import PAIREDContextVectorAgent
from .paired_episodic_memory import PAIREDEpisodicMemoryAgent
from .paired_next_env_prediction import PAIREDNextEnvPredictionAgent     # TODO: bad naming

__all__ = [
    # Base classes
    "BaseAgent",
    "PAIREDBaseAgent",
    # Standard agents (ACCEL/PLR/DR)
    "AccelProbeAgent",
    "PersistentLSTMAgent",
    "ContextVectorAgent",
    "EpisodicMemoryAgent",
    "NextEnvPredictionAgent",
    # PAIRED agents
    "PAIREDAccelProbeAgent",
    "PAIREDPersistentLSTMAgent",
    "PAIREDContextVectorAgent",
    "PAIREDEpisodicMemoryAgent",
    "PAIREDNextEnvPredictionAgent",
    # Factory functions
    "create_agent",
    "get_agent_class",
]


# Agent class registry
AGENT_CLASSES = {
    # Base agents
    "accel_probe": AccelProbeAgent,
    "persistent_lstm": PersistentLSTMAgent,
    "context_vector": ContextVectorAgent,
    "episodic_memory": EpisodicMemoryAgent,
    "next_env_prediction": NextEnvPredictionAgent,
    # PAIRED agents
    "paired_accel_probe": PAIREDAccelProbeAgent,
    "paired_persistent_lstm": PAIREDPersistentLSTMAgent,
    "paired_context_vector": PAIREDContextVectorAgent,
    "paired_episodic_memory": PAIREDEpisodicMemoryAgent,
    "paired_next_env_prediction": PAIREDNextEnvPredictionAgent,
}


def get_agent_class(agent_type: str):
    """
    Get agent class by type name.

    Args:
        agent_type: One of the registered agent types

    Returns:
        Agent class
    """
    if agent_type not in AGENT_CLASSES:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Available: {list(AGENT_CLASSES.keys())}"
        )
    return AGENT_CLASSES[agent_type]


def create_agent(config: dict):
    """
    Factory function to create the appropriate agent based on config.

    Args:
        config: Configuration dictionary containing:
            - training_method: One of "accel", "plr", "robust_plr", "dr", "paired"
            - agent_type: One of "accel_probe", "persistent_lstm", "context_vector",
                         "episodic_memory", "next_env_prediction"

    Returns:
        Agent instance appropriate for the training method and agent type.
    """
    training_method = config.get("training_method", "accel")
    agent_type = config.get("agent_type", "accel_probe")

    if training_method == "paired":
        # Use PAIRED agent variants
        paired_agents = {
            "accel_probe": PAIREDAccelProbeAgent,
            "persistent_lstm": PAIREDPersistentLSTMAgent,
            "context_vector": PAIREDContextVectorAgent,
            "episodic_memory": PAIREDEpisodicMemoryAgent,
            "next_env_prediction": PAIREDNextEnvPredictionAgent,
        }
        # Strip paired_ prefix if present (configs use "paired_accel_probe" etc.)
        lookup_key = agent_type.removeprefix("paired_")
        if lookup_key not in paired_agents:
            raise ValueError(f"Unknown agent type for PAIRED: {agent_type}")
        return paired_agents[lookup_key](config)

    else:
        # Use standard agents (ACCEL/PLR/Robust-PLR/DR all use same agent classes)
        standard_agents = {
            "accel_probe": AccelProbeAgent,
            "persistent_lstm": PersistentLSTMAgent,
            "context_vector": ContextVectorAgent,
            "episodic_memory": EpisodicMemoryAgent,
            "next_env_prediction": NextEnvPredictionAgent,
        }
        if agent_type not in standard_agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return standard_agents[agent_type](config)
