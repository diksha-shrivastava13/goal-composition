"""
Configuration presets for curriculum awareness ablations.

Training methods:
- accel: ACCEL with DR→Replay→Mutate branches
- plr: Prioritized Level Replay (DR→Replay)
- robust_plr: Robust PLR (no exploratory updates on replay)
- dr: Domain Randomization only
- paired: PAIRED adversarial curriculum

Agent types:
- accel_probe: Baseline (reset LSTM + external probe)
- persistent_lstm: Non-resetting LSTM + external probe
- context_vector: EMA context + external probe
- episodic_memory: Episodic buffer + external probe
- next_env_prediction: Integrated prediction head (upper bound)

PAIRED variants (prepend 'paired_' to agent type):
- paired_accel_probe, paired_persistent_lstm, etc.
"""

from .presets import (
    get_config,
    get_agent_config,
    get_training_method_config,
    ALL_AGENTS,
    ALL_TRAINING_METHODS,
    ALL_CONFIGURATIONS,
)

__all__ = [
    'get_config',
    'get_agent_config',
    'get_training_method_config',
    'ALL_AGENTS',
    'ALL_TRAINING_METHODS',
    'ALL_CONFIGURATIONS',
]
