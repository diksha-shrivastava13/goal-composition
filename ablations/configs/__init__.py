"""
Configuration presets for curriculum awareness ablations.

Training methods:
- accel: ACCEL with DR->Replay->Mutate branches
- plr: Prioritized Level Replay (DR->Replay)
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
    BASE_AGENTS,
    PAIRED_AGENTS,
)

from .defaults import get_default_config

from .experiment_defaults import (
    UNIVERSAL_EXPERIMENTS,
    PAIRED_EXPERIMENTS,
    TRAINING_TIME_EXPERIMENTS,
    TRAINING_TIME_SET,
    EXPERIMENT_CADENCE,
    EXPERIMENT_DEFAULTS,
    get_experiments_for_method,
    get_experiment_config,
    get_all_experiment_defaults,
)

from .cli import (
    add_common_args,
    add_training_args,
    add_experiment_selection_args,
    add_experiment_param_args,
    add_posthoc_args,
    build_config_from_args,
    get_experiment_param_overrides,
    EXPERIMENT_PARAM_KEYS,
)

__all__ = [
    # presets
    "get_config",
    "get_agent_config",
    "get_training_method_config",
    "ALL_AGENTS",
    "ALL_TRAINING_METHODS",
    "ALL_CONFIGURATIONS",
    "BASE_AGENTS",
    "PAIRED_AGENTS",
    # defaults
    "get_default_config",
    # experiment_defaults
    "UNIVERSAL_EXPERIMENTS",
    "PAIRED_EXPERIMENTS",
    "TRAINING_TIME_EXPERIMENTS",
    "TRAINING_TIME_SET",
    "EXPERIMENT_CADENCE",
    "EXPERIMENT_DEFAULTS",
    "get_experiments_for_method",
    "get_experiment_config",
    "get_all_experiment_defaults",
    # cli
    "add_common_args",
    "add_training_args",
    "add_experiment_selection_args",
    "add_experiment_param_args",
    "add_posthoc_args",
    "build_config_from_args",
    "get_experiment_param_overrides",
    "EXPERIMENT_PARAM_KEYS",
]
