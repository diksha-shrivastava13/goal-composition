"""Utility functions for interpretability experiments."""

from .calibration_utils import (
    compute_multi_point_calibration,
    compute_branch_conditioned_ece,
    compute_value_gradient,
    compute_temporal_consistency,
)
from .rsa_cka import (
    compute_rdm,
    compute_rsa,
    compute_cka,
    compute_layer_wise_cka,
)
from .time_series_analysis import (
    compute_predictive_signal_strength,
    compute_rolling_correlation,
    compute_granger_causality,
    detect_change_points,
)
from .transfer_metrics import (
    compute_behavioral_transfer,
    compute_representational_transfer,
    compute_td_error_surprise,
)
from .distribution_shift import (
    compute_mmd,
    create_branch_ablation,
    create_difficulty_manipulation,
)
from .history_injection import (
    create_failure_history,
    create_success_history,
    inject_hidden_state,
    measure_injection_effect,
)
from .activation_patching import (
    compute_saliency_map,
    patch_activations,
    identify_task_controlling_subspace,
)
from .transition_metrics import (
    estimate_fisher_information,
    compute_effective_dimensionality,
    compute_synergy_measure,
)
from .memory_probing import (
    inject_distinctive_pattern,
    test_memory_capacity,
    analyze_selective_memory,
)
from .agent_aware_loss import (
    detect_agent_type,
    uses_prediction_head,
    compute_agent_prediction_loss,
    create_observation_from_level,
    create_level_object,
    extract_curriculum_features,
    compute_random_baseline_loss,
    compute_information_gain,
    compute_normalized_loss,
)

__all__ = [
    # Calibration
    "compute_multi_point_calibration",
    "compute_branch_conditioned_ece",
    "compute_value_gradient",
    "compute_temporal_consistency",
    # RSA/CKA
    "compute_rdm",
    "compute_rsa",
    "compute_cka",
    "compute_layer_wise_cka",
    # Time series
    "compute_predictive_signal_strength",
    "compute_rolling_correlation",
    "compute_granger_causality",
    "detect_change_points",
    # Transfer
    "compute_behavioral_transfer",
    "compute_representational_transfer",
    "compute_td_error_surprise",
    # Distribution shift
    "compute_mmd",
    "create_branch_ablation",
    "create_difficulty_manipulation",
    # History injection
    "create_failure_history",
    "create_success_history",
    "inject_hidden_state",
    "measure_injection_effect",
    # Activation patching
    "compute_saliency_map",
    "patch_activations",
    "identify_task_controlling_subspace",
    # Transition metrics
    "estimate_fisher_information",
    "compute_effective_dimensionality",
    "compute_synergy_measure",
    # Memory probing
    "inject_distinctive_pattern",
    "test_memory_capacity",
    "analyze_selective_memory",
    # Agent-aware loss computation
    "detect_agent_type",
    "uses_prediction_head",
    "compute_agent_prediction_loss",
    "create_observation_from_level",
    "create_level_object",
    "extract_curriculum_features",
    "compute_random_baseline_loss",
    "compute_information_gain",
    "compute_normalized_loss",
]
