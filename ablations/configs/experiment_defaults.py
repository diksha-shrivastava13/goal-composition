"""
Experiment registry, cadence, and per-experiment defaults.

Single source of truth for:
- Which experiments exist and which training methods they apply to
- How often each experiment runs (cadence)
- Default parameter values for every experiment (namespaced as exp.<name>.<param>)
"""

from typing import Dict, Any, List, Optional


# =============================================================================
# EXPERIMENT LISTS
# =============================================================================

# Universal post-hoc experiments (run on any checkpoint, any method)
UNIVERSAL_EXPERIMENTS = [
    "level_probing",
    "value_calibration",
    "activation_analysis",
    "cross_agent_comparison",
    "mutation_adaptation",
    "causal_intervention",
    "counterfactual",
    "output_probing",
    "goal_extraction",
    "cross_episode_flow",
    "dr_coverage",
    "n_env_prediction",
    "n_step_prediction",
]

# PAIRED-specific experiments (only for training_method == "paired")
PAIRED_EXPERIMENTS = [
    "adversary_dynamics",
    "regret_transfer",
    # A: Utility Function Extraction
    "utility_extraction",
    "adversary_policy_extraction",
    "bilateral_utility",
    # B: Causal Interventions
    "adversary_ablation",
    "regret_decomposition",
    "teaching_signal_intervention",
    "counterfactual_curriculum",
    "activation_patching",
    # C: Three-Agent Dynamics
    "representation_divergence",
    "antagonist_audit",
    "adversary_strategy_clustering",
    "coalition_dynamics",
    # D: Belief Revision Tracking
    "representation_trajectory",
    "belief_revision_detection",
    "goal_evolution",
    # F: Theoretical Validation
    "causal_model_extraction",
    "multiscale_goals",
    "shard_dynamics",
    "belief_behaviour_divergence",
    "teaching_opacity",
]

# Training-time experiments (require hooks, cannot run post-hoc)
TRAINING_TIME_EXPERIMENTS = [
    "behavioral_coupling",
    "symbolic_regression",
    "phase_transition",
]

# Fast lookup set
TRAINING_TIME_SET = frozenset(TRAINING_TIME_EXPERIMENTS)


# =============================================================================
# EXPERIMENT CADENCE
# =============================================================================

# Value N means: run every Nth eval step. Default (not listed) = 1.
EXPERIMENT_CADENCE: Dict[str, int] = {
    "adversary_ablation": 3,
}


# =============================================================================
# PER-EXPERIMENT DEFAULTS
# =============================================================================

EXPERIMENT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # --- Universal checkpoint experiments (13) ---
    "level_probing": {
        "n_levels": 500,
        "collection_steps": [1, 10, 50, -1],
        "max_episode_length": 256,
    },
    "value_calibration": {
        "n_episodes": 500,
        "calibration_timesteps": [1, 10, 50, 100, 200],
        "max_episode_length": 256,
        "gamma": 0.995,
    },
    "activation_analysis": {
        "n_episodes": 200,
        "n_components_pca": 50,
        "n_components_viz": 2,
        "compute_sparse_ae": True,
        "sparse_ae_hidden": 2048,
        "sparse_ae_sparsity": 0.1,
    },
    "cross_agent_comparison": {
        "n_test_levels": 200,
        "wall_density_range": [0.05, 0.35],
        "seed": 42,
    },
    "mutation_adaptation": {
        "n_level_pairs": 200,
        "mutation_distances": [1, 2, 3, 5],
        "n_random_baselines": 10,
        "max_episode_steps": 256,
        "n_clusters": 5,
        "n_recovery_episodes": 5,
        "recovery_threshold": 0.9,
    },
    "causal_intervention": {
        "n_episodes_per_intervention": 100,
        "adaptation_episodes": 20,
        "progressive_difficulty_steps": 10,
        "interventions": None,  # None = use all compatible; or list of InterventionType values
    },
    "counterfactual": {
        "n_episodes_per_condition": 100,
        "injection_strength": 1.0,
        "n_injection_episodes": 10,
        "antagonist_rollout_steps": 10,
        "regret_conditioning_steps": 20,
    },
    "output_probing": {
        "n_episodes": 200,
        "n_steps_per_episode": 50,
    },
    "goal_extraction": {
        "n_samples": 200,
        "n_patching_pairs": 50,
        "n_attribution_steps": 50,
    },
    "cross_episode_flow": {
        "n_episode_sequences": 100,
        "sequence_length": 10,
        "max_lag_to_test": 5,
    },
    "dr_coverage": {
        "n_levels": 1000,
        "n_grid_bins": 10,
        "feature_names": None,  # None = use default feature set
    },
    "n_env_prediction": {
        "n_sequences": 100,
        "sequence_length": 15,
        "horizons": [1, 2, 5, 10],
        "max_steps": 256,
    },
    "n_step_prediction": {
        "n_levels": 100,
        "max_steps": 128,
        "horizons": [1, 5, 10, 25],
    },

    # --- Training-time experiments (3) ---
    "behavioral_coupling": {
        "collection_interval": 100,
        "probe_n_samples": 100,
        "rolling_window": 20,
        "granger_max_lag": 10,
        "random_baseline_samples": 100,
    },
    "symbolic_regression": {
        "collection_interval": 100,
        "n_samples_per_collection": 20,
        "use_pysr": True,
        "pysr_iterations": 50,
    },
    "phase_transition": {
        "collection_interval": 100,
        "n_gradient_samples": 10,
        "n_representation_samples": 50,
    },

    # --- PAIRED-specific experiments (22) ---
    "adversary_dynamics": {
        "n_episodes": 200,
        "window_size": 20,
    },
    "regret_transfer": {
        "n_levels_per_subset": 200,
        "max_steps": 256,
    },
    "utility_extraction": {
        "n_samples": 500,
        "use_pysr": True,
        "pysr_iterations": 100,
    },
    "adversary_policy_extraction": {
        "n_levels": 200,
        "max_steps": 256,
    },
    "bilateral_utility": {
        "n_levels": 500,
    },
    "adversary_ablation": {
        "n_levels_per_condition": 200,
    },
    "regret_decomposition": {
        "n_levels": 500,
        "min_regret_threshold": 0.1,
        "dominance_ratio": 1.5,
        "adaptive_threshold": False,
        "solvability_threshold": 0.8,
    },
    "teaching_signal_intervention": {
        "baseline_steps": 500,
        "intervention_steps": 1000,
        "post_steps": 500,
        "hidden_dim": 256,
    },
    "counterfactual_curriculum": {
        "n_eval_levels": 200,
        "max_steps": 256,
    },
    "activation_patching": {
        "n_pairs": 200,
        "hidden_dim": 256,
        "top_k_variance": 50,
    },
    "representation_divergence": {
        "n_levels_per_checkpoint": 200,
        "hidden_dim": 256,
    },
    "antagonist_audit": {
        "n_levels_per_type": 100,
        "n_probe_levels": 300,
        "max_steps": 256,
    },
    "adversary_strategy_clustering": {
        "n_rollouts_per_checkpoint": 100,
        "min_cluster_size": 10,
        "use_hdbscan": True,
        "n_clusters_kmeans": 5,
    },
    "coalition_dynamics": {
        "n_samples_per_step": 50,
        "trajectory_length": 100,
        "hidden_dim": 256,
    },
    "representation_trajectory": {
        "n_samples_per_step": 50,
        "trajectory_length": 100,
        "hidden_dim": 256,
        "reduced_dim": 20,
    },
    "belief_revision_detection": {
        "n_samples_per_step": 50,
        "trajectory_length": 200,
        "detection_window": 10,
        "sigma_threshold": 2.0,
        "hidden_dim": 256,
    },
    "goal_evolution": {
        "n_samples_per_step": 100,
        "trajectory_length": 50,
        "hidden_dim": 256,
        "n_shard_components": 10,
        "policy_effect_threshold": 0.1,
    },
    "causal_model_extraction": {
        "n_samples": 500,
        "n_interventions": 100,
        "hidden_dim": 256,
        "n_permutation_trials": 100,
        "edge_corr_threshold": 0.1,
        "partial_corr_threshold": 0.05,
        "final_edge_threshold": 0.15,
    },
    "multiscale_goals": {
        "n_episodes": 100,
        "max_steps_per_episode": 256,
        "hidden_dim": 256,
    },
    "shard_dynamics": {
        "n_samples_per_step": 100,
        "trajectory_length": 50,
        "hidden_dim": 256,
        "n_shard_components": 15,
        "competition_threshold": 0.3,
    },
    "belief_behaviour_divergence": {
        "n_samples": 500,
        "hidden_dim": 256,
        "divergence_threshold": 0.3,
    },
    "teaching_opacity": {
        "n_samples": 500,
        "n_adversary_strategies": 5,
        "hidden_dim": 256,
        "n_strategy_clusters": 5,
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def get_experiments_for_method(
    training_method: str,
    include_training_time: bool = False,
) -> List[str]:
    """Return experiment names applicable to a training method.

    Args:
        training_method: One of accel, plr, robust_plr, dr, paired
        include_training_time: If True, also include training-time experiments
    """
    experiments = list(UNIVERSAL_EXPERIMENTS)
    if training_method == "paired":
        experiments.extend(PAIRED_EXPERIMENTS)
    if include_training_time:
        experiments.extend(TRAINING_TIME_EXPERIMENTS)
    return experiments


def get_experiment_config(name: str) -> Dict[str, Any]:
    """Get default config dict for a single experiment."""
    if name not in EXPERIMENT_DEFAULTS:
        raise ValueError(
            f"Unknown experiment: {name}. "
            f"Available: {sorted(EXPERIMENT_DEFAULTS.keys())}"
        )
    return EXPERIMENT_DEFAULTS[name].copy()


def get_all_experiment_defaults() -> Dict[str, Any]:
    """Return all experiment defaults namespaced as ``exp.<name>.<param>``.

    Example: ``exp.level_probing.n_levels`` = 500
    """
    flat: Dict[str, Any] = {}
    for exp_name, params in EXPERIMENT_DEFAULTS.items():
        for param, value in params.items():
            flat[f"exp.{exp_name}.{param}"] = value
    return flat
