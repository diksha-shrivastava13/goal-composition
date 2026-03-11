"""
Interpretability experiments for curriculum awareness analysis.

This module contains experiments for studying goal emergence and evolution
in curriculum learning, with support for multiple training methods:
- ACCEL: Autocurricula via Emergent Complexity (branches: DR, Replay, Mutate)
- PLR: Prioritized Level Replay (branches: DR, Replay)
- PAIRED: Protagonist Antagonist Induced Regret Environment Design (adversarial)
- DR: Domain Randomization (no curriculum structure)

Core Experiments (1-5):
- LevelProbingExperiment: Probe hidden states for level properties
- ValueCalibrationExperiment: Analyze value function calibration
- ActivationAnalysisExperiment: RSA, CKA, and dimensionality reduction
- BehavioralCouplingExperiment: Signal-performance correlation over training
- CrossAgentComparisonExperiment: Compare prediction loss across agent types

Transfer & Robustness (6-8):
- MutationAdaptationExperiment: Knowledge transfer (method-appropriate)
- CausalInterventionExperiment: Robustness to curriculum shifts
- CounterfactualExperiment: Memory injection testing

Advanced Interpretability (9-10):
- SymbolicRegressionExperiment: Predictability proxy analysis
- OutputProbingExperiment: Policy/value output analysis

Novel Experiments (11-13):
- GoalExtractionExperiment: Task-relevant activation analysis
- PhaseTransitionExperiment: Training dynamics characterization
- CrossEpisodeFlowExperiment: Cross-episode information flow

Method-Specific Experiments:
- DRCoverageExperiment: DR-specific level space coverage analysis

Note: PAIRED-specific experiments (AdversaryDynamicsExperiment, RegretTransferExperiment)
are in the paired/ subdirectory.
"""

from .base import (
    BaseExperiment,
    CheckpointExperiment,
    TrainingTimeExperiment,
    TRAINING_METHODS,
    get_method_properties,
)

# Core experiments
from .level_probing import LevelProbingExperiment
from .value_calibration import ValueCalibrationExperiment
from .activation_analysis import ActivationAnalysisExperiment
from .behavioral_coupling import BehavioralCouplingExperiment
from .cross_agent_comparison import CrossAgentComparisonExperiment

# Transfer & robustness experiments
from .mutation_adaptation import MutationAdaptationExperiment
from .causal_intervention import CausalInterventionExperiment
from .counterfactual import CounterfactualExperiment

# Advanced interpretability experiments
from .symbolic_regression import SymbolicRegressionExperiment
from .output_probing import OutputProbingExperiment

# Novel experiments
from .goal_extraction import GoalExtractionExperiment
from .phase_transition import PhaseTransitionExperiment
from .cross_episode_flow import CrossEpisodeFlowExperiment

# Prediction experiments (N-ENV and N-STEP)
from .n_env_prediction import NEnvPredictionExperiment
from .n_step_prediction import NStepPredictionExperiment

# Method-specific experiments (DR-specific only; PAIRED-specific moved to paired/)
from .dr_coverage import DRCoverageExperiment

# PAIRED-specific experiments
from .paired import (
    # A. Utility Function Extraction
    UtilityExtractionExperiment,
    AdversaryPolicyExtractionExperiment,
    BilateralUtilityExperiment,
    # B. Causal Interventions
    AdversaryAblationExperiment,
    RegretDecompositionExperiment,
    TeachingSignalInterventionExperiment,
    CounterfactualCurriculumExperiment,
    ActivationPatchingExperiment,
    # C. Three-Agent Dynamics
    RepresentationDivergenceExperiment,
    AntagonistAuditExperiment,
    AdversaryStrategyClusteringExperiment,
    CoalitionDynamicsExperiment,
    # D. Belief Revision Tracking
    RepresentationTrajectoryExperiment,
    BeliefRevisionDetectionExperiment,
    GoalEvolutionExperiment,
    # F. Theoretical Validation
    CausalModelExtractionExperiment,
    MultiscaleGoalsExperiment,
    ShardDynamicsExperiment,
    BeliefBehaviourDivergenceExperiment,
    TeachingOpacityExperiment,
    # Moved from main experiments folder (PAIRED-only)
    AdversaryDynamicsExperiment,
    RegretTransferExperiment,
)

# Experiment runners
from .run_experiment import run_experiment, get_experiment_class
from .run_all import run_all_experiments, UNIVERSAL_EXPERIMENTS, PAIRED_EXPERIMENTS, BASE_AGENTS, PAIRED_AGENTS
from .run_paired_suite import run_paired_suite, get_paired_experiments, PAIRED_EXPERIMENT_ORDER

__all__ = [
    # Base classes and utilities
    "BaseExperiment",
    "CheckpointExperiment",
    "TrainingTimeExperiment",
    "TRAINING_METHODS",
    "get_method_properties",

    # Core experiments
    "LevelProbingExperiment",
    "ValueCalibrationExperiment",
    "ActivationAnalysisExperiment",
    "BehavioralCouplingExperiment",
    "CrossAgentComparisonExperiment",

    # Transfer & robustness experiments
    "MutationAdaptationExperiment",
    "CausalInterventionExperiment",
    "CounterfactualExperiment",

    # Advanced interpretability experiments
    "SymbolicRegressionExperiment",
    "OutputProbingExperiment",

    # Novel experiments
    "GoalExtractionExperiment",
    "PhaseTransitionExperiment",
    "CrossEpisodeFlowExperiment",

    # Prediction experiments
    "NEnvPredictionExperiment",
    "NStepPredictionExperiment",

    # Method-specific experiments (DR-specific; PAIRED-specific in paired/)
    "DRCoverageExperiment",
    # Re-exported from paired/ for backwards compatibility
    "AdversaryDynamicsExperiment",
    "RegretTransferExperiment",

    # Runners
    "run_experiment",
    "get_experiment_class",
    "run_all_experiments",
    "UNIVERSAL_EXPERIMENTS",
    "PAIRED_EXPERIMENTS",
    "BASE_AGENTS",
    "PAIRED_AGENTS",
]


def get_experiment_by_name(name: str):
    """Get experiment class by name.

    Delegates to get_experiment_class which has the complete registry
    of all 38 experiments (16 universal + 22 PAIRED-specific).
    """
    return get_experiment_class(name)


def get_compatible_experiments(training_method: str) -> list:
    """Get experiments compatible with a training method."""
    from .run_all import get_experiments_for_method
    return get_experiments_for_method(training_method)
