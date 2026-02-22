"""
PAIRED Experiment Suite.

Experiments for analyzing PAIRED (Protagonist Antagonist Induced Regret Environment Design):
- Utility function extraction from prediction losses
- Causal interventions on adversary/antagonist
- Three-agent dynamics analysis
- Belief revision tracking
- Theoretical validation

Categories:
    A. Utility Function Extraction (A1-A3)
    B. Causal Interventions (B1-B5)
    C. Three-Agent Dynamics (C1-C4)
    D. Belief Revision Tracking (D1-D3)
    F. Theoretical Validation (F1-F5)

Note: E-series experiments (E1-E7) are modifications to existing experiments
in the parent directory, not separate files here.
"""

# A. Utility Function Extraction
from .utility_extraction import UtilityExtractionExperiment
from .adversary_policy_extraction import AdversaryPolicyExtractionExperiment
from .bilateral_utility import BilateralUtilityExperiment

# B. Causal Interventions
from .adversary_ablation import AdversaryAblationExperiment
from .regret_decomposition import RegretDecompositionExperiment
from .teaching_signal_intervention import TeachingSignalInterventionExperiment
from .counterfactual_curriculum import CounterfactualCurriculumExperiment
from .activation_patching import ActivationPatchingExperiment

# C. Three-Agent Dynamics
from .representation_divergence import RepresentationDivergenceExperiment
from .antagonist_audit import AntagonistAuditExperiment
from .adversary_strategy_clustering import AdversaryStrategyClusteringExperiment
from .coalition_dynamics import CoalitionDynamicsExperiment

# D. Belief Revision Tracking
from .representation_trajectory import RepresentationTrajectoryExperiment
from .belief_revision_detection import BeliefRevisionDetectionExperiment
from .goal_evolution import GoalEvolutionExperiment

# F. Theoretical Validation
from .causal_model_extraction import CausalModelExtractionExperiment
from .multiscale_goals import MultiscaleGoalsExperiment
from .shard_dynamics import ShardDynamicsExperiment
from .belief_behaviour_divergence import BeliefBehaviourDivergenceExperiment
from .teaching_opacity import TeachingOpacityExperiment

# Moved from main experiments folder (PAIRED-only)
from .adversary_dynamics import AdversaryDynamicsExperiment
from .regret_transfer import RegretTransferExperiment

__all__ = [
    # A. Utility Function Extraction
    'UtilityExtractionExperiment',
    'AdversaryPolicyExtractionExperiment',
    'BilateralUtilityExperiment',
    # B. Causal Interventions
    'AdversaryAblationExperiment',
    'RegretDecompositionExperiment',
    'TeachingSignalInterventionExperiment',
    'CounterfactualCurriculumExperiment',
    'ActivationPatchingExperiment',
    # C. Three-Agent Dynamics
    'RepresentationDivergenceExperiment',
    'AntagonistAuditExperiment',
    'AdversaryStrategyClusteringExperiment',
    'CoalitionDynamicsExperiment',
    # D. Belief Revision Tracking
    'RepresentationTrajectoryExperiment',
    'BeliefRevisionDetectionExperiment',
    'GoalEvolutionExperiment',
    # F. Theoretical Validation
    'CausalModelExtractionExperiment',
    'MultiscaleGoalsExperiment',
    'ShardDynamicsExperiment',
    'BeliefBehaviourDivergenceExperiment',
    'TeachingOpacityExperiment',
    # Moved from main experiments folder (PAIRED-only)
    'AdversaryDynamicsExperiment',
    'RegretTransferExperiment',
]
