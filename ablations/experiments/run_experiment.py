"""
Single experiment runner.

Runs a single experiment on a checkpoint and saves results.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import jax


class _PAIREDTrainStateWrapper:
    """Wraps PAIREDTrainState to expose protagonist's apply_fn/params for experiments.

    Experiments access self.train_state.apply_fn and self.train_state.params,
    which don't exist on PAIREDTrainState. This wrapper delegates those to the
    protagonist's train state while preserving access to ant_train_state/adv_train_state.
    """

    def __init__(self, paired_state):
        self._paired_state = paired_state
        # Expose protagonist's apply_fn and params at top level
        self.apply_fn = paired_state.pro_train_state.apply_fn
        self.params = paired_state.pro_train_state.params
        # Expose sub-states for experiments that access them via getattr
        self.pro_train_state = paired_state.pro_train_state
        self.ant_train_state = paired_state.ant_train_state
        self.adv_train_state = paired_state.adv_train_state

    def __getattr__(self, name):
        # Fall back to the underlying PAIREDTrainState for other attributes
        return getattr(self._paired_state, name)


def _wrap_paired_train_state(paired_state):
    return _PAIREDTrainStateWrapper(paired_state)


def get_experiment_class(experiment_name: str):
    """Get experiment class by name."""
    from . import (
        LevelProbingExperiment,
        ValueCalibrationExperiment,
        ActivationAnalysisExperiment,
        BehavioralCouplingExperiment,
        CrossAgentComparisonExperiment,
        MutationAdaptationExperiment,
        CausalInterventionExperiment,
        CounterfactualExperiment,
        SymbolicRegressionExperiment,
        OutputProbingExperiment,
        GoalExtractionExperiment,
        PhaseTransitionExperiment,
        CrossEpisodeFlowExperiment,
        # Prediction experiments
        NEnvPredictionExperiment,
        NStepPredictionExperiment,
        # Method-specific experiments
        AdversaryDynamicsExperiment,
        RegretTransferExperiment,
        DRCoverageExperiment,
    )

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
    )

    experiments = {
        'level_probing': LevelProbingExperiment,
        'value_calibration': ValueCalibrationExperiment,
        'activation_analysis': ActivationAnalysisExperiment,
        'behavioral_coupling': BehavioralCouplingExperiment,
        'cross_agent_comparison': CrossAgentComparisonExperiment,
        'mutation_adaptation': MutationAdaptationExperiment,
        'causal_intervention': CausalInterventionExperiment,
        'counterfactual': CounterfactualExperiment,
        'symbolic_regression': SymbolicRegressionExperiment,
        'output_probing': OutputProbingExperiment,
        'goal_extraction': GoalExtractionExperiment,
        'phase_transition': PhaseTransitionExperiment,
        'cross_episode_flow': CrossEpisodeFlowExperiment,
        # Prediction experiments
        'n_env_prediction': NEnvPredictionExperiment,
        'n_step_prediction': NStepPredictionExperiment,
        # Method-specific experiments
        'adversary_dynamics': AdversaryDynamicsExperiment,
        'regret_transfer': RegretTransferExperiment,
        'dr_coverage': DRCoverageExperiment,
        # PAIRED-specific experiments (A-series: Utility Function Extraction)
        'utility_extraction': UtilityExtractionExperiment,
        'adversary_policy_extraction': AdversaryPolicyExtractionExperiment,
        'bilateral_utility': BilateralUtilityExperiment,
        # PAIRED-specific experiments (B-series: Causal Interventions)
        'adversary_ablation': AdversaryAblationExperiment,
        'regret_decomposition': RegretDecompositionExperiment,
        'teaching_signal_intervention': TeachingSignalInterventionExperiment,
        'counterfactual_curriculum': CounterfactualCurriculumExperiment,
        'activation_patching': ActivationPatchingExperiment,
        # PAIRED-specific experiments (C-series: Three-Agent Dynamics)
        'representation_divergence': RepresentationDivergenceExperiment,
        'antagonist_audit': AntagonistAuditExperiment,
        'adversary_strategy_clustering': AdversaryStrategyClusteringExperiment,
        'coalition_dynamics': CoalitionDynamicsExperiment,
        # PAIRED-specific experiments (D-series: Belief Revision Tracking)
        'representation_trajectory': RepresentationTrajectoryExperiment,
        'belief_revision_detection': BeliefRevisionDetectionExperiment,
        'goal_evolution': GoalEvolutionExperiment,
        # PAIRED-specific experiments (F-series: Theoretical Validation)
        'causal_model_extraction': CausalModelExtractionExperiment,
        'multiscale_goals': MultiscaleGoalsExperiment,
        'shard_dynamics': ShardDynamicsExperiment,
        'belief_behaviour_divergence': BeliefBehaviourDivergenceExperiment,
        'teaching_opacity': TeachingOpacityExperiment,
    }

    if experiment_name not in experiments:
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {list(experiments.keys())}")

    return experiments[experiment_name]


def load_checkpoint(checkpoint_path: str, agent=None, seed: int = 0):
    """Load checkpoint from path.

    Args:
        checkpoint_path: Path to checkpoint directory (containing config.json and models/)
        agent: Optional agent instance for creating train_state template.
               If None, attempts to create one from the config in checkpoint.
        seed: Random seed for creating template train state

    Returns:
        (train_state, config) tuple
    """
    from ..common.utils import load_checkpoint as _load_checkpoint

    # Read config from checkpoint dir to get agent info
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    # Create agent if not provided (needed for train_state template)
    if agent is None:
        agent_type = config.get("agent_type", "accel_probe")
        agent = load_agent(agent_type, config)

    # Create template train state for structure
    rng = jax.random.PRNGKey(seed)
    train_state_template = agent.create_train_state(rng)

    return _load_checkpoint(checkpoint_path, train_state_template)


def load_agent(agent_type: str, config: Optional[Dict[str, Any]] = None):
    """Load agent by type."""
    from ..agents import get_agent_class

    agent_cls = get_agent_class(agent_type)
    agent = agent_cls(config or {})
    return agent


def run_experiment(
    experiment_name: str,
    checkpoint_path: str,
    agent_type: str,
    output_dir: str,
    seed: int = 0,
    training_method: str = "accel",
    experiment_kwargs: Optional[Dict[str, Any]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a single experiment.

    Args:
        experiment_name: Name of the experiment to run
        checkpoint_path: Path to the checkpoint directory
        agent_type: Type of agent (e.g., 'persistent_lstm')
        output_dir: Directory to save results
        seed: Random seed
        training_method: Training method used (accel, plr, robust_plr, paired, dr)
        experiment_kwargs: Additional experiment parameters
        config_overrides: Runtime config overrides (e.g. from CLI --n_levels, --max_steps)

    Returns:
        Dict with experiment results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get experiment class
    ExperimentClass = get_experiment_class(experiment_name)

    # Read config from checkpoint dir first so agent gets proper config
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    config['training_method'] = training_method

    # Apply runtime config overrides (e.g. CLI experiment params)
    if config_overrides:
        config.update(config_overrides)

    # Load agent with checkpoint config (needed for checkpoint template)
    agent = load_agent(agent_type, config)

    # Load checkpoint (uses agent for train_state template)
    train_state, checkpoint_config = load_checkpoint(checkpoint_path, agent=agent, seed=seed)

    # Merge any additional checkpoint config
    if checkpoint_config:
        config.update(checkpoint_config)

    # Re-apply runtime overrides after checkpoint config merge (CLI takes precedence)
    if config_overrides:
        config.update(config_overrides)

    # For PAIRED agents, wrap the train_state so experiments can access
    # apply_fn/params (protagonist) while still accessing ant_train_state etc.
    from ..common.types import PAIREDTrainState
    if isinstance(train_state, PAIREDTrainState):
        train_state = _wrap_paired_train_state(train_state)

    # Create experiment with required config arg
    kwargs = experiment_kwargs or {}
    kwargs['training_method'] = training_method

    # Handle secondary checkpoint for cross-adversary experiments (e.g., regret_transfer)
    sec_path = kwargs.pop('secondary_checkpoint_path', None)
    if sec_path:
        sec_agent = load_agent(agent_type, config)
        sec_ts, _ = load_checkpoint(sec_path, agent=sec_agent, seed=seed)
        if isinstance(sec_ts, PAIREDTrainState):
            sec_ts = _wrap_paired_train_state(sec_ts)
        kwargs['secondary_train_state'] = sec_ts
        kwargs['secondary_agent'] = sec_agent

    experiment = ExperimentClass(
        agent=agent,
        train_state=train_state,
        config=config,
        output_dir=str(output_path),
        **kwargs,
    )

    # Run experiment
    rng = jax.random.PRNGKey(seed)

    print(f"Running {experiment_name}...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Agent: {agent_type}")

    # Collect data
    print("  Collecting data...")
    experiment.data = experiment.collect_data(rng)

    # Analyze
    print("  Analyzing...")
    results = experiment.analyze()

    # Visualize
    print("  Generating visualizations...")
    viz_data = experiment.visualize()

    # Save results
    results_path = output_path / f"{experiment_name}_results.json"

    def convert_to_serializable(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            # Skip non-serializable objects (e.g., matplotlib Figures)
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return None

    results_serializable = convert_to_serializable(results)
    viz_serializable = convert_to_serializable(viz_data)

    output = {
        'experiment': experiment_name,
        'checkpoint': checkpoint_path,
        'agent_type': agent_type,
        'training_method': training_method,
        'seed': seed,
        'results': results_serializable,
        'visualization': viz_serializable,
    }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Results saved to: {results_path}")

    return output


def main():
    from ..configs.cli import add_common_args, add_experiment_param_args, EXPERIMENT_PARAM_KEYS

    parser = argparse.ArgumentParser(description="Run a single experiment")

    # Shared args
    add_common_args(parser)
    add_experiment_param_args(parser)

    # Entry-point-specific
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name (e.g., level_probing)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--secondary_checkpoint', type=str, default=None,
                        help='Secondary checkpoint path for cross-adversary experiments '
                             '(e.g., regret_transfer). Can be a different seed (cross-seed) '
                             'or a different step from the same run (cross-stage).')

    args = parser.parse_args()

    if args.dry_run:
        # Validate that experiment is compatible with training method
        from . import TRAINING_METHODS
        from ..configs import PAIRED_EXPERIMENTS
        if args.training_method not in TRAINING_METHODS:
            print(f"Unknown training method: {args.training_method}")
            return

        print(f"Experiment: {args.experiment}")
        print(f"Training method: {args.training_method}")
        print(f"Method properties: {TRAINING_METHODS[args.training_method]}")

        paired_only = set(PAIRED_EXPERIMENTS)
        if args.experiment in paired_only and args.training_method != 'paired':
            print(f"WARNING: {args.experiment} is PAIRED-specific, will return error for {args.training_method}")
        else:
            print("Compatibility check passed")
        return

    experiment_kwargs = {}
    if args.secondary_checkpoint:
        experiment_kwargs['secondary_checkpoint_path'] = args.secondary_checkpoint

    # Collect CLI experiment param overrides (only if explicitly set)
    config_overrides = {}
    for key in EXPERIMENT_PARAM_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            config_overrides[key] = val

    run_experiment(
        experiment_name=args.experiment,
        checkpoint_path=args.checkpoint,
        agent_type=args.agent_type,
        output_dir=args.output_dir or ".",
        seed=args.seed,
        training_method=args.training_method,
        experiment_kwargs=experiment_kwargs or None,
        config_overrides=config_overrides or None,
    )


if __name__ == '__main__':
    main()
