#!/usr/bin/env python3
"""
Train agents with experiments running at each eval checkpoint.

ALL checkpoint-runnable experiments run at every eval_freq checkpoint.
Training-time experiments (requiring hooks) run if explicitly requested.

Usage:
    # Train one agent with all applicable experiments at each checkpoint
    python -m ablations.scripts.train_with_experiments \
        --agent_type persistent_lstm --training_method accel --seed 0

    # Train with specific experiments only
    python -m ablations.scripts.train_with_experiments \
        --agent_type persistent_lstm --training_method accel \
        --experiments level_probing value_calibration n_env_prediction

    # Train all 5 agents under a method with all experiments
    python -m ablations.scripts.train_with_experiments \
        --training_method accel --all_agents --seed 0

    # Train PAIRED (auto-includes 22 PAIRED-specific experiments)
    python -m ablations.scripts.train_with_experiments \
        --training_method paired --agent_type paired_persistent_lstm --seed 0

    # Train without experiments (just training)
    python -m ablations.scripts.train_with_experiments \
        --agent_type accel_probe --training_method dr --no_experiments

    # Dry run to see what would be run
    python -m ablations.scripts.train_with_experiments \
        --training_method paired --all_agents --dry_run
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
import wandb

from ablations.common.utils import parse_args, get_default_config, setup_checkpointing
from ablations.agents import get_agent_class, AGENT_CLASSES


# =========================================================================
# Experiment categories
# =========================================================================

# Universal experiments: run on any training method at every eval checkpoint
UNIVERSAL_EXPERIMENTS = [
    # Core experiments (1-5)
    'level_probing',
    'value_calibration',
    'activation_analysis',
    'cross_agent_comparison',
    # Transfer & robustness (6-8)
    'mutation_adaptation',
    'causal_intervention',
    'counterfactual',
    # Advanced interpretability (9-10)
    'output_probing',
    # Novel experiments (11-13)
    'goal_extraction',
    'cross_episode_flow',
    'dr_coverage',
    # Prediction experiments
    'n_env_prediction',
    'n_step_prediction',
]

# PAIRED-specific experiments: only for training_method=paired
PAIRED_EXPERIMENTS = [
    # Method-specific
    'adversary_dynamics',
    'regret_transfer',
    # A: Utility Function Extraction
    'utility_extraction',
    'adversary_policy_extraction',
    'bilateral_utility',
    # B: Causal Interventions
    'adversary_ablation',
    'regret_decomposition',
    'teaching_signal_intervention',
    'counterfactual_curriculum',
    'activation_patching',
    # C: Three-Agent Dynamics
    'representation_divergence',
    'antagonist_audit',
    'adversary_strategy_clustering',
    'coalition_dynamics',
    # D: Belief Revision Tracking
    'representation_trajectory',
    'belief_revision_detection',
    'goal_evolution',
    # F: Theoretical Validation
    'causal_model_extraction',
    'multiscale_goals',
    'shard_dynamics',
    'belief_behaviour_divergence',
    'teaching_opacity',
]

# Training-time experiments: need hooks during training loop, must be explicitly requested
TRAINING_TIME_EXPERIMENTS = [
    'behavioral_coupling',
    'symbolic_regression',
    'phase_transition',
]

# Base agents (for non-PAIRED methods)
BASE_AGENTS = [
    "accel_probe",
    "persistent_lstm",
    "context_vector",
    "episodic_memory",
    "next_env_prediction",
]

# PAIRED agents
PAIRED_AGENTS = [f"paired_{a}" for a in BASE_AGENTS]


def get_checkpoint_experiments_for_method(training_method: str) -> List[str]:
    """Get all checkpoint-runnable experiments for a training method."""
    experiments = list(UNIVERSAL_EXPERIMENTS)
    if training_method == "paired":
        experiments.extend(PAIRED_EXPERIMENTS)
    return experiments


def run_checkpoint_experiments(
    checkpoint_path: str,
    agent_type: str,
    training_method: str,
    experiments: List[str],
    output_dir: str,
    step: int,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run experiments on a checkpoint."""
    from ablations.experiments.run_experiment import run_experiment

    results = {}
    for exp_name in experiments:
        try:
            exp_output = os.path.join(output_dir, f"step_{step}", exp_name)
            result = run_experiment(
                experiment_name=exp_name,
                checkpoint_path=checkpoint_path,
                agent_type=agent_type,
                output_dir=exp_output,
                seed=seed,
                training_method=training_method,
            )
            results[exp_name] = {'status': 'success'}
        except Exception as e:
            results[exp_name] = {'status': 'error', 'error': str(e)}

    return results


def create_training_experiments(
    experiment_names: List[str],
    agent,
    train_state,
    training_method: str,
) -> list:
    """
    Create training-time experiment instances.

    Args:
        experiment_names: Names of training-time experiments to create
        agent: Agent instance
        train_state: Initial train state
        training_method: Training method being used

    Returns:
        List of TrainingTimeExperiment instances
    """
    from ablations.experiments import (
        BehavioralCouplingExperiment,
        SymbolicRegressionExperiment,
        PhaseTransitionExperiment,
    )

    experiment_classes = {
        'behavioral_coupling': BehavioralCouplingExperiment,
        'symbolic_regression': SymbolicRegressionExperiment,
        'phase_transition': PhaseTransitionExperiment,
    }

    experiments = []
    for name in experiment_names:
        if name in experiment_classes:
            exp = experiment_classes[name](
                agent=agent,
                train_state=train_state,
                config={},
                training_method=training_method,
            )
            experiments.append(exp)

    return experiments


def train_with_experiments(
    config: dict,
    checkpoint_experiments: List[str],
    training_time_experiments: List[str] = None,
) -> Dict[str, Any]:
    """
    Train agent with experiments at each eval checkpoint and training-time hooks.

    Args:
        config: Training configuration
        checkpoint_experiments: Experiments to run at each eval checkpoint
        training_time_experiments: Experiments requiring training hooks

    Returns:
        Dict with training summary and experiment results
    """
    import time as time_module
    from ablations.common.utils import setup_checkpointing

    agent_type = config["agent_type"]
    training_method = config["training_method"]
    seed = config["seed"]
    training_time_experiments = training_time_experiments or []

    # Get agent class
    agent_class = get_agent_class(agent_type)

    # Output directories
    checkpoint_dir = os.path.join(
        os.getcwd(), "checkpoints", training_method, agent_type, str(seed)
    )
    experiment_dir = os.path.join(
        os.getcwd(), "experiments", training_method, agent_type, str(seed)
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)

    # Initialize
    print(f"=" * 60)
    print(f"Training with Experiments")
    print(f"=" * 60)
    print(f"Training method: {training_method}")
    print(f"Agent type: {agent_type}")
    print(f"Seed: {seed}")
    print(f"Checkpoint experiments ({len(checkpoint_experiments)}): run at EVERY eval checkpoint")
    for exp in checkpoint_experiments:
        print(f"  - {exp}")
    if training_time_experiments:
        print(f"Training-time experiments ({len(training_time_experiments)}): hooks in training loop")
        for exp in training_time_experiments:
            print(f"  - {exp}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Experiment dir: {experiment_dir}")
    print(f"=" * 60)

    # Initialize wandb
    if config.get("use_wandb", True):
        wandb.init(
            project=config.get("project", "curriculum-awareness-ablation"),
            name=config.get("run_name", f"{training_method}_{agent_type}_seed{seed}"),
            config=config,
        )

    # Create agent (without training experiments initially)
    agent = agent_class(config)
    rng = jax.random.PRNGKey(seed)
    rng_init, rng_train = jax.random.split(rng)
    train_state = agent.create_train_state(rng_init)

    # Create and attach training-time experiments
    if training_time_experiments:
        print(f"Setting up {len(training_time_experiments)} training-time experiments...")
        tt_exps = create_training_experiments(
            training_time_experiments, agent, train_state, training_method
        )
        agent.training_experiments = tt_exps
        print(f"  Attached: {[e.name for e in tt_exps]}")

    # Setup checkpointing
    checkpoint_manager = setup_checkpointing(config, config["run_name"], seed)

    # Training loop with experiments
    eval_freq = config["eval_freq"]
    num_updates = config["num_updates"]
    num_eval_steps = num_updates // eval_freq

    results_summary = {
        'config': config,
        'checkpoint_experiments': {},
        'training_metrics': [],
    }

    runner_state = (rng_train, train_state)

    for eval_step in range(num_eval_steps):
        step_num = (eval_step + 1) * eval_freq
        start_time = time_module.time()

        # Train for eval_freq steps
        runner_state, metrics = agent.train_and_eval_step(runner_state)

        # Log training metrics
        train_time = time_module.time() - start_time
        metrics["time_delta"] = train_time
        agent.log_metrics(metrics, runner_state[1])

        results_summary['training_metrics'].append({
            'step': step_num,
            'metrics': {k: float(v) if hasattr(v, 'item') else v
                       for k, v in metrics.items() if isinstance(v, (int, float))}
        })

        # Save checkpoint
        train_state = runner_state[1]
        if hasattr(train_state, 'params'):
            checkpoint_items = {"params": train_state.params}
        else:
            # PAIREDTrainState: save all three network params
            checkpoint_items = {
                "pro_params": train_state.pro_train_state.params,
                "ant_params": train_state.ant_train_state.params,
                "adv_params": train_state.adv_train_state.params,
            }
        checkpoint_manager.save(eval_step, items=checkpoint_items)
        checkpoint_manager.wait_until_finished()

        print(f"\n[Step {step_num}/{num_updates}] Train time: {train_time:.1f}s")

        # Run ALL checkpoint experiments at this checkpoint
        if checkpoint_experiments:
            print(f"  Running {len(checkpoint_experiments)} experiments at checkpoint...")
            exp_start = time_module.time()

            exp_results = run_checkpoint_experiments(
                checkpoint_path=checkpoint_dir,
                agent_type=agent_type,
                training_method=training_method,
                experiments=checkpoint_experiments,
                output_dir=experiment_dir,
                step=step_num,
                seed=seed,
            )

            exp_time = time_module.time() - exp_start
            n_success = sum(1 for r in exp_results.values() if r['status'] == 'success')
            n_error = sum(1 for r in exp_results.values() if r['status'] == 'error')
            print(f"  Experiments: {n_success} success, {n_error} errors in {exp_time:.1f}s")

            results_summary['checkpoint_experiments'][step_num] = exp_results

            # Log to wandb
            if config.get("use_wandb", True):
                for exp_name, result in exp_results.items():
                    wandb.log({
                        f"experiment/{exp_name}/status": 1 if result['status'] == 'success' else 0,
                        "step": step_num,
                    })

    # Save summary
    summary_path = os.path.join(experiment_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Summary saved to: {summary_path}")

    return results_summary


def main():
    parser = argparse.ArgumentParser(
        description="Train agents with ALL experiments at each eval checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train one agent, all experiments at every checkpoint:
  %(prog)s --agent_type persistent_lstm --training_method accel --seed 0

  # Train all 5 PAIRED agents with 35 experiments each:
  %(prog)s --training_method paired --all_agents --seed 0

  # Only specific experiments:
  %(prog)s --agent_type accel_probe --experiments level_probing n_env_prediction n_step_prediction

  # Dry run to see what experiments would be selected:
  %(prog)s --training_method paired --all_agents --dry_run

  # Custom hyperparameters:
  %(prog)s --agent_type persistent_lstm --lr 3e-4 --num_updates 50000 --num_train_envs 64
""",
    )

    # ---- Run config ----
    parser.add_argument("--project", type=str, default="curriculum-awareness-ablation",
                        help="Wandb project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Wandb run name (default: {method}_{agent}_seed{seed})")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="Run mode (train or eval)")
    parser.add_argument("--checkpoint_directory", type=str, default=None,
                        help="Checkpoint directory for eval mode")
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1,
                        help="Specific checkpoint index to evaluate (-1 = latest)")

    # ---- Agent & method selection ----
    parser.add_argument("--agent_type", type=str, default=None,
                        help="Agent type (e.g., persistent_lstm, paired_persistent_lstm)")
    parser.add_argument("--training_method", type=str, default="accel",
                        choices=["accel", "plr", "robust_plr", "dr", "paired"])
    parser.add_argument("--all_agents", action="store_true",
                        help="Train all 5 agents under the training method")

    # ---- Checkpointing ----
    parser.add_argument("--checkpoint_save_interval", type=int, default=1,
                        help="Save checkpoint every N eval steps")
    parser.add_argument("--max_number_of_checkpoints", type=int, default=120,
                        help="Max checkpoints to keep on disk")

    # ---- Eval ----
    parser.add_argument("--eval_freq", type=int, default=250,
                        help="Eval (and experiment) frequency in training updates")
    parser.add_argument("--eval_num_attempts", type=int, default=10,
                        help="Number of eval attempts per level")
    parser.add_argument("--eval_levels", nargs="+", default=[
        "SixteenRooms", "SixteenRooms2",
        "Labyrinth", "LabyrinthFlipped", "Labyrinth2",
        "StandardMaze", "StandardMaze2", "StandardMaze3",
    ], help="Eval level names")

    # ---- PPO / Training hyperparams ----
    train_group = parser.add_argument_group("Training params")
    train_group.add_argument("--lr", type=float, default=1e-4)
    train_group.add_argument("--max_grad_norm", type=float, default=0.5)
    mut_group = train_group.add_mutually_exclusive_group()
    mut_group.add_argument("--num_updates", type=int, default=30000)
    mut_group.add_argument("--num_env_steps", type=int, default=None,
                           help="Total env steps (alternative to --num_updates)")
    train_group.add_argument("--num_steps", type=int, default=256,
                             help="Rollout length per environment")
    train_group.add_argument("--num_train_envs", type=int, default=32,
                             help="Number of parallel training environments")
    train_group.add_argument("--num_minibatches", type=int, default=1)
    train_group.add_argument("--gamma", type=float, default=0.995)
    train_group.add_argument("--epoch_ppo", type=int, default=5)
    train_group.add_argument("--clip_eps", type=float, default=0.2)
    train_group.add_argument("--gae_lambda", type=float, default=0.98)
    train_group.add_argument("--entropy_coeff", type=float, default=1e-3)
    train_group.add_argument("--critic_coeff", type=float, default=0.5)

    # ---- PLR ----
    plr_group = parser.add_argument_group("PLR params")
    plr_group.add_argument("--score_function", type=str, default="MaxMC",
                           choices=["MaxMC", "pvl"])
    plr_group.add_argument("--exploratory_grad_updates",
                           action=argparse.BooleanOptionalAction, default=False)
    plr_group.add_argument("--level_buffer_capacity", type=int, default=4000)
    plr_group.add_argument("--replay_prob", type=float, default=0.8)
    plr_group.add_argument("--staleness_coeff", type=float, default=0.3)
    plr_group.add_argument("--temperature", type=float, default=0.3)
    plr_group.add_argument("--top_k", type=int, default=4)
    plr_group.add_argument("--minimum_fill_ratio", type=float, default=0.5)
    plr_group.add_argument("--prioritization", type=str, default="rank",
                           choices=["rank", "topk"])
    plr_group.add_argument("--buffer_duplicate_check",
                           action=argparse.BooleanOptionalAction, default=True)

    # ---- ACCEL ----
    accel_group = parser.add_argument_group("ACCEL params")
    accel_group.add_argument("--use_accel",
                             action=argparse.BooleanOptionalAction, default=False)
    accel_group.add_argument("--num_edits", type=int, default=5)

    # ---- Environment ----
    env_group = parser.add_argument_group("Environment params")
    env_group.add_argument("--agent_view_size", type=int, default=5)
    env_group.add_argument("--n_walls", type=int, default=25)

    # ---- Probe ----
    probe_group = parser.add_argument_group("Probe params")
    probe_group.add_argument("--use_probe",
                             action=argparse.BooleanOptionalAction, default=True)
    probe_group.add_argument("--probe_lr", type=float, default=1e-3)
    probe_group.add_argument("--probe_tracking_buffer_size", type=int, default=500)

    # ---- Prediction / masking ----
    pred_group = parser.add_argument_group("Prediction params")
    pred_group.add_argument("--wall_loss_region", type=str, default="full",
                            choices=["full", "explored", "frontier"],
                            help="Wall loss masking region (full=all cells, "
                                 "explored=observed cells, frontier=observed+adjacent)")

    # ---- Experiment selection ----
    exp_group = parser.add_argument_group("Experiment selection")
    exp_group.add_argument("--experiments", type=str, nargs="+", default=None,
                           help="Specific experiments to run at each checkpoint "
                                "(default: ALL applicable for the training method)")
    exp_group.add_argument("--training_experiments", type=str, nargs="+", default=None,
                           choices=TRAINING_TIME_EXPERIMENTS,
                           help="Training-time experiments requiring hooks "
                                "(behavioral_coupling, symbolic_regression, phase_transition)")
    exp_group.add_argument("--no_experiments", action="store_true",
                           help="Disable all experiment running (just train)")

    # ---- Other ----
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print configuration without running")

    args = parser.parse_args()

    # Convert num_env_steps to num_updates if provided
    if args.num_env_steps is not None:
        args.num_updates = args.num_env_steps // (args.num_train_envs * args.num_steps)

    # Determine agents to train
    if args.all_agents:
        if args.training_method == "paired":
            agents = PAIRED_AGENTS
        else:
            agents = BASE_AGENTS
    else:
        if args.agent_type is None:
            parser.error("Either --agent_type or --all_agents is required")
        agents = [args.agent_type]

    # Validate agent types
    for agent in agents:
        if agent not in AGENT_CLASSES:
            parser.error(f"Unknown agent type: {agent}")
        if args.training_method == "paired" and not agent.startswith("paired_"):
            parser.error(f"PAIRED method requires paired_* agent, got {agent}")
        if args.training_method != "paired" and agent.startswith("paired_"):
            parser.error(f"Non-PAIRED method cannot use paired_* agent")

    # Determine checkpoint experiments: ALL applicable experiments at every checkpoint
    if args.no_experiments:
        checkpoint_exps = []
    elif args.experiments:
        checkpoint_exps = args.experiments
    else:
        # Default: all experiments appropriate for the training method
        checkpoint_exps = get_checkpoint_experiments_for_method(args.training_method)

    # Determine training-time experiments (require hooks, must be explicitly requested)
    if args.no_experiments:
        training_exps = []
    elif args.training_experiments:
        training_exps = args.training_experiments
    else:
        training_exps = []  # Must be explicitly requested (expensive, need hooks)

    # Build config from defaults + CLI overrides
    base_config = get_default_config()
    # Apply all CLI arguments that match config keys
    cli_overrides = {
        "training_method": args.training_method,
        "seed": args.seed,
        "num_updates": args.num_updates,
        "eval_freq": args.eval_freq,
        "eval_num_attempts": args.eval_num_attempts,
        "eval_levels": args.eval_levels,
        "use_wandb": not args.no_wandb,
        "project": args.project,
        "checkpoint_save_interval": args.checkpoint_save_interval,
        "max_number_of_checkpoints": args.max_number_of_checkpoints,
        # PPO
        "lr": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "num_steps": args.num_steps,
        "num_train_envs": args.num_train_envs,
        "num_minibatches": args.num_minibatches,
        "gamma": args.gamma,
        "epoch_ppo": args.epoch_ppo,
        "clip_eps": args.clip_eps,
        "gae_lambda": args.gae_lambda,
        "entropy_coeff": args.entropy_coeff,
        "critic_coeff": args.critic_coeff,
        # PLR
        "score_function": args.score_function,
        "exploratory_grad_updates": args.exploratory_grad_updates,
        "level_buffer_capacity": args.level_buffer_capacity,
        "replay_prob": args.replay_prob,
        "staleness_coeff": args.staleness_coeff,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "minimum_fill_ratio": args.minimum_fill_ratio,
        "prioritization": args.prioritization,
        "buffer_duplicate_check": args.buffer_duplicate_check,
        # ACCEL
        "use_accel": args.use_accel,
        "num_edits": args.num_edits,
        # Environment
        "agent_view_size": args.agent_view_size,
        "n_walls": args.n_walls,
        # Probe
        "use_probe": args.use_probe,
        "probe_lr": args.probe_lr,
        "probe_tracking_buffer_size": args.probe_tracking_buffer_size,
        # Prediction masking
        "wall_loss_region": args.wall_loss_region,
    }
    base_config.update(cli_overrides)

    if args.mode == "eval":
        os.environ["WANDB_MODE"] = "disabled"

    # Print summary
    print("=" * 60)
    print("Train with Experiments - Configuration")
    print("=" * 60)
    print(f"Project: {args.project}")
    print(f"Training method: {args.training_method}")
    print(f"Agents: {agents}")
    print(f"Seeds: [{args.seed}]")
    print(f"num_updates: {args.num_updates} | lr: {args.lr} | num_train_envs: {args.num_train_envs}")
    print(f"eval_freq: {args.eval_freq} | eval_levels: {len(args.eval_levels)}")
    print(f"Checkpoint experiments ({len(checkpoint_exps)}) - run at EVERY eval checkpoint:")
    for exp in checkpoint_exps:
        print(f"  - {exp}")
    print(f"Training-time experiments ({len(training_exps)}):")
    if training_exps:
        for exp in training_exps:
            print(f"  - {exp} (with hooks)")
    else:
        print("  (none - use --training_experiments to enable)")
    print(f"Total checkpoints: {args.num_updates // args.eval_freq}")
    print(f"Experiments per checkpoint: {len(checkpoint_exps)}")
    total_exp_runs = len(checkpoint_exps) * (args.num_updates // args.eval_freq) * len(agents)
    print(f"Total experiment runs: {total_exp_runs} ({len(checkpoint_exps)} x {args.num_updates // args.eval_freq} checkpoints x {len(agents)} agents)")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] No training performed.")
        return

    # Train each agent
    for agent_type in agents:
        config = base_config.copy()
        config["agent_type"] = agent_type
        config["run_name"] = args.run_name or f"{args.training_method}_{agent_type}_seed{args.seed}"

        print(f"\n{'#'*60}")
        print(f"Training: {agent_type}")
        print(f"{'#'*60}")

        train_with_experiments(config, checkpoint_exps, training_exps)


if __name__ == "__main__":
    main()
