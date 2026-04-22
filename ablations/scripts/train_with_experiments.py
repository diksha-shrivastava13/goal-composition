#!/usr/bin/env python3
"""
Canonical training script with integrated experiments.

ALL 38 experiments run by default at every eval checkpoint (every eval_freq
training updates). The 3 training-time experiments (behavioral_coupling,
symbolic_regression, phase_transition) maintain persistent state across
checkpoint calls to track trends over the training process.

Usage:
    # Train one agent with ALL experiments (default):
    python -m ablations.scripts.train_with_experiments \
        --agent_type persistent_lstm --training_method accel --seed 0

    # Train with specific experiments only (auto-partitions checkpoint vs hooks):
    python -m ablations.scripts.train_with_experiments \
        --agent_type persistent_lstm --training_method accel \
        --experiments level_probing behavioral_coupling

    # Train all 5 agents under a method with all experiments:
    python -m ablations.scripts.train_with_experiments \
        --training_method accel --all_agents --seed 0

    # Train PAIRED (auto-includes 22 PAIRED-specific experiments + 3 training-time):
    python -m ablations.scripts.train_with_experiments \
        --training_method paired --agent_type paired_persistent_lstm --seed 0

    # Train without experiments (just training):
    python -m ablations.scripts.train_with_experiments \
        --agent_type accel_probe --training_method dr --no_experiments

    # Resume from checkpoint:
    python -m ablations.scripts.train_with_experiments \
        --agent_type persistent_lstm --training_method accel --seed 0 \
        --resume checkpoints/accel/persistent_lstm/0

    # Load config from file:
    python -m ablations.scripts.train_with_experiments --config my_config.json

    # Dry run to see what would be run:
    python -m ablations.scripts.train_with_experiments \
        --training_method paired --all_agents --dry_run
"""

import logging
import os
import sys
import time
import json
import argparse

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
import wandb

from ablations.common.utils import setup_checkpointing
from ablations.agents import get_agent_class, AGENT_CLASSES
from ablations.configs import (
    BASE_AGENTS,
    PAIRED_AGENTS,
    UNIVERSAL_EXPERIMENTS,
    PAIRED_EXPERIMENTS,
    TRAINING_TIME_EXPERIMENTS,
    TRAINING_TIME_SET,
    EXPERIMENT_CADENCE,
    get_experiments_for_method,
    get_experiment_param_overrides,
    EXPERIMENT_PARAM_KEYS,
)
from ablations.configs.cli import (
    add_common_args,
    add_training_args,
    add_experiment_selection_args,
    add_experiment_param_args,
    build_config_from_args,
)


def get_checkpoint_experiments_for_method(training_method: str) -> List[str]:
    """Get all checkpoint-runnable experiments for a training method."""
    return get_experiments_for_method(training_method, include_training_time=False)


def run_checkpoint_experiments(
    checkpoint_path: str,
    agent_type: str,
    training_method: str,
    experiments: List[str],
    output_dir: str,
    step: int,
    seed: int = 0,
    config_overrides: Optional[Dict[str, Any]] = None,
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
                config_overrides=config_overrides,
            )
            results[exp_name] = {'status': 'success', 'result': result}
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


def _format_eta(seconds: float) -> str:
    """Format seconds into human-readable ETA string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def train_with_experiments(
    config: dict,
    checkpoint_experiments: List[str],
    training_time_experiments: List[str] = None,
    resume_dir: str = None,
) -> Dict[str, Any]:
    """
    Train agent with experiments at each eval checkpoint and training-time hooks.

    Args:
        config: Training configuration
        checkpoint_experiments: Experiments to run at each eval checkpoint
        training_time_experiments: Experiments requiring training hooks
        resume_dir: Directory to resume training from (optional)

    Returns:
        Dict with training summary and experiment results
    """
    import time as time_module

    agent_type = config["agent_type"]
    training_method = config["training_method"]
    seed = config["seed"]
    training_time_experiments = training_time_experiments or []
    output_base = config.get("output_dir", os.getcwd())

    # Get agent class
    agent_class = get_agent_class(agent_type)

    # Output directories
    checkpoint_dir = os.path.join(
        output_base, "checkpoints", training_method, agent_type, str(seed)
    )
    experiment_dir = os.path.join(
        output_base, "experiments", training_method, agent_type, str(seed)
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
        print(f"Training-time experiments ({len(training_time_experiments)}): hooks at eval checkpoints")
        for exp in training_time_experiments:
            print(f"  - {exp}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Experiment dir: {experiment_dir}")
    if resume_dir:
        print(f"Resuming from: {resume_dir}")
    print(f"=" * 60)

    # Initialize wandb
    if config.get("use_wandb", True):
        wandb.login()
        wandb.init(
            project=config.get("project", "JaxUED-minigrid-maze"),
            name=config.get("run_name", f"{training_method}_{agent_type}_seed{seed}"),
            group=config.get("group_name", None),
            config=config,
        )

    # Create agent
    agent = agent_class(config)
    rng = jax.random.PRNGKey(seed)
    rng_init, rng_train = jax.random.split(rng)
    train_state = agent.create_train_state(rng_init)

    # Resume from checkpoint if requested
    start_eval_step = 0
    if resume_dir:
        from ablations.common.utils import load_checkpoint
        try:
            train_state, loaded_config = load_checkpoint(resume_dir, train_state)
            # Detect last completed eval step from checkpoint manager
            import orbax.checkpoint as ocp
            resume_ckpt_mgr = ocp.CheckpointManager(
                os.path.join(resume_dir, "models"),
                ocp.PyTreeCheckpointer(),
            )
            latest_step = resume_ckpt_mgr.latest_step()
            if latest_step is not None:
                start_eval_step = latest_step + 1
                print(f"Resuming from eval step {start_eval_step} (checkpoint {latest_step})")
        except Exception as e:
            print(f"Warning: Could not resume from {resume_dir}: {e}")
            print("Starting from scratch.")

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

    # Timing for ETA estimation
    step_times = deque(maxlen=10)  # Rolling window of recent step times

    for eval_step in range(start_eval_step, num_eval_steps):
        step_num = (eval_step + 1) * eval_freq
        step_start = time_module.time()

        # Train for eval_freq steps
        runner_state, metrics = agent.train_and_eval_step(runner_state)

        # Log training metrics
        train_time = time_module.time() - step_start
        metrics["time_delta"] = train_time
        agent.log_metrics(metrics, runner_state[1])

        # Call training-time experiment hooks
        agent._call_training_hooks(runner_state[1], metrics, step_num)

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

        # Run checkpoint experiments (filtered by cadence)
        exp_time = 0.0
        active_checkpoint_exps = [
            name for name in checkpoint_experiments
            if (eval_step + 1) % EXPERIMENT_CADENCE.get(name, 1) == 0
        ]
        if active_checkpoint_exps:
            exp_start = time_module.time()

            # Extract experiment-specific config overrides from config
            _exp_overrides = {k: config[k] for k in EXPERIMENT_PARAM_KEYS if k in config}

            exp_results = run_checkpoint_experiments(
                checkpoint_path=checkpoint_dir,
                agent_type=agent_type,
                training_method=training_method,
                experiments=active_checkpoint_exps,
                output_dir=experiment_dir,
                step=step_num,
                seed=seed,
                config_overrides=_exp_overrides or None,
            )

            exp_time = time_module.time() - exp_start
            n_success = sum(1 for r in exp_results.values() if r['status'] == 'success')
            n_error = sum(1 for r in exp_results.values() if r['status'] == 'error')

            results_summary['checkpoint_experiments'][step_num] = exp_results

            # Log to wandb with richer metrics
            if config.get("use_wandb", True):
                exp_log = {}
                for exp_name, result in exp_results.items():
                    exp_log[f"experiment/{exp_name}/status"] = 1 if result['status'] == 'success' else 0
                    # Log key metric values from experiment results
                    if result['status'] == 'success' and isinstance(result.get('result'), dict):
                        for k, v in result['result'].items():
                            if isinstance(v, (int, float)):
                                exp_log[f"experiment/{exp_name}/{k}"] = v
                exp_log["step"] = step_num
                wandb.log(exp_log)

        # ETA estimation
        total_step_time = time_module.time() - step_start
        step_times.append(total_step_time)
        avg_step_time = sum(step_times) / len(step_times)
        remaining_steps = num_eval_steps - eval_step - 1
        eta = avg_step_time * remaining_steps

        # Progress display
        progress_parts = [f"[Step {step_num}/{num_updates}]"]
        progress_parts.append(f"Train: {train_time:.1f}s")
        if checkpoint_experiments:
            progress_parts.append(f"Exps: {exp_time:.1f}s ({n_success}ok/{n_error}err)")
        progress_parts.append(f"ETA: {_format_eta(eta)}")
        print(" | ".join(progress_parts))

    # Finalize training-time experiments
    agent._finalize_training_experiments()

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
  # Train one agent, all 38 experiments at every checkpoint:
  %(prog)s --agent_type persistent_lstm --training_method accel --seed 0

  # Train all 5 PAIRED agents with 38 experiments each:
  %(prog)s --training_method paired --all_agents --seed 0

  # Only specific experiments (auto-partitions checkpoint vs training-time):
  %(prog)s --agent_type accel_probe --experiments level_probing behavioral_coupling

  # Resume training from checkpoint:
  %(prog)s --agent_type persistent_lstm --training_method accel --seed 0 \
      --resume checkpoints/accel/persistent_lstm/0

  # Load config from JSON file:
  %(prog)s --config my_config.json --agent_type persistent_lstm

  # Dry run to see what experiments would be selected:
  %(prog)s --training_method paired --all_agents --dry_run
""",
    )

    # Shared argparse builders
    add_common_args(parser)
    add_training_args(parser)
    add_experiment_selection_args(parser)
    add_experiment_param_args(parser)

    # Entry-point-specific args
    parser.add_argument("--all_agents", action="store_true",
                        help="Train all 5 agents under the training method")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint directory")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")

    args = parser.parse_args()

    # Build config via shared resolver
    base_config = build_config_from_args(args)
    base_config["use_wandb"] = not args.no_wandb

    if args.mode == "eval":
        os.environ["WANDB_MODE"] = "disabled"

    # Generate group_name
    try:
        group_name = "".join([
            str(base_config.get(key, ""))
            for key in sorted([a.dest for a in parser._action_groups[2]._group_actions])
        ])
        base_config["group_name"] = group_name
    except Exception:
        logger.debug("Failed to generate group_name", exc_info=True)
        base_config["group_name"] = None

    # Determine agents to train
    if args.all_agents:
        agents = PAIRED_AGENTS if args.training_method == "paired" else BASE_AGENTS
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

    # Unified experiment selection: auto-partition into checkpoint vs training-time
    if args.no_experiments:
        checkpoint_exps = []
        training_exps = []
    elif args.experiments:
        # User specified -- auto-partition
        checkpoint_exps = [e for e in args.experiments if e not in TRAINING_TIME_SET]
        training_exps = [e for e in args.experiments if e in TRAINING_TIME_SET]
    else:
        # Default: ALL experiments
        checkpoint_exps = get_checkpoint_experiments_for_method(args.training_method)
        training_exps = list(TRAINING_TIME_EXPERIMENTS)

    # Print summary
    total_exps = len(checkpoint_exps) + len(training_exps)
    print("=" * 60)
    print("Train with Experiments - Configuration")
    print("=" * 60)
    print(f"Project: {base_config.get('project', 'JaxUED-minigrid-maze')}")
    print(f"Training method: {args.training_method}")
    print(f"Agents: {agents}")
    print(f"Seeds: [{args.seed}]")
    print(f"num_updates: {base_config['num_updates']} | lr: {base_config['lr']} | num_train_envs: {base_config['num_train_envs']}")
    print(f"eval_freq: {base_config['eval_freq']} | eval_levels: {len(base_config['eval_levels'])}")
    print(f"All experiments ({total_exps}) - run at EVERY eval checkpoint:")
    print(f"  Checkpoint experiments ({len(checkpoint_exps)}):")
    for exp in checkpoint_exps:
        print(f"    - {exp}")
    print(f"  Training-time experiments ({len(training_exps)}) [persistent state]:")
    for exp in training_exps:
        print(f"    - {exp}")
    eval_freq = base_config["eval_freq"]
    num_updates = base_config["num_updates"]
    print(f"Total checkpoints: {num_updates // eval_freq}")
    print(f"Experiments per checkpoint: {total_exps}")
    total_exp_runs = total_exps * (num_updates // eval_freq) * len(agents)
    print(f"Total experiment runs: {total_exp_runs} ({total_exps} x {num_updates // eval_freq} checkpoints x {len(agents)} agents)")
    if args.resume:
        print(f"Resume from: {args.resume}")
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

        train_with_experiments(
            config, checkpoint_exps, training_exps,
            resume_dir=args.resume,
        )


if __name__ == "__main__":
    main()
