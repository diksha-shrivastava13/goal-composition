"""
Run all experiments on multiple agents and checkpoints.

This script runs the full experiment suite on saved checkpoints:
- Non-PAIRED methods: 11 universal post-hoc experiments per checkpoint
- PAIRED method: 11 universal + 22 PAIRED-specific = 33 post-hoc experiments per checkpoint

Usage:
    # Run universal experiments on ACCEL checkpoints
    python -m ablations.experiments.run_all --results_dir checkpoints/accel --output_dir results --training_method accel

    # Run all 33 experiments on PAIRED checkpoints
    python -m ablations.experiments.run_all --results_dir checkpoints/paired --output_dir results --training_method paired

    # Run specific experiments only
    python -m ablations.experiments.run_all --results_dir checkpoints/accel --output_dir results --experiments level_probing value_calibration
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob

import jax


# Universal post-hoc experiments (run on any checkpoint, any method)
UNIVERSAL_EXPERIMENTS = [
    # Core experiments
    'level_probing',
    'value_calibration',
    'activation_analysis',
    'cross_agent_comparison',
    # Transfer & robustness
    'mutation_adaptation',
    'causal_intervention',
    'counterfactual',
    # Advanced interpretability
    'output_probing',
    # Novel experiments
    'goal_extraction',
    'cross_episode_flow',
    'dr_coverage',
    # Prediction experiments
    'n_env_prediction',
    'n_step_prediction',
]

# PAIRED-specific experiments (22 experiments, only for paired training method)
PAIRED_EXPERIMENTS = [
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
    # Moved from main (PAIRED-only)
    'adversary_dynamics',
    'regret_transfer',
]

# Training-time experiments (require hooks, cannot run post-hoc on checkpoints)
TRAINING_TIME_EXPERIMENTS = [
    'behavioral_coupling',
    'symbolic_regression',
    'phase_transition',
]

# Base agent types (for non-PAIRED methods)
BASE_AGENTS = [
    'next_env_prediction',
    'accel_probe',
    'persistent_lstm',
    'context_vector',
    'episodic_memory',
]

# PAIRED agent types
PAIRED_AGENTS = [f'paired_{a}' for a in BASE_AGENTS]


def get_experiments_for_method(training_method: str) -> List[str]:
    """Get the list of experiments applicable to a training method."""
    experiments = list(UNIVERSAL_EXPERIMENTS)
    if training_method == "paired":
        experiments.extend(PAIRED_EXPERIMENTS)
    return experiments


def find_checkpoints(results_dir: str, agent_type: str) -> List[str]:
    """Find all checkpoints for an agent."""
    pattern = os.path.join(results_dir, agent_type, 'seed_*', 'checkpoint_*')
    checkpoints = glob.glob(pattern)

    # Sort by step number
    def get_step(path):
        try:
            return int(path.split('checkpoint_')[-1])
        except ValueError:
            return 0

    return sorted(checkpoints, key=get_step)


def run_single_experiment(
    experiment_name: str,
    checkpoint_path: str,
    agent_type: str,
    output_dir: str,
    seed: int = 0,
    training_method: str = "accel",
) -> Dict[str, Any]:
    """Run a single experiment (wrapper for subprocess)."""
    from .run_experiment import run_experiment

    try:
        result = run_experiment(
            experiment_name=experiment_name,
            checkpoint_path=checkpoint_path,
            agent_type=agent_type,
            output_dir=output_dir,
            seed=seed,
            training_method=training_method,
        )
        return {'status': 'success', 'result': result}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def run_all_experiments(
    results_dir: str,
    output_dir: str,
    training_method: str = "accel",
    agents: Optional[List[str]] = None,
    experiments: Optional[List[str]] = None,
    checkpoints_per_agent: Optional[int] = None,
    parallel: int = 1,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Run all experiments on all agents and checkpoints.

    Args:
        results_dir: Directory containing trained agent checkpoints
        output_dir: Directory to save experiment results
        training_method: Training method (accel, plr, robust_plr, dr, paired)
        agents: List of agent types to run (default: method-appropriate agents)
        experiments: List of experiments to run (default: method-appropriate set)
        checkpoints_per_agent: Max checkpoints per agent (default: all)
        parallel: Number of parallel workers
        seed: Random seed

    Returns:
        Dict with run summary
    """
    if agents is None:
        agents = PAIRED_AGENTS if training_method == "paired" else BASE_AGENTS
    if experiments is None:
        experiments = get_experiments_for_method(training_method)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all tasks
    tasks = []

    for agent_type in agents:
        checkpoints = find_checkpoints(results_dir, agent_type)

        if checkpoints_per_agent is not None:
            # Sample evenly across training
            n_checkpoints = len(checkpoints)
            if n_checkpoints > checkpoints_per_agent:
                indices = [int(i * n_checkpoints / checkpoints_per_agent)
                          for i in range(checkpoints_per_agent)]
                checkpoints = [checkpoints[i] for i in indices]

        for checkpoint_path in checkpoints:
            # Extract step from checkpoint path
            step = checkpoint_path.split('checkpoint_')[-1]

            for experiment_name in experiments:
                exp_output_dir = output_path / agent_type / f"step_{step}" / experiment_name

                tasks.append({
                    'experiment': experiment_name,
                    'checkpoint': checkpoint_path,
                    'agent_type': agent_type,
                    'output_dir': str(exp_output_dir),
                    'seed': seed,
                    'training_method': training_method,
                })

    print(f"Running {len(tasks)} experiment tasks:")
    print(f"  Training method: {training_method}")
    print(f"  Agents: {agents}")
    print(f"  Experiments: {len(experiments)} ({len(UNIVERSAL_EXPERIMENTS)} universal"
          + (f" + {len(PAIRED_EXPERIMENTS)} PAIRED" if training_method == "paired" else "")
          + ")")
    print(f"  Parallel workers: {parallel}")

    # Run tasks
    results = []
    errors = []

    if parallel == 1:
        # Sequential execution
        for i, task in enumerate(tasks):
            print(f"\n[{i+1}/{len(tasks)}] {task['experiment']} on {task['agent_type']}")
            result = run_single_experiment(**task)

            if result['status'] == 'success':
                results.append(result)
            else:
                errors.append({'task': task, 'error': result['error']})
                print(f"  ERROR: {result['error']}")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(run_single_experiment, **task): task
                for task in tasks
            }

            for i, future in enumerate(as_completed(futures)):
                task = futures[future]
                print(f"\n[{i+1}/{len(tasks)}] Completed: {task['experiment']} on {task['agent_type']}")

                try:
                    result = future.result()
                    if result['status'] == 'success':
                        results.append(result)
                    else:
                        errors.append({'task': task, 'error': result['error']})
                        print(f"  ERROR: {result['error']}")
                except Exception as e:
                    errors.append({'task': task, 'error': str(e)})
                    print(f"  ERROR: {e}")

    # Save summary
    summary = {
        'n_tasks': len(tasks),
        'n_success': len(results),
        'n_errors': len(errors),
        'agents': agents,
        'experiments': experiments,
        'errors': errors,
    }

    summary_path = output_path / 'run_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Run complete!")
    print(f"  Successful: {len(results)}/{len(tasks)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Summary saved to: {summary_path}")

    return summary


def generate_summary_report(output_dir: str) -> Dict[str, Any]:
    """Generate summary report from experiment results."""
    output_path = Path(output_dir)

    # Collect all results
    all_results = {}

    for agent_dir in output_path.iterdir():
        if not agent_dir.is_dir() or agent_dir.name.startswith('.'):
            continue

        agent_type = agent_dir.name
        all_results[agent_type] = {}

        for step_dir in agent_dir.iterdir():
            if not step_dir.is_dir():
                continue

            step = step_dir.name.replace('step_', '')
            all_results[agent_type][step] = {}

            for exp_dir in step_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                experiment_name = exp_dir.name
                results_file = exp_dir / f"{experiment_name}_results.json"

                if results_file.exists():
                    with open(results_file) as f:
                        all_results[agent_type][step][experiment_name] = json.load(f)

    # Generate summary statistics
    summary = {
        'agents': list(all_results.keys()),
        'experiments_by_agent': {},
    }

    for agent_type, steps in all_results.items():
        experiments_run = set()
        for step, experiments in steps.items():
            experiments_run.update(experiments.keys())
        summary['experiments_by_agent'][agent_type] = list(experiments_run)

    # Save summary
    summary_path = output_path / 'analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run all experiments on checkpoints")
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing trained agent checkpoints')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for experiment results')
    parser.add_argument('--training_method', type=str, default='accel',
                        choices=['accel', 'plr', 'robust_plr', 'dr', 'paired'],
                        help='Training method (determines which experiments to run)')
    parser.add_argument('--agents', type=str, nargs='+', default=None,
                        help='Agent types to run (default: method-appropriate agents)')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='Experiments to run (default: method-appropriate set)')
    parser.add_argument('--checkpoints_per_agent', type=int, default=None,
                        help='Max checkpoints per agent (default: all)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--summarize', action='store_true',
                        help='Only generate summary report from existing results')

    args = parser.parse_args()

    if args.summarize:
        generate_summary_report(args.output_dir)
    else:
        run_all_experiments(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            training_method=args.training_method,
            agents=args.agents,
            experiments=args.experiments,
            checkpoints_per_agent=args.checkpoints_per_agent,
            parallel=args.parallel,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()
