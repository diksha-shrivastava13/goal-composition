"""
Run all experiments on multiple agents and checkpoints.

This script runs the full experiment suite on saved checkpoints:
- Non-PAIRED methods: 13 universal post-hoc experiments per checkpoint
- PAIRED method: 13 universal + 22 PAIRED-specific = 35 post-hoc experiments per checkpoint

Usage:
    # Run universal experiments on ACCEL checkpoints
    python -m ablations.experiments.run_all --results_dir checkpoints/accel --output_dir results --training_method accel

    # Run all 35 experiments on PAIRED checkpoints
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

from ..configs import (
    UNIVERSAL_EXPERIMENTS,
    PAIRED_EXPERIMENTS,
    TRAINING_TIME_EXPERIMENTS,
    BASE_AGENTS,
    PAIRED_AGENTS,
    get_experiments_for_method,
)


def find_checkpoints(results_dir: str, agent_type: str, seed: int = 0) -> List[str]:
    """Find all checkpoints for an agent.

    The checkpoint structure from training is:
        {results_dir}/{agent_type}/{seed}/models/{step}/

    Each checkpoint is identified by its parent dir:
        {results_dir}/{agent_type}/{seed}/
    plus a step number from Orbax.

    Returns:
        List of (checkpoint_dir, step) tuples sorted by step,
        where checkpoint_dir contains config.json and models/.
    """
    checkpoint_dir = os.path.join(results_dir, agent_type, str(seed))
    models_dir = os.path.join(checkpoint_dir, "models")

    if not os.path.isdir(models_dir):
        return []

    # Orbax saves steps as numbered subdirectories under models/
    steps = []
    for entry in os.listdir(models_dir):
        entry_path = os.path.join(models_dir, entry)
        if os.path.isdir(entry_path):
            try:
                steps.append(int(entry))
            except ValueError:
                pass

    return sorted(steps)


def run_single_experiment(
    experiment_name: str,
    checkpoint_path: str,
    agent_type: str,
    output_dir: str,
    seed: int = 0,
    training_method: str = "accel",
    config_overrides: Optional[Dict[str, Any]] = None,
    step: int = -1,
) -> Dict[str, Any]:
    """Run a single experiment on a specific checkpoint step.

    Args:
        step: Orbax checkpoint step to load (-1 for latest).
    """
    from .run_experiment import run_experiment

    try:
        result = run_experiment(
            experiment_name=experiment_name,
            checkpoint_path=checkpoint_path,
            agent_type=agent_type,
            output_dir=output_dir,
            seed=seed,
            training_method=training_method,
            config_overrides=config_overrides,
            step=step,
        )
        return {'status': 'success', 'result': result}
    except Exception as e:
        import traceback
        print(f"  ERROR in {experiment_name}: {e}")
        traceback.print_exc()
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
    config_overrides: Optional[Dict[str, Any]] = None,
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
        config_overrides: Runtime config overrides for experiment params (n_levels, max_steps, etc.)

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
        steps = find_checkpoints(results_dir, agent_type, seed=seed)

        if checkpoints_per_agent is not None:
            # Sample evenly across training
            n_steps = len(steps)
            if n_steps > checkpoints_per_agent:
                indices = [int(i * n_steps / checkpoints_per_agent)
                          for i in range(checkpoints_per_agent)]
                steps = [steps[i] for i in indices]

        # checkpoint_dir is the parent dir with config.json and models/
        checkpoint_dir = os.path.join(results_dir, agent_type, str(seed))

        for step in steps:
            for experiment_name in experiments:
                exp_output_dir = output_path / agent_type / f"step_{step}" / experiment_name

                tasks.append({
                    'experiment_name': experiment_name,
                    'checkpoint_path': checkpoint_dir,
                    'agent_type': agent_type,
                    'output_dir': str(exp_output_dir),
                    'seed': seed,
                    'training_method': training_method,
                    'config_overrides': config_overrides,
                    'step': step,
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
        import gc
        for i, task in enumerate(tasks):
            print(f"\n[{i+1}/{len(tasks)}] {task['experiment_name']} on {task['agent_type']}")
            result = run_single_experiment(**task)

            if result['status'] == 'success':
                results.append(result)
            else:
                errors.append({'task': task, 'error': result['error']})
                print(f"  ERROR: {result['error']}")

            # Free memory between experiments to prevent cumulative OOM
            gc.collect()
            try:
                jax.clear_caches()
            except AttributeError:
                pass  # older JAX versions
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(run_single_experiment, **task): task
                for task in tasks
            }

            for i, future in enumerate(as_completed(futures)):
                task = futures[future]
                print(f"\n[{i+1}/{len(tasks)}] Completed: {task['experiment_name']} on {task['agent_type']}")

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


def compute_cross_experiment_correlations(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Compute cross-experiment correlations from collected results.

    Key correlation (PDF Exp 29): R² between probe accuracy (from level_probing)
    and adaptation speed (from mutation_adaptation) across agents.

    Args:
        all_results: Nested dict of agent_type → step → experiment_name → results

    Returns:
        Dict with cross-experiment correlation metrics
    """
    import numpy as np

    correlations = {}

    # --- Exp 29 × Exp 1: Probe accuracy ↔ adaptation speed ---
    # For each agent at each step, extract:
    #   - level_probing: mean probe R² across features
    #   - mutation_adaptation: mean adaptation speed
    probe_accuracies = []
    adaptation_speeds = []
    agent_labels = []

    for agent_type, steps in all_results.items():
        for step, experiments in steps.items():
            lp = experiments.get('level_probing', {})
            ma = experiments.get('mutation_adaptation', {})

            # Extract probe R² from level_probing results
            lp_results = lp.get('results', {})
            probe_r2s = []
            # level_probing stores per-feature R² in probe_results
            probe_results = lp_results.get('probe_results', {})
            for feat_name, feat_data in probe_results.items():
                if isinstance(feat_data, dict):
                    r2 = feat_data.get('r2', feat_data.get('mean_score'))
                    if r2 is not None:
                        probe_r2s.append(float(r2))

            # Fallback: try summary metrics
            if not probe_r2s:
                for key, val in lp_results.items():
                    if 'r2' in str(key).lower() and isinstance(val, (int, float)):
                        probe_r2s.append(float(val))

            # Extract adaptation speed from mutation_adaptation results
            ma_results = ma.get('results', {})
            speed_data = ma_results.get('adaptation_speed', {})
            mean_speed = speed_data.get('overall_mean_speed')

            if probe_r2s and mean_speed is not None:
                probe_accuracies.append(float(np.mean(probe_r2s)))
                adaptation_speeds.append(float(mean_speed))
                agent_labels.append(f"{agent_type}@{step}")

    if len(probe_accuracies) >= 3:
        probe_arr = np.array(probe_accuracies)
        speed_arr = np.array(adaptation_speeds)

        # Pearson correlation
        corr = float(np.corrcoef(probe_arr, speed_arr)[0, 1])

        # R² via linear regression
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        model = LinearRegression()
        model.fit(probe_arr.reshape(-1, 1), speed_arr)
        r2 = float(r2_score(speed_arr, model.predict(probe_arr.reshape(-1, 1))))

        correlations['probe_accuracy_vs_adaptation_speed'] = {
            'pearson_r': corr,
            'r2': r2,
            'slope': float(model.coef_[0]),
            'intercept': float(model.intercept_),
            'n_datapoints': len(probe_accuracies),
            'agent_labels': agent_labels,
            'probe_accuracies': probe_accuracies,
            'adaptation_speeds': adaptation_speeds,
            'interpretation': (
                'Positive correlation supports Exp 29 hypothesis: '
                'agents with stronger training dynamics representations '
                'adapt faster to mutations.'
            ),
        }
    else:
        correlations['probe_accuracy_vs_adaptation_speed'] = {
            'error': f'Insufficient data: need >=3 agent×step pairs with both '
                     f'level_probing and mutation_adaptation results, found {len(probe_accuracies)}',
        }

    # --- Exp 5 × Exp 29: Horizon decay rate ↔ adaptation speed ---
    decay_rates = []
    speeds_for_decay = []
    labels_for_decay = []

    for agent_type, steps in all_results.items():
        for step, experiments in steps.items():
            nep = experiments.get('n_env_prediction', {})
            ma = experiments.get('mutation_adaptation', {})

            nep_results = nep.get('results', {})
            decay_params = nep_results.get('decay_params', {})

            # Get mean decay rate across features
            feat_decay_rates = []
            for feat, params in decay_params.items():
                if isinstance(params, dict) and params.get('fit_success'):
                    feat_decay_rates.append(params['b'])

            ma_results = ma.get('results', {})
            speed_data = ma_results.get('adaptation_speed', {})
            mean_speed = speed_data.get('overall_mean_speed')

            if feat_decay_rates and mean_speed is not None:
                decay_rates.append(float(np.mean(feat_decay_rates)))
                speeds_for_decay.append(float(mean_speed))
                labels_for_decay.append(f"{agent_type}@{step}")

    if len(decay_rates) >= 3:
        decay_arr = np.array(decay_rates)
        speed_arr = np.array(speeds_for_decay)
        corr = float(np.corrcoef(decay_arr, speed_arr)[0, 1])

        correlations['horizon_decay_vs_adaptation_speed'] = {
            'pearson_r': corr,
            'n_datapoints': len(decay_rates),
            'interpretation': (
                'Negative correlation (slower decay = faster adaptation) supports '
                'the hypothesis that longer-horizon curriculum models aid generalization.'
            ),
        }

    return correlations


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

    # Cross-experiment correlations
    cross_experiment = compute_cross_experiment_correlations(all_results)
    if cross_experiment:
        summary['cross_experiment'] = cross_experiment

    # Save summary
    summary_path = output_path / 'analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    from ..configs.cli import (
        add_common_args,
        add_experiment_param_args,
        add_posthoc_args,
        _EXPERIMENT_PARAM_CLI_KEYS,
    )

    parser = argparse.ArgumentParser(description="Run all experiments on checkpoints")

    # Shared args
    add_common_args(parser)
    add_experiment_param_args(parser)
    add_posthoc_args(parser)

    # Entry-point-specific
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='Experiments to run (default: method-appropriate set)')
    parser.add_argument('--summarize', action='store_true',
                        help='Only generate summary report from existing results')

    args = parser.parse_args()

    # Route warnings through logging for clean log ordering
    import logging
    logging.captureWarnings(True)

    if args.summarize:
        output_dir = args.output_dir or "."
        generate_summary_report(output_dir)
    else:
        # Build config overrides from experiment params
        # CLI uses exp_adv_num_steps but experiments expect adv_num_steps
        _cli_to_config = {"exp_adv_num_steps": "adv_num_steps"}
        config_overrides = {}
        for cli_key in _EXPERIMENT_PARAM_CLI_KEYS:
            val = getattr(args, cli_key, None)
            if val is not None:
                config_key = _cli_to_config.get(cli_key, cli_key)
                config_overrides[config_key] = val

        run_all_experiments(
            results_dir=args.results_dir,
            output_dir=args.output_dir or "results",
            training_method=args.training_method,
            agents=args.agents,
            experiments=args.experiments,
            checkpoints_per_agent=args.checkpoints_per_agent,
            parallel=args.parallel,
            seed=args.seed,
            config_overrides=config_overrides or None,
        )


if __name__ == '__main__':
    main()
