"""
PAIRED Experiment Suite Runner.

Runs all PAIRED-specific experiments in the recommended order,
respecting dependencies between experiments.

Experiment Categories:
    A. Utility Function Extraction (A1-A3)
    B. Causal Interventions (B1-B5)
    C. Three-Agent Dynamics (C1-C4)
    D. Belief Revision Tracking (D1-D3)
    E. Mechanistic Interpretability Elevations (E1-E7) - via existing experiments
    F. Theoretical Validation (F1-F5)

Dependencies:
    C3 (adversary strategy clustering)
     └──→ E1, E2, E5 (use clusters as conditioning)
     └──→ F3 (shard dynamics uses adversary strategies)

    A1 (utility extraction + causal validation)
     └──→ F1 (causal model extraction validates A1's causal claims)
     └──→ F5 (opacity uses A1's Û as student model)

    A2 (adversary policy extraction)
     └──→ F5 (adversary encoding needs A2)

    B3 (teaching signal intervention)
     └──→ D2 (belief revision detection can use B3 as controlled triggers)
     └──→ F4 (belief-behaviour divergence on intervention levels)

    D3 (shard-aware goal tracking)
     └──→ F2 (multi-scale goals use D3's shard identification)
     └──→ F3 (shard competition uses D3's shard clusters)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import jax

from .run_experiment import run_experiment, get_experiment_class


# PAIRED experiment execution order (respects dependencies)
PAIRED_EXPERIMENT_ORDER = [
    # Phase 1: Independent experiments (no dependencies)
    {
        'phase': 1,
        'name': 'Independent Experiments',
        'experiments': [
            'adversary_strategy_clustering',  # C3 - needed by many
            'utility_extraction',  # A1 - core contribution
            'adversary_policy_extraction',  # A2
            'adversary_dynamics',  # Legacy PAIRED experiment (moved from main)
            'regret_transfer',  # Legacy PAIRED experiment (moved from main)
        ],
    },
    # Phase 2: Depends on Phase 1
    {
        'phase': 2,
        'name': 'Utility & Strategy Analysis',
        'experiments': [
            'bilateral_utility',  # A3 - needs A1, A2
            'adversary_ablation',  # B1
            'regret_decomposition',  # B2
        ],
    },
    # Phase 3: Causal Interventions
    {
        'phase': 3,
        'name': 'Causal Interventions',
        'experiments': [
            'teaching_signal_intervention',  # B3
            'counterfactual_curriculum',  # B4
            'activation_patching',  # B5
        ],
    },
    # Phase 4: Three-Agent Dynamics (except C3 which was in Phase 1)
    {
        'phase': 4,
        'name': 'Three-Agent Dynamics',
        'experiments': [
            'representation_divergence',  # C1
            'antagonist_audit',  # C2
            'coalition_dynamics',  # C4
        ],
    },
    # Phase 5: Belief Revision
    {
        'phase': 5,
        'name': 'Belief Revision',
        'experiments': [
            'representation_trajectory',  # D1
            'belief_revision_detection',  # D2 - can use B3 results
            'goal_evolution',  # D3
        ],
    },
    # Phase 6: Elevated Mechanistic Interpretability (E1-E7)
    # These are the existing experiments modified for PAIRED
    {
        'phase': 6,
        'name': 'Mechanistic Interpretability',
        'experiments': [
            'level_probing',  # E1 - bilateral probing
            'activation_analysis',  # E2 - regret-conditioned
            'value_calibration',  # E3 - regret-aware
            'cross_episode_flow',  # E4 - adversary pattern retention
            'symbolic_regression',  # E5 - regret structure
            'counterfactual',  # E6 - bilateral injection
            'output_probing',  # E7 - adversary prediction
        ],
    },
    # Phase 7: Theoretical Validation (depends on D3, A1, A2)
    {
        'phase': 7,
        'name': 'Theoretical Validation',
        'experiments': [
            'causal_model_extraction',  # F1 - validates A1
            'multiscale_goals',  # F2 - uses D3
            'shard_dynamics',  # F3 - uses D3, C3
            'belief_behaviour_divergence',  # F4 - uses B3
            'teaching_opacity',  # F5 - uses A1, A2
        ],
    },
]


def get_paired_experiments(phase: Optional[int] = None) -> List[str]:
    """Get list of PAIRED experiments, optionally filtered by phase."""
    experiments = []
    for phase_info in PAIRED_EXPERIMENT_ORDER:
        if phase is None or phase_info['phase'] == phase:
            experiments.extend(phase_info['experiments'])
    return experiments


def run_paired_suite(
    checkpoint_path: str,
    agent_type: str,
    output_dir: str,
    seed: int = 0,
    phases: Optional[List[int]] = None,
    experiments: Optional[List[str]] = None,
    skip_existing: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the PAIRED experiment suite.

    Args:
        checkpoint_path: Path to the PAIRED checkpoint
        agent_type: Type of agent (must be PAIRED-compatible)
        output_dir: Directory to save results
        seed: Random seed
        phases: Optional list of phases to run (default: all)
        experiments: Optional list of specific experiments to run
        skip_existing: Skip experiments with existing results
        verbose: Print progress

    Returns:
        Dict with suite results summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine experiments to run
    if experiments:
        experiment_list = experiments
    elif phases:
        experiment_list = []
        for phase_info in PAIRED_EXPERIMENT_ORDER:
            if phase_info['phase'] in phases:
                experiment_list.extend(phase_info['experiments'])
    else:
        experiment_list = get_paired_experiments()

    # Suite metadata
    suite_results = {
        'suite': 'paired',
        'checkpoint': checkpoint_path,
        'agent_type': agent_type,
        'seed': seed,
        'start_time': datetime.now().isoformat(),
        'experiments': {},
        'summary': {
            'total': len(experiment_list),
            'completed': 0,
            'failed': 0,
            'skipped': 0,
        },
    }

    if verbose:
        print(f"PAIRED Experiment Suite")
        print(f"=" * 50)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Agent: {agent_type}")
        print(f"Experiments: {len(experiment_list)}")
        print(f"=" * 50)

    for exp_name in experiment_list:
        if verbose:
            print(f"\n[{suite_results['summary']['completed'] + 1}/{len(experiment_list)}] {exp_name}")

        # Check if result exists
        result_path = output_path / f"{exp_name}_results.json"
        if skip_existing and result_path.exists():
            if verbose:
                print(f"  Skipping (results exist)")
            suite_results['experiments'][exp_name] = {'status': 'skipped'}
            suite_results['summary']['skipped'] += 1
            continue

        try:
            result = run_experiment(
                experiment_name=exp_name,
                checkpoint_path=checkpoint_path,
                agent_type=agent_type,
                output_dir=str(output_path),
                seed=seed,
                training_method='paired',
            )

            suite_results['experiments'][exp_name] = {
                'status': 'completed',
                'result_path': str(result_path),
            }
            suite_results['summary']['completed'] += 1

        except Exception as e:
            if verbose:
                print(f"  FAILED: {e}")
            suite_results['experiments'][exp_name] = {
                'status': 'failed',
                'error': str(e),
            }
            suite_results['summary']['failed'] += 1

    # Save suite summary
    suite_results['end_time'] = datetime.now().isoformat()
    summary_path = output_path / 'paired_suite_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(suite_results, f, indent=2)

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Suite Complete")
        print(f"  Completed: {suite_results['summary']['completed']}")
        print(f"  Failed: {suite_results['summary']['failed']}")
        print(f"  Skipped: {suite_results['summary']['skipped']}")
        print(f"  Summary: {summary_path}")

    return suite_results


def main():
    parser = argparse.ArgumentParser(description="Run PAIRED experiment suite")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to PAIRED checkpoint')
    parser.add_argument('--agent_type', type=str, required=True,
                        help='Agent type (e.g., paired_persistent_lstm)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--phases', type=int, nargs='+',
                        help='Phases to run (1-7)')
    parser.add_argument('--experiments', type=str, nargs='+',
                        help='Specific experiments to run')
    parser.add_argument('--no-skip', action='store_true',
                        help='Rerun experiments even if results exist')
    parser.add_argument('--list', action='store_true',
                        help='List all experiments and exit')

    args = parser.parse_args()

    if args.list:
        print("PAIRED Experiment Suite")
        print("=" * 50)
        for phase_info in PAIRED_EXPERIMENT_ORDER:
            print(f"\nPhase {phase_info['phase']}: {phase_info['name']}")
            for exp in phase_info['experiments']:
                print(f"  - {exp}")
        return

    run_paired_suite(
        checkpoint_path=args.checkpoint,
        agent_type=args.agent_type,
        output_dir=args.output_dir,
        seed=args.seed,
        phases=args.phases,
        experiments=args.experiments,
        skip_existing=not args.no_skip,
    )


if __name__ == '__main__':
    main()
