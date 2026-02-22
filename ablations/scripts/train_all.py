#!/usr/bin/env python3
"""
Train all 25 agent configurations.

Usage:
    # Train all 25 configurations
    python -m ablations.scripts.train_all --seeds 0 1 2

    # Train specific training methods
    python -m ablations.scripts.train_all --methods accel paired --seeds 0

    # Train specific agents
    python -m ablations.scripts.train_all --agents accel_probe persistent_lstm --seeds 0

    # Dry run (show what would be trained)
    python -m ablations.scripts.train_all --dry_run
"""

import argparse
import subprocess
import sys
from itertools import product
from typing import List, Optional


# 5 training methods
ALL_METHODS = ["accel", "plr", "robust_plr", "dr", "paired"]

# 5 base agents (for non-PAIRED methods)
BASE_AGENTS = [
    "accel_probe",
    "persistent_lstm",
    "context_vector",
    "episodic_memory",
    "next_env_prediction",
]

# 5 PAIRED agents (for PAIRED method)
PAIRED_AGENTS = [f"paired_{agent}" for agent in BASE_AGENTS]


def get_configurations(
    methods: Optional[List[str]] = None,
    agents: Optional[List[str]] = None,
) -> List[tuple]:
    """
    Get list of (method, agent) configurations to train.

    Returns list of (training_method, agent_type) tuples.
    """
    methods = methods or ALL_METHODS
    configurations = []

    for method in methods:
        if method == "paired":
            # PAIRED uses paired_* agents
            method_agents = PAIRED_AGENTS
        else:
            # Other methods use base agents
            method_agents = BASE_AGENTS

        # Filter by specified agents if provided
        if agents:
            # Handle both base and paired_* agent names
            method_agents = [
                a for a in method_agents
                if a in agents or a.replace("paired_", "") in agents
            ]

        for agent in method_agents:
            configurations.append((method, agent))

    return configurations


def train_configuration(
    method: str,
    agent: str,
    seed: int,
    extra_args: List[str] = None,
) -> int:
    """Train a single configuration. Returns exit code."""
    cmd = [
        sys.executable, "-m", "ablations.scripts.train",
        "--training_method", method,
        "--agent_type", agent,
        "--seed", str(seed),
    ]

    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Training: {method} / {agent} / seed={seed}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Train all 25 agent configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train all 25 configurations with 3 seeds
    python -m ablations.scripts.train_all --seeds 0 1 2

    # Train only ACCEL and PAIRED methods
    python -m ablations.scripts.train_all --methods accel paired --seeds 0

    # Train only specific agents across all methods
    python -m ablations.scripts.train_all --agents persistent_lstm context_vector --seeds 0
        """,
    )

    parser.add_argument(
        "--methods", nargs="+", default=None,
        choices=ALL_METHODS,
        help="Training methods to run (default: all 5)",
    )
    parser.add_argument(
        "--agents", nargs="+", default=None,
        help="Agent types to run (default: all 5 per method)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0],
        help="Random seeds to run (default: [0])",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print configurations without training",
    )
    parser.add_argument(
        "--continue_on_error", action="store_true",
        help="Continue training other configs if one fails",
    )

    args, extra_args = parser.parse_known_args()

    # Get configurations
    configurations = get_configurations(args.methods, args.agents)

    # Calculate total runs
    total_runs = len(configurations) * len(args.seeds)

    print("=" * 60)
    print("Curriculum Awareness Ablation Study: Training Suite")
    print("=" * 60)
    print(f"Configurations: {len(configurations)}")
    print(f"Seeds: {args.seeds}")
    print(f"Total training runs: {total_runs}")
    print("=" * 60)

    # Print configurations
    print("\nConfigurations to train:")
    for method, agent in configurations:
        print(f"  - {method} / {agent}")

    if args.dry_run:
        print("\n[DRY RUN] No training performed.")
        return

    # Train each configuration
    completed = 0
    failed = []

    for seed in args.seeds:
        for method, agent in configurations:
            run_id = f"{method}/{agent}/seed={seed}"
            try:
                exit_code = train_configuration(method, agent, seed, extra_args)
                if exit_code != 0:
                    failed.append((run_id, f"exit code {exit_code}"))
                    if not args.continue_on_error:
                        print(f"\nTraining failed for {run_id}. Stopping.")
                        break
                else:
                    completed += 1
            except Exception as e:
                failed.append((run_id, str(e)))
                if not args.continue_on_error:
                    print(f"\nError training {run_id}: {e}. Stopping.")
                    break

        if failed and not args.continue_on_error:
            break

    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Completed: {completed}/{total_runs}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed runs:")
        for run_id, error in failed:
            print(f"  - {run_id}: {error}")


if __name__ == "__main__":
    main()