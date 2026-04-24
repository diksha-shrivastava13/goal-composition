#!/usr/bin/env python3
"""
Smoke test for all 25 agent configurations.

Runs each configuration with minimal settings to verify everything works:
- 12 training updates (4 eval checkpoints with eval_freq=3)
- ALL applicable experiments run at each eval checkpoint
- No wandb logging

Usage:
    # Run all 25 configurations
    python -m ablations.scripts.smoke_test

    # Run specific method
    python -m ablations.scripts.smoke_test --methods accel

    # Run one agent across all methods
    python -m ablations.scripts.smoke_test --agents persistent_lstm

    # Run a single configuration
    python -m ablations.scripts.smoke_test --methods dr --agents accel_probe
"""

import argparse
import json
import os
import subprocess
import sys
import time
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
    """Get list of (method, agent) configurations to test."""
    methods = methods or ALL_METHODS
    configurations = []

    for method in methods:
        if method == "paired":
            method_agents = PAIRED_AGENTS
        else:
            method_agents = BASE_AGENTS

        if agents:
            method_agents = [
                a for a in method_agents
                if a in agents or a.replace("paired_", "") in agents
            ]

        for agent in method_agents:
            configurations.append((method, agent))

    return configurations


def get_experiment_results(method: str, agent: str, seed: int = 0) -> dict:
    """Parse experiment results from training summary JSON."""
    summary_path = os.path.join(
        "experiments", method, agent, str(seed), "training_summary.json"
    )
    if not os.path.exists(summary_path):
        return {}

    try:
        with open(summary_path) as f:
            data = json.load(f)
        ce = data.get("checkpoint_experiments", {})
        # Use the last checkpoint step for summary
        if not ce:
            return {}
        last_step = max(ce.keys(), key=int)
        exps = ce[last_step]
        ok = sum(1 for e in exps.values() if isinstance(e, dict) and e.get("status") == "success")
        err = sum(1 for e in exps.values() if isinstance(e, dict) and e.get("status") != "success")
        failed = {
            name: e.get("error", "unknown")[:100]
            for name, e in exps.items()
            if isinstance(e, dict) and e.get("status") != "success"
        }
        return {"ok": ok, "err": err, "failed": failed, "total": ok + err}
    except Exception:
        return {}


def run_smoke_test(method: str, agent: str) -> tuple:
    """
    Run smoke test for a single configuration.

    Returns (success: bool, duration: float, error_msg: str or None)
    """
    cmd = [
        sys.executable, "-m", "ablations.scripts.train_with_experiments",
        "--training_method", method,
        "--agent_type", agent,
        "--seed", "0",
        "--num_updates", "10",
        "--eval_freq", "3",
        "--num_train_envs", "4",
        "--num_steps", "16",
        "--no_wandb",
        # Reduce experiment sizes to avoid OOM on small instances
        "--n_samples", "20",
        "--n_levels", "20",
        "--n_episodes", "20",
        "--max_steps", "32",
        "--n_samples_per_step", "10",
        "--trajectory_length", "5",
        "--max_steps_per_episode", "16",
        "--n_levels_per_type", "10",
        "--n_probe_levels", "10",
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per config
        )
        duration = time.time() - start

        if result.returncode == 0:
            return True, duration, None
        else:
            error = result.stderr[-500:] if result.stderr else f"exit code {result.returncode}"
            return False, duration, error
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return False, duration, "TIMEOUT (>10 min)"
    except Exception as e:
        duration = time.time() - start
        return False, duration, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test all 25 agent configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Each configuration runs with minimal settings:
  - 12 training updates, eval_freq=3 (4 eval checkpoints)
  - 4 parallel envs, 16-step rollouts
  - ALL applicable experiments at each checkpoint
  - No wandb logging

Expected: 25/25 pass
  Non-PAIRED: 13 checkpoint exps + 3 training-time = 16 per checkpoint
  PAIRED: 35 checkpoint exps + 3 training-time = 38 per checkpoint
        """,
    )
    parser.add_argument("--methods", nargs="+", default=None, choices=ALL_METHODS,
                        help="Training methods to test (default: all 5)")
    parser.add_argument("--agents", nargs="+", default=None,
                        help="Agent types to test (default: all 5 per method)")
    parser.add_argument("--continue_on_error", action="store_true", default=True,
                        help="Continue testing other configs if one fails (default: True)")
    parser.add_argument("--stop_on_error", action="store_true",
                        help="Stop on first failure")

    args = parser.parse_args()
    if args.stop_on_error:
        args.continue_on_error = False

    configurations = get_configurations(args.methods, args.agents)

    print("=" * 60)
    print("Smoke Test: All Agent Configurations")
    print("=" * 60)
    print(f"Configurations to test: {len(configurations)}")
    print(f"Settings: num_updates=10, eval_freq=3, num_train_envs=4, num_steps=16")
    print(f"All applicable experiments run at each of 3 eval checkpoints")
    print("=" * 60)

    for method, agent in configurations:
        print(f"  {method:12s} / {agent}")
    print()

    # Run tests
    results = []
    passed = 0
    failed = 0

    for i, (method, agent) in enumerate(configurations, 1):
        config_name = f"{method}/{agent}"
        print(f"[{i}/{len(configurations)}] Testing {config_name}...", end=" ", flush=True)

        success, duration, error = run_smoke_test(method, agent)
        results.append((config_name, success, duration, error))

        exp_results = get_experiment_results(method, agent)

        if success:
            passed += 1
            if exp_results:
                exp_info = f" [exps: {exp_results['ok']}/{exp_results['total']} ok"
                if exp_results['err'] > 0:
                    exp_info += f", {exp_results['err']} err"
                exp_info += "]"
            else:
                exp_info = ""
            print(f"PASS ({duration:.1f}s){exp_info}")
            if exp_results.get('failed'):
                for exp_name, err_msg in exp_results['failed'].items():
                    print(f"    exp err: {exp_name}: {err_msg}")
        else:
            failed += 1
            print(f"FAIL ({duration:.1f}s)")
            if error:
                # Print last few lines of error
                for line in error.strip().split("\n")[-3:]:
                    print(f"    {line}")

            if not args.continue_on_error:
                print("\nStopping on first failure.")
                break

    # Summary
    total = passed + failed
    print(f"\n{'='*60}")
    print(f"Smoke Test Results: {passed}/{total} PASSED")
    print(f"{'='*60}")

    if failed > 0:
        print(f"\nFailed configurations:")
        for name, success, duration, error in results:
            if not success:
                print(f"  FAIL: {name}")
                if error:
                    print(f"        {error[:200]}")

    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
