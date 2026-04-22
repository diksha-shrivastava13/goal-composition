#!/usr/bin/env python3
"""
Quick test of all PAIRED experiments with reduced config for speed.

Skips adversary_ablation (16 rollouts = too slow for quick testing).
Uses n_levels=30, max_steps=50 to minimize XLA compile + rollout time.
"""

import sys
import os
import json
import time
import traceback

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax

from ablations.configs.experiment_defaults import (
    UNIVERSAL_EXPERIMENTS,
    PAIRED_EXPERIMENTS,
)

SKIP_EXPERIMENTS = {"adversary_ablation"}


def run_single_experiment(exp_name, checkpoint_path, agent_type, config_overrides, seed=0):
    """Run a single experiment with config overrides injected."""
    from pathlib import Path
    from ablations.experiments.run_experiment import (
        get_experiment_class, load_agent, load_checkpoint, _wrap_paired_train_state
    )
    from ablations.common.types import PAIREDTrainState

    output_path = Path(f"/tmp/paired_test/{exp_name}")
    output_path.mkdir(parents=True, exist_ok=True)

    ExperimentClass = get_experiment_class(exp_name)

    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    config['training_method'] = 'paired'
    # Inject speed overrides
    config.update(config_overrides)

    agent = load_agent(agent_type, config)
    train_state, checkpoint_config = load_checkpoint(checkpoint_path, agent=agent, seed=seed)
    if checkpoint_config:
        config.update(checkpoint_config)
        # Re-apply overrides (checkpoint_config might clobber them)
        config.update(config_overrides)

    if isinstance(train_state, PAIREDTrainState):
        train_state = _wrap_paired_train_state(train_state)

    experiment = ExperimentClass(
        agent=agent,
        train_state=train_state,
        config=config,
        output_dir=str(output_path),
        training_method='paired',
    )

    rng = jax.random.PRNGKey(seed)

    experiment.data = experiment.collect_data(rng)
    results = experiment.analyze()
    experiment.visualize()

    # Save results
    results_path = output_path / f"{exp_name}_results.json"

    def convert(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return None

    with open(results_path, 'w') as f:
        json.dump(convert(results), f, indent=2)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(os.getcwd(), "checkpoints/paired/paired_accel_probe/0"))
    parser.add_argument("--agent_type", default="paired_accel_probe")
    parser.add_argument("--skip", nargs="*", default=list(SKIP_EXPERIMENTS))
    parser.add_argument("--only_paired", action="store_true")
    parser.add_argument("--only_universal", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Run only these experiments")
    parser.add_argument("--n_levels", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    skip = set(args.skip)

    if args.only:
        experiments = args.only
    elif args.only_paired:
        experiments = [e for e in PAIRED_EXPERIMENTS if e not in skip]
    elif args.only_universal:
        experiments = [e for e in UNIVERSAL_EXPERIMENTS if e not in skip]
    else:
        experiments = [e for e in UNIVERSAL_EXPERIMENTS + PAIRED_EXPERIMENTS if e not in skip]

    # Config overrides for speed
    config_overrides = {
        "n_levels": args.n_levels,
        "n_episodes": args.n_levels,
        "n_samples": args.n_levels,
        "n_levels_per_condition": max(10, args.n_levels // 3),
        "n_episodes_per_condition": max(10, args.n_levels // 3),
        "n_episodes_per_intervention": max(10, args.n_levels // 3),
        "max_steps": args.max_steps,
        "max_episode_length": args.max_steps,
        "max_steps_per_episode": args.max_steps,
    }

    print(f"Testing {len(experiments)} experiments (skipping: {skip})")
    print(f"Config overrides: n_levels={args.n_levels}, max_steps={args.max_steps}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Agent: {args.agent_type}")
    print("=" * 60)

    results = {}
    passed = 0
    failed = 0

    for i, exp_name in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {exp_name}...", flush=True)
        t0 = time.time()

        try:
            result = run_single_experiment(
                exp_name, args.checkpoint, args.agent_type,
                config_overrides, args.seed,
            )
            dt = time.time() - t0
            passed += 1
            results[exp_name] = {"status": "success", "time": dt}
            print(f"  PASS ({dt:.1f}s)")

        except Exception as e:
            dt = time.time() - t0
            failed += 1
            err_msg = str(e)
            results[exp_name] = {"status": "error", "error": err_msg, "time": dt}
            print(f"  FAIL ({dt:.1f}s)")
            print(f"    {err_msg[:300]}")
            traceback.print_exc()

    # Summary
    total = passed + failed
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} PASSED, {failed}/{total} FAILED")
    print(f"{'='*60}")

    if failed > 0:
        print("\nFailed experiments:")
        for name, r in results.items():
            if r["status"] == "error":
                print(f"  {name}: {r['error'][:300]}")

    with open("/tmp/paired_test/summary.json", "w") as f:
        json.dump(results, f, indent=2)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
