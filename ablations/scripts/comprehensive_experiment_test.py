#!/usr/bin/env python3
"""
Comprehensive experiment test: verifies ALL experiments run without crashing
across ALL 25 agent configurations.

Two-phase design:
  Phase 1: Generate minimal checkpoints for all 25 configs (via subprocess)
  Phase 2: Run all applicable experiments in-process with aggressive speed overrides

Usage:
    # Full suite (all 25 configs, all experiments)
    python -m ablations.scripts.comprehensive_experiment_test

    # Single config (quick validation of the script itself)
    python -m ablations.scripts.comprehensive_experiment_test \
        --methods accel --agents accel_probe

    # Skip checkpoint generation (reuse existing)
    python -m ablations.scripts.comprehensive_experiment_test \
        --skip_checkpoint_generation --checkpoint_base test_checkpoints

    # Stop on first failure
    python -m ablations.scripts.comprehensive_experiment_test --stop_on_error
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax


# ============================================================================
# SPEED OVERRIDES — injected into config to make experiments fast
# ============================================================================

SPEED_OVERRIDES: Dict[str, Any] = {
    # Flat keys (caught by exp_config fallback)
    "n_levels": 20,
    "n_episodes": 20,
    "n_samples": 20,
    "max_steps": 32,
    # NOTE: Do NOT override adv_num_steps here — it's baked into the adversary
    # network's time embedding table shape at training time. Overriding causes
    # shape mismatch: checkpoint has (N+1, 10) but override creates (M+1, 10).
    # value_calibration: timesteps must be < max_steps
    "calibration_timesteps": [1, 5, 10, 20],
    # Per-experiment param overrides
    "n_levels_per_condition": 5,
    "n_episodes_per_condition": 5,
    "n_episodes_per_intervention": 5,
    "max_episode_length": 32,
    "max_steps_per_episode": 32,
    "max_episode_steps": 32,
    "n_test_levels": 10,
    "n_level_pairs": 5,
    "adaptation_episodes": 3,
    "progressive_difficulty_steps": 3,
    "n_episode_sequences": 3,
    "sequence_length": 3,
    "n_sequences": 5,
    "n_rollouts_per_checkpoint": 10,
    "n_levels_per_subset": 10,
    "n_levels_per_checkpoint": 10,
    "n_probe_levels": 20,
    "n_levels_per_type": 5,
    "n_eval_levels": 10,
    "n_pairs": 10,
    "n_samples_per_step": 10,
    "trajectory_length": 5,
    "n_interventions": 5,
    "n_permutation_trials": 5,
    "baseline_steps": 10,
    "intervention_steps": 20,
    "post_steps": 10,
    "n_steps_per_episode": 10,
    # Analysis params (reduce computation)
    "n_components_pca": 5,
    "n_components_viz": 2,
    "reduced_dim": 3,
    "compute_sparse_ae": False,
    "n_gradient_samples": 3,
    "n_representation_samples": 5,
    "n_patching_pairs": 5,
    "n_attribution_steps": 5,
    "n_injection_episodes": 3,
    "n_random_baselines": 3,
    "n_recovery_episodes": 2,
    "use_pysr": False,
    "pysr_iterations": 5,
    "n_adversary_strategies": 2,
    "n_strategy_clusters": 2,
    "n_shard_components": 3,
    "n_clusters": 2,
    "n_clusters_kmeans": 2,
    "min_cluster_size": 3,
    "random_baseline_samples": 10,
    "granger_max_lag": 3,
    "max_lag_to_test": 2,
    # Training-time experiment params
    "collection_interval": 3,
    "probe_n_samples": 5,
    "rolling_window": 3,
    "n_samples_per_collection": 5,
}


# ============================================================================
# STATUS CONSTANTS
# ============================================================================

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"


# ============================================================================
# PHASE 1: CHECKPOINT GENERATION
# ============================================================================

def generate_checkpoints(
    methods: List[str],
    agents_for_method: Dict[str, List[str]],
    output_dir: str,
    timeout: int = 900,
) -> Dict[str, Dict[str, Any]]:
    """Generate minimal checkpoints via subprocess.

    Runs train_with_experiments with 10 updates, eval_freq=5 (2 eval checkpoints),
    no experiments (fast). Training-time experiments are tested via simulated
    hooks in Phase 2.

    Returns:
        Dict mapping "method/agent" to dict with:
          - "path": checkpoint directory path
          - "phase1_status": "ok", "FAILED", "TIMEOUT", or "ERROR"
          - "phase1_output": subprocess stdout (for debugging)
    """
    # Write speed overrides to a temp config JSON for subprocess to pick up
    config_overrides = dict(SPEED_OVERRIDES)
    config_file = os.path.join(output_dir, "_speed_overrides.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(config_file, "w") as f:
        json.dump(config_overrides, f, indent=2)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))

    checkpoint_info = {}
    total = sum(len(agents) for agents in agents_for_method.values())
    done = 0
    phase1_failures = []

    for method in methods:
        agents = agents_for_method[method]
        for agent in agents:
            done += 1
            key = f"{method}/{agent}"
            ckpt_dir = os.path.abspath(os.path.join(output_dir, "checkpoints", method, agent, "0"))

            # Skip if checkpoint already exists
            if os.path.exists(os.path.join(ckpt_dir, "config.json")):
                print(f"  [{done}/{total}] {key} — cached")
                checkpoint_info[key] = {
                    "path": ckpt_dir,
                    "phase1_status": "cached",
                }
                continue

            print(f"  [{done}/{total}] {key} — training + experiments...",
                  end=" ", flush=True)
            t0 = time.time()

            # Run with --no_experiments for fast checkpoint generation.
            # Note: setup_checkpointing writes to cwd/checkpoints/ regardless
            # of --output_dir, so we run the subprocess from output_dir as cwd.
            cmd = [
                sys.executable, "-m", "ablations.scripts.train_with_experiments",
                "--training_method", method,
                "--agent_type", agent,
                "--seed", "0",
                "--num_updates", "10",
                "--eval_freq", "5",
                "--num_train_envs", "4",
                "--num_steps", "16",
                "--no_wandb",
                "--no_experiments",
            ]

            try:
                # Run from output_dir so setup_checkpointing's os.getcwd()
                # writes checkpoints under output_dir/checkpoints/
                subprocess_cwd = os.path.abspath(output_dir)
                os.makedirs(subprocess_cwd, exist_ok=True)
                env = os.environ.copy()
                env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=subprocess_cwd,
                    env=env,
                )
                dt = time.time() - t0

                if result.returncode != 0:
                    print(f"FAILED ({dt:.1f}s)")
                    # Show last few lines of stderr for quick diagnosis
                    stderr_tail = result.stderr.strip().split("\n")[-5:]
                    for line in stderr_tail:
                        print(f"    {line}")
                    checkpoint_info[key] = {
                        "path": ckpt_dir,
                        "phase1_status": "FAILED",
                        "phase1_output": result.stdout[-2000:] + "\n---STDERR---\n" + result.stderr[-2000:],
                    }
                    phase1_failures.append(key)
                    continue

                print(f"ok ({dt:.1f}s)")
                checkpoint_info[key] = {
                    "path": ckpt_dir,
                    "phase1_status": "ok",
                }

            except subprocess.TimeoutExpired:
                dt = time.time() - t0
                print(f"TIMEOUT ({timeout}s)")
                checkpoint_info[key] = {
                    "path": ckpt_dir,
                    "phase1_status": "TIMEOUT",
                }
                phase1_failures.append(key)
            except Exception as e:
                dt = time.time() - t0
                print(f"ERROR: {e}")
                checkpoint_info[key] = {
                    "path": ckpt_dir,
                    "phase1_status": "ERROR",
                    "phase1_output": str(e),
                }
                phase1_failures.append(key)

    # Phase 1 summary
    n_ok = sum(1 for v in checkpoint_info.values() if v["phase1_status"] in ("ok", "cached"))
    print(f"\n  Phase 1 complete: {n_ok}/{total} configs succeeded")
    if phase1_failures:
        print(f"  Phase 1 failures ({len(phase1_failures)}):")
        for key in phase1_failures:
            print(f"    - {key}: {checkpoint_info[key]['phase1_status']}")

    return checkpoint_info


# ============================================================================
# PHASE 2: EXPERIMENT EXECUTION
# ============================================================================

def run_single_experiment(
    exp_name: str,
    checkpoint_path: str,
    agent_type: str,
    training_method: str,
    config_overrides: Dict[str, Any],
    seed: int = 0,
) -> Dict[str, Any]:
    """Run a single checkpoint experiment in-process.

    Returns:
        Dict with status, duration, error info.
    """
    from ablations.experiments.run_experiment import (
        get_experiment_class,
        load_agent,
        load_checkpoint,
        _wrap_paired_train_state,
    )
    from ablations.common.types import PAIREDTrainState

    output_path = Path(f"/tmp/comprehensive_test/{training_method}/{agent_type}/{exp_name}")
    output_path.mkdir(parents=True, exist_ok=True)

    # Build config from checkpoint + overrides
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    config["training_method"] = training_method
    config.update(config_overrides)
    # Override namespaced experiment params so exp_config() picks them up
    for key, value in config_overrides.items():
        namespaced = f"exp.{exp_name}.{key}"
        if namespaced in config:
            config[namespaced] = value

    # Load agent and checkpoint
    agent = load_agent(agent_type, config)
    train_state, checkpoint_config = load_checkpoint(
        checkpoint_path, agent=agent, seed=seed
    )
    if checkpoint_config:
        config.update(checkpoint_config)
        config.update(config_overrides)  # Re-apply after merge
        for key, value in config_overrides.items():
            namespaced = f"exp.{exp_name}.{key}"
            if namespaced in config:
                config[namespaced] = value

    if isinstance(train_state, PAIREDTrainState):
        train_state = _wrap_paired_train_state(train_state)

    # Instantiate experiment
    ExperimentClass = get_experiment_class(exp_name)
    experiment = ExperimentClass(
        agent=agent,
        train_state=train_state,
        config=config,
        output_dir=str(output_path),
        training_method=training_method,
    )

    # Run: collect, analyze, visualize
    rng = jax.random.PRNGKey(seed)
    experiment.data = experiment.collect_data(rng)
    results = experiment.analyze()
    experiment.visualize()

    return results


def run_training_time_experiment(
    exp_name: str,
    agent_type: str,
    training_method: str,
    config_overrides: Dict[str, Any],
    seed: int = 0,
) -> Dict[str, Any]:
    """Test a training-time experiment by simulating hook calls.

    Instantiates the experiment directly (not via create_training_experiments,
    which passes config={}), calls training_hook() with dummy metrics
    a few times, then calls analyze() and visualize().
    """
    from ablations.experiments.run_experiment import get_experiment_class
    from ablations.agents import get_agent_class
    from ablations.configs import get_config

    # Build config with experiment defaults + speed overrides.
    # Speed overrides must also be applied to namespaced keys (exp.<name>.<param>)
    # since exp_config() checks namespaced keys first.
    config = get_config(training_method, agent_type, include_experiments=True)
    config.update(config_overrides)
    # Override namespaced experiment params so exp_config() picks them up
    for key, value in config_overrides.items():
        namespaced = f"exp.{exp_name}.{key}"
        if namespaced in config:
            config[namespaced] = value

    # Create agent + train_state
    agent_cls = get_agent_class(agent_type)
    agent_obj = agent_cls(config)
    rng = jax.random.PRNGKey(seed)
    train_state = agent_obj.create_train_state(rng)

    # Instantiate experiment directly with full config
    ExperimentClass = get_experiment_class(exp_name)
    exp = ExperimentClass(
        agent=agent_obj,
        train_state=train_state,
        config=config,
        training_method=training_method,
    )

    # Simulate a few hook calls with dummy metrics
    collection_interval = config_overrides.get("collection_interval", 3)
    dummy_metrics = {
        "solve_rate": 0.5,
        "mean_return": 1.0,
        "policy_entropy": 0.3,
        "value_mean": 0.5,
        "grad_norm": 0.1,
    }
    for step in range(0, collection_interval * 5, collection_interval):
        exp.training_hook(train_state, dummy_metrics, step)

    # Analyze and visualize
    results = exp.analyze()
    exp.visualize()

    return results


# ============================================================================
# RESULT TRACKING
# ============================================================================

class TestResult:
    """Single experiment test result."""

    def __init__(
        self,
        method: str,
        agent: str,
        experiment: str,
        status: str,
        duration: float,
        error: Optional[str] = None,
        traceback_str: Optional[str] = None,
    ):
        self.method = method
        self.agent = agent
        self.experiment = experiment
        self.status = status
        self.duration = duration
        self.error = error
        self.traceback_str = traceback_str

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "method": self.method,
            "agent": self.agent,
            "experiment": self.experiment,
            "status": self.status,
            "duration": round(self.duration, 2),
            "error": self.error,
        }
        return d


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def get_agents_for_method(
    method: str, agent_filter: Optional[List[str]] = None
) -> List[str]:
    """Return agent list for a training method, optionally filtered."""
    from ablations.configs import BASE_AGENTS, PAIRED_AGENTS

    if method == "paired":
        agents = list(PAIRED_AGENTS)
    else:
        agents = list(BASE_AGENTS)

    if agent_filter:
        agents = [a for a in agents if a in agent_filter]

    return agents


def run_comprehensive_test(
    methods: List[str],
    agent_filter: Optional[List[str]],
    checkpoint_base: str,
    output_dir: str,
    skip_checkpoint_generation: bool,
    stop_on_error: bool,
    timeout: int,
) -> List[TestResult]:
    """Run the full comprehensive test."""
    from ablations.configs import (
        UNIVERSAL_EXPERIMENTS,
        PAIRED_EXPERIMENTS,
        TRAINING_TIME_EXPERIMENTS,
        get_experiments_for_method,
    )

    # Build agents_for_method mapping
    agents_for_method = {}
    for method in methods:
        agents = get_agents_for_method(method, agent_filter)
        if agents:
            agents_for_method[method] = agents

    if not agents_for_method:
        print("No valid method/agent combinations to test.")
        return []

    # Print test matrix
    total_configs = sum(len(a) for a in agents_for_method.values())
    print("=" * 70)
    print("COMPREHENSIVE EXPERIMENT TEST")
    print("=" * 70)
    print(f"Methods: {methods}")
    print(f"Configs: {total_configs}")
    for m, agents in agents_for_method.items():
        exps = get_experiments_for_method(m, include_training_time=True)
        print(f"  {m}: {len(agents)} agents x {len(exps)} experiments")
    print(f"Checkpoint base: {checkpoint_base}")
    print(f"Output dir: {output_dir}")
    print("=" * 70)

    # Phase 1: Checkpoint generation (with experiments enabled)
    if not skip_checkpoint_generation:
        print("\n--- PHASE 1: Checkpoint Generation (2 eval checkpoints each) ---\n")
        checkpoint_info = generate_checkpoints(
            methods, agents_for_method, checkpoint_base, timeout=900
        )
    else:
        print("\n--- PHASE 1: Skipped (using existing checkpoints) ---\n")
        checkpoint_info = {}
        for method, agents in agents_for_method.items():
            for agent in agents:
                key = f"{method}/{agent}"
                ckpt_dir = os.path.abspath(os.path.join(checkpoint_base, "checkpoints", method, agent, "0"))
                if os.path.exists(ckpt_dir):
                    checkpoint_info[key] = {"path": ckpt_dir, "phase1_status": "cached"}
                else:
                    print(f"  WARNING: No checkpoint at {ckpt_dir}")

    # Extract usable checkpoint paths (only those that succeeded)
    checkpoint_paths: Dict[str, str] = {}
    phase1_results: List[TestResult] = []
    for key, info in checkpoint_info.items():
        method, agent = key.split("/", 1)
        if info["phase1_status"] in ("ok", "cached"):
            if os.path.exists(info["path"]):
                checkpoint_paths[key] = info["path"]
            else:
                # Checkpoint dir doesn't exist despite "ok" status
                phase1_results.append(TestResult(
                    method=method, agent=agent,
                    experiment="[phase1] train+experiments",
                    status=FAIL, duration=0.0,
                    error=f"Checkpoint dir missing: {info['path']}",
                ))
        else:
            # Phase 1 failed for this config — record it
            phase1_results.append(TestResult(
                method=method, agent=agent,
                experiment="[phase1] train+experiments",
                status=FAIL, duration=0.0,
                error=f"Phase 1 {info['phase1_status']}: {info.get('phase1_output', '')[:500]}",
                traceback_str=info.get("phase1_output"),
            ))

    if not checkpoint_paths:
        print("ERROR: No checkpoints available. Cannot proceed to Phase 2.")
        _write_results(phase1_results, [], output_dir)
        return phase1_results

    # Phase 2: Experiment execution (isolated, per-experiment)
    print(f"\n--- PHASE 2: Isolated Experiment Execution ---\n")

    all_results: List[TestResult] = list(phase1_results)
    failures_log = []

    # Collect Phase 1 failure tracebacks
    for r in phase1_results:
        if r.status == FAIL and r.traceback_str:
            failures_log.append(
                f"\n{'='*60}\n{r.method}/{r.agent}/{r.experiment}\n{'='*60}\n{r.traceback_str}"
            )

    for method in methods:
        agents = agents_for_method.get(method, [])
        checkpoint_exps = list(UNIVERSAL_EXPERIMENTS)
        if method == "paired":
            checkpoint_exps.extend(PAIRED_EXPERIMENTS)
        training_exps = list(TRAINING_TIME_EXPERIMENTS)

        for agent in agents:
            key = f"{method}/{agent}"
            ckpt_path = checkpoint_paths.get(key)

            if ckpt_path is None:
                # Mark all experiments as SKIP for missing checkpoint
                for exp in checkpoint_exps + training_exps:
                    all_results.append(TestResult(
                        method=method, agent=agent, experiment=exp,
                        status=SKIP, duration=0.0,
                        error="No checkpoint (Phase 1 failed)",
                    ))
                continue

            # --- Checkpoint experiments ---
            for exp_name in checkpoint_exps:
                label = f"{method}/{agent}/{exp_name}"
                print(f"  {label}...", end=" ", flush=True)
                t0 = time.time()

                try:
                    results = run_single_experiment(
                        exp_name, ckpt_path, agent, method,
                        SPEED_OVERRIDES, seed=0,
                    )
                    dt = time.time() - t0

                    if results and isinstance(results, dict) and len(results) > 0:
                        status = PASS
                    else:
                        status = WARN

                    print(f"{status} ({dt:.1f}s)")
                    all_results.append(TestResult(
                        method=method, agent=agent, experiment=exp_name,
                        status=status, duration=dt,
                    ))

                except Exception as e:
                    dt = time.time() - t0
                    tb = traceback.format_exc()
                    print(f"FAIL ({dt:.1f}s)")
                    print(f"    {str(e)[:200]}")
                    all_results.append(TestResult(
                        method=method, agent=agent, experiment=exp_name,
                        status=FAIL, duration=dt,
                        error=str(e), traceback_str=tb,
                    ))
                    failures_log.append(f"\n{'='*60}\n{label}\n{'='*60}\n{tb}")

                    if stop_on_error:
                        print("\n  STOPPING ON FIRST ERROR (--stop_on_error)")
                        _write_results(all_results, failures_log, output_dir)
                        return all_results

            # --- Training-time experiments ---
            for exp_name in training_exps:
                label = f"{method}/{agent}/{exp_name} (training-time)"
                print(f"  {label}...", end=" ", flush=True)
                t0 = time.time()

                try:
                    results = run_training_time_experiment(
                        exp_name, agent, method,
                        SPEED_OVERRIDES, seed=0,
                    )
                    dt = time.time() - t0

                    if results and isinstance(results, dict) and len(results) > 0:
                        status = PASS
                    else:
                        status = WARN

                    print(f"{status} ({dt:.1f}s)")
                    all_results.append(TestResult(
                        method=method, agent=agent, experiment=exp_name,
                        status=status, duration=dt,
                    ))

                except Exception as e:
                    dt = time.time() - t0
                    tb = traceback.format_exc()
                    print(f"FAIL ({dt:.1f}s)")
                    print(f"    {str(e)[:200]}")
                    all_results.append(TestResult(
                        method=method, agent=agent, experiment=exp_name,
                        status=FAIL, duration=dt,
                        error=str(e), traceback_str=tb,
                    ))
                    failures_log.append(f"\n{'='*60}\n{label}\n{'='*60}\n{tb}")

                    if stop_on_error:
                        print("\n  STOPPING ON FIRST ERROR (--stop_on_error)")
                        _write_results(all_results, failures_log, output_dir)
                        return all_results

    _write_results(all_results, failures_log, output_dir)
    return all_results


# ============================================================================
# OUTPUT
# ============================================================================

def _write_results(
    results: List[TestResult],
    failures_log: List[str],
    output_dir: str,
):
    """Write JSON report, failures log, and print console summary."""
    os.makedirs(output_dir, exist_ok=True)

    # Count by status
    counts = defaultdict(int)
    for r in results:
        counts[r.status] += 1

    # JSON report
    report = {
        "summary": {
            "total": len(results),
            "pass": counts[PASS],
            "warn": counts[WARN],
            "fail": counts[FAIL],
            "skip": counts[SKIP],
        },
        "results": [r.to_dict() for r in results],
    }
    report_path = os.path.join(output_dir, "comprehensive_test_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Failures log
    if failures_log:
        failures_path = os.path.join(output_dir, "failures.log")
        with open(failures_path, "w") as f:
            f.write("\n".join(failures_log))

    # Console summary table
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(
        f"{'Method':<12} | {'Agent':<25} | {'Phase1':>6} | {'Universal':>9} | "
        f"{'PAIRED':>6} | {'Training':>8} | {'Total':>7} | Status"
    )
    print("-" * 100)

    # Group results by method/agent
    grouped: Dict[str, Dict[str, List[TestResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        grouped[r.method][r.agent].append(r)

    from ablations.configs import (
        UNIVERSAL_EXPERIMENTS,
        PAIRED_EXPERIMENTS,
        TRAINING_TIME_EXPERIMENTS,
    )
    universal_set = set(UNIVERSAL_EXPERIMENTS)
    paired_set = set(PAIRED_EXPERIMENTS)
    training_set = set(TRAINING_TIME_EXPERIMENTS)

    for method in sorted(grouped.keys()):
        for agent in sorted(grouped[method].keys()):
            agent_results = grouped[method][agent]

            # Phase 1 result
            p1 = [r for r in agent_results if r.experiment.startswith("[phase1]")]
            p1_str = "FAIL" if any(r.status == FAIL for r in p1) else ("ok" if not p1 else "ok")

            # Classify Phase 2 results
            uni_pass = sum(
                1 for r in agent_results
                if r.experiment in universal_set and r.status in (PASS, WARN)
            )
            uni_total = sum(
                1 for r in agent_results if r.experiment in universal_set
            )
            paired_pass = sum(
                1 for r in agent_results
                if r.experiment in paired_set and r.status in (PASS, WARN)
            )
            paired_total = sum(
                1 for r in agent_results if r.experiment in paired_set
            )
            train_pass = sum(
                1 for r in agent_results
                if r.experiment in training_set and r.status in (PASS, WARN)
            )
            train_total = sum(
                1 for r in agent_results if r.experiment in training_set
            )

            total_pass = uni_pass + paired_pass + train_pass
            total_all = uni_total + paired_total + train_total

            has_fail = any(r.status == FAIL for r in agent_results)
            row_status = "FAIL" if has_fail else "PASS"

            paired_str = (
                f"{paired_pass}/{paired_total}"
                if paired_total > 0
                else "SKIP"
            )

            print(
                f"{method:<12} | {agent:<25} | "
                f"{p1_str:>6} | "
                f"{uni_pass:>4}/{uni_total:<4} | "
                f"{paired_str:>6} | "
                f"{train_pass:>3}/{train_total:<4} | "
                f"{total_pass:>3}/{total_all:<3} | {row_status}"
            )

    print("-" * 100)
    print(
        f"TOTAL: {counts[PASS]} PASS, {counts[WARN]} WARN, "
        f"{counts[FAIL]} FAIL, {counts[SKIP]} SKIP "
        f"(out of {len(results)})"
    )
    print(f"\nJSON report: {report_path}")
    if failures_log:
        print(f"Failures log: {os.path.join(output_dir, 'failures.log')}")
    print("=" * 100)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive experiment test across all configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--methods", nargs="*", default=None,
        help="Training methods to test (default: all 5)",
    )
    parser.add_argument(
        "--agents", nargs="*", default=None,
        help="Agent types to test (default: all for each method)",
    )
    parser.add_argument(
        "--checkpoint_base", type=str, default="test_checkpoints",
        help="Base directory for checkpoints (default: test_checkpoints)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="test_results",
        help="Output directory for reports (default: test_results)",
    )
    parser.add_argument(
        "--skip_checkpoint_generation", action="store_true",
        help="Skip Phase 1, use existing checkpoints",
    )
    parser.add_argument(
        "--stop_on_error", action="store_true",
        help="Halt on the first experiment failure",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Per-experiment timeout in seconds (default: 120)",
    )
    args = parser.parse_args()

    # Default methods
    from ablations.configs import ALL_TRAINING_METHODS
    methods = args.methods or list(ALL_TRAINING_METHODS)

    results = run_comprehensive_test(
        methods=methods,
        agent_filter=args.agents,
        checkpoint_base=args.checkpoint_base,
        output_dir=args.output_dir,
        skip_checkpoint_generation=args.skip_checkpoint_generation,
        stop_on_error=args.stop_on_error,
        timeout=args.timeout,
    )

    # Exit with non-zero if any failures
    n_fail = sum(1 for r in results if r.status == FAIL)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
