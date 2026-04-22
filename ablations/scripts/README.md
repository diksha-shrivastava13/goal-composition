# Scripts Reference

This document covers every script under `ablations/scripts/` and `ablations/experiments/`.
For configuration resolution order, experiment namespacing, and agent types, see [`../configs/README.md`](../configs/README.md).

## Overview

| Script                             | Purpose                                     | Scope                                 |
|------------------------------------|---------------------------------------------|---------------------------------------|
| `train_with_experiments.py`        | Train agent(s) with integrated experiments  | 1 or all 5 agents under 1 method      |
| `train_all.py`                     | Orchestrate all 25 configurations           | All methods × all agents              |
| `experiments/run_experiment.py`    | Run one post-hoc experiment on a checkpoint | 1 experiment × 1 checkpoint           |
| `experiments/run_all.py`           | Batch post-hoc experiments on checkpoints   | N experiments × N checkpoints         |
| `smoke_test.py`                    | Quick smoke test (minimal settings)         | Up to 25 configs                      |
| `comprehensive_experiment_test.py` | Full experiment crash test                  | All experiments × all configs         |
| `test_paired_experiments.py`       | Quick PAIRED experiment test                | All PAIRED experiments × 1 checkpoint |

---

## Training Scripts

### `train_with_experiments.py` — primary entry point

Trains a single agent (or all 5 via `--all_agents`) under one training method, running all applicable experiments at every eval checkpoint.

**Key flags:**

| Flag                | Type | Default | Description                                                               |
|---------------------|------|---------|---------------------------------------------------------------------------|
| `--training_method` | str  | `accel` | One of: `accel`, `plr`, `robust_plr`, `dr`, `paired`                      |
| `--agent_type`      | str  | —       | Agent type (e.g., `persistent_lstm`, `paired_persistent_lstm`)            |
| `--all_agents`      | flag | —       | Train all 5 agents under the selected method                              |
| `--experiments`     | str+ | all     | Specific experiments to run (auto-partitions checkpoint vs training-time) |
| `--no_experiments`  | flag | —       | Disable all experiment running (just train)                               |
| `--resume`          | str  | —       | Resume training from checkpoint directory                                 |
| `--no_wandb`        | flag | —       | Disable wandb logging                                                     |
| `--dry_run`         | flag | —       | Print configuration and experiment list without training                  |

**Wandb:** Each agent gets its own `wandb.init()` run, named `{method}_{agent}_seed{seed}`. Use `--group_name` to group related runs for comparison. `--no_wandb` disables entirely.

**Experiment auto-partitioning:** `--experiments` accepts any mix of experiment names; the script auto-splits into checkpoint experiments (run at each eval step) and training-time experiments (maintain persistent state across training). Without `--experiments`, all 38 applicable experiments are enabled.

**Cadence:** `adversary_ablation` runs every 3rd eval step (configured in `experiment_defaults.py:EXPERIMENT_CADENCE`). All other experiments run at every eval step.

```bash
# Train one agent with all 38 experiments
python -m ablations.scripts.train_with_experiments \
    --agent_type persistent_lstm --training_method accel --seed 0

# Train all 5 PAIRED agents
python -m ablations.scripts.train_with_experiments \
    --training_method paired --all_agents --seed 0

# Only specific experiments
python -m ablations.scripts.train_with_experiments \
    --agent_type accel_probe --experiments level_probing behavioral_coupling

# Resume from checkpoint
python -m ablations.scripts.train_with_experiments \
    --agent_type persistent_lstm --training_method accel --seed 0 \
    --resume checkpoints/accel/persistent_lstm/0

# Dry run to preview experiment selection
python -m ablations.scripts.train_with_experiments \
    --training_method paired --all_agents --dry_run
```

### `train_all.py` — orchestrator

Trains all 25 agent × method configurations via subprocess calls to `train_with_experiments.py`.

**Key flags:**

| Flag                  | Type | Default          | Description                           |
|-----------------------|------|------------------|---------------------------------------|
| `--methods`           | str+ | all 5            | Training methods to run               |
| `--agents`            | str+ | all 5 per method | Agent types to run                    |
| `--seeds`             | int+ | `[0]`            | Random seeds to run                   |
| `--dry_run`           | flag | —                | Print configurations without training |
| `--continue_on_error` | flag | —                | Continue if a subprocess fails        |

**Wandb:** 25 separate wandb runs (one per subprocess). Pass `--group_name` via extra args to group them.

**Extra CLI args** after the known flags are forwarded verbatim to each `train_with_experiments.py` subprocess.

```bash
# Train all 25 configurations with 3 seeds
python -m ablations.scripts.train_all --seeds 0 1 2

# Train only ACCEL and PAIRED methods
python -m ablations.scripts.train_all --methods accel paired --seeds 0

# Forward extra args (e.g., disable wandb for all)
python -m ablations.scripts.train_all --seeds 0 -- --no_wandb --num_updates 10000
```

### `experiments/run_experiment.py` — single post-hoc experiment

Runs a single experiment on one checkpoint and saves results as JSON.

**Key flags:**

| Flag                     | Type | Default      | Description                                                                    |
|--------------------------|------|--------------|--------------------------------------------------------------------------------|
| `--experiment`           | str  | **required** | Experiment name (e.g., `level_probing`)                                        |
| `--checkpoint`           | str  | **required** | Path to checkpoint directory                                                   |
| `--secondary_checkpoint` | str  | —            | Secondary checkpoint for cross-adversary experiments (e.g., `regret_transfer`) |
| `--agent_type`           | str  | —            | Agent type                                                                     |
| `--training_method`      | str  | `accel`      | Training method used                                                           |
| `--dry_run`              | flag | —            | Validate experiment/method compatibility without running                       |

Plus all common args and experiment param overrides (see [CLI Parameter Reference](#cli-parameter-reference)).

```bash
python -m ablations.experiments.run_experiment \
    --experiment level_probing \
    --checkpoint checkpoints/accel/accel_probe/seed_0/checkpoint_100 \
    --agent_type accel_probe --training_method accel

# Cross-seed regret_transfer
python -m ablations.experiments.run_experiment \
    --experiment regret_transfer \
    --checkpoint checkpoints/paired/paired_persistent_lstm/seed_0/checkpoint_100 \
    --secondary_checkpoint checkpoints/paired/paired_persistent_lstm/seed_1/checkpoint_100 \
    --agent_type paired_persistent_lstm --training_method paired
```

### `experiments/run_all.py` — batch post-hoc experiments

Runs the full experiment suite on saved checkpoints, with optional parallelism and cross-experiment correlations.

**Key flags:**

| Flag                      | Type | Default            | Description                                        |
|---------------------------|------|--------------------|----------------------------------------------------|
| `--results_dir`           | str  | —                  | Directory containing trained agent checkpoints     |
| `--experiments`           | str+ | method-appropriate | Experiments to run                                 |
| `--agents`                | str+ | method-appropriate | Agent types to run                                 |
| `--checkpoints_per_agent` | int  | all                | Max checkpoints per agent (evenly sampled)         |
| `--parallel`              | int  | `1`                | Number of parallel workers                         |
| `--summarize`             | flag | —                  | Only generate summary report from existing results |

**Cross-experiment correlations** computed automatically when summarizing:
- Probe accuracy (Exp 1: level_probing) vs adaptation speed (Exp 29: mutation_adaptation)
- Horizon decay rate (Exp 5: n_env_prediction) vs adaptation speed (Exp 29)

```bash
# Run all experiments on ACCEL checkpoints
python -m ablations.experiments.run_all \
    --results_dir checkpoints/accel --output_dir results --training_method accel

# Run all 35 experiments on PAIRED checkpoints with 4 workers
python -m ablations.experiments.run_all \
    --results_dir checkpoints/paired --output_dir results \
    --training_method paired --parallel 4

# Only re-generate summary from existing results
python -m ablations.experiments.run_all --summarize --output_dir results
```

---

## Test Scripts

### `smoke_test.py`

Runs each of the 25 configurations with minimal settings to verify the full pipeline works end-to-end.

| Flag              | Type | Default        | Description                               |
|-------------------|------|----------------|-------------------------------------------|
| `--methods`       | str+ | all 5          | Training methods to test                  |
| `--agents`        | str+ | all per method | Agent types to test                       |
| `--stop_on_error` | flag | —              | Stop on first failure (default: continue) |

Settings: 10 training updates, `eval_freq=3`, 4 parallel envs, 16-step rollouts, no wandb. All applicable experiments run at each checkpoint.

```bash
python -m ablations.scripts.smoke_test
python -m ablations.scripts.smoke_test --methods accel --agents persistent_lstm
```

### `comprehensive_experiment_test.py`

Two-phase test: (1) generate minimal checkpoints, (2) run all applicable experiments in-process with aggressive speed overrides.

| Flag                           | Type | Default            | Description                               |
|--------------------------------|------|--------------------|-------------------------------------------|
| `--methods`                    | str+ | all 5              | Training methods to test                  |
| `--agents`                     | str+ | all per method     | Agent types to test                       |
| `--checkpoint_base`            | str  | `test_checkpoints` | Base directory for checkpoints            |
| `--output_dir`                 | str  | `test_results`     | Output directory for reports              |
| `--skip_checkpoint_generation` | flag | —                  | Reuse existing checkpoints (skip Phase 1) |
| `--stop_on_error`              | flag | —                  | Halt on first experiment failure          |
| `--timeout`                    | int  | `120`              | Per-experiment timeout in seconds         |

```bash
python -m ablations.scripts.comprehensive_experiment_test
python -m ablations.scripts.comprehensive_experiment_test \
    --methods accel --agents accel_probe --stop_on_error
```

### `test_paired_experiments.py`

Quick test of all PAIRED experiments on a single `paired_persistent_lstm` checkpoint. Skips `adversary_ablation` (too slow). Run directly:

```bash
python ablations/scripts/test_paired_experiments.py
```

---

## CLI Parameter Reference

All parameters below are defined in `ablations/configs/cli.py` and available to scripts that use the corresponding argparse builder.

### Common (`add_common_args`)

| Flag                | Type | Default | Description                                    |
|---------------------|------|---------|------------------------------------------------|
| `--seed`            | int  | `0`     | Random seed                                    |
| `--training_method` | str  | `accel` | `accel`, `plr`, `robust_plr`, `dr`, `paired`   |
| `--agent_type`      | str  | —       | Agent type (e.g., `persistent_lstm`)           |
| `--config`          | str  | —       | Load config from JSON file (CLI args override) |
| `--output_dir`      | str  | cwd     | Base output directory                          |
| `--dry_run`         | flag | —       | Print configuration without running            |

### Wandb & Run Config

| Flag                     | Type | Default                | Description                                             |
|--------------------------|------|------------------------|---------------------------------------------------------|
| `--project`              | str  | `JaxUED-minigrid-maze` | Wandb project name                                      |
| `--run_name`             | str  | auto                   | Wandb run name (default: `{method}_{agent}_seed{seed}`) |
| `--group_name`           | str  | —                      | Wandb group name                                        |
| `--mode`                 | str  | `train`                | `train` or `eval`                                       |
| `--checkpoint_directory` | str  | —                      | Checkpoint directory for eval mode                      |
| `--checkpoint_to_eval`   | int  | `-1`                   | Specific checkpoint index (`-1` = latest)               |

### Training / PPO

| Flag                | Type  | Default | Description                                                          |
|---------------------|-------|---------|----------------------------------------------------------------------|
| `--lr`              | float | `1e-4`  | Learning rate                                                        |
| `--max_grad_norm`   | float | `0.5`   | Max gradient norm                                                    |
| `--num_updates`     | int   | `30000` | Number of training updates                                           |
| `--num_env_steps`   | int   | —       | Total env steps (alternative to `--num_updates`, mutually exclusive) |
| `--num_steps`       | int   | `256`   | Rollout length per environment                                       |
| `--num_train_envs`  | int   | `32`    | Number of parallel training environments                             |
| `--num_minibatches` | int   | `1`     | Number of minibatches                                                |
| `--gamma`           | float | `0.995` | Discount factor                                                      |
| `--epoch_ppo`       | int   | `5`     | PPO epochs per update                                                |
| `--clip_eps`        | float | `0.2`   | PPO clip epsilon                                                     |
| `--gae_lambda`      | float | `0.98`  | GAE lambda                                                           |
| `--entropy_coeff`   | float | `1e-3`  | Entropy coefficient                                                  |
| `--critic_coeff`    | float | `0.5`   | Critic loss coefficient                                              |

### PLR

| Flag                         | Type  | Default | Description                                                    |
|------------------------------|-------|---------|----------------------------------------------------------------|
| `--score_function`           | str   | `MaxMC` | `MaxMC` or `pvl`                                               |
| `--exploratory_grad_updates` | bool  | `False` | `--exploratory_grad_updates` / `--no-exploratory_grad_updates` |
| `--level_buffer_capacity`    | int   | `4000`  | Level buffer capacity                                          |
| `--replay_prob`              | float | `0.8`   | Replay probability                                             |
| `--staleness_coeff`          | float | `0.3`   | Staleness coefficient                                          |
| `--temperature`              | float | `0.3`   | Score temperature                                              |
| `--top_k`                    | int   | `4`     | Top-k for prioritization                                       |
| `--minimum_fill_ratio`       | float | `0.5`   | Minimum buffer fill before replay                              |
| `--prioritization`           | str   | `rank`  | `rank` or `topk`                                               |
| `--buffer_duplicate_check`   | bool  | `True`  | `--buffer_duplicate_check` / `--no-buffer_duplicate_check`     |

### ACCEL

| Flag          | Type | Default | Description                        |
|---------------|------|---------|------------------------------------|
| `--use_accel` | bool | `False` | `--use_accel` / `--no-use_accel`   |
| `--num_edits` | int  | `5`     | Number of level edits per mutation |

### PAIRED

| Flag                       | Type  | Default | Description                       |
|----------------------------|-------|---------|-----------------------------------|
| `--adv_num_steps`          | int   | —       | Adversary rollout steps per level |
| `--adv_lr`                 | float | —       | Adversary learning rate           |
| `--adv_max_grad_norm`      | float | —       | Adversary max gradient norm       |
| `--adv_num_minibatches`    | int   | —       | Adversary number of minibatches   |
| `--adv_gamma`              | float | —       | Adversary discount factor         |
| `--adv_epoch_ppo`          | int   | —       | Adversary PPO epochs per update   |
| `--adv_clip_eps`           | float | —       | Adversary PPO clip epsilon        |
| `--adv_gae_lambda`         | float | —       | Adversary GAE lambda              |
| `--adv_entropy_coeff`      | float | —       | Adversary entropy coefficient     |
| `--adv_critic_coeff`       | float | —       | Adversary critic loss coefficient |
| `--adv_random_z_dimension` | int   | —       | Adversary random noise dimension  |
| `--adv_zero_out_random_z`  | bool  | —       | Zero out adversary random noise   |

Defaults for PAIRED params come from `presets.py:TRAINING_METHOD_CONFIGS["paired"]`; only set these to override.

### Environment

| Flag                | Type | Default | Description                           |
|---------------------|------|---------|---------------------------------------|
| `--agent_view_size` | int  | `5`     | Agent's partial observation view size |
| `--n_walls`         | int  | `25`    | Number of walls in generated levels   |

### Evaluation

| Flag                  | Type | Default  | Description                                         |
|-----------------------|------|----------|-----------------------------------------------------|
| `--eval_freq`         | int  | `250`    | Eval (and experiment) frequency in training updates |
| `--eval_num_attempts` | int  | `10`     | Number of evaluation attempts per level             |
| `--eval_levels`       | str+ | 8 levels | Named evaluation levels                             |

### Checkpointing

| Flag                          | Type | Default | Description                    |
|-------------------------------|------|---------|--------------------------------|
| `--checkpoint_save_interval`  | int  | `2`     | Save every Nth eval checkpoint |
| `--max_number_of_checkpoints` | int  | `60`    | Max checkpoints to keep        |

### Probe

| Flag                           | Type  | Default | Description                        |
|--------------------------------|-------|---------|------------------------------------|
| `--use_probe`                  | bool  | `True`  | Enable/disable online linear probe |
| `--probe_lr`                   | float | `1e-3`  | Probe learning rate                |
| `--probe_tracking_buffer_size` | int   | `500`   | Probe tracking buffer size         |

### Curriculum Prediction

| Flag                            | Type  | Default | Description                        |
|---------------------------------|-------|---------|------------------------------------|
| `--use_curriculum_prediction`   | bool  | `False` | Enable curriculum prediction head  |
| `--curriculum_hidden_size`      | int   | `128`   | Hidden size for prediction network |
| `--curriculum_pred_coeff`       | float | `1.0`   | Loss coefficient                   |
| `--curriculum_pred_eval_freq`   | int   | `100`   | Eval frequency for prediction      |
| `--n_prediction_eval`           | int   | `20`    | Number of prediction eval episodes |
| `--wall_loss_region`            | str   | `full`  | `full`, `explored`, or `frontier`  |
| `--curriculum_history_length`   | int   | `64`    | History length                     |
| `--curriculum_wall_weight`      | float | `1.0`   | Wall prediction weight             |
| `--curriculum_goal_weight`      | float | `1.0`   | Goal prediction weight             |
| `--curriculum_agent_pos_weight` | float | `1.0`   | Agent position prediction weight   |
| `--curriculum_agent_dir_weight` | float | `1.0`   | Agent direction prediction weight  |

### Experiment Selection (`add_experiment_selection_args`)

| Flag               | Type | Default | Description                                                               |
|--------------------|------|---------|---------------------------------------------------------------------------|
| `--experiments`    | str+ | all     | Specific experiments to run (auto-partitions checkpoint vs training-time) |
| `--no_experiments` | flag | —       | Disable all experiments                                                   |

### Experiment Parameter Overrides (`add_experiment_param_args`)

Flat overrides that apply globally to all experiments using that key. For per-experiment overrides, use `--config` with a JSON file (see [Non-CLI Configuration](#non-cli-configuration)).

| Flag              | Type | Default | Description                                  |
|-------------------|------|---------|----------------------------------------------|
| `--n_levels`      | int  | —       | Number of levels for experiments             |
| `--max_steps`     | int  | —       | Max rollout steps                            |
| `--n_samples`     | int  | —       | Number of samples                            |
| `--n_episodes`    | int  | —       | Number of episodes                           |
| `--adv_num_steps` | int  | —       | Adversary rollout steps (PAIRED experiments) |

### Post-hoc Runner (`add_posthoc_args`)

| Flag                      | Type | Default            | Description                                    |
|---------------------------|------|--------------------|------------------------------------------------|
| `--results_dir`           | str  | —                  | Directory containing trained agent checkpoints |
| `--checkpoint`            | str  | —                  | Path to a single checkpoint                    |
| `--agents`                | str+ | method-appropriate | Agent types to run                             |
| `--checkpoints_per_agent` | int  | all                | Max checkpoints per agent                      |
| `--parallel`              | int  | `1`                | Number of parallel workers                     |

---

## Non-CLI Configuration

### Per-experiment namespaced defaults

Each experiment has its own defaults namespaced as `exp.<experiment_name>.<param>` in the config dict (e.g., `exp.level_probing.n_levels`). These are defined in `experiment_defaults.py:EXPERIMENT_DEFAULTS`.

Experiments access their params via `self.exp_config("param")`, which resolves:
1. `exp.<self.name>.<param>` (namespaced)
2. `config[param]` (flat, backward compatible)
3. fallback value

### JSON config file

Pass `--config path/to/config.json` to any script. The JSON is merged into the config after method/agent presets but before CLI args (CLI always wins).

Example overriding a specific experiment's params:

```json
{
  "exp.level_probing.n_levels": 1000,
  "exp.dr_coverage.n_levels": 500,
  "exp.mutation_adaptation.n_episodes": 50,
  "eval_freq": 500,
  "lr": 3e-4
}
```

### `experiment_defaults.py`

Single source of truth for experiment parameters:
- `EXPERIMENT_DEFAULTS` — per-experiment default params (38 entries)
- `EXPERIMENT_CADENCE` — run frequency overrides (`adversary_ablation: 3`)
- `UNIVERSAL_EXPERIMENTS` — 13 experiments applicable to all methods
- `PAIRED_EXPERIMENTS` — 22 PAIRED-specific experiments
- `TRAINING_TIME_EXPERIMENTS` — 3 experiments with persistent state across training

See [`../configs/README.md`](../configs/README.md) for the full config resolution order.

---

## Deprecated Scripts

| Script               | Replacement                                                 |
|----------------------|-------------------------------------------------------------|
| `train.py`           | `train_with_experiments.py`                                 |
| `evaluate.py`        | `experiments/run_experiment.py` or `experiments/run_all.py` |
| `shell/train_all.sh` | `train_all.py`                                              |
