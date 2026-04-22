# Configuration Reference

All configuration lives under `ablations/configs/`. The override order is:

```
get_default_config()           # training defaults (configs/defaults.py)
  -> method config             # TRAINING_METHOD_CONFIGS (presets.py)
  -> agent config              # AGENT_CONFIGS (presets.py)
  -> experiment defaults       # EXPERIMENT_DEFAULTS namespaced as exp.* (experiment_defaults.py)
  -> JSON file (--config)      # user-provided JSON
  -> checkpoint config.json    # loaded at runtime (post-hoc runners only)
  -> CLI args                  # highest precedence
```

## Files

| File                     | Contents                                                        |
|--------------------------|-----------------------------------------------------------------|
| `defaults.py`            | `get_default_config()` — ~74-key base training defaults         |
| `presets.py`             | `TRAINING_METHOD_CONFIGS`, `AGENT_CONFIGS`, `get_config()`      |
| `experiment_defaults.py` | `EXPERIMENT_DEFAULTS` (38 experiments), lists, cadence, helpers |
| `cli.py`                 | Shared argparse builders, `build_config_from_args()`            |
| `__init__.py`            | Re-exports everything                                           |

## Experiment Namespacing

Experiment defaults are namespaced as `exp.<experiment_name>.<param>` in the config dict
to avoid collisions (e.g., `n_levels` means 500 for `level_probing` but 1000 for `dr_coverage`).

Experiments access their params via `self.exp_config("param")`, which resolves:
1. `exp.<self.name>.<param>` (namespaced)
2. `config[param]` (flat, backward compatible)
3. fallback value

## Agent Types

| Agent Type            | Memory Mechanism   | Description                          |
|-----------------------|--------------------|--------------------------------------|
| `accel_probe`         | Reset per episode  | Baseline - no memory across episodes |
| `persistent_lstm`     | Non-resetting LSTM | Tests emergent curriculum awareness  |
| `context_vector`      | EMA context vector | Compressed episode history           |
| `episodic_memory`     | Discrete buffer    | Attention-based retrieval            |
| `next_env_prediction` | Integrated head    | Upper bound with prediction loss     |

PAIRED variants: prepend `paired_` (e.g., `paired_persistent_lstm`).

## Shared CLI Args

All entry points use composable argparse builders from `cli.py`:

- `add_common_args` — seed, training_method, agent_type, config, output_dir, dry_run
- `add_training_args` — PPO, PLR, ACCEL, environment, eval, checkpoint, probe
- `add_experiment_selection_args` — --experiments, --no_experiments
- `add_experiment_param_args` — --n_levels, --max_steps, --n_samples, --n_episodes, --adv_num_steps
- `add_posthoc_args` — results_dir, checkpoint, agents, checkpoints_per_agent, parallel

## Example Commands

```bash
# Train one agent with all 38 experiments
python -m ablations.scripts.train_with_experiments \
    --agent_type persistent_lstm --training_method accel --seed 0

# Train all PAIRED agents
python -m ablations.scripts.train_with_experiments \
    --training_method paired --all_agents --seed 0

# Run post-hoc experiments on checkpoints
python -m ablations.experiments.run_all \
    --results_dir checkpoints/accel --output_dir results --training_method accel

# Run a single experiment
python -m ablations.experiments.run_experiment \
    --experiment level_probing --checkpoint checkpoints/accel/accel_probe/seed_0/checkpoint_100 \
    --agent_type accel_probe --training_method accel

# Train all 25 configurations
python -m ablations.scripts.train_all --seeds 0 1 2
```
