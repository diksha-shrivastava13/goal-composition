# Configuration Reference

This document lists all available command-line arguments for the ablation study.

## Agent Types

```
--agent_type {accel_probe, persistent_lstm, context_vector, episodic_memory}
```

| Agent Type | Memory Mechanism | Description |
|------------|------------------|-------------|
| `accel_probe` | Reset per episode | Baseline - no memory across episodes |
| `persistent_lstm` | Non-resetting LSTM | Tests emergent curriculum awareness |
| `context_vector` | EMA context vector | Compressed episode history |
| `episodic_memory` | Discrete buffer | Attention-based retrieval |

## Training Arguments

```bash
# Basic
--seed INT                  # Random seed (default: 0)
--run_name STR              # W&B run name (default: "ablation")
--project STR               # W&B project (default: "curriculum-awareness-ablation")
--num_updates INT           # Training steps (default: 30000)

# PPO
--lr FLOAT                  # Learning rate (default: 1e-4)
--gamma FLOAT               # Discount factor (default: 0.995)
--gae_lambda FLOAT          # GAE lambda (default: 0.98)
--clip_eps FLOAT            # PPO clip epsilon (default: 0.2)
--entropy_coeff FLOAT       # Entropy bonus (default: 1e-3)
--critic_coeff FLOAT        # Value loss coeff (default: 0.5)
--epoch_ppo INT             # PPO epochs (default: 5)
--num_minibatches INT       # Minibatches (default: 1)
--max_grad_norm FLOAT       # Gradient clipping (default: 0.5)
--num_train_envs INT        # Parallel envs (default: 32)
--num_steps INT             # Steps per rollout (default: 256)
```

## PLR/ACCEL Arguments

```bash
# PLR Mode Selection
--use_accel                 # Enable ACCEL (DR → Replay → Mutate)
--no_accel                  # PLR only (DR → Replay, no mutation)
--exploratory_grad_updates  # PLR (update on DR), default is Robust-PLR

# Level Buffer
--level_buffer_capacity INT # Buffer size (default: 4000)
--replay_prob FLOAT         # P(replay) (default: 0.8)
--staleness_coeff FLOAT     # Staleness penalty (default: 0.3)
--minimum_fill_ratio FLOAT  # Min fill before replay (default: 0.5)
--score_function {MaxMC,pvl}# Score function (default: MaxMC)
--prioritization {rank,topk}# Prioritization (default: rank)
--temperature FLOAT         # Sampling temperature (default: 0.3)
--top_k INT                 # Top-k for sampling (default: 4)
--no_buffer_duplicate_check # Disable duplicate checking
```

## ACCEL Arguments

```bash
--num_edits INT             # Mutations per level (default: 5)
```

## Probe Arguments

```bash
--use_probe                 # Enable probe (default)
--no_probe                  # Disable probe
--probe_lr FLOAT            # Probe learning rate (default: 1e-3)
--probe_tracking_buffer_size INT  # History buffer (default: 500)
```

## Memory-Specific Arguments

```bash
# Context Vector Agent
--context_dim INT           # Context vector size (default: 64)
--context_decay FLOAT       # EMA decay factor (default: 0.9)

# Episodic Memory Agent
--memory_buffer_size INT    # Episode buffer size (default: 64)
--memory_top_k INT          # Retrieval top-k (default: 8)
```

## Environment Arguments

```bash
--agent_view_size INT       # Agent observation size (default: 5)
--n_walls INT               # Walls for random levels (default: 25)
```

## Evaluation Arguments

```bash
--eval_freq INT             # Eval every N steps (default: 250)
--eval_num_attempts INT     # Eval attempts per level (default: 10)
--eval_levels STR+          # Eval level names
--n_env_predictions INT     # N-env prediction count (default: 100)
```

## Checkpointing Arguments

```bash
--checkpoint_save_interval INT   # Save every N evals (default: 2)
--max_number_of_checkpoints INT  # Max checkpoints (default: 60)
--checkpoint_directory STR       # For evaluation mode
--checkpoint_to_eval INT         # Checkpoint step (-1 for latest)
```

## Example Commands

### Train baseline (accel_probe)
```bash
python -m ablations.scripts.train \
    --agent_type accel_probe \
    --seed 0 \
    --use_accel \
    --num_updates 30000
```

### Train persistent LSTM
```bash
python -m ablations.scripts.train \
    --agent_type persistent_lstm \
    --seed 0 \
    --use_accel \
    --num_updates 30000
```

### Train with Robust-PLR (no exploratory updates)
```bash
python -m ablations.scripts.train \
    --agent_type accel_probe \
    --seed 0 \
    --use_accel \
    # --exploratory_grad_updates is False by default (Robust-PLR)
```

### Train with standard PLR (exploratory updates)
```bash
python -m ablations.scripts.train \
    --agent_type accel_probe \
    --seed 0 \
    --use_accel \
    --exploratory_grad_updates  # Enable updates on DR/Mutate branches
```

### PLR only (no ACCEL mutation)
```bash
python -m ablations.scripts.train \
    --agent_type accel_probe \
    --seed 0 \
    --no_accel
```

### Evaluate checkpoint
```bash
python -m ablations.scripts.evaluate \
    --checkpoint_dir ./checkpoints/accel_probe_seed0/0 \
    --experiment n_env_predictions \
    --n_env_predictions 100
```
