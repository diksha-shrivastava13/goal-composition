#!/usr/bin/env bash
# =============================================================================
# Pipelined training + experiments for all 25 configs on a single GPU.
#
# Strategy:
#   tmux pane 0 (GPU):  Train configs sequentially
#   tmux pane 1 (CPU):  Run post-hoc experiments on completed configs
#
# The experiment pane polls for new checkpoints and runs experiments as
# training completes, so GPU and CPU stay busy simultaneously.
#
# Usage:
#   # Start the full pipeline (creates a tmux session called "pipeline")
#   bash ablations/scripts/shell/run_pipeline.sh
#
#   # Attach to monitor
#   tmux attach -t pipeline
#
#   # Customize
#   NUM_UPDATES=50000 EVAL_FREQ=500 SEEDS="0" bash ablations/scripts/shell/run_pipeline.sh
#
# Requires: tmux, wandb login
# =============================================================================

set -euo pipefail

# --- Configuration (override via environment) ---
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
NUM_UPDATES="${NUM_UPDATES:-50000}"
EVAL_FREQ="${EVAL_FREQ:-500}"
CHECKPOINT_SAVE_INTERVAL="${CHECKPOINT_SAVE_INTERVAL:-500}"
SEEDS="${SEEDS:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-goal-composition}"
WANDB_GROUP="${WANDB_GROUP:-pipeline-$(date +%Y%m%d-%H%M)}"
CHECKPOINTS_PER_AGENT="${CHECKPOINTS_PER_AGENT:-10}"
SESSION_NAME="${SESSION_NAME:-pipeline}"

# 5 methods × 5 agents = 25 configs
METHODS="accel plr robust_plr dr paired"

cd "$PROJECT_DIR"

# --- Write the training script ---
cat > /tmp/pipeline_train.sh << 'TRAIN_EOF'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$1"
NUM_UPDATES="$2"
EVAL_FREQ="$3"
CHECKPOINT_SAVE_INTERVAL="$4"
SEEDS="$5"
WANDB_PROJECT="$6"
WANDB_GROUP="$7"

cd "$PROJECT_DIR"

METHODS=(accel plr robust_plr dr paired)
TOTAL=${#METHODS[@]}
DONE=0

for METHOD in "${METHODS[@]}"; do
    DONE=$((DONE + 1))
    echo ""
    echo "============================================================"
    echo "[TRAIN $DONE/$TOTAL] Method: $METHOD"
    echo "============================================================"

    for SEED in $SEEDS; do
        python -m ablations.scripts.train_with_experiments \
            --training_method "$METHOD" \
            --all_agents \
            --seed "$SEED" \
            --num_updates "$NUM_UPDATES" \
            --eval_freq "$EVAL_FREQ" \
            --checkpoint_save_interval "$CHECKPOINT_SAVE_INTERVAL" \
            --project "$WANDB_PROJECT" \
            --group_name "${WANDB_GROUP}" \
            --no_experiments \
            || echo "WARNING: $METHOD seed=$SEED failed, continuing..."

        # Signal that this method+seed is done
        touch "$PROJECT_DIR/.pipeline_done_${METHOD}_${SEED}"
    done
done

echo ""
echo "============================================================"
echo "ALL TRAINING COMPLETE"
echo "============================================================"
touch "$PROJECT_DIR/.pipeline_all_training_done"
TRAIN_EOF
chmod +x /tmp/pipeline_train.sh

# --- Write the experiment script ---
cat > /tmp/pipeline_experiments.sh << 'EXP_EOF'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$1"
CHECKPOINTS_PER_AGENT="$2"
SEEDS="$3"

cd "$PROJECT_DIR"

METHODS=(accel plr robust_plr dr paired)

echo "Experiment runner waiting for training to complete configs..."
echo "Will run experiments on each method as it finishes."
echo ""

for METHOD in "${METHODS[@]}"; do
    for SEED in $SEEDS; do
        SIGNAL="$PROJECT_DIR/.pipeline_done_${METHOD}_${SEED}"

        # Wait for training to finish this config
        while [ ! -f "$SIGNAL" ]; do
            sleep 30
        done

        echo ""
        echo "============================================================"
        echo "[EXPERIMENTS] Running on $METHOD (seed=$SEED)"
        echo "============================================================"

        python -m ablations.experiments.run_all \
            --results_dir "$PROJECT_DIR/checkpoints/$METHOD" \
            --output_dir "$PROJECT_DIR/experiments_posthoc/$METHOD" \
            --training_method "$METHOD" \
            --seed "$SEED" \
            --checkpoints_per_agent "$CHECKPOINTS_PER_AGENT" \
            || echo "WARNING: experiments for $METHOD seed=$SEED failed, continuing..."
    done
done

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
EXP_EOF
chmod +x /tmp/pipeline_experiments.sh

# --- Clean up old signals ---
rm -f "$PROJECT_DIR"/.pipeline_done_* "$PROJECT_DIR"/.pipeline_all_training_done

# --- Launch tmux session ---
echo "============================================================"
echo "Pipeline Configuration"
echo "============================================================"
echo "  Project dir:    $PROJECT_DIR"
echo "  Updates:        $NUM_UPDATES"
echo "  Eval freq:      $EVAL_FREQ"
echo "  Seeds:          $SEEDS"
echo "  Wandb project:  $WANDB_PROJECT"
echo "  Wandb group:    $WANDB_GROUP"
echo "  Checkpoints/agent for experiments: $CHECKPOINTS_PER_AGENT"
echo "  Session:        $SESSION_NAME"
echo "============================================================"
echo ""

# Kill existing session if present
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Create session with training pane
tmux new-session -d -s "$SESSION_NAME" -n "train" \
    "bash /tmp/pipeline_train.sh '$PROJECT_DIR' '$NUM_UPDATES' '$EVAL_FREQ' '$CHECKPOINT_SAVE_INTERVAL' '$SEEDS' '$WANDB_PROJECT' '$WANDB_GROUP'; bash"

# Split horizontally for experiments pane
tmux split-window -h -t "$SESSION_NAME:train" \
    "bash /tmp/pipeline_experiments.sh '$PROJECT_DIR' '$CHECKPOINTS_PER_AGENT' '$SEEDS'; bash"

# Set pane titles
tmux select-pane -t "$SESSION_NAME:train.0" -T "TRAINING (GPU)"
tmux select-pane -t "$SESSION_NAME:train.1" -T "EXPERIMENTS (CPU)"

echo "Pipeline launched in tmux session '$SESSION_NAME'."
echo ""
echo "  Attach:   tmux attach -t $SESSION_NAME"
echo "  Detach:   Ctrl+B, then D"
echo "  Kill:     tmux kill-session -t $SESSION_NAME"
echo ""
echo "Left pane = training (GPU), Right pane = experiments (CPU)"