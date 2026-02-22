#!/bin/bash
# Run full experiment matrix: 5 agents x N experiments
#
# Usage:
#   ./run_matrix.sh                                    # All agents, default experiments
#   ./run_matrix.sh --agents "accel_probe persistent_lstm" --experiments "level_probing"
#   ./run_matrix.sh --training_method paired --agents "paired_accel_probe"

set -e

# Configuration
CHECKPOINT_BASE="${CHECKPOINT_BASE:-./checkpoints}"
OUTPUT_BASE="${OUTPUT_BASE:-./results}"
SEEDS="${SEEDS:-0}"
TRAINING_METHOD="${TRAINING_METHOD:-accel}"

# Agent types (all 5 non-PAIRED by default)
AGENTS="${AGENTS:-accel_probe persistent_lstm context_vector episodic_memory next_env_prediction}"

# Available experiments
EXPERIMENTS="${EXPERIMENTS:-level_probing value_calibration activation_analysis cross_agent_comparison mutation_adaptation causal_intervention counterfactual output_probing goal_extraction cross_episode_flow dr_coverage n_env_prediction n_step_prediction}"

# Parse command line overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        --agents) AGENTS="$2"; shift 2 ;;
        --experiments) EXPERIMENTS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --training_method) TRAINING_METHOD="$2"; shift 2 ;;
        --checkpoint_base) CHECKPOINT_BASE="$2"; shift 2 ;;
        --output_base) OUTPUT_BASE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo "Curriculum Awareness Ablation Study"
echo "Running Experiment Matrix"
echo "  Training method: $TRAINING_METHOD"
echo "  Agents: $AGENTS"
echo "  Experiments: $EXPERIMENTS"
echo "  Seeds: $SEEDS"
echo "=============================================="

for agent in $AGENTS; do
    for seed in $SEEDS; do
        # Path matches setup_checkpointing: checkpoints/{method}/{agent}/{seed}
        checkpoint_dir="${CHECKPOINT_BASE}/${TRAINING_METHOD}/${agent}/${seed}"

        if [ ! -d "$checkpoint_dir" ]; then
            echo "Skipping $agent (seed=$seed) - checkpoint not found at $checkpoint_dir"
            continue
        fi

        for exp in $EXPERIMENTS; do
            echo ""
            echo "Running: $agent (seed=$seed) - $exp"
            echo "----------------------------------------------"

            output_dir="${OUTPUT_BASE}/${TRAINING_METHOD}/${agent}/seed${seed}/${exp}"

            python -m ablations.experiments.run_experiment \
                --experiment "$exp" \
                --checkpoint "$checkpoint_dir" \
                --agent_type "$agent" \
                --output_dir "$output_dir" \
                --seed "$seed" \
                --training_method "$TRAINING_METHOD"

            echo "Completed: $exp"
        done
    done
done

echo ""
echo "=============================================="
echo "Experiment matrix complete!"
echo "=============================================="
