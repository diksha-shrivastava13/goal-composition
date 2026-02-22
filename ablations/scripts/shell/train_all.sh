#!/bin/bash
# Train all agent variants with multiple seeds

set -e

# Configuration
SEEDS="0 1 2"
PROJECT="curriculum-awareness-ablation"
NUM_UPDATES=30000

# Agent types to train
AGENTS="accel_probe persistent_lstm context_vector episodic_memory next_env_prediction"

echo "=============================================="
echo "Curriculum Awareness Ablation Study"
echo "Training all agents with seeds: $SEEDS"
echo "=============================================="

for agent in $AGENTS; do
    for seed in $SEEDS; do
        echo ""
        echo "Training: $agent (seed=$seed)"
        echo "----------------------------------------------"

        python -m ablations.scripts.train \
            --agent_type $agent \
            --seed $seed \
            --project $PROJECT \
            --run_name "${agent}_seed${seed}" \
            --num_updates $NUM_UPDATES \
            --use_accel \
            --use_probe

        echo "Completed: $agent (seed=$seed)"
    done
done

echo ""
echo "=============================================="
echo "All training runs complete!"
echo "=============================================="
