#PYTHONPATH=$(pwd)
#export PYTHONPATH
#SHARED="--no-use_accel --n_walls 25 --eval_freq 500 --project dcd --num_updates 30000 --lr_annealing --tmaze --tmaze_not_goal_reward -1"
#
#for j in {0..9}; do
#    for i in 0; do
#        CUDA_VISIBLE_DEVICES=$i python x_minigrid_remidi.py \
#        --seed $((i+1*j)) \
#        --run_name tmaze_remidi \
#        --replay_prob 0.8 \
#        --score_function=perfect_regret \
#        --entropy_coeff 0.0 \
#        --gae_lambda 0.95 \
#        --lr 0.001 \
#        --temperature 1.0 \
#        --tau_buffer_size 4 \
#        --level_buffer_capacity 256 \
#        --stay_at_last \
#        --number_of_replays_per_buffer 1000 \
#        "$SHARED" &
#    done
#    wait;
#done
#wait;


PYTHONPATH=$(pwd)
export PYTHONPATH
SHARED="--no-use_accel --n_walls 25 --eval_freq 5 --project dcd --num_updates 25 --lr_annealing --tmaze --tmaze_not_goal_reward -1"

# Changed {0..9} to {0..0} for single run instead of 10
for j in {0..0}; do
  for i in 0; do
    CUDA_VISIBLE_DEVICES=$i python x_minigrid_remidi.py \
     --seed $j \
     --run_name tmaze_remidi_test \
     --replay_prob 0.8 \
     --score_function=perfect_regret \
     --entropy_coeff 0.0 \
     --gae_lambda 0.95 \
     --lr 0.001 \
     --temperature 1.0 \
     --tau_buffer_size 2 \
     --level_buffer_capacity 50 \
     --stay_at_last \
     --number_of_replays_per_buffer 5 \
     --num_train_envs 6 \
     --num_steps 32 \
     --checkpoint_save_interval 0 \
     --num_outer_adversaries 2 \
     $SHARED &
  done
  wait;
done;
wait;