"""
Training defaults for the ablation study.

This is the single source of truth for all default training configuration values.
Previously lived in common/utils.py as get_default_config().
"""

from typing import Dict, Any


def get_default_config() -> Dict[str, Any]:
    """Get default configuration dictionary.

    These are base training defaults. They are overridden (in order) by:
      1. Training method config (presets.py TRAINING_METHOD_CONFIGS)
      2. Agent config (presets.py AGENT_CONFIGS)
      3. Experiment defaults (experiment_defaults.py, namespaced as exp.*)
      4. JSON config file (--config)
      5. Checkpoint config.json (post-hoc runners only)
      6. CLI arguments (highest precedence)
    """
    return {
        # Run config
        "seed": 0,
        "run_name": "ablation",
        "mode": "train",

        # Agent type
        "agent_type": "accel_probe",

        # Training method: accel, plr, robust_plr, dr, paired
        "training_method": "accel",

        # Training
        "num_updates": 30000,
        "num_train_envs": 32,
        "num_steps": 256,
        "lr": 1e-4,
        "max_grad_norm": 0.5,
        "num_minibatches": 1,
        "gamma": 0.995,
        "epoch_ppo": 5,
        "clip_eps": 0.2,
        "gae_lambda": 0.98,
        "entropy_coeff": 1e-3,
        "critic_coeff": 0.5,

        # PLR
        "score_function": "MaxMC",
        "exploratory_grad_updates": False,  # True = PLR, False = Robust-PLR
        "level_buffer_capacity": 4000,
        "replay_prob": 0.8,
        "staleness_coeff": 0.3,
        "temperature": 0.3,
        "top_k": 4,
        "minimum_fill_ratio": 0.5,
        "prioritization": "rank",
        "buffer_duplicate_check": True,

        # ACCEL
        "use_accel": True,
        "num_edits": 5,

        # Environment
        "agent_view_size": 5,
        "n_walls": 25,

        # Evaluation
        "eval_freq": 250,
        "eval_num_attempts": 10,
        "eval_levels": [
            "SixteenRooms", "SixteenRooms2",
            "Labyrinth", "LabyrinthFlipped", "Labyrinth2",
            "StandardMaze", "StandardMaze2", "StandardMaze3",
        ],

        # Checkpointing
        "checkpoint_save_interval": 2,
        "max_number_of_checkpoints": 60,

        # Probe config
        "use_probe": True,
        "probe_lr": 1e-3,
        "probe_tracking_buffer_size": 500,

        # Memory-specific (for variants)
        "context_dim": 64,
        "context_decay": 0.9,
        "memory_buffer_size": 64,
        "memory_top_k": 8,

        # Post-training evaluation
        "n_env_predictions": 100,

        # Curriculum prediction config (for next_env_prediction agent)
        "use_curriculum_prediction": False,
        "curriculum_hidden_size": 128,
        "curriculum_pred_coeff": 1.0,
        "curriculum_pred_eval_freq": 100,
        "n_prediction_eval": 20,
        "wall_loss_region": "full",
        "prediction_coeff": 0.1,
        "curriculum_history_length": 64,
        "curriculum_wall_weight": 1.0,
        "curriculum_goal_weight": 1.0,
        "curriculum_agent_pos_weight": 1.0,
        "curriculum_agent_dir_weight": 1.0,
        "nl_buffer_size": 100,

        # PAIRED-specific config (when training_method == "paired")
        "adv_random_z_dimension": 16,
        "adv_zero_out_random_z": False,
        "adv_num_steps": 50,
        "adv_lr": 1e-4,
        "adv_max_grad_norm": 0.5,
        "adv_num_minibatches": 1,
        "adv_gamma": 0.995,
        "adv_epoch_ppo": 5,
        "adv_clip_eps": 0.2,
        "adv_gae_lambda": 0.98,
        "adv_entropy_coeff": 1e-3,
        "adv_critic_coeff": 0.5,
        # Student (protagonist/antagonist) hyperparams for PAIRED
        "student_num_steps": 256,
        "student_lr": 1e-4,
        "student_max_grad_norm": 0.5,
        "student_num_minibatches": 1,
        "student_gamma": 0.995,
        "student_epoch_ppo": 5,
        "student_clip_eps": 0.2,
        "student_gae_lambda": 0.98,
        "student_entropy_coeff": 1e-3,
        "student_critic_coeff": 0.5,
    }
