"""
Utility functions for curriculum awareness ablations.

Contains:
- Checkpointing utilities
- Logging utilities
- Configuration parsing
- Misc helpers
"""

import os
import json
from typing import Optional
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import chex

from jaxued.level_sampler import LevelSampler


def setup_checkpointing(
    config: dict,
    run_name: str,
    seed: int,
) -> ocp.CheckpointManager:
    """
    Setup Orbax checkpoint manager.

    Args:
        config: Config dictionary
        run_name: Name of the run
        seed: Random seed

    Returns:
        CheckpointManager instance

    Directory structure:
        checkpoints/{training_method}/{agent_type}/{seed}/
        - config.json
        - models/
    """
    training_method = config.get("training_method", "accel")
    agent_type = config.get("agent_type", "accel_probe")

    overall_save_dir = os.path.join(
        os.getcwd(), "checkpoints", training_method, agent_type, str(seed)
    )
    os.makedirs(overall_save_dir, exist_ok=True)

    with open(os.path.join(overall_save_dir, "config.json"), "w+") as f:
        f.write(json.dumps(config, indent=True))

    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, "models"),
        ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config.get("checkpoint_save_interval", 2),
            max_to_keep=config.get("max_number_of_checkpoints", 60),
        ),
    )
    return checkpoint_manager


def load_checkpoint(
    checkpoint_dir: str,
    train_state_template,
    step: int = -1,
) -> tuple:
    """
    Load checkpoint from directory.

    Args:
        checkpoint_dir: Path to checkpoint directory
        train_state_template: Template train state for structure
        step: Checkpoint step to load (-1 for latest)

    Returns:
        (train_state, config)
    """
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        config = json.load(f)

    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(checkpoint_dir, "models"),
        ocp.PyTreeCheckpointer(),
    )

    if step == -1:
        step = checkpoint_manager.latest_step()

    loaded_checkpoint = checkpoint_manager.restore(step)
    params = loaded_checkpoint["params"]
    train_state = train_state_template.replace(params=params)

    # Load probe params if present
    if "probe_params" in loaded_checkpoint:
        train_state = train_state.replace(probe_params=loaded_checkpoint["probe_params"])

    return train_state, config


def train_state_to_log_dict(
    train_state,
    level_sampler: LevelSampler,
) -> dict:
    """
    Extract loggable information from train state.

    Prevents copying entire train state to CPU for logging.
    """
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)

    return {
        "log": {
            "level_sampler/size": sampler["size"],
            'level_sampler/episode_count': sampler["episode_count"],
            "level_sampler/max_score": sampler["scores"].max(),
            "level_sampler/weighted_score": (sampler["scores"] * level_sampler.level_weights(sampler)).sum(),
            "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
        },
        "info": {
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        }
    }


def flatten_hstate(hstate: tuple) -> chex.Array:
    """Flatten LSTM hidden state (c, h) tuple to single array."""
    h_c, h_h = hstate
    if h_c.ndim == 2:  # (batch, features)
        return jnp.concatenate([h_c, h_h], axis=-1)
    elif h_c.ndim == 1:  # (features,)
        return jnp.concatenate([h_c, h_h], axis=-1)
    else:
        raise ValueError(f"Unexpected hstate shape: {h_c.shape}")


def unflatten_hstate(hstate_flat: chex.Array, feature_dim: int = 256) -> tuple:
    """Unflatten single array back to LSTM hidden state tuple."""
    if hstate_flat.ndim == 2:  # (batch, 2*features)
        h_c = hstate_flat[:, :feature_dim]
        h_h = hstate_flat[:, feature_dim:]
    elif hstate_flat.ndim == 1:  # (2*features,)
        h_c = hstate_flat[:feature_dim]
        h_h = hstate_flat[feature_dim:]
    else:
        raise ValueError(f"Unexpected hstate_flat shape: {hstate_flat.shape}")
    return (h_c, h_h)


def get_default_config() -> dict:
    """Get default configuration dictionary."""
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
        "checkpoint_save_interval": 1,
        "max_number_of_checkpoints": 120,

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


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Curriculum Awareness Ablation Study")

    # Run config
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="ablation")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--project", type=str, default="curriculum-awareness-ablation")

    # Training method
    parser.add_argument("--training_method", type=str, default="accel",
                        choices=["accel", "plr", "robust_plr", "dr", "paired"],
                        help="Training method for curriculum learning")

    # Agent type
    parser.add_argument("--agent_type", type=str, default="accel_probe",
                        choices=["accel_probe", "persistent_lstm", "context_vector", "episodic_memory", "next_env_prediction"])

    # Training
    parser.add_argument("--num_updates", type=int, default=30000)
    parser.add_argument("--num_train_envs", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--num_minibatches", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--epoch_ppo", type=int, default=5)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--gae_lambda", type=float, default=0.98)
    parser.add_argument("--entropy_coeff", type=float, default=1e-3)
    parser.add_argument("--critic_coeff", type=float, default=0.5)

    # PLR
    parser.add_argument("--score_function", type=str, default="MaxMC", choices=["MaxMC", "pvl"])
    parser.add_argument("--exploratory_grad_updates", action="store_true", default=False)
    parser.add_argument("--level_buffer_capacity", type=int, default=4000)
    parser.add_argument("--replay_prob", type=float, default=0.8)
    parser.add_argument("--staleness_coeff", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--minimum_fill_ratio", type=float, default=0.5)
    parser.add_argument("--prioritization", type=str, default="rank", choices=["rank", "topk"])
    parser.add_argument("--no_buffer_duplicate_check", action="store_true", default=False)

    # ACCEL
    parser.add_argument("--use_accel", action="store_true", default=True)
    parser.add_argument("--no_accel", action="store_true", default=False)
    parser.add_argument("--num_edits", type=int, default=5)

    # Environment
    parser.add_argument("--agent_view_size", type=int, default=5)
    parser.add_argument("--n_walls", type=int, default=25)

    # Evaluation
    parser.add_argument("--eval_freq", type=int, default=250)
    parser.add_argument("--eval_num_attempts", type=int, default=10)
    parser.add_argument("--eval_levels", nargs="+", default=[
        "SixteenRooms", "SixteenRooms2",
        "Labyrinth", "LabyrinthFlipped", "Labyrinth2",
        "StandardMaze", "StandardMaze2", "StandardMaze3",
    ])

    # Checkpointing
    parser.add_argument("--checkpoint_save_interval", type=int, default=2)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)
    parser.add_argument("--checkpoint_directory", type=str, default=None)
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1)

    # Probe config
    parser.add_argument("--use_probe", action="store_true", default=True)
    parser.add_argument("--no_probe", action="store_true", default=False)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_tracking_buffer_size", type=int, default=500)

    # Memory-specific
    parser.add_argument("--context_dim", type=int, default=64)
    parser.add_argument("--context_decay", type=float, default=0.9)
    parser.add_argument("--memory_buffer_size", type=int, default=64)
    parser.add_argument("--memory_top_k", type=int, default=8)

    # Post-training evaluation
    parser.add_argument("--n_env_predictions", type=int, default=100)

    # Curriculum prediction args (for next_env_prediction agent)
    parser.add_argument("--prediction_coeff", type=float, default=0.1,
                        help="Weight for curriculum prediction loss")
    parser.add_argument("--curriculum_history_length", type=int, default=64,
                        help="Length of curriculum history buffer")
    parser.add_argument("--curriculum_wall_weight", type=float, default=1.0,
                        help="Weight for wall prediction loss")
    parser.add_argument("--curriculum_goal_weight", type=float, default=1.0,
                        help="Weight for goal position prediction loss")
    parser.add_argument("--curriculum_agent_pos_weight", type=float, default=1.0,
                        help="Weight for agent position prediction loss")
    parser.add_argument("--curriculum_agent_dir_weight", type=float, default=1.0,
                        help="Weight for agent direction prediction loss")
    parser.add_argument("--nl_buffer_size", type=int, default=100,
                        help="Buffer size for novelty/learnability tracking")

    # PAIRED-specific arguments (when training_method == "paired")
    parser.add_argument("--adv_random_z_dimension", type=int, default=16,
                        help="Dimension of random z input for adversary diversity")
    parser.add_argument("--adv_zero_out_random_z", action="store_true", default=False,
                        help="Zero out random z (for ablation)")
    parser.add_argument("--adv_num_steps", type=int, default=50,
                        help="Number of steps for adversary level generation")
    parser.add_argument("--adv_lr", type=float, default=1e-4,
                        help="Adversary learning rate")
    parser.add_argument("--adv_max_grad_norm", type=float, default=0.5,
                        help="Adversary gradient clipping")
    parser.add_argument("--adv_num_minibatches", type=int, default=1,
                        help="Adversary minibatches")
    parser.add_argument("--adv_gamma", type=float, default=0.995,
                        help="Adversary discount factor")
    parser.add_argument("--adv_epoch_ppo", type=int, default=5,
                        help="Adversary PPO epochs")
    parser.add_argument("--adv_clip_eps", type=float, default=0.2,
                        help="Adversary PPO clip epsilon")
    parser.add_argument("--adv_gae_lambda", type=float, default=0.98,
                        help="Adversary GAE lambda")
    parser.add_argument("--adv_entropy_coeff", type=float, default=1e-3,
                        help="Adversary entropy coefficient")
    parser.add_argument("--adv_critic_coeff", type=float, default=0.5,
                        help="Adversary critic loss coefficient")

    # Student (protagonist/antagonist) args for PAIRED
    parser.add_argument("--student_num_steps", type=int, default=256,
                        help="Number of steps for student rollouts in PAIRED")
    parser.add_argument("--student_lr", type=float, default=1e-4,
                        help="Student learning rate in PAIRED")
    parser.add_argument("--student_max_grad_norm", type=float, default=0.5,
                        help="Student gradient clipping in PAIRED")
    parser.add_argument("--student_num_minibatches", type=int, default=1,
                        help="Student minibatches in PAIRED")
    parser.add_argument("--student_gamma", type=float, default=0.995,
                        help="Student discount factor in PAIRED")
    parser.add_argument("--student_epoch_ppo", type=int, default=5,
                        help="Student PPO epochs in PAIRED")
    parser.add_argument("--student_clip_eps", type=float, default=0.2,
                        help="Student PPO clip epsilon in PAIRED")
    parser.add_argument("--student_gae_lambda", type=float, default=0.98,
                        help="Student GAE lambda in PAIRED")
    parser.add_argument("--student_entropy_coeff", type=float, default=1e-3,
                        help="Student entropy coefficient in PAIRED")
    parser.add_argument("--student_critic_coeff", type=float, default=0.5,
                        help="Student critic loss coefficient in PAIRED")

    args = parser.parse_args()

    # Handle negation flags
    config = vars(args)
    if config.pop("no_accel"):
        config["use_accel"] = False
    if config.pop("no_probe"):
        config["use_probe"] = False
    config["buffer_duplicate_check"] = not config.pop("no_buffer_duplicate_check")

    # Set use_accel and exploratory_grad_updates based on training_method for backward compatibility
    training_method = config.get("training_method", "accel")
    if training_method == "accel":
        config["use_accel"] = True
        config["exploratory_grad_updates"] = True
    elif training_method == "plr":
        config["use_accel"] = False
        config["exploratory_grad_updates"] = True
    elif training_method == "robust_plr":
        config["use_accel"] = False
        config["exploratory_grad_updates"] = False
    elif training_method == "dr":
        config["use_accel"] = False
        config["exploratory_grad_updates"] = True  # Always update in DR
    elif training_method == "paired":
        config["use_accel"] = False
        config["exploratory_grad_updates"] = True

    return config
