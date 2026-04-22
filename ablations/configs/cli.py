"""
Shared argparse builders and config resolution.

All entry points (train_with_experiments, run_all, run_experiment, train_all)
import from here instead of maintaining their own argparse definitions.
"""

import argparse
import json
from typing import Dict, Any, Optional

from .defaults import get_default_config
from .presets import (
    ALL_TRAINING_METHODS,
    ALL_AGENTS,
    TRAINING_METHOD_CONFIGS,
    AGENT_CONFIGS,
)
from .experiment_defaults import (
    get_all_experiment_defaults,
    get_experiments_for_method,
    TRAINING_TIME_SET,
)


# =============================================================================
# COMPOSABLE ARGPARSE BUILDERS
# =============================================================================

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by all entry points."""
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--training_method", type=str, default="accel",
        choices=ALL_TRAINING_METHODS,
        help="Training method for curriculum learning",
    )
    parser.add_argument(
        "--agent_type", type=str, default=None,
        help="Agent type (e.g., persistent_lstm, paired_persistent_lstm)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Load config from JSON file (CLI args override)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Base output directory (default: current working directory)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print configuration without running",
    )


def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add training / PPO / PLR / ACCEL / environment / eval / checkpoint / probe args."""
    # Run config
    parser.add_argument("--project", type=str, default="JaxUED-minigrid-maze",
                        help="Wandb project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Wandb run name (default: {method}_{agent}_seed{seed})")
    parser.add_argument("--group_name", type=str, default=None,
                        help="Wandb group name (groups related runs for comparison)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--checkpoint_directory", type=str, default=None,
                        help="Checkpoint directory for eval mode")
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1,
                        help="Specific checkpoint index to evaluate (-1 = latest)")

    # PPO / Training hyperparams
    train_group = parser.add_argument_group("Training params")
    train_group.add_argument("--lr", type=float, default=1e-4)
    train_group.add_argument("--max_grad_norm", type=float, default=0.5)
    mut_group = train_group.add_mutually_exclusive_group()
    mut_group.add_argument("--num_updates", type=int, default=30000)
    mut_group.add_argument("--num_env_steps", type=int, default=None,
                           help="Total env steps (alternative to --num_updates)")
    train_group.add_argument("--num_steps", type=int, default=256,
                             help="Rollout length per environment")
    train_group.add_argument("--num_train_envs", type=int, default=32,
                             help="Number of parallel training environments")
    train_group.add_argument("--num_minibatches", type=int, default=1)
    train_group.add_argument("--gamma", type=float, default=0.995)
    train_group.add_argument("--epoch_ppo", type=int, default=5)
    train_group.add_argument("--clip_eps", type=float, default=0.2)
    train_group.add_argument("--gae_lambda", type=float, default=0.98)
    train_group.add_argument("--entropy_coeff", type=float, default=1e-3)
    train_group.add_argument("--critic_coeff", type=float, default=0.5)

    # PLR
    plr_group = parser.add_argument_group("PLR params")
    plr_group.add_argument("--score_function", type=str, default="MaxMC",
                           choices=["MaxMC", "pvl"])
    plr_group.add_argument("--exploratory_grad_updates",
                           action=argparse.BooleanOptionalAction, default=False)
    plr_group.add_argument("--level_buffer_capacity", type=int, default=4000)
    plr_group.add_argument("--replay_prob", type=float, default=0.8)
    plr_group.add_argument("--staleness_coeff", type=float, default=0.3)
    plr_group.add_argument("--temperature", type=float, default=0.3)
    plr_group.add_argument("--top_k", type=int, default=4)
    plr_group.add_argument("--minimum_fill_ratio", type=float, default=0.5)
    plr_group.add_argument("--prioritization", type=str, default="rank",
                           choices=["rank", "topk"])
    plr_group.add_argument("--buffer_duplicate_check",
                           action=argparse.BooleanOptionalAction, default=True)

    # ACCEL
    accel_group = parser.add_argument_group("ACCEL params")
    accel_group.add_argument("--use_accel",
                             action=argparse.BooleanOptionalAction, default=False)
    accel_group.add_argument("--num_edits", type=int, default=5)

    # PAIRED
    paired_group = parser.add_argument_group("PAIRED params")
    paired_group.add_argument("--adv_num_steps", type=int, default=None,
                              help="Adversary rollout steps per level")
    paired_group.add_argument("--adv_lr", type=float, default=None,
                              help="Adversary learning rate")
    paired_group.add_argument("--adv_max_grad_norm", type=float, default=None,
                              help="Adversary max gradient norm")
    paired_group.add_argument("--adv_num_minibatches", type=int, default=None,
                              help="Adversary number of minibatches")
    paired_group.add_argument("--adv_gamma", type=float, default=None,
                              help="Adversary discount factor")
    paired_group.add_argument("--adv_epoch_ppo", type=int, default=None,
                              help="Adversary PPO epochs per update")
    paired_group.add_argument("--adv_clip_eps", type=float, default=None,
                              help="Adversary PPO clip epsilon")
    paired_group.add_argument("--adv_gae_lambda", type=float, default=None,
                              help="Adversary GAE lambda")
    paired_group.add_argument("--adv_entropy_coeff", type=float, default=None,
                              help="Adversary entropy coefficient")
    paired_group.add_argument("--adv_critic_coeff", type=float, default=None,
                              help="Adversary critic loss coefficient")
    paired_group.add_argument("--adv_random_z_dimension", type=int, default=None,
                              help="Adversary random noise dimension")
    paired_group.add_argument("--adv_zero_out_random_z",
                              action=argparse.BooleanOptionalAction, default=None,
                              help="Zero out adversary random noise")

    # Environment
    env_group = parser.add_argument_group("Environment params")
    env_group.add_argument("--agent_view_size", type=int, default=5)
    env_group.add_argument("--n_walls", type=int, default=25)

    # Evaluation
    parser.add_argument("--eval_freq", type=int, default=250,
                        help="Eval (and experiment) frequency in training updates")
    parser.add_argument("--eval_num_attempts", type=int, default=10)
    parser.add_argument("--eval_levels", nargs="+", default=[
        "SixteenRooms", "SixteenRooms2",
        "Labyrinth", "LabyrinthFlipped", "Labyrinth2",
        "StandardMaze", "StandardMaze2", "StandardMaze3",
    ])

    # Checkpointing
    parser.add_argument("--checkpoint_save_interval", type=int, default=2)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)

    # Probe config
    probe_group = parser.add_argument_group("Probe params")
    probe_group.add_argument("--use_probe",
                             action=argparse.BooleanOptionalAction, default=True)
    probe_group.add_argument("--probe_lr", type=float, default=1e-3)
    probe_group.add_argument("--probe_tracking_buffer_size", type=int, default=500)

    # Prediction / curriculum prediction
    pred_group = parser.add_argument_group("Prediction params")
    pred_group.add_argument("--use_curriculum_prediction",
                            action=argparse.BooleanOptionalAction, default=False,
                            help="Enable curriculum prediction head")
    pred_group.add_argument("--curriculum_hidden_size", type=int, default=128,
                            help="Hidden size for curriculum prediction network")
    pred_group.add_argument("--curriculum_pred_coeff", type=float, default=1.0,
                            help="Loss coefficient for curriculum prediction")
    pred_group.add_argument("--curriculum_pred_eval_freq", type=int, default=100,
                            help="Eval frequency for curriculum prediction")
    pred_group.add_argument("--n_prediction_eval", type=int, default=20,
                            help="Number of prediction evaluation episodes")
    pred_group.add_argument("--wall_loss_region", type=str, default="full",
                            choices=["full", "explored", "frontier"])
    pred_group.add_argument("--curriculum_history_length", type=int, default=64)
    pred_group.add_argument("--curriculum_wall_weight", type=float, default=1.0)
    pred_group.add_argument("--curriculum_goal_weight", type=float, default=1.0)
    pred_group.add_argument("--curriculum_agent_pos_weight", type=float, default=1.0)
    pred_group.add_argument("--curriculum_agent_dir_weight", type=float, default=1.0)


def add_experiment_selection_args(parser: argparse.ArgumentParser) -> None:
    """Add --experiments / --no_experiments."""
    exp_group = parser.add_argument_group("Experiment selection")
    exp_group.add_argument(
        "--experiments", type=str, nargs="+", default=None,
        help="Specific experiments to run (auto-partitions checkpoint vs "
             "training-time). Default: ALL applicable.",
    )
    exp_group.add_argument(
        "--no_experiments", action="store_true",
        help="Disable all experiment running (just train)",
    )


def add_experiment_param_args(parser: argparse.ArgumentParser) -> None:
    """Add legacy flat overrides that apply to ALL experiments using that key."""
    exp_params = parser.add_argument_group("Experiment params")
    exp_params.add_argument("--n_levels", type=int, default=None,
                            help="Number of levels for experiments")
    exp_params.add_argument("--max_steps", type=int, default=None,
                            help="Max rollout steps for experiments")
    exp_params.add_argument("--n_samples", type=int, default=None,
                            help="Number of samples for experiments")
    exp_params.add_argument("--n_episodes", type=int, default=None,
                            help="Number of episodes for experiments")
    exp_params.add_argument("--adv_num_steps", type=int, default=None,
                            help="Adversary rollout steps for PAIRED experiments")


# Legacy flat-override keys (the keys added by add_experiment_param_args)
EXPERIMENT_PARAM_KEYS = ["n_levels", "max_steps", "n_samples", "n_episodes", "adv_num_steps"]


def add_posthoc_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to post-hoc experiment runners."""
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory containing trained agent checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a single checkpoint")
    parser.add_argument("--agents", type=str, nargs="+", default=None,
                        help="Agent types to run (default: method-appropriate)")
    parser.add_argument("--checkpoints_per_agent", type=int, default=None,
                        help="Max checkpoints per agent (default: all)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel workers")


# =============================================================================
# CONFIG RESOLUTION
# =============================================================================

def build_config_from_args(
    args: argparse.Namespace,
    include_experiments: bool = True,
) -> Dict[str, Any]:
    """Build a fully-resolved config dict from parsed CLI args.

    Override order:
      get_default_config()
        -> method config (TRAINING_METHOD_CONFIGS)
        -> agent config (AGENT_CONFIGS)
        -> experiment defaults (exp.* namespaced)
        -> JSON file (--config)
        -> CLI args (highest precedence)

    Args:
        args: Parsed argparse namespace
        include_experiments: Whether to merge experiment defaults (exp.* keys)
    """
    config = get_default_config()

    # Layer 1: training method config
    training_method = getattr(args, "training_method", "accel")
    if training_method in TRAINING_METHOD_CONFIGS:
        config.update(TRAINING_METHOD_CONFIGS[training_method])
    config["training_method"] = training_method

    # Layer 2: agent config
    agent_type = getattr(args, "agent_type", None)
    if agent_type and agent_type in AGENT_CONFIGS:
        config.update(AGENT_CONFIGS[agent_type])
    if agent_type:
        config["agent_type"] = agent_type

    # Layer 3: experiment defaults (namespaced)
    if include_experiments:
        config.update(get_all_experiment_defaults())

    # Layer 4: JSON config file
    json_config_path = getattr(args, "config", None)
    if json_config_path:
        with open(json_config_path) as f:
            config.update(json.load(f))

    # Layer 5: CLI args — only override if the user explicitly set them.
    # We collect all known CLI keys and apply non-None values.
    _cli_keys = [
        "seed", "training_method", "agent_type", "output_dir",
        # Training
        "project", "run_name", "group_name", "mode", "checkpoint_directory", "checkpoint_to_eval",
        "lr", "max_grad_norm", "num_updates", "num_steps", "num_train_envs",
        "num_minibatches", "gamma", "epoch_ppo", "clip_eps", "gae_lambda",
        "entropy_coeff", "critic_coeff",
        # PLR
        "score_function", "exploratory_grad_updates", "level_buffer_capacity",
        "replay_prob", "staleness_coeff", "temperature", "top_k",
        "minimum_fill_ratio", "prioritization", "buffer_duplicate_check",
        # ACCEL
        "use_accel", "num_edits",
        # PAIRED
        "adv_num_steps", "adv_lr", "adv_max_grad_norm", "adv_num_minibatches",
        "adv_gamma", "adv_epoch_ppo", "adv_clip_eps", "adv_gae_lambda",
        "adv_entropy_coeff", "adv_critic_coeff",
        "adv_random_z_dimension", "adv_zero_out_random_z",
        # Environment
        "agent_view_size", "n_walls",
        # Eval
        "eval_freq", "eval_num_attempts", "eval_levels",
        # Checkpointing
        "checkpoint_save_interval", "max_number_of_checkpoints",
        # Probe
        "use_probe", "probe_lr", "probe_tracking_buffer_size",
        # Prediction / curriculum
        "use_curriculum_prediction", "curriculum_hidden_size",
        "curriculum_pred_coeff", "curriculum_pred_eval_freq", "n_prediction_eval",
        "wall_loss_region",
        "curriculum_history_length", "curriculum_wall_weight",
        "curriculum_goal_weight", "curriculum_agent_pos_weight",
        "curriculum_agent_dir_weight",
    ]

    for key in _cli_keys:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    # Experiment param overrides (flat, only if explicitly set)
    for key in EXPERIMENT_PARAM_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    # Handle num_env_steps -> num_updates conversion
    num_env_steps = getattr(args, "num_env_steps", None)
    if num_env_steps is not None:
        config["num_updates"] = num_env_steps // (
            config["num_train_envs"] * config["num_steps"]
        )

    return config


def get_experiment_param_overrides(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract experiment-param overrides from config (only if explicitly set)."""
    overrides = {k: config[k] for k in EXPERIMENT_PARAM_KEYS if k in config}
    return overrides or None
