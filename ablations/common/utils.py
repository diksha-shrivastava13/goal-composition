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

    base_dir = config.get("output_dir", os.getcwd())
    overall_save_dir = os.path.join(
        base_dir, "checkpoints", training_method, agent_type, str(seed)
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

    if "params" in loaded_checkpoint:
        # Standard (non-PAIRED) checkpoint
        train_state = train_state_template.replace(params=loaded_checkpoint["params"])
    elif "pro_params" in loaded_checkpoint:
        # PAIRED checkpoint with 3 network params
        train_state = train_state_template.replace(
            pro_train_state=train_state_template.pro_train_state.replace(
                params=loaded_checkpoint["pro_params"]
            ),
            ant_train_state=train_state_template.ant_train_state.replace(
                params=loaded_checkpoint["ant_params"]
            ),
            adv_train_state=train_state_template.adv_train_state.replace(
                params=loaded_checkpoint["adv_params"]
            ),
        )
    else:
        train_state = train_state_template

    # Load probe params if present and train state supports it
    if "probe_params" in loaded_checkpoint and hasattr(train_state, 'probe_params'):
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
    """Get default configuration dictionary.

    .. deprecated:: Moved to ablations.configs.defaults.get_default_config().
        This shim exists for backward compatibility.
    """
    from ablations.configs.defaults import get_default_config as _get_default_config
    return _get_default_config()


def parse_args():
    """Parse command line arguments.

    .. deprecated:: Replaced by ablations.configs.cli.build_config_from_args().
        This shim exists for backward compatibility.
    """
    from ablations.configs.cli import (
        add_common_args,
        add_training_args,
        add_experiment_param_args,
        build_config_from_args,
    )
    import argparse

    parser = argparse.ArgumentParser(description="Curriculum Awareness Ablation Study")
    add_common_args(parser)
    add_training_args(parser)
    add_experiment_param_args(parser)
    args = parser.parse_args()
    return build_config_from_args(args, include_experiments=False)
