"""
Type definitions and dataclasses for curriculum awareness ablations.

This module defines the TrainState hierarchy and tracking states used across
all agent variants.
"""

from typing import Optional
from enum import IntEnum, Enum
import jax.numpy as jnp
import chex
from flax import core, struct
from flax.training.train_state import TrainState as FlaxTrainState


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_ENV_HEIGHT = 13
DEFAULT_ENV_WIDTH = 13
DEFAULT_HSTATE_DIM = 512  # LSTM c + h, each 256
DEFAULT_CALIBRATION_BINS = 10


class UpdateState(IntEnum):
    """ACCEL curriculum branch states."""
    DR = 0         # Domain Randomization (new random levels)
    REPLAY = 1     # Replay from buffer
    MUTATION = 2   # Mutation of replayed levels (ACCEL)


class TrainingMethod(str, Enum):
    """Training method for curriculum learning.

    Controls how levels are sampled and whether gradient updates are applied.

    Methods:
        ACCEL: DR → Replay → Mutate cycle with gradient updates on all branches
        PLR: DR + Replay with exploratory gradient updates on DR
        ROBUST_PLR: DR + Replay without gradient updates on DR (robust exploration)
        DR: Pure domain randomization - continuous rollouts with random levels
        PAIRED: Adversarial level generation with protagonist/antagonist agents
    """
    ACCEL = "accel"           # DR → Replay → Mutate cycle
    PLR = "plr"               # DR + Replay with exploratory grads
    ROBUST_PLR = "robust_plr" # DR + Replay without exploratory grads
    DR = "dr"                 # Pure domain randomization
    PAIRED = "paired"         # Adversarial level generation


# =============================================================================
# PROBE TRACKING STATE
# =============================================================================

@struct.dataclass
class ProbeTrackingState:
    """
    Tracks probe metrics over time for computing novelty, learnability,
    calibration, and correlation with agent performance.
    """
    # Rolling buffer of probe losses with associated training steps
    loss_history: chex.Array              # (buffer_size,)
    training_step_history: chex.Array     # (buffer_size,)
    # Per-branch loss history for comparison
    branch_loss_history: chex.Array       # (3, buffer_size) - [DR, replay, mutate]
    branch_step_history: chex.Array       # (3, buffer_size)
    # Agent performance correlation tracking
    agent_returns_history: chex.Array     # (buffer_size,)
    probe_accuracy_history: chex.Array    # (buffer_size,)
    # Hidden state statistics per branch
    hstate_mean_by_branch: chex.Array     # (3, hstate_dim)
    hstate_var_by_branch: chex.Array      # (3, hstate_dim)
    hstate_count_by_branch: chex.Array    # (3,)
    # Sample counts per branch
    branch_sample_counts: chex.Array      # (3,)
    # Buffer management
    buffer_ptr: int
    total_samples: int
    current_training_step: int
    branch_ptrs: chex.Array               # (3,)
    hstate_dim: int

    # =====================================================================
    # Per-instance tracking for R→M and PAIRED (1-to-1 correspondence)
    # =====================================================================
    # Per-instance accuracy tracking (meaningful only for 1-to-1 cases)
    per_instance_wall_accuracy_history: chex.Array     # (buffer_size,)
    per_instance_goal_accuracy_history: chex.Array     # (buffer_size,)
    per_instance_agent_pos_accuracy_history: chex.Array  # (buffer_size,)
    per_instance_agent_dir_accuracy_history: chex.Array  # (buffer_size,)
    per_instance_combined_accuracy_history: chex.Array  # (buffer_size,)
    per_instance_loss_history: chex.Array              # (buffer_size,)
    per_instance_ptr: int
    per_instance_total: int
    # Mask for valid per-instance entries (only R→M and PAIRED)
    is_per_instance_valid: chex.Array                  # (buffer_size,) bool

    # =====================================================================
    # R→M specific tracking (replay_to_mutate transition metrics)
    # =====================================================================
    replay_to_mutate_wall_accuracy: chex.Array         # (buffer_size,)
    replay_to_mutate_goal_accuracy: chex.Array         # (buffer_size,)
    replay_to_mutate_loss: chex.Array                  # (buffer_size,)
    replay_to_mutate_count: int
    replay_to_mutate_ptr: int

    # =====================================================================
    # R→M visualization data (last batch for heatmap visualization)
    # =====================================================================
    last_r2m_wall_logits: chex.Array          # (batch_size, height, width)
    last_r2m_goal_logits: chex.Array          # (batch_size, height*width)
    last_r2m_wall_map: chex.Array             # (batch_size, height, width)
    last_r2m_goal_pos: chex.Array             # (batch_size, 2)
    last_r2m_agent_pos: chex.Array            # (batch_size, 2)
    last_r2m_valid: bool                      # Whether visualization data is valid

    # =====================================================================
    # Distributional calibration tracking (per-branch)
    # =====================================================================
    dist_calibration_wall_history: chex.Array          # (3, buffer_size)
    dist_calibration_goal_history: chex.Array          # (3, buffer_size)
    dist_accuracy_wall_history: chex.Array             # (3, buffer_size)
    dist_accuracy_goal_mode_match_history: chex.Array  # (3, buffer_size)

    # =====================================================================
    # Per-component loss tracking (wall, goal, agent_pos, agent_dir separately)
    # =====================================================================
    wall_loss_history: chex.Array                      # (buffer_size,)
    goal_loss_history: chex.Array                      # (buffer_size,)
    agent_pos_loss_history: chex.Array                 # (buffer_size,)
    agent_dir_loss_history: chex.Array                 # (buffer_size,)

    # Per-branch per-component losses
    branch_wall_loss_history: chex.Array               # (3, buffer_size)
    branch_goal_loss_history: chex.Array               # (3, buffer_size)
    branch_agent_pos_loss_history: chex.Array          # (3, buffer_size)
    branch_agent_dir_loss_history: chex.Array          # (3, buffer_size)

    # Per-branch zone-decomposed wall losses
    branch_zone_wall_loss_observed: chex.Array        # (3, buffer_size)
    branch_zone_wall_loss_adjacent: chex.Array        # (3, buffer_size)
    branch_zone_wall_loss_distant: chex.Array         # (3, buffer_size)
    branch_zone_wall_accuracy_observed: chex.Array    # (3, buffer_size)
    branch_zone_wall_accuracy_adjacent: chex.Array    # (3, buffer_size)
    branch_zone_wall_accuracy_distant: chex.Array     # (3, buffer_size)

    # Per-branch aggregated accuracy (agent_pos, agent_dir, combined)
    dist_accuracy_agent_pos_history: chex.Array        # (3, buffer_size)
    dist_accuracy_agent_dir_history: chex.Array        # (3, buffer_size)
    dist_accuracy_combined_history: chex.Array         # (3, buffer_size)

    # =====================================================================
    # Last predictions and levels (for divergence, heatmaps, matched accuracy)
    # =====================================================================
    last_predictions_wall_logits: chex.Array           # (batch_size, height, width)
    last_predictions_goal_logits: chex.Array           # (batch_size, height*width)
    last_predictions_agent_pos_logits: chex.Array      # (batch_size, height*width)
    last_predictions_agent_dir_logits: chex.Array      # (batch_size, 4)
    last_levels_wall_map: chex.Array                   # (batch_size, height, width)
    last_levels_goal_pos: chex.Array                   # (batch_size, 2)
    last_levels_agent_pos: chex.Array                  # (batch_size, 2)
    last_levels_agent_dir: chex.Array                  # (batch_size,)
    last_branch: int                                   # Branch of last update

    # =====================================================================
    # Tier 1/2/3 prediction loss tracking
    # =====================================================================
    # Per-tier aggregate loss histories
    tier1_loss_history: chex.Array                     # (buffer_size,)
    tier2_loss_history: chex.Array                     # (buffer_size,)
    tier3_loss_history: chex.Array                     # (buffer_size,)

    # Per-component loss histories
    tier1_regret_loss_history: chex.Array              # (buffer_size,)
    tier1_difficulty_loss_history: chex.Array           # (buffer_size,)
    tier1_branch_loss_history: chex.Array              # (buffer_size,)
    tier1_score_loss_history: chex.Array               # (buffer_size,)
    tier2_return_loss_history: chex.Array              # (buffer_size,)
    tier2_novelty_loss_history: chex.Array             # (buffer_size,)
    tier2_unusualness_loss_history: chex.Array         # (buffer_size,)
    tier3_drift_loss_history: chex.Array               # (buffer_size,)

    # Per-branch per-tier losses
    branch_tier1_loss_history: chex.Array              # (3, buffer_size)
    branch_tier2_loss_history: chex.Array              # (3, buffer_size)
    branch_tier3_loss_history: chex.Array              # (3, buffer_size)


def create_probe_tracking_state(
    buffer_size: int = 500,
    hstate_dim: int = DEFAULT_HSTATE_DIM,
    batch_size: int = 32,
    env_height: int = 13,
    env_width: int = 13,
) -> ProbeTrackingState:
    """Initialize probe tracking state with all tracking fields."""
    return ProbeTrackingState(
        # Original fields
        loss_history=jnp.zeros(buffer_size),
        training_step_history=jnp.zeros(buffer_size, dtype=jnp.int32),
        branch_loss_history=jnp.zeros((3, buffer_size)),
        branch_step_history=jnp.zeros((3, buffer_size), dtype=jnp.int32),
        agent_returns_history=jnp.zeros(buffer_size),
        probe_accuracy_history=jnp.zeros(buffer_size),
        hstate_mean_by_branch=jnp.zeros((3, hstate_dim)),
        hstate_var_by_branch=jnp.zeros((3, hstate_dim)),
        hstate_count_by_branch=jnp.zeros(3),
        branch_sample_counts=jnp.zeros(3, dtype=jnp.int32),
        buffer_ptr=0,
        total_samples=0,
        current_training_step=0,
        branch_ptrs=jnp.zeros(3, dtype=jnp.int32),
        hstate_dim=hstate_dim,
        # Per-instance tracking (for R→M and PAIRED)
        per_instance_wall_accuracy_history=jnp.zeros(buffer_size),
        per_instance_goal_accuracy_history=jnp.zeros(buffer_size),
        per_instance_agent_pos_accuracy_history=jnp.zeros(buffer_size),
        per_instance_agent_dir_accuracy_history=jnp.zeros(buffer_size),
        per_instance_combined_accuracy_history=jnp.zeros(buffer_size),
        per_instance_loss_history=jnp.zeros(buffer_size),
        per_instance_ptr=0,
        per_instance_total=0,
        is_per_instance_valid=jnp.zeros(buffer_size, dtype=jnp.bool_),
        # R→M specific tracking
        replay_to_mutate_wall_accuracy=jnp.zeros(buffer_size),
        replay_to_mutate_goal_accuracy=jnp.zeros(buffer_size),
        replay_to_mutate_loss=jnp.zeros(buffer_size),
        replay_to_mutate_count=0,
        replay_to_mutate_ptr=0,
        # R→M visualization data (last batch)
        last_r2m_wall_logits=jnp.zeros((batch_size, env_height, env_width)),
        last_r2m_goal_logits=jnp.zeros((batch_size, env_height * env_width)),
        last_r2m_wall_map=jnp.zeros((batch_size, env_height, env_width), dtype=jnp.bool_),
        last_r2m_goal_pos=jnp.zeros((batch_size, 2), dtype=jnp.uint32),
        last_r2m_agent_pos=jnp.zeros((batch_size, 2), dtype=jnp.uint32),
        last_r2m_valid=False,
        # Distributional calibration tracking
        dist_calibration_wall_history=jnp.zeros((3, buffer_size)),
        dist_calibration_goal_history=jnp.zeros((3, buffer_size)),
        dist_accuracy_wall_history=jnp.zeros((3, buffer_size)),
        dist_accuracy_goal_mode_match_history=jnp.zeros((3, buffer_size)),
        # Per-component loss tracking
        wall_loss_history=jnp.zeros(buffer_size),
        goal_loss_history=jnp.zeros(buffer_size),
        agent_pos_loss_history=jnp.zeros(buffer_size),
        agent_dir_loss_history=jnp.zeros(buffer_size),
        branch_wall_loss_history=jnp.zeros((3, buffer_size)),
        branch_goal_loss_history=jnp.zeros((3, buffer_size)),
        branch_agent_pos_loss_history=jnp.zeros((3, buffer_size)),
        branch_agent_dir_loss_history=jnp.zeros((3, buffer_size)),
        branch_zone_wall_loss_observed=jnp.zeros((3, buffer_size)),
        branch_zone_wall_loss_adjacent=jnp.zeros((3, buffer_size)),
        branch_zone_wall_loss_distant=jnp.zeros((3, buffer_size)),
        branch_zone_wall_accuracy_observed=jnp.zeros((3, buffer_size)),
        branch_zone_wall_accuracy_adjacent=jnp.zeros((3, buffer_size)),
        branch_zone_wall_accuracy_distant=jnp.zeros((3, buffer_size)),
        dist_accuracy_agent_pos_history=jnp.zeros((3, buffer_size)),
        dist_accuracy_agent_dir_history=jnp.zeros((3, buffer_size)),
        dist_accuracy_combined_history=jnp.zeros((3, buffer_size)),
        # Last predictions and levels
        last_predictions_wall_logits=jnp.zeros((batch_size, env_height, env_width)),
        last_predictions_goal_logits=jnp.zeros((batch_size, env_height * env_width)),
        last_predictions_agent_pos_logits=jnp.zeros((batch_size, env_height * env_width)),
        last_predictions_agent_dir_logits=jnp.zeros((batch_size, 4)),
        last_levels_wall_map=jnp.zeros((batch_size, env_height, env_width), dtype=jnp.bool_),
        last_levels_goal_pos=jnp.zeros((batch_size, 2), dtype=jnp.uint32),
        last_levels_agent_pos=jnp.zeros((batch_size, 2), dtype=jnp.uint32),
        last_levels_agent_dir=jnp.zeros(batch_size, dtype=jnp.uint8),
        last_branch=0,
        # Tier 1/2/3 loss tracking
        tier1_loss_history=jnp.zeros(buffer_size),
        tier2_loss_history=jnp.zeros(buffer_size),
        tier3_loss_history=jnp.zeros(buffer_size),
        tier1_regret_loss_history=jnp.zeros(buffer_size),
        tier1_difficulty_loss_history=jnp.zeros(buffer_size),
        tier1_branch_loss_history=jnp.zeros(buffer_size),
        tier1_score_loss_history=jnp.zeros(buffer_size),
        tier2_return_loss_history=jnp.zeros(buffer_size),
        tier2_novelty_loss_history=jnp.zeros(buffer_size),
        tier2_unusualness_loss_history=jnp.zeros(buffer_size),
        tier3_drift_loss_history=jnp.zeros(buffer_size),
        branch_tier1_loss_history=jnp.zeros((3, buffer_size)),
        branch_tier2_loss_history=jnp.zeros((3, buffer_size)),
        branch_tier3_loss_history=jnp.zeros((3, buffer_size)),
    )


@struct.dataclass
class ParetoHistoryState:
    """Tracks novelty/learnability history at eval checkpoints."""
    novelty_history: chex.Array           # (max_checkpoints,)
    learnability_history: chex.Array      # (max_checkpoints,)
    training_steps: chex.Array            # (max_checkpoints,)
    checkpoint_ptr: int
    num_checkpoints: int


def create_pareto_history_state(max_checkpoints: int = 200) -> ParetoHistoryState:
    """Create state for tracking Pareto history."""
    return ParetoHistoryState(
        novelty_history=jnp.zeros(max_checkpoints),
        learnability_history=jnp.zeros(max_checkpoints),
        training_steps=jnp.zeros(max_checkpoints, dtype=jnp.int32),
        checkpoint_ptr=0,
        num_checkpoints=0,
    )


# =============================================================================
# AGENT-CENTRIC TRACKING STATE
# =============================================================================

@struct.dataclass
class AgentTrackingState:
    """
    Agent-centric tracking (NOT probe-based).
    Tracks policy, value, and performance metrics.

    This replaces probe-based novelty/learnability with agent-centric metrics:
    - Policy entropy diversity across level types
    - Value calibration (|V(s) - actual_return|)
    - Performance on different branches

    The probe is an interpretability tool. These are the ACTUAL agent metrics.
    """
    # Policy tracking
    policy_entropy_history: chex.Array      # (buffer_size,)
    policy_kl_history: chex.Array           # (buffer_size,) KL from previous

    # Value tracking
    value_predictions: chex.Array           # (buffer_size,)
    actual_returns: chex.Array              # (buffer_size,)

    # Level features (for clustering by difficulty)
    level_wall_density_history: chex.Array  # (buffer_size,)
    level_goal_distance_history: chex.Array # (buffer_size,) Manhattan distance

    # Branch tracking
    branch_history: chex.Array              # (buffer_size,) 0=DR, 1=Replay, 2=Mutate

    # Training steps
    training_step_history: chex.Array       # (buffer_size,)

    # Buffer management
    buffer_ptr: int
    total_samples: int


def create_agent_tracking_state(buffer_size: int = 1000) -> AgentTrackingState:
    """Initialize agent-centric tracking state."""
    return AgentTrackingState(
        # Policy tracking
        policy_entropy_history=jnp.zeros(buffer_size),
        policy_kl_history=jnp.zeros(buffer_size),
        # Value tracking
        value_predictions=jnp.zeros(buffer_size),
        actual_returns=jnp.zeros(buffer_size),
        # Level features
        level_wall_density_history=jnp.zeros(buffer_size),
        level_goal_distance_history=jnp.zeros(buffer_size),
        # Branch tracking
        branch_history=jnp.zeros(buffer_size, dtype=jnp.int32),
        # Training steps
        training_step_history=jnp.zeros(buffer_size, dtype=jnp.int32),
        # Buffer management
        buffer_ptr=0,
        total_samples=0,
    )


@struct.dataclass
class VisualizationData:
    """
    Per-branch visualization data for matched pairs.

    For DR and Replay branches, predictions don't have 1-to-1 correspondence
    with levels, so we use greedy matching for visualization.
    R->M has natural 1-to-1 correspondence.
    """
    # DR branch visualization (requires greedy matching)
    last_dr_predictions_wall: chex.Array    # (batch_size, height, width)
    last_dr_predictions_goal: chex.Array    # (batch_size, height*width)
    last_dr_levels_wall: chex.Array         # (batch_size, height, width)
    last_dr_levels_goal: chex.Array         # (batch_size, 2)
    dr_valid: bool

    # Replay branch visualization (requires greedy matching)
    last_replay_predictions_wall: chex.Array
    last_replay_predictions_goal: chex.Array
    last_replay_levels_wall: chex.Array
    last_replay_levels_goal: chex.Array
    replay_valid: bool

    # R->M has 1-to-1 correspondence (inherited from ProbeTrackingState)


def create_visualization_data(
    batch_size: int = 32,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> VisualizationData:
    """Initialize visualization data buffers."""
    return VisualizationData(
        # DR branch
        last_dr_predictions_wall=jnp.zeros((batch_size, env_height, env_width)),
        last_dr_predictions_goal=jnp.zeros((batch_size, env_height * env_width)),
        last_dr_levels_wall=jnp.zeros((batch_size, env_height, env_width)),
        last_dr_levels_goal=jnp.zeros((batch_size, 2), dtype=jnp.int32),
        dr_valid=False,
        # Replay branch
        last_replay_predictions_wall=jnp.zeros((batch_size, env_height, env_width)),
        last_replay_predictions_goal=jnp.zeros((batch_size, env_height * env_width)),
        last_replay_levels_wall=jnp.zeros((batch_size, env_height, env_width)),
        last_replay_levels_goal=jnp.zeros((batch_size, 2), dtype=jnp.int32),
        replay_valid=False,
    )


# =============================================================================
# CONTEXT VECTOR STATE (for context_vector agent)
# =============================================================================

@struct.dataclass
class ContextState:
    """
    Compressed context vector that persists across episodes.
    Updated via EMA after each episode.
    """
    context_vector: chex.Array    # (context_dim,)
    episode_count: int


def create_context_state(context_dim: int = 64) -> ContextState:
    """Initialize context state with zeros."""
    return ContextState(
        context_vector=jnp.zeros(context_dim),
        episode_count=0,
    )


# =============================================================================
# EPISODIC MEMORY STATE (for episodic_memory agent)
# =============================================================================

@struct.dataclass
class EpisodicMemoryState:
    """
    Discrete buffer of recent episode summaries for retrieval.
    """
    episode_embeddings: chex.Array   # (buffer_size, embed_dim)
    episode_returns: chex.Array      # (buffer_size,)
    episode_lengths: chex.Array      # (buffer_size,)
    episode_solved: chex.Array       # (buffer_size,)
    pointer: int                     # Circular buffer pointer
    size: int                        # Current number of stored episodes


def create_episodic_memory_state(
    buffer_size: int = 64,
    embed_dim: int = 64,
) -> EpisodicMemoryState:
    """Initialize episodic memory buffer."""
    return EpisodicMemoryState(
        episode_embeddings=jnp.zeros((buffer_size, embed_dim)),
        episode_returns=jnp.zeros(buffer_size),
        episode_lengths=jnp.zeros(buffer_size, dtype=jnp.int32),
        episode_solved=jnp.zeros(buffer_size, dtype=jnp.bool_),
        pointer=0,
        size=0,
    )


# =============================================================================
# TRAIN STATE HIERARCHY
# =============================================================================

class BaseTrainState(FlaxTrainState):
    """
    Base TrainState with fields common to all agent variants.

    This includes the level sampler state and ACCEL branch tracking.

    NOTE: Probe fields are NOT included here by default. Use ProbeTrainState
    for agents that need probe-based interpretability analysis. The probe
    is an EXTERNAL interpretability tool, not part of the agent itself.
    """
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)

    # Update counters for logging
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int

    # Last level batches for visualization
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)

    # Training step counter
    training_step: int = struct.field(pytree_node=True, default=0)

    # DR mode: continuous rollout state (for pure domain randomization)
    # These track state across train_step calls for continuous rollouts
    last_hstate: Optional[chex.ArrayTree] = struct.field(pytree_node=True, default=None)
    last_obs: Optional[chex.ArrayTree] = struct.field(pytree_node=True, default=None)
    last_env_state: Optional[chex.ArrayTree] = struct.field(pytree_node=True, default=None)

    # Agent-centric tracking (NOT probe-based)
    agent_tracking: Optional[AgentTrackingState] = struct.field(pytree_node=True, default=None)

    # Visualization data for matched pairs
    visualization_data: Optional[VisualizationData] = struct.field(pytree_node=True, default=None)


class ProbeTrainState(BaseTrainState):
    """
    TrainState extended with probe fields.
    Used by: accel_probe, persistent_lstm, context_vector, episodic_memory
    """
    # Probe network params and optimizer state
    probe_params: Optional[chex.ArrayTree] = struct.field(pytree_node=True, default=None)
    probe_opt_state: Optional[chex.ArrayTree] = struct.field(pytree_node=True, default=None)

    # Hidden state tracking for probe
    current_hstate: Optional[chex.ArrayTree] = struct.field(pytree_node=True, default=None)
    prev_hstate: Optional[chex.ArrayTree] = struct.field(pytree_node=True, default=None)
    current_branch: int = struct.field(pytree_node=True, default=0)
    prev_branch: int = struct.field(pytree_node=True, default=0)
    has_valid_prev_hstate: bool = struct.field(pytree_node=True, default=False)

    # Probe tracking and Pareto history
    probe_tracking: Optional[ProbeTrackingState] = struct.field(pytree_node=True, default=None)
    pareto_history: Optional[ParetoHistoryState] = struct.field(pytree_node=True, default=None)

    # Hidden state samples for t-SNE visualization
    hstate_samples: Optional[chex.Array] = struct.field(pytree_node=True, default=None)
    hstate_sample_branches: Optional[chex.Array] = struct.field(pytree_node=True, default=None)
    hstate_sample_ptr: int = struct.field(pytree_node=True, default=0)

    # Episode context for probe
    last_episode_return: Optional[chex.Array] = struct.field(pytree_node=True, default=None)
    last_episode_solved: Optional[chex.Array] = struct.field(pytree_node=True, default=None)
    last_episode_length: Optional[chex.Array] = struct.field(pytree_node=True, default=None)
    last_agent_return: jnp.ndarray = struct.field(pytree_node=True, default=0.0)


class PersistentMemoryTrainState(ProbeTrainState):
    """
    TrainState for persistent_lstm agent.

    Key difference: Hidden state persists across episodes and branches.
    """
    # Persistent hidden state (not reset on episode boundary)
    persistent_hstate: Optional[chex.ArrayTree] = struct.field(pytree_node=True, default=None)


class ContextVectorTrainState(ProbeTrainState):
    """
    TrainState for context_vector agent.

    Includes compressed context vector updated via EMA.
    """
    context_state: Optional[ContextState] = struct.field(pytree_node=True, default=None)


class EpisodicMemoryTrainState(ProbeTrainState):
    """
    TrainState for episodic_memory agent.

    Includes discrete episode buffer for retrieval.
    """
    episodic_memory: Optional[EpisodicMemoryState] = struct.field(pytree_node=True, default=None)


class NextEnvPredictionTrainState(BaseTrainState):
    """
    TrainState for next_env_prediction agent with curriculum memory.

    This agent has explicit curriculum info fed to the network (upper-bound baseline).
    Unlike probe agents, predictions are computed during forward pass and gradients
    flow through the shared backbone.
    """
    # Curriculum state (cross-episode memory)
    curriculum_state: Optional[chex.ArrayTree] = struct.field(pytree_node=True, default=None)
    # Novelty/Learnability tracking
    nl_state: Optional[chex.ArrayTree] = struct.field(pytree_node=True, default=None)
    # Accumulated prediction metrics for logging
    pred_metrics_accumulator: Optional[dict] = struct.field(pytree_node=True, default=None)


# =============================================================================
# PAIRED TRAIN STATE
# =============================================================================

@struct.dataclass
class PAIREDTrainState:
    """
    TrainState for PAIRED training with 3 networks.

    PAIRED uses:
    - Protagonist: Learns to solve levels (standard ActorCritic)
    - Antagonist: Finds exploits in levels (standard ActorCritic)
    - Adversary: Generates levels to maximize regret (AdversaryActorCritic)

    The adversary receives reward = antagonist_max_return - protagonist_mean_return

    AGENT-CENTRIC DESIGN:
    - PAIRED has natural 1-to-1 correspondence: (pro[i], ant[i], level[i])
    - Per-instance metrics ARE meaningful
    - Probe is OPTIONAL and EXTERNAL
    - agent_tracking tracks protagonist metrics
    """
    update_count: int

    # The three network train states
    pro_train_state: FlaxTrainState  # Protagonist
    ant_train_state: FlaxTrainState  # Antagonist
    adv_train_state: FlaxTrainState  # Adversary

    # Optional: Memory architecture state (for memory variants)
    memory_state: Optional[chex.ArrayTree] = None

    # =========================================================================
    # AGENT-CENTRIC TRACKING (NOT probe-based)
    # =========================================================================
    # Agent-centric tracking for protagonist value calibration
    agent_tracking: Optional["AgentTrackingState"] = None

    # Optional: Probe params for curriculum awareness analysis (EXTERNAL)
    probe_params: Optional[chex.ArrayTree] = None
    probe_opt_state: Optional[chex.ArrayTree] = None

    # Tracking state for probes (EXTERNAL interpretability tool)
    probe_tracking: Optional[ProbeTrackingState] = None

    # Hidden state samples for t-SNE visualization
    hstate_samples: Optional[chex.Array] = None
    hstate_sample_branches: Optional[chex.Array] = None
    hstate_sample_ptr: int = 0

    # PAIRED-specific history tracking for visualizations
    pro_returns_history: Optional[chex.Array] = None       # (history_size,)
    ant_returns_history: Optional[chex.Array] = None       # (history_size,)
    regret_history: Optional[chex.Array] = None            # (history_size,)
    training_steps_history: Optional[chex.Array] = None    # (history_size,)
    novelty_history: Optional[chex.Array] = None           # (history_size,)
    learnability_history: Optional[chex.Array] = None      # (history_size,)
    level_history_wall_maps: Optional[chex.Array] = None   # (history_size, H, W) for novelty
    history_ptr: int = 0
    history_total: int = 0

    # Bilateral probe support (antagonist also gets probe)
    ant_probe_params: Optional[chex.ArrayTree] = None
    ant_probe_opt_state: Optional[chex.ArrayTree] = None
    ant_memory_state: Optional[chex.ArrayTree] = None

    # Last adversary-generated level batch for displacement tracking
    last_adversary_level: Optional[chex.ArrayTree] = None

    # PAIRED-specific tracking for experiments
    adversary_level_features: Optional[chex.Array] = None
    adversary_strategy_labels: Optional[chex.Array] = None  # from C3 clustering
    prediction_data_buffer: Optional[chex.ArrayTree] = None  # PAIREDPredictionData buffer


# =============================================================================
# PAIRED EXPERIMENT DATA STRUCTURES
# =============================================================================

@struct.dataclass
class PAIREDPredictionData:
    """
    For A1 utility extraction - uses prediction losses to extract Û.

    This is ONLY for the symbolic regression experiment (A1) that extracts
    the protagonist's implicit model of the adversary's utility function.
    """
    # Level features (observable)
    wall_density: float
    goal_distance: float
    path_length: float
    num_corridors: int = 0
    bottleneck_count: int = 0
    open_space_ratio: float = 0.0
    shortest_path_wall_adjacency: float = 0.0

    # Returns and regret
    protagonist_return: float = 0.0
    antagonist_return: float = 0.0
    regret: float = 0.0  # ant_return - pro_return

    # Prediction losses (A1 specifically needs these)
    wall_prediction_loss: float = 0.0
    goal_prediction_loss: float = 0.0
    next_wall_density_prediction_loss: float = 0.0
    next_goal_distance_prediction_loss: float = 0.0

    # Value errors
    value_prediction_error: float = 0.0
    protagonist_advantage: float = 0.0

    # Temporal context
    training_step: int = 0
    adversary_entropy: float = 0.0


def create_paired_prediction_data(
    wall_density: float,
    goal_distance: float,
    path_length: float,
    training_step: int = 0,
) -> PAIREDPredictionData:
    """Create a PAIREDPredictionData instance with basic required fields."""
    return PAIREDPredictionData(
        wall_density=wall_density,
        goal_distance=goal_distance,
        path_length=path_length,
        training_step=training_step,
    )


@struct.dataclass
class PAIREDCurriculumState:
    """
    Replaces branch-based CurriculumState for PAIRED.

    Tracks recent level statistics and regret dynamics instead of
    DR/Replay/Mutate branch information.
    """
    # Recent level feature statistics
    recent_wall_densities: chex.Array      # (buffer_size,)
    recent_goal_distances: chex.Array      # (buffer_size,)
    recent_path_lengths: chex.Array        # (buffer_size,)
    recent_regrets: chex.Array             # (buffer_size,)
    recent_adversary_entropies: chex.Array # (buffer_size,)

    # Return tracking
    recent_protagonist_returns: chex.Array # (buffer_size,)
    recent_antagonist_returns: chex.Array  # (buffer_size,)
    recent_regret_sources: chex.Array      # (buffer_size,) - 0=ant_strong, 1=pro_weak, 2=mixed

    # Aggregate statistics
    mean_regret: float
    max_regret: float
    adversary_feature_focus: float         # entropy of level feature distribution
    training_step: int

    # Buffer management
    buffer_ptr: int
    buffer_size: int


def create_paired_curriculum_state(buffer_size: int = 100) -> PAIREDCurriculumState:
    """Initialize PAIRED curriculum state with empty buffers."""
    return PAIREDCurriculumState(
        recent_wall_densities=jnp.zeros(buffer_size),
        recent_goal_distances=jnp.zeros(buffer_size),
        recent_path_lengths=jnp.zeros(buffer_size),
        recent_regrets=jnp.zeros(buffer_size),
        recent_adversary_entropies=jnp.zeros(buffer_size),
        recent_protagonist_returns=jnp.zeros(buffer_size),
        recent_antagonist_returns=jnp.zeros(buffer_size),
        recent_regret_sources=jnp.zeros(buffer_size, dtype=jnp.int32),
        mean_regret=0.0,
        max_regret=0.0,
        adversary_feature_focus=1.0,  # Max entropy initially
        training_step=0,
        buffer_ptr=0,
        buffer_size=buffer_size,
    )


@struct.dataclass
class PAIREDProbeTrackingState:
    """
    Replaces branch-based ProbeTrackingState for PAIRED.

    Tracks probe metrics by regret condition and adversary strategy
    instead of by curriculum branch.
    """
    # Overall loss history
    loss_history: chex.Array               # (buffer_size,)
    training_step_history: chex.Array      # (buffer_size,)

    # Regret-conditioned loss (by tercile: low=0, medium=1, high=2)
    regret_conditioned_loss: chex.Array    # (3, buffer_size)
    regret_step_history: chex.Array        # (3, buffer_size)
    regret_sample_counts: chex.Array       # (3,)

    # Adversary strategy-conditioned loss (K strategies from clustering)
    adversary_strategy_loss: chex.Array    # (K, buffer_size)
    strategy_sample_counts: chex.Array     # (K,)

    # Bilateral hidden state statistics (protagonist and antagonist)
    pro_hstate_mean_by_regret: chex.Array  # (3, hidden_dim)
    pro_hstate_var_by_regret: chex.Array   # (3, hidden_dim)
    ant_hstate_mean_by_regret: chex.Array  # (3, hidden_dim)
    ant_hstate_var_by_regret: chex.Array   # (3, hidden_dim)

    # Buffer management
    buffer_ptr: int
    total_samples: int
    current_training_step: int
    regret_ptrs: chex.Array                # (3,)
    hstate_dim: int
    n_adversary_strategies: int


def create_paired_probe_tracking_state(
    buffer_size: int = 500,
    hstate_dim: int = DEFAULT_HSTATE_DIM,
    n_adversary_strategies: int = 5,
) -> PAIREDProbeTrackingState:
    """Initialize PAIRED probe tracking state."""
    return PAIREDProbeTrackingState(
        loss_history=jnp.zeros(buffer_size),
        training_step_history=jnp.zeros(buffer_size, dtype=jnp.int32),
        regret_conditioned_loss=jnp.zeros((3, buffer_size)),
        regret_step_history=jnp.zeros((3, buffer_size), dtype=jnp.int32),
        regret_sample_counts=jnp.zeros(3, dtype=jnp.int32),
        adversary_strategy_loss=jnp.zeros((n_adversary_strategies, buffer_size)),
        strategy_sample_counts=jnp.zeros(n_adversary_strategies, dtype=jnp.int32),
        pro_hstate_mean_by_regret=jnp.zeros((3, hstate_dim)),
        pro_hstate_var_by_regret=jnp.zeros((3, hstate_dim)),
        ant_hstate_mean_by_regret=jnp.zeros((3, hstate_dim)),
        ant_hstate_var_by_regret=jnp.zeros((3, hstate_dim)),
        buffer_ptr=0,
        total_samples=0,
        current_training_step=0,
        regret_ptrs=jnp.zeros(3, dtype=jnp.int32),
        hstate_dim=hstate_dim,
        n_adversary_strategies=n_adversary_strategies,
    )


@struct.dataclass
class BilateralProbeData:
    """
    Data for bilateral probing (E1) - probe both protagonist and antagonist.

    Used to compare what each agent encodes about the level.
    """
    # Hidden states
    pro_hstate: chex.Array                 # Protagonist hidden state
    ant_hstate: chex.Array                 # Antagonist hidden state

    # Level features
    wall_density: float
    goal_distance: float
    path_length: float

    # Returns and regret
    protagonist_return: float
    antagonist_return: float
    regret: float

    # Probe predictions (after probing)
    pro_wall_prediction: Optional[chex.Array] = None
    ant_wall_prediction: Optional[chex.Array] = None
    pro_goal_prediction: Optional[chex.Array] = None
    ant_goal_prediction: Optional[chex.Array] = None

    # Probe losses
    pro_probe_loss: float = 0.0
    ant_probe_loss: float = 0.0

    # Additional PAIRED-specific targets
    adversary_strategy_cluster: int = 0
    regret_estimate: float = 0.0
    opponent_return_estimate: float = 0.0  # Theory of mind target


@struct.dataclass
class ShardCluster:
    """
    Represents a contextually-activated shard (D3, F2, F3).

    Following shard theory: goals are contextually-activated, weighted,
    composable decision-influencing computations.
    """
    # Dimensions in hidden state that constitute this shard
    dimension_indices: chex.Array          # (n_dims,)

    # Contexts that activate this shard
    activation_contexts: chex.Array        # Feature patterns that activate

    # Policy influence
    policy_influence: float                # How much this shard affects policy

    # Lifecycle tracking
    birth_step: int = 0
    last_active_step: int = 0
    activation_count: int = 0

    # Shard identity
    shard_id: int = 0