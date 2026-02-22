"""
Standalone Probe Runner for interpretability experiments.

This module provides a ProbeRunner class that handles probe evaluation
OUTSIDE of agent training. The probe is an interpretability tool, NOT
part of the agent.

Key Design Principles:
- Probe does NOT affect agent training (stop_gradient)
- Probe can be enabled/disabled without affecting agent performance
- Probe metrics are separate from agent metrics
- Supports both online analysis and post-training evaluation
"""

from typing import Dict, Optional, Tuple, Callable
import jax
import jax.numpy as jnp
import optax
import chex

from .types import (
    ProbeTrackingState,
    create_probe_tracking_state,
    DEFAULT_HSTATE_DIM,
    DEFAULT_ENV_HEIGHT,
    DEFAULT_ENV_WIDTH,
)
from .networks import CurriculumProbe
from .metrics import (
    compute_probe_loss_batch,
    compute_per_instance_calibration_batch,
    compute_distributional_calibration_metrics,
    compute_greedy_matching,
    compute_matched_accuracy_metrics,
)
from .utils import flatten_hstate


class ProbeRunner:
    """
    Standalone probe runner - NOT part of agent training.
    Used for interpretability experiments only.

    The ProbeRunner can:
    1. Evaluate predictions from hidden states
    2. Compute probe loss for interpretability analysis
    3. Run batched analysis across multiple episodes
    4. Compute matched pairs for visualization (greedy matching for DR/Replay)

    This class is SEPARATE from the agent's training loop. Agents can
    optionally use a ProbeRunner for analysis, but the probe:
    - Does NOT affect agent gradients (stop_gradient enforced)
    - Does NOT update during training steps by default
    - Can be run post-hoc on saved checkpoints
    """

    def __init__(
        self,
        probe_network: Optional[CurriculumProbe] = None,
        probe_params: Optional[chex.ArrayTree] = None,
        learning_rate: float = 1e-3,
        env_height: int = DEFAULT_ENV_HEIGHT,
        env_width: int = DEFAULT_ENV_WIDTH,
        use_episode_context: bool = True,
    ):
        """
        Initialize ProbeRunner.

        Args:
            probe_network: CurriculumProbe network. If None, creates default.
            probe_params: Initial probe parameters. If None, initializes randomly.
            learning_rate: Learning rate for probe updates.
            env_height: Environment height for predictions.
            env_width: Environment width for predictions.
            use_episode_context: Whether to use episode context in probe.
        """
        self.env_height = env_height
        self.env_width = env_width
        self.learning_rate = learning_rate
        self.use_episode_context = use_episode_context

        # Create probe network if not provided
        if probe_network is None:
            self.probe = CurriculumProbe(
                env_height=env_height,
                env_width=env_width,
                use_episode_context=use_episode_context,
            )
        else:
            self.probe = probe_network

        # Initialize params if not provided
        self.params = probe_params
        self.opt_state = None

        # Create optimizer
        self.optimizer = optax.adam(learning_rate=learning_rate)

    def initialize(self, rng: chex.PRNGKey) -> None:
        """Initialize probe parameters if not already set."""
        if self.params is None:
            dummy_hstate = jnp.zeros((1, DEFAULT_HSTATE_DIM))
            if self.use_episode_context:
                self.params = self.probe.init(
                    rng, dummy_hstate,
                    episode_return=jnp.zeros(1),
                    episode_solved=jnp.zeros(1),
                    episode_length=jnp.zeros(1),
                )
            else:
                self.params = self.probe.init(rng, dummy_hstate)
            self.opt_state = self.optimizer.init(self.params)

    def evaluate(
        self,
        hstate: chex.ArrayTree,
        episode_return: Optional[chex.Array] = None,
        episode_solved: Optional[chex.Array] = None,
        episode_length: Optional[chex.Array] = None,
    ) -> Dict[str, chex.Array]:
        """
        Run probe on hidden state, return predictions.

        Args:
            hstate: LSTM hidden state tuple (c, h) or flattened array
            episode_return: Optional episode returns for context
            episode_solved: Optional episode solved flags for context
            episode_length: Optional episode lengths for context

        Returns:
            predictions: Dict with wall_logits, goal_logits, agent_pos_logits, agent_dir_logits
        """
        if self.params is None:
            raise ValueError("ProbeRunner not initialized. Call initialize() first.")

        # Flatten hidden state if needed
        if isinstance(hstate, tuple):
            hstate_flat = flatten_hstate(hstate)
        else:
            hstate_flat = hstate

        # Ensure stop_gradient - probe should NOT affect agent
        hstate_flat = jax.lax.stop_gradient(hstate_flat)

        # Run probe
        if self.use_episode_context and episode_return is not None:
            predictions = self.probe.apply(
                self.params,
                hstate_flat,
                episode_return=episode_return,
                episode_solved=episode_solved if episode_solved is not None else jnp.zeros_like(episode_return),
                episode_length=episode_length if episode_length is not None else jnp.zeros_like(episode_return, dtype=jnp.int32),
            )
        else:
            predictions = self.probe.apply(self.params, hstate_flat)

        return predictions

    def compute_loss(
        self,
        hstate: chex.ArrayTree,
        levels,
        episode_return: Optional[chex.Array] = None,
        episode_solved: Optional[chex.Array] = None,
        episode_length: Optional[chex.Array] = None,
    ) -> Tuple[float, Dict]:
        """
        Compute probe loss for interpretability analysis.

        Args:
            hstate: LSTM hidden state tuple (c, h) or flattened array
            levels: Batch of Level objects
            episode_return: Optional episode returns for context
            episode_solved: Optional episode solved flags for context
            episode_length: Optional episode lengths for context

        Returns:
            loss: Scalar loss value
            metrics: Dict with detailed loss breakdown
        """
        predictions = self.evaluate(
            hstate, episode_return, episode_solved, episode_length
        )
        loss, metrics = compute_probe_loss_batch(
            predictions, levels, self.env_height, self.env_width
        )
        return float(loss), metrics

    def train_step(
        self,
        hstate: chex.ArrayTree,
        levels,
        episode_return: Optional[chex.Array] = None,
        episode_solved: Optional[chex.Array] = None,
        episode_length: Optional[chex.Array] = None,
    ) -> Tuple[float, Dict]:
        """
        Run one training step on the probe (NOT the agent).

        This is for interpretability experiments where we want to
        train the probe to analyze agent representations.

        Args:
            hstate: LSTM hidden state tuple (c, h) or flattened array
            levels: Batch of Level objects
            episode_return: Optional episode returns for context
            episode_solved: Optional episode solved flags for context
            episode_length: Optional episode lengths for context

        Returns:
            loss: Scalar loss value
            metrics: Dict with detailed loss breakdown
        """
        if self.params is None or self.opt_state is None:
            raise ValueError("ProbeRunner not initialized. Call initialize() first.")

        # Flatten hidden state if needed
        if isinstance(hstate, tuple):
            hstate_flat = flatten_hstate(hstate)
        else:
            hstate_flat = hstate

        # Stop gradient on hidden state
        hstate_flat = jax.lax.stop_gradient(hstate_flat)

        def loss_fn(params):
            if self.use_episode_context and episode_return is not None:
                preds = self.probe.apply(
                    params, hstate_flat,
                    episode_return=episode_return,
                    episode_solved=episode_solved if episode_solved is not None else jnp.zeros_like(episode_return),
                    episode_length=episode_length if episode_length is not None else jnp.zeros_like(episode_return, dtype=jnp.int32),
                )
            else:
                preds = self.probe.apply(params, hstate_flat)
            loss, metrics = compute_probe_loss_batch(
                preds, levels, self.env_height, self.env_width
            )
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.params)

        # Update probe params
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        self.params = new_params
        self.opt_state = new_opt_state

        return float(loss), metrics

    def evaluate_batch_with_matching(
        self,
        hstates: chex.ArrayTree,
        levels,
        episode_returns: Optional[chex.Array] = None,
        episode_solved: Optional[chex.Array] = None,
        episode_lengths: Optional[chex.Array] = None,
    ) -> Dict:
        """
        Evaluate probe with greedy matching for DR/Replay branches.

        For DR and Replay branches, there's no 1-to-1 correspondence between
        predictions and levels. This method uses greedy matching to find
        the best pairing for visualization.

        Args:
            hstates: Batch of hidden states
            levels: Batch of Level objects
            episode_returns: Optional episode returns for context
            episode_solved: Optional episode solved flags for context
            episode_lengths: Optional episode lengths for context

        Returns:
            Dict with:
                - predictions: Raw probe predictions
                - matched_indices: Greedy matched level indices
                - matched_metrics: Accuracy metrics with matched pairs
                - distributional_metrics: Batch-level distributional metrics
        """
        predictions = self.evaluate(
            hstates, episode_returns, episode_solved, episode_lengths
        )

        # Compute greedy matching
        matched_indices = compute_greedy_matching(
            predictions, levels, self.env_height, self.env_width
        )

        # Compute metrics with matched pairs
        matched_metrics = compute_matched_accuracy_metrics(
            predictions, levels, matched_indices, self.env_height, self.env_width
        )

        # Compute distributional metrics (no matching needed)
        dist_metrics = compute_distributional_calibration_metrics(
            predictions, levels, self.env_height, self.env_width
        )

        return {
            'predictions': predictions,
            'matched_indices': matched_indices,
            'matched_metrics': matched_metrics,
            'distributional_metrics': dist_metrics,
        }

    def evaluate_per_instance(
        self,
        hstates: chex.ArrayTree,
        levels,
        episode_returns: Optional[chex.Array] = None,
        episode_solved: Optional[chex.Array] = None,
        episode_lengths: Optional[chex.Array] = None,
    ) -> Dict:
        """
        Evaluate probe with per-instance metrics.

        ONLY valid for R->M transition or PAIRED where there's 1-to-1
        correspondence between hidden states and levels.

        Args:
            hstates: Batch of hidden states
            levels: Batch of Level objects
            episode_returns: Optional episode returns for context
            episode_solved: Optional episode solved flags for context
            episode_lengths: Optional episode lengths for context

        Returns:
            Dict with:
                - predictions: Raw probe predictions
                - per_instance_metrics: Per-instance accuracy metrics
                - distributional_metrics: Batch-level distributional metrics
        """
        predictions = self.evaluate(
            hstates, episode_returns, episode_solved, episode_lengths
        )

        # Per-instance metrics (assumes 1-to-1 correspondence)
        per_instance = compute_per_instance_calibration_batch(
            predictions, levels, self.env_height, self.env_width
        )

        # Distributional metrics
        dist_metrics = compute_distributional_calibration_metrics(
            predictions, levels, self.env_height, self.env_width
        )

        return {
            'predictions': predictions,
            'per_instance_metrics': per_instance,
            'distributional_metrics': dist_metrics,
        }

    def get_params(self) -> chex.ArrayTree:
        """Get current probe parameters."""
        return self.params

    def set_params(self, params: chex.ArrayTree) -> None:
        """Set probe parameters (e.g., from checkpoint)."""
        self.params = params
        if self.opt_state is None:
            self.opt_state = self.optimizer.init(params)


class ProbeAnalysisState:
    """
    Separate state for probe-based interpretability analysis.

    This is NOT part of the agent's TrainState. It can be:
    - Created independently for post-hoc analysis
    - Maintained alongside training for online monitoring
    - Serialized/loaded separately from agent checkpoints
    """

    def __init__(
        self,
        buffer_size: int = 500,
        hstate_dim: int = DEFAULT_HSTATE_DIM,
        batch_size: int = 32,
        env_height: int = DEFAULT_ENV_HEIGHT,
        env_width: int = DEFAULT_ENV_WIDTH,
    ):
        """
        Initialize probe analysis state.

        Args:
            buffer_size: Size of tracking buffers
            hstate_dim: Hidden state dimension
            batch_size: Batch size for visualization data
            env_height: Environment height
            env_width: Environment width
        """
        self.tracking = create_probe_tracking_state(
            buffer_size=buffer_size,
            hstate_dim=hstate_dim,
            batch_size=batch_size,
            env_height=env_height,
            env_width=env_width,
        )
        self.buffer_size = buffer_size

    def update(
        self,
        loss: float,
        training_step: int,
        branch: int,
        agent_return: float,
        predictions: Optional[Dict] = None,
        levels=None,
        is_replay_to_mutate: bool = False,
    ) -> None:
        """
        Update tracking state with new probe results.

        Args:
            loss: Probe loss value
            training_step: Current training step
            branch: Branch index (0=DR, 1=Replay, 2=Mutate)
            agent_return: Mean agent return
            predictions: Optional predictions dict for visualization
            levels: Optional levels for visualization
            is_replay_to_mutate: Whether this is R->M transition
        """
        tracking = self.tracking
        buffer_size = tracking.loss_history.shape[0]
        ptr = tracking.buffer_ptr % buffer_size

        # Update loss history
        new_loss_history = tracking.loss_history.at[ptr].set(loss)
        new_step_history = tracking.training_step_history.at[ptr].set(training_step)

        # Update branch history
        branch_ptr = tracking.branch_ptrs[branch] % buffer_size
        new_branch_loss = tracking.branch_loss_history.at[branch, branch_ptr].set(loss)
        new_branch_ptrs = tracking.branch_ptrs.at[branch].set(branch_ptr + 1)
        new_branch_counts = tracking.branch_sample_counts.at[branch].add(1)

        # Update agent returns
        new_agent_returns = tracking.agent_returns_history.at[ptr].set(agent_return)

        # Update R->M visualization data if applicable
        new_r2m_wall_logits = tracking.last_r2m_wall_logits
        new_r2m_goal_logits = tracking.last_r2m_goal_logits
        new_r2m_wall_map = tracking.last_r2m_wall_map
        new_r2m_goal_pos = tracking.last_r2m_goal_pos
        new_r2m_agent_pos = tracking.last_r2m_agent_pos
        new_r2m_valid = tracking.last_r2m_valid

        if is_replay_to_mutate and predictions is not None and levels is not None:
            new_r2m_wall_logits = predictions['wall_logits']
            new_r2m_goal_logits = predictions['goal_logits']
            new_r2m_wall_map = levels.wall_map
            new_r2m_goal_pos = levels.goal_pos
            new_r2m_agent_pos = levels.agent_pos
            new_r2m_valid = True

        self.tracking = tracking.replace(
            loss_history=new_loss_history,
            training_step_history=new_step_history,
            branch_loss_history=new_branch_loss,
            branch_ptrs=new_branch_ptrs,
            branch_sample_counts=new_branch_counts,
            buffer_ptr=(ptr + 1) % buffer_size,
            total_samples=tracking.total_samples + 1,
            current_training_step=training_step + 1,
            agent_returns_history=new_agent_returns,
            last_r2m_wall_logits=new_r2m_wall_logits,
            last_r2m_goal_logits=new_r2m_goal_logits,
            last_r2m_wall_map=new_r2m_wall_map,
            last_r2m_goal_pos=new_r2m_goal_pos,
            last_r2m_agent_pos=new_r2m_agent_pos,
            last_r2m_valid=new_r2m_valid,
        )

    def get_tracking(self) -> ProbeTrackingState:
        """Get current tracking state."""
        return self.tracking
