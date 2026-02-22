"""
Visualization utilities for curriculum awareness ablations.

Contains plotting functions for:
- Wall prediction heatmaps
- Position prediction visualizations
- Probe loss by branch
- Novelty-learnability plots
- Hidden state t-SNE
- Information content dashboards
- Curriculum prediction comparison figures
- Pareto frontier figures
"""

from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
import chex
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .types import ProbeTrackingState, ParetoHistoryState, DEFAULT_ENV_HEIGHT, DEFAULT_ENV_WIDTH
from .metrics import compute_random_baselines, compute_per_instance_calibration_batch


def create_wall_prediction_heatmap(
    predictions: dict,
    actual_level,
    env_height: int = 13,
    env_width: int = 13,
) -> np.ndarray:
    """Create visualization comparing predicted vs actual wall maps."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    wall_probs = jax.nn.sigmoid(predictions['wall_logits'])
    im0 = axes[0].imshow(np.array(wall_probs), cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Predicted Wall Probs')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(np.array(actual_level.wall_map), cmap='binary', vmin=0, vmax=1)
    axes[1].set_title('Actual Wall Map')
    plt.colorbar(im1, ax=axes[1])

    error = np.abs(np.array(wall_probs) - np.array(actual_level.wall_map.astype(jnp.float32)))
    im2 = axes[2].imshow(error, cmap='Reds', vmin=0, vmax=1)
    axes[2].set_title('Prediction Error')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_batch_wall_prediction_summary(
    predictions_batch: dict,
    levels_batch,
    env_height: int = 13,
    env_width: int = 13,
    n_samples: int = 4,
) -> np.ndarray:
    """Create visualization showing prediction variance across batch."""
    batch_size = predictions_batch['wall_logits'].shape[0]
    n_samples = min(n_samples, batch_size)

    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(2, 4, 1)
    wall_probs_batch = jax.nn.sigmoid(predictions_batch['wall_logits'])
    mean_pred = np.array(wall_probs_batch.mean(axis=0))
    im1 = ax1.imshow(mean_pred, cmap='hot', vmin=0, vmax=1)
    ax1.set_title(f'Mean Predicted (n={batch_size})')
    plt.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(2, 4, 2)
    mean_actual = np.array(levels_batch.wall_map.astype(jnp.float32).mean(axis=0))
    im2 = ax2.imshow(mean_actual, cmap='hot', vmin=0, vmax=1)
    ax2.set_title(f'Mean Actual (n={batch_size})')
    plt.colorbar(im2, ax=ax2)

    ax3 = fig.add_subplot(2, 4, 3)
    pred_variance = np.array(wall_probs_batch.var(axis=0))
    im3 = ax3.imshow(pred_variance, cmap='Purples', vmin=0, vmax=0.25)
    ax3.set_title('Prediction Variance')
    plt.colorbar(im3, ax=ax3)

    ax4 = fig.add_subplot(2, 4, 4)
    mean_error = np.abs(mean_pred - mean_actual)
    im4 = ax4.imshow(mean_error, cmap='Reds', vmin=0, vmax=1)
    ax4.set_title('Mean Absolute Error')
    plt.colorbar(im4, ax=ax4)

    indices = np.linspace(0, batch_size - 1, n_samples, dtype=int)
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(2, 4, 5 + i)
        pred = np.array(jax.nn.sigmoid(predictions_batch['wall_logits'][idx]))
        actual = np.array(levels_batch.wall_map[idx])
        ax.imshow(pred, cmap='hot', vmin=0, vmax=1)
        ax.contour(actual, levels=[0.5], colors='cyan', linewidths=1)
        acc = ((pred > 0.5) == actual).mean()
        ax.set_title(f'Sample {idx}\nAcc: {acc:.1%}')
        ax.axis('off')

    plt.suptitle('Batch Wall Prediction Summary', y=1.02)
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_position_prediction_heatmap(
    predictions: dict,
    actual_level,
    env_height: int = 13,
    env_width: int = 13,
) -> np.ndarray:
    """Create visualization comparing predicted vs actual positions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    goal_probs = jax.nn.softmax(predictions['goal_logits']).reshape(env_height, env_width)
    im0 = axes[0].imshow(np.array(goal_probs), cmap='hot')
    actual_goal = actual_level.goal_pos
    axes[0].scatter(actual_goal[0], actual_goal[1], c='green', s=100, marker='*', label='Actual')
    axes[0].set_title('Goal Position Prediction')
    axes[0].legend()
    plt.colorbar(im0, ax=axes[0])

    agent_probs = jax.nn.softmax(predictions['agent_pos_logits']).reshape(env_height, env_width)
    im1 = axes[1].imshow(np.array(agent_probs), cmap='hot')
    actual_agent = actual_level.agent_pos
    axes[1].scatter(actual_agent[0], actual_agent[1], c='blue', s=100, marker='^', label='Actual')
    axes[1].set_title('Agent Position Prediction')
    axes[1].legend()
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_probe_loss_by_branch_plot(
    branch_loss_history: chex.Array,
    branch_ptrs: chex.Array,
) -> np.ndarray:
    """Create bar chart comparing probe losses across branches."""
    fig, ax = plt.subplots(figsize=(8, 5))

    branch_names = ['Random (DR)', 'Replay', 'Mutate']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    buffer_size = branch_loss_history.shape[1]
    losses = []
    stds = []

    for i in range(3):
        valid = jnp.minimum(branch_ptrs[i], buffer_size)
        mask = jnp.arange(buffer_size) < valid
        mean_loss = jnp.where(
            mask.sum() > 0,
            (branch_loss_history[i] * mask).sum() / mask.sum(),
            0.0
        )
        std_loss = jnp.where(
            mask.sum() > 1,
            jnp.sqrt(((branch_loss_history[i] - mean_loss) ** 2 * mask).sum() / (mask.sum() - 1)),
            0.0
        )
        losses.append(float(mean_loss))
        stds.append(float(std_loss))

    bars = ax.bar(branch_names, losses, yerr=stds, color=colors, capsize=5, alpha=0.8)
    ax.set_ylabel('Probe Loss (lower = more predictable)')
    ax.set_title('Probe Loss by Curriculum Branch')
    ax.set_ylim(bottom=0)

    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{loss:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_novelty_learnability_plot(
    novelty: float,
    learnability: float,
    score: float,
    regime: str,
) -> np.ndarray:
    """Create Pareto frontier visualization of novelty vs learnability."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    ax.text(0.7, 0.7, 'Open-ended\n(ideal)', ha='center', va='center',
            fontsize=10, alpha=0.5, transform=ax.transAxes)
    ax.text(0.3, 0.7, 'Chaotic', ha='center', va='center',
            fontsize=10, alpha=0.5, transform=ax.transAxes)
    ax.text(0.7, 0.3, 'Converging', ha='center', va='center',
            fontsize=10, alpha=0.5, transform=ax.transAxes)
    ax.text(0.3, 0.3, 'Stagnant', ha='center', va='center',
            fontsize=10, alpha=0.5, transform=ax.transAxes)

    color = {'open-ended': 'green', 'chaotic': 'red', 'converging': 'blue', 'stagnant': 'gray'}[regime]
    ax.scatter([novelty], [learnability], s=200, c=color, marker='o', zorder=5)
    ax.annotate(f'Current\n({regime})\nScore: {score:.3f}',
                (novelty, learnability), textcoords="offset points",
                xytext=(10, 10), fontsize=9)

    ax.set_xlabel('Novelty (curriculum surprise)')
    ax.set_ylabel('Learnability (probe improvement)')
    ax.set_title('Novelty-Learnability Space')

    max_val = max(abs(novelty), abs(learnability), 0.5) * 1.5
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_pareto_trajectory_plot(
    pareto_history: ParetoHistoryState,
) -> np.ndarray:
    """Create Pareto frontier visualization with training trajectory."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    num_points = int(pareto_history.num_checkpoints)
    if num_points < 2:
        for ax in axes:
            ax.text(0.5, 0.5, 'Not enough checkpoints yet',
                   ha='center', va='center', transform=ax.transAxes)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return img

    novelty = np.array(pareto_history.novelty_history[:num_points])
    learnability = np.array(pareto_history.learnability_history[:num_points])
    steps = np.array(pareto_history.training_steps[:num_points])

    ax = axes[0]
    colors = np.linspace(0, 1, num_points)
    scatter = ax.scatter(learnability, novelty, c=colors, cmap='viridis', s=50, alpha=0.7)
    ax.plot(learnability, novelty, 'k-', alpha=0.3, linewidth=1)

    ax.scatter([learnability[0]], [novelty[0]], c='green', s=200, marker='o', label='Start', zorder=5)
    ax.scatter([learnability[-1]], [novelty[-1]], c='red', s=200, marker='s', label='End', zorder=5)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Learnability')
    ax.set_ylabel('Novelty')
    ax.set_title('Pareto Frontier: Novelty vs Learnability')
    ax.legend(loc='upper left')
    plt.colorbar(scatter, ax=ax, label='Training Progress')

    ax = axes[1]
    ax.plot(steps, novelty, 'b-', label='Novelty', linewidth=2)
    ax.plot(steps, learnability, 'g-', label='Learnability', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Metric Value')
    ax.set_title('Novelty and Learnability Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_hidden_state_tsne_plot(
    hstate_samples: chex.Array,
    branch_labels: chex.Array,
    max_samples: int = 500,
) -> np.ndarray:
    """Create t-SNE visualization of hidden states colored by branch."""
    try:
        from sklearn.manifold import TSNE
        use_tsne = True
    except ImportError:
        use_tsne = False

    fig, ax = plt.subplots(figsize=(8, 6))

    n_samples = min(len(hstate_samples), max_samples)
    if n_samples < 10:
        ax.text(0.5, 0.5, 'Not enough samples yet', ha='center', va='center', transform=ax.transAxes)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return img

    indices = np.random.choice(len(hstate_samples), n_samples, replace=False)
    X = np.array(hstate_samples[indices])
    labels = np.array(branch_labels[indices])

    if use_tsne:
        tsne = TSNE(n_components=2, perplexity=min(30, n_samples - 1), random_state=42)
        X_2d = tsne.fit_transform(X)
        method = 't-SNE'
    else:
        X_centered = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        X_2d = X_centered @ Vt[:2].T
        method = 'PCA'

    branch_names = ['Random', 'Replay', 'Mutate']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for i, (name, color) in enumerate(zip(branch_names, colors)):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=name, alpha=0.6, s=20)

    ax.legend()
    ax.set_xlabel(f'{method} Component 1')
    ax.set_ylabel(f'{method} Component 2')
    ax.set_title(f'Hidden State {method} by Branch')

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_n_env_prediction_summary(
    results: dict,
    n_samples_to_show: int = 6,
) -> np.ndarray:
    """
    Create visualization summarizing N-environment predictions.

    Shows accuracy distributions and sample predictions.
    """
    fig = plt.figure(figsize=(16, 10))

    # Top row: Accuracy histograms
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(results['raw_results']['wall_accuracy'], bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(results['mean_wall_accuracy'], color='red', linestyle='--',
                label=f'Mean: {results["mean_wall_accuracy"]:.1%}')
    ax1.axvline(0.5, color='gray', linestyle=':', label='Random: 50%')
    ax1.set_xlabel('Wall Accuracy')
    ax1.set_ylabel('Count')
    ax1.set_title('Wall Prediction Accuracy Distribution')
    ax1.legend()

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(results['raw_results']['goal_accuracy'], bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(results['mean_goal_accuracy'], color='red', linestyle='--',
                label=f'Mean: {results["mean_goal_accuracy"]:.1%}')
    ax2.set_xlabel('Goal Accuracy')
    ax2.set_title('Goal Position Accuracy Distribution')
    ax2.legend()

    ax3 = fig.add_subplot(2, 3, 3)
    metrics = ['Wall', 'Goal', 'Agent Pos', 'Agent Dir']
    means = [
        results['mean_wall_accuracy'],
        results['mean_goal_accuracy'],
        results['mean_agent_pos_accuracy'],
        results['mean_agent_dir_accuracy'],
    ]
    stds = [
        results['std_wall_accuracy'],
        results['std_goal_accuracy'],
        results['std_agent_pos_accuracy'],
        results['std_agent_dir_accuracy'],
    ]
    baselines = [0.5, 1/169, 1/169, 0.25]

    x = np.arange(len(metrics))
    bars = ax3.bar(x, means, yerr=stds, capsize=5, alpha=0.8, label='Probe')
    ax3.scatter(x, baselines, color='red', s=100, marker='x', label='Random Baseline', zorder=5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy Summary')
    ax3.legend()

    # Bottom row: Sample predictions
    n_samples = min(n_samples_to_show, len(results['raw_results']['wall_predictions']))
    indices = np.linspace(0, len(results['raw_results']['wall_predictions']) - 1, n_samples, dtype=int)

    for i, idx in enumerate(indices[:3]):
        ax = fig.add_subplot(2, 3, 4 + i)
        pred = np.array(results['raw_results']['wall_predictions'][idx])
        actual = np.array(results['raw_results']['actual_walls'][idx])
        ax.imshow(pred, cmap='hot', vmin=0, vmax=1)
        ax.contour(actual, levels=[0.5], colors='cyan', linewidths=1)
        acc = results['raw_results']['wall_accuracy'][idx]
        ax.set_title(f'Env {idx}: Acc={acc:.1%}')
        ax.axis('off')

    plt.suptitle(f'N-Environment Prediction Summary (n={results["n_envs"]})', y=1.02)
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


# =============================================================================
# CURRICULUM PREDICTION LOGGING AND VISUALIZATION
# =============================================================================

def log_curriculum_prediction_metrics(
    metrics: dict,
    branch: int,
    update_count: int,
) -> dict:
    """
    Format curriculum prediction metrics for logging by branch type.

    Args:
        metrics: Dict from compute_curriculum_prediction_loss
        branch: Which branch (0=DR, 1=replay, 2=mutate)
        update_count: Current update step

    Returns:
        Dict formatted for wandb logging
    """
    branch_names = {0: "random", 1: "replay", 2: "mutate"}
    branch_name = branch_names.get(branch, "unknown")

    log_dict = {
        f"curriculum_pred/{branch_name}/wall_loss": float(metrics.get("wall_loss", 0.0)),
        f"curriculum_pred/{branch_name}/goal_loss": float(metrics.get("goal_loss", 0.0)),
        f"curriculum_pred/{branch_name}/agent_pos_loss": float(metrics.get("agent_pos_loss", 0.0)),
        f"curriculum_pred/{branch_name}/agent_dir_loss": float(metrics.get("agent_dir_loss", 0.0)),
        f"curriculum_pred/{branch_name}/total_loss": float(metrics.get("total_loss", 0.0)),
        f"curriculum_pred/all/total_loss": float(metrics.get("total_loss", 0.0)),
    }

    return log_dict


def create_prediction_comparison_figure(
    predictions: dict,
    actual_level,
    env_height: int = 13,
    env_width: int = 13,
    n_display: int = 1,
) -> np.ndarray:
    """
    Create side-by-side comparison of predicted vs actual level.

    Shows:
    - Predicted wall probs vs actual wall map
    - Goal distribution vs actual goal
    - Agent position distribution vs actual agent
    - Agent direction distribution

    Args:
        predictions: Dict from CurriculumPredictionHead
        actual_level: The actual Level
        env_height: Environment height
        env_width: Environment width
        n_display: Number of samples (if batch)

    Returns:
        RGB image as numpy array
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Wall predictions
    ax = axes[0, 0]
    wall_probs = jax.nn.sigmoid(predictions['wall_logits'])
    if wall_probs.ndim == 3:  # Batch
        wall_probs = wall_probs[0]
    im = ax.imshow(np.array(wall_probs), cmap='hot', vmin=0, vmax=1)
    ax.set_title('Predicted Wall Probs')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    wall_actual = actual_level.wall_map
    if wall_actual.ndim == 3:  # Batch
        wall_actual = wall_actual[0]
    im = ax.imshow(np.array(wall_actual), cmap='binary', vmin=0, vmax=1)
    ax.set_title('Actual Wall Map')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    error = np.abs(np.array(wall_probs) - np.array(wall_actual.astype(jnp.float32)))
    im = ax.imshow(error, cmap='Reds', vmin=0, vmax=1)
    ax.set_title('Wall Prediction Error')
    plt.colorbar(im, ax=ax)

    # Row 2: Position predictions
    ax = axes[1, 0]
    goal_probs = jax.nn.softmax(predictions['goal_logits'])
    if goal_probs.ndim == 2:  # Batch
        goal_probs = goal_probs[0]
    goal_probs = goal_probs.reshape(env_height, env_width)
    im = ax.imshow(np.array(goal_probs), cmap='hot')
    goal_pos = actual_level.goal_pos
    if goal_pos.ndim == 2:  # Batch
        goal_pos = goal_pos[0]
    ax.scatter(goal_pos[0], goal_pos[1], c='green', s=150, marker='*', edgecolors='white', linewidths=2)
    ax.set_title('Goal Position Distribution')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    agent_probs = jax.nn.softmax(predictions['agent_pos_logits'])
    if agent_probs.ndim == 2:  # Batch
        agent_probs = agent_probs[0]
    agent_probs = agent_probs.reshape(env_height, env_width)
    im = ax.imshow(np.array(agent_probs), cmap='hot')
    agent_pos = actual_level.agent_pos
    if agent_pos.ndim == 2:  # Batch
        agent_pos = agent_pos[0]
    ax.scatter(agent_pos[0], agent_pos[1], c='blue', s=150, marker='^', edgecolors='white', linewidths=2)
    ax.set_title('Agent Position Distribution')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 2]
    dir_probs = jax.nn.softmax(predictions['agent_dir_logits'])
    if dir_probs.ndim == 2:  # Batch
        dir_probs = dir_probs[0]
    dir_names = ['Right', 'Down', 'Left', 'Up']
    bars = ax.bar(dir_names, np.array(dir_probs), alpha=0.8)
    agent_dir = actual_level.agent_dir
    if hasattr(agent_dir, 'shape') and agent_dir.ndim > 0:  # Batch
        agent_dir = agent_dir[0]
    bars[int(agent_dir)].set_color('green')
    ax.set_title('Agent Direction Distribution')
    ax.set_ylabel('Probability')

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_pareto_frontier_figure(
    novelty_history: np.ndarray,
    learnability_history: np.ndarray,
    update_steps: np.ndarray,
) -> np.ndarray:
    """
    Create Pareto frontier visualization with training trajectory.

    Shows:
    - Left: Novelty vs Learnability scatter with trajectory
    - Right: Time series of both metrics

    Args:
        novelty_history: Array of novelty values
        learnability_history: Array of learnability values
        update_steps: Array of training steps

    Returns:
        RGB image as numpy array
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    num_points = len(novelty_history)
    if num_points < 2:
        for ax in axes:
            ax.text(0.5, 0.5, 'Not enough data points',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return img

    # Left plot: Scatter with trajectory
    ax = axes[0]
    colors = np.linspace(0, 1, num_points)
    scatter = ax.scatter(learnability_history, novelty_history, c=colors, cmap='viridis', s=50, alpha=0.7)
    ax.plot(learnability_history, novelty_history, 'k-', alpha=0.3, linewidth=1)

    # Mark start and end
    ax.scatter([learnability_history[0]], [novelty_history[0]], c='green', s=200, marker='o',
              label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax.scatter([learnability_history[-1]], [novelty_history[-1]], c='red', s=200, marker='s',
              label='End', zorder=5, edgecolors='black', linewidths=2)

    # Quadrant lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Quadrant labels
    ax.text(0.75, 0.85, 'Open-ended', ha='center', va='center',
            fontsize=10, alpha=0.5, transform=ax.transAxes, color='green')
    ax.text(0.25, 0.85, 'Chaotic', ha='center', va='center',
            fontsize=10, alpha=0.5, transform=ax.transAxes, color='red')
    ax.text(0.75, 0.15, 'Converging', ha='center', va='center',
            fontsize=10, alpha=0.5, transform=ax.transAxes, color='blue')
    ax.text(0.25, 0.15, 'Stagnant', ha='center', va='center',
            fontsize=10, alpha=0.5, transform=ax.transAxes, color='gray')

    ax.set_xlabel('Learnability')
    ax.set_ylabel('Novelty')
    ax.set_title('Pareto Frontier: Novelty vs Learnability')
    ax.legend(loc='upper left')
    plt.colorbar(scatter, ax=ax, label='Training Progress')

    # Right plot: Time series
    ax = axes[1]
    ax.plot(update_steps, novelty_history, 'b-', label='Novelty', linewidth=2)
    ax.plot(update_steps, learnability_history, 'g-', label='Learnability', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Metric Value')
    ax.set_title('Novelty and Learnability Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_curriculum_prediction_dashboard(
    predictions_batch: dict,
    levels_batch,
    metrics: dict,
    branch_name: str,
    env_height: int = 13,
    env_width: int = 13,
) -> np.ndarray:
    """
    Create comprehensive dashboard for curriculum prediction analysis.

    Shows:
    - Mean predicted vs actual wall patterns
    - Prediction uncertainty
    - Loss breakdown by component
    - Position prediction accuracy

    Args:
        predictions_batch: Batched predictions
        levels_batch: Batch of actual levels
        metrics: Loss metrics dict
        branch_name: Name of the curriculum branch
        env_height: Environment height
        env_width: Environment width

    Returns:
        RGB image as numpy array
    """
    fig = plt.figure(figsize=(18, 12))

    batch_size = predictions_batch['wall_logits'].shape[0]

    # Row 1: Wall predictions
    ax = fig.add_subplot(3, 4, 1)
    wall_probs = jax.nn.sigmoid(predictions_batch['wall_logits'])
    mean_pred = np.array(wall_probs.mean(axis=0))
    im = ax.imshow(mean_pred, cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'Mean Pred (n={batch_size})')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(3, 4, 2)
    mean_actual = np.array(levels_batch.wall_map.astype(jnp.float32).mean(axis=0))
    im = ax.imshow(mean_actual, cmap='hot', vmin=0, vmax=1)
    ax.set_title('Mean Actual')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(3, 4, 3)
    pred_var = np.array(wall_probs.var(axis=0))
    im = ax.imshow(pred_var, cmap='Purples', vmin=0, vmax=0.25)
    ax.set_title('Prediction Variance')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(3, 4, 4)
    mae = np.abs(mean_pred - mean_actual)
    im = ax.imshow(mae, cmap='Reds', vmin=0, vmax=1)
    ax.set_title('Mean Abs Error')
    plt.colorbar(im, ax=ax)

    # Row 2: Position distributions
    ax = fig.add_subplot(3, 4, 5)
    goal_probs = jax.nn.softmax(predictions_batch['goal_logits']).mean(axis=0)
    goal_probs = goal_probs.reshape(env_height, env_width)
    im = ax.imshow(np.array(goal_probs), cmap='hot')
    ax.set_title('Mean Goal Distribution')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(3, 4, 6)
    agent_probs = jax.nn.softmax(predictions_batch['agent_pos_logits']).mean(axis=0)
    agent_probs = agent_probs.reshape(env_height, env_width)
    im = ax.imshow(np.array(agent_probs), cmap='hot')
    ax.set_title('Mean Agent Pos Distribution')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(3, 4, 7)
    dir_probs = jax.nn.softmax(predictions_batch['agent_dir_logits']).mean(axis=0)
    dir_names = ['R', 'D', 'L', 'U']
    ax.bar(dir_names, np.array(dir_probs), alpha=0.8)
    ax.set_title('Mean Dir Distribution')
    ax.set_ylabel('Probability')

    # Row 2 cont: Loss breakdown
    ax = fig.add_subplot(3, 4, 8)
    loss_components = ['Wall', 'Goal', 'Agent Pos', 'Agent Dir']
    loss_values = [
        float(metrics.get('wall_loss', 0)),
        float(metrics.get('goal_loss', 0)),
        float(metrics.get('agent_pos_loss', 0)),
        float(metrics.get('agent_dir_loss', 0)),
    ]
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
    ax.bar(loss_components, loss_values, color=colors, alpha=0.8)
    ax.set_title('Loss Components')
    ax.set_ylabel('Loss')

    # Row 3: Sample predictions
    sample_indices = np.linspace(0, batch_size - 1, 4, dtype=int)
    for i, idx in enumerate(sample_indices):
        ax = fig.add_subplot(3, 4, 9 + i)
        pred = np.array(jax.nn.sigmoid(predictions_batch['wall_logits'][idx]))
        actual = np.array(levels_batch.wall_map[idx])
        ax.imshow(pred, cmap='hot', vmin=0, vmax=1)
        ax.contour(actual, levels=[0.5], colors='cyan', linewidths=1)
        acc = ((pred > 0.5) == actual).mean()
        ax.set_title(f'Sample {idx}: {acc:.1%}')
        ax.axis('off')

    plt.suptitle(f'Curriculum Prediction Dashboard - {branch_name} Branch', y=1.02, fontsize=14)
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


# =============================================================================
# ADDITIONAL VISUALIZATION FUNCTIONS FOR PARITY WITH SCALABLE_OVERSIGHT
# =============================================================================

def create_per_cell_correlation_heatmap(
    correlations: np.ndarray,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> np.ndarray:
    """
    Create heatmap showing per-cell prediction correlation.

    Shows how well the probe predicts walls at each cell position
    across a batch of levels.

    Args:
        correlations: Per-cell correlation values, shape (H, W)
        env_height: Environment height
        env_width: Environment width

    Returns:
        RGB image as numpy array
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    correlations = np.array(correlations)
    im = ax.imshow(correlations, cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_title('Per-Cell Prediction Correlation')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.colorbar(im, ax=ax, label='Correlation')

    # Add text annotations for extreme values
    for i in range(env_height):
        for j in range(env_width):
            if abs(correlations[i, j]) > 0.5:
                color = 'white' if abs(correlations[i, j]) > 0.7 else 'black'
                ax.text(j, i, f'{correlations[i, j]:.2f}', ha='center', va='center',
                       fontsize=6, color=color)

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_distribution_divergence_plot(
    divergence_history: dict,
    update_steps: np.ndarray,
) -> np.ndarray:
    """
    Create plot showing KL/JS divergence over training.

    Args:
        divergence_history: Dict with 'goal_kl', 'goal_js' etc. lists
        update_steps: Array of training steps

    Returns:
        RGB image as numpy array
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: KL/JS divergence over time
    ax = axes[0]
    if 'goal_kl' in divergence_history and len(divergence_history['goal_kl']) > 0:
        ax.plot(update_steps[:len(divergence_history['goal_kl'])],
                divergence_history['goal_kl'], 'b-', label='Goal KL', linewidth=2)
    if 'goal_js' in divergence_history and len(divergence_history['goal_js']) > 0:
        ax.plot(update_steps[:len(divergence_history['goal_js'])],
                divergence_history['goal_js'], 'r--', label='Goal JS', linewidth=2)
    if 'agent_pos_kl' in divergence_history and len(divergence_history['agent_pos_kl']) > 0:
        ax.plot(update_steps[:len(divergence_history['agent_pos_kl'])],
                divergence_history['agent_pos_kl'], 'g-', label='Agent Pos KL', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Divergence')
    ax.set_title('Distribution Divergence Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Wall density error
    ax = axes[1]
    if 'wall_density_error' in divergence_history and len(divergence_history['wall_density_error']) > 0:
        ax.plot(update_steps[:len(divergence_history['wall_density_error'])],
                divergence_history['wall_density_error'], 'b-', linewidth=2)
        ax.fill_between(update_steps[:len(divergence_history['wall_density_error'])],
                        0, divergence_history['wall_density_error'], alpha=0.3)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Wall Density Prediction Error')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_gradient_flow_visualization(
    gradient_norms: dict,
    update_steps: np.ndarray,
) -> np.ndarray:
    """
    Create visualization of gradient flow through the network.

    Shows gradient magnitudes for different network components over training.

    Args:
        gradient_norms: Dict mapping layer names to lists of gradient norms
        update_steps: Array of training steps

    Returns:
        RGB image as numpy array
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Gradient norms by layer
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(gradient_norms)))
    for (layer_name, norms), color in zip(gradient_norms.items(), colors):
        if len(norms) > 0:
            ax.plot(update_steps[:len(norms)], norms, label=layer_name, color=color, alpha=0.8)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Flow by Layer')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Right: Gradient statistics
    ax = axes[1]
    layer_names = list(gradient_norms.keys())
    mean_norms = [np.mean(norms) if len(norms) > 0 else 0 for norms in gradient_norms.values()]
    std_norms = [np.std(norms) if len(norms) > 1 else 0 for norms in gradient_norms.values()]

    x = np.arange(len(layer_names))
    bars = ax.bar(x, mean_norms, yerr=std_norms, capsize=5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Gradient Norm')
    ax.set_title('Gradient Statistics by Layer')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_prediction_confidence_histogram(
    predictions_batch: dict,
    actual_levels_batch,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> np.ndarray:
    """
    Create histogram of prediction confidences and their accuracy.

    Shows calibration: are confident predictions actually correct?

    Args:
        predictions_batch: Dict with batched predictions
        actual_levels_batch: Batch of actual levels
        env_height: Environment height
        env_width: Environment width

    Returns:
        RGB image as numpy array
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Wall prediction confidence
    ax = axes[0]
    wall_probs = jax.nn.sigmoid(predictions_batch['wall_logits']).flatten()
    wall_confidence = np.maximum(np.array(wall_probs), 1 - np.array(wall_probs))
    wall_actual = np.array(actual_levels_batch.wall_map).flatten()
    wall_correct = ((np.array(wall_probs) > 0.5) == wall_actual).astype(float)

    # Bin by confidence
    bins = np.linspace(0.5, 1.0, 11)
    bin_indices = np.digitize(wall_confidence, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    bin_counts = np.zeros(len(bins) - 1)
    bin_accuracy = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        bin_counts[i] = mask.sum()
        if mask.sum() > 0:
            bin_accuracy[i] = wall_correct[mask].mean()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.bar(bin_centers, bin_counts / max(bin_counts.sum(), 1), width=0.04, alpha=0.7, label='Proportion')
    ax2 = ax.twinx()
    ax2.plot(bin_centers, bin_accuracy, 'r-o', label='Accuracy', markersize=6)
    ax2.plot([0.5, 1.0], [0.5, 1.0], 'k--', alpha=0.5, label='Perfect calibration')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Proportion of Predictions')
    ax2.set_ylabel('Accuracy', color='r')
    ax.set_title('Wall Prediction Calibration')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Goal prediction confidence
    ax = axes[1]
    goal_probs = jax.nn.softmax(predictions_batch['goal_logits'])
    goal_max_probs = np.array(goal_probs.max(axis=-1))
    ax.hist(goal_max_probs, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(goal_max_probs.mean(), color='r', linestyle='--',
               label=f'Mean: {goal_max_probs.mean():.3f}')
    ax.set_xlabel('Max Probability')
    ax.set_ylabel('Count')
    ax.set_title('Goal Position Prediction Confidence')
    ax.legend()

    # Agent position confidence
    ax = axes[2]
    agent_pos_probs = jax.nn.softmax(predictions_batch['agent_pos_logits'])
    agent_pos_max_probs = np.array(agent_pos_probs.max(axis=-1))
    ax.hist(agent_pos_max_probs, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(agent_pos_max_probs.mean(), color='r', linestyle='--',
               label=f'Mean: {agent_pos_max_probs.mean():.3f}')
    ax.set_xlabel('Max Probability')
    ax.set_ylabel('Count')
    ax.set_title('Agent Position Prediction Confidence')
    ax.legend()

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_curriculum_trajectory_plot(
    curriculum_state,
    max_points: int = 500,
) -> np.ndarray:
    """
    Create visualization of curriculum trajectory over training.

    Shows how curriculum features evolve: wall density, branch distribution,
    scores, etc.

    Args:
        curriculum_state: CurriculumState with history
        max_points: Maximum number of points to display

    Returns:
        RGB image as numpy array
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get history data
    history_length = curriculum_state.recent_wall_densities.shape[0]
    num_filled = min(curriculum_state.head_pointer if not curriculum_state.history_filled else history_length,
                     max_points)

    if num_filled < 2:
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return img

    # Wall density over time
    ax = axes[0, 0]
    wall_densities = np.array(curriculum_state.recent_wall_densities[:num_filled])
    ax.plot(wall_densities, 'b-', alpha=0.7)
    ax.axhline(wall_densities.mean(), color='r', linestyle='--', label=f'Mean: {wall_densities.mean():.3f}')
    ax.set_xlabel('Curriculum Step')
    ax.set_ylabel('Wall Density')
    ax.set_title('Wall Density Over Curriculum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scores over time
    ax = axes[0, 1]
    scores = np.array(curriculum_state.recent_scores[:num_filled])
    ax.plot(scores, 'g-', alpha=0.7)
    ax.axhline(scores.mean(), color='r', linestyle='--', label=f'Mean: {scores.mean():.3f}')
    ax.set_xlabel('Curriculum Step')
    ax.set_ylabel('Regret Score')
    ax.set_title('Level Scores Over Curriculum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Branch distribution (rolling window)
    ax = axes[1, 0]
    branches = np.array(curriculum_state.recent_branches[:num_filled])
    window_size = min(50, num_filled // 5)
    if window_size > 0:
        dr_prop = np.convolve((branches == 0).astype(float), np.ones(window_size)/window_size, mode='valid')
        replay_prop = np.convolve((branches == 1).astype(float), np.ones(window_size)/window_size, mode='valid')
        mutate_prop = np.convolve((branches == 2).astype(float), np.ones(window_size)/window_size, mode='valid')
        x = np.arange(len(dr_prop))
        ax.fill_between(x, 0, dr_prop, alpha=0.7, label='DR', color='#2ecc71')
        ax.fill_between(x, dr_prop, dr_prop + replay_prop, alpha=0.7, label='Replay', color='#3498db')
        ax.fill_between(x, dr_prop + replay_prop, 1, alpha=0.7, label='Mutate', color='#e74c3c')
    ax.set_xlabel('Curriculum Step')
    ax.set_ylabel('Branch Proportion')
    ax.set_title('Branch Distribution (Rolling Window)')
    ax.legend()
    ax.set_ylim(0, 1)

    # Goal/agent position heatmap
    ax = axes[1, 1]
    goal_positions = np.array(curriculum_state.recent_goal_positions[:num_filled])
    goal_heatmap = np.zeros((DEFAULT_ENV_HEIGHT, DEFAULT_ENV_WIDTH))
    for pos in goal_positions:
        if 0 <= pos[0] < DEFAULT_ENV_WIDTH and 0 <= pos[1] < DEFAULT_ENV_HEIGHT:
            goal_heatmap[pos[1], pos[0]] += 1
    goal_heatmap /= max(goal_heatmap.sum(), 1)
    im = ax.imshow(goal_heatmap, cmap='hot')
    ax.set_title('Goal Position Distribution')
    plt.colorbar(im, ax=ax, label='Frequency')

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


# =============================================================================
# BATCH POSITION PREDICTION SUMMARY
# =============================================================================

def create_batch_position_prediction_summary(
    predictions_batch: dict,
    levels_batch,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
    n_samples: int = 4,
) -> np.ndarray:
    """
    Create batch-level position prediction visualization.

    Shows:
    - Row 1: Mean goal distribution, actual goal heatmap, entropy histogram
    - Row 2: Mean agent position, actual agent heatmap, direction accuracy
    - Row 3: Sample predictions with markers

    Args:
        predictions_batch: Dict with batched predictions
        levels_batch: Batch of actual Levels
        env_height: Environment height
        env_width: Environment width
        n_samples: Number of individual samples to show

    Returns:
        RGB image as numpy array
    """
    batch_size = predictions_batch['goal_logits'].shape[0]
    grid_size = env_height * env_width
    n_samples = min(n_samples, batch_size)

    fig = plt.figure(figsize=(16, 12))

    # Row 1: Goal predictions
    ax = fig.add_subplot(3, 4, 1)
    goal_probs = jax.nn.softmax(predictions_batch['goal_logits'])
    mean_goal = goal_probs.mean(axis=0).reshape(env_height, env_width)
    im = ax.imshow(np.array(mean_goal), cmap='hot')
    ax.set_title(f'Mean Goal Distribution (n={batch_size})')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(3, 4, 2)
    # Create actual goal heatmap
    goal_heatmap = np.zeros((env_height, env_width))
    for i in range(batch_size):
        y, x = int(levels_batch.goal_pos[i, 1]), int(levels_batch.goal_pos[i, 0])
        if 0 <= x < env_width and 0 <= y < env_height:
            goal_heatmap[y, x] += 1
    goal_heatmap /= batch_size
    im = ax.imshow(goal_heatmap, cmap='hot')
    ax.set_title('Actual Goal Distribution')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(3, 4, 3)
    # Goal entropy per prediction
    goal_entropy = -jnp.sum(goal_probs * jnp.log(goal_probs + 1e-10), axis=1)
    ax.hist(np.array(goal_entropy), bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.log(grid_size), color='r', linestyle='--', label='Max entropy')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Count')
    ax.set_title('Goal Prediction Entropy')
    ax.legend()

    ax = fig.add_subplot(3, 4, 4)
    # Goal accuracy
    goal_indices = levels_batch.goal_pos[:, 1] * env_width + levels_batch.goal_pos[:, 0]
    goal_correct = (goal_probs.argmax(axis=1) == goal_indices).astype(jnp.float32)
    ax.bar(['Correct', 'Incorrect'], [float(goal_correct.mean()), float(1 - goal_correct.mean())],
           color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Proportion')
    ax.set_title(f'Goal Accuracy: {goal_correct.mean():.1%}')

    # Row 2: Agent position predictions
    ax = fig.add_subplot(3, 4, 5)
    agent_probs = jax.nn.softmax(predictions_batch['agent_pos_logits'])
    mean_agent = agent_probs.mean(axis=0).reshape(env_height, env_width)
    im = ax.imshow(np.array(mean_agent), cmap='hot')
    ax.set_title('Mean Agent Pos Distribution')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(3, 4, 6)
    # Create actual agent heatmap
    agent_heatmap = np.zeros((env_height, env_width))
    for i in range(batch_size):
        y, x = int(levels_batch.agent_pos[i, 1]), int(levels_batch.agent_pos[i, 0])
        if 0 <= x < env_width and 0 <= y < env_height:
            agent_heatmap[y, x] += 1
    agent_heatmap /= batch_size
    im = ax.imshow(agent_heatmap, cmap='hot')
    ax.set_title('Actual Agent Distribution')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(3, 4, 7)
    # Direction distribution comparison
    dir_probs = jax.nn.softmax(predictions_batch['agent_dir_logits'])
    mean_dir_probs = dir_probs.mean(axis=0)
    # Actual direction distribution
    dir_counts = jnp.zeros(4).at[levels_batch.agent_dir].add(1.0)
    actual_dir_dist = dir_counts / batch_size
    x = np.arange(4)
    width = 0.35
    ax.bar(x - width/2, np.array(mean_dir_probs), width, label='Predicted', alpha=0.7)
    ax.bar(x + width/2, np.array(actual_dir_dist), width, label='Actual', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['R', 'D', 'L', 'U'])
    ax.set_ylabel('Probability')
    ax.set_title('Direction Distribution')
    ax.legend()

    ax = fig.add_subplot(3, 4, 8)
    # Agent position accuracy
    agent_indices = levels_batch.agent_pos[:, 1] * env_width + levels_batch.agent_pos[:, 0]
    agent_correct = (agent_probs.argmax(axis=1) == agent_indices).astype(jnp.float32)
    dir_correct = (dir_probs.argmax(axis=1) == levels_batch.agent_dir).astype(jnp.float32)
    metrics = ['Agent Pos', 'Agent Dir']
    values = [float(agent_correct.mean()), float(dir_correct.mean())]
    ax.bar(metrics, values, color=['blue', 'purple'], alpha=0.7)
    ax.set_ylabel('Accuracy')
    ax.set_title('Position/Direction Accuracy')
    ax.set_ylim(0, 1)

    # Row 3: Sample predictions
    sample_indices = np.linspace(0, batch_size - 1, n_samples, dtype=int)
    for i, idx in enumerate(sample_indices):
        ax = fig.add_subplot(3, 4, 9 + i)
        # Show goal distribution
        goal_prob = goal_probs[idx].reshape(env_height, env_width)
        ax.imshow(np.array(goal_prob), cmap='hot')
        # Mark actual goal
        actual_goal = levels_batch.goal_pos[idx]
        ax.scatter(actual_goal[0], actual_goal[1], c='green', s=100, marker='*', edgecolors='white', linewidths=2)
        # Mark actual agent
        actual_agent = levels_batch.agent_pos[idx]
        ax.scatter(actual_agent[0], actual_agent[1], c='blue', s=80, marker='^', edgecolors='white', linewidths=1)
        ax.set_title(f'Sample {idx}')
        ax.axis('off')

    plt.suptitle('Batch Position Prediction Summary', y=1.02)
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


# =============================================================================
# MATCHED PAIRS VISUALIZATION
# =============================================================================

def create_matched_pairs_visualization(
    predictions_batch: dict,
    levels_batch,
    matched_indices: chex.Array,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
    n_pairs: int = 4,
) -> np.ndarray:
    """
    Show greedy-matched prediction/actual pairs.

    Each row shows: Predicted walls | Matched actual walls | Error map

    Args:
        predictions_batch: Dict with batched predictions
        levels_batch: Batch of actual Levels
        matched_indices: Array of matched level indices
        env_height: Environment height
        env_width: Environment width
        n_pairs: Number of pairs to display

    Returns:
        RGB image as numpy array
    """
    batch_size = predictions_batch['wall_logits'].shape[0]
    n_pairs = min(n_pairs, batch_size)

    fig, axes = plt.subplots(n_pairs, 3, figsize=(12, 4 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    # Select pairs spread across batch
    pair_indices = np.linspace(0, batch_size - 1, n_pairs, dtype=int)

    for row, pred_idx in enumerate(pair_indices):
        matched_idx = int(matched_indices[pred_idx])

        # Predicted walls
        wall_probs = np.array(jax.nn.sigmoid(predictions_batch['wall_logits'][pred_idx]))
        axes[row, 0].imshow(wall_probs, cmap='hot', vmin=0, vmax=1)
        axes[row, 0].set_title(f'Prediction {pred_idx}')
        if row == 0:
            axes[row, 0].set_ylabel('Predicted')

        # Matched actual walls
        actual_walls = np.array(levels_batch.wall_map[matched_idx])
        axes[row, 1].imshow(actual_walls, cmap='binary', vmin=0, vmax=1)
        axes[row, 1].set_title(f'Matched Actual {matched_idx}')

        # Error map
        error = np.abs(wall_probs - actual_walls.astype(np.float32))
        axes[row, 2].imshow(error, cmap='Reds', vmin=0, vmax=1)
        acc = ((wall_probs > 0.5) == actual_walls).mean()
        axes[row, 2].set_title(f'Error (Acc: {acc:.1%})')

        for ax in axes[row]:
            ax.axis('off')

    plt.suptitle('Greedy-Matched Prediction Pairs', y=1.02)
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


# =============================================================================
# REPLAY TO MUTATE HEATMAP (CRITICAL)
# =============================================================================

def create_replay_to_mutate_heatmap(
    predictions_batch: dict,
    mutate_levels_batch,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
    n_samples: int = 4,
) -> np.ndarray:
    """
    CRITICAL: Show per-instance R→M correspondence.

    For Replay->Mutate transition, prediction[i] corresponds to mutate(replay[i]).
    This visualization shows the true 1-to-1 matching.

    Args:
        predictions_batch: Dict with predictions (made from replay hidden state)
        mutate_levels_batch: Batch of mutated levels (mutation of replay levels)
        env_height: Environment height
        env_width: Environment width
        n_samples: Number of samples to display

    Returns:
        RGB image as numpy array
    """
    batch_size = predictions_batch['wall_logits'].shape[0]
    n_samples = min(n_samples, batch_size)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    # Compute per-instance metrics
    per_instance = compute_per_instance_calibration_batch(
        predictions_batch, mutate_levels_batch, env_height, env_width
    )

    # Select samples spread across batch
    sample_indices = np.linspace(0, batch_size - 1, n_samples, dtype=int)

    for row, idx in enumerate(sample_indices):
        # Predicted walls
        wall_probs = np.array(jax.nn.sigmoid(predictions_batch['wall_logits'][idx]))
        im = axes[row, 0].imshow(wall_probs, cmap='hot', vmin=0, vmax=1)
        axes[row, 0].set_title(f'Pred[{idx}] Wall Probs')

        # Actual mutated walls
        actual_walls = np.array(mutate_levels_batch.wall_map[idx])
        axes[row, 1].imshow(actual_walls, cmap='binary', vmin=0, vmax=1)
        axes[row, 1].set_title(f'Actual[{idx}] Mutated')

        # Error map
        error = np.abs(wall_probs - actual_walls.astype(np.float32))
        axes[row, 2].imshow(error, cmap='Reds', vmin=0, vmax=1)
        wall_acc = ((wall_probs > 0.5) == actual_walls).mean()
        axes[row, 2].set_title(f'Error (Acc: {wall_acc:.1%})')

        # Position predictions
        ax = axes[row, 3]
        goal_probs = jax.nn.softmax(predictions_batch['goal_logits'][idx]).reshape(env_height, env_width)
        ax.imshow(np.array(goal_probs), cmap='hot')
        # Mark actual goal
        actual_goal = mutate_levels_batch.goal_pos[idx]
        ax.scatter(actual_goal[0], actual_goal[1], c='green', s=150, marker='*',
                   edgecolors='white', linewidths=2, label='Goal')
        # Mark actual agent
        actual_agent = mutate_levels_batch.agent_pos[idx]
        ax.scatter(actual_agent[0], actual_agent[1], c='blue', s=100, marker='^',
                   edgecolors='white', linewidths=2, label='Agent')
        ax.set_title('Goal/Agent Pred')
        ax.legend(loc='lower right', fontsize=8)

        for ax in axes[row]:
            ax.axis('off')

    # Add overall metrics
    plt.suptitle(
        f'Replay→Mutate Per-Instance (1-to-1) | '
        f'Wall Acc: {per_instance["wall_accuracy"]:.1%} | '
        f'Goal Acc: {per_instance["goal_accuracy"]:.1%} | '
        f'Combined: {per_instance["combined_accuracy"]:.1%}',
        y=1.02, fontsize=12
    )
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


# =============================================================================
# CORRELATION SCATTER PLOT
# =============================================================================

def create_correlation_scatter_plot(
    probe_accuracy_history: chex.Array,
    agent_returns_history: chex.Array,
    valid_samples: int,
) -> np.ndarray:
    """
    Create scatter plot of probe accuracy vs agent returns.

    Shows correlation between environment understanding and performance.

    Args:
        probe_accuracy_history: Buffer of probe accuracy values
        agent_returns_history: Buffer of agent return values
        valid_samples: Number of valid samples

    Returns:
        RGB image as numpy array
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if valid_samples < 10:
        ax.text(0.5, 0.5, 'Not enough samples yet', ha='center', va='center', transform=ax.transAxes)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return img

    buffer_size = probe_accuracy_history.shape[0]
    n = min(valid_samples, buffer_size)

    accuracy = np.array(probe_accuracy_history[:n])
    returns = np.array(agent_returns_history[:n])

    # Scatter plot
    scatter = ax.scatter(accuracy, returns, alpha=0.5, s=20, c=np.arange(n), cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Time Step')

    # Trend line
    if n > 2:
        z = np.polyfit(accuracy, returns, 1)
        p = np.poly1d(z)
        x_line = np.linspace(accuracy.min(), accuracy.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.3f})')

        # Compute correlation
        corr = np.corrcoef(accuracy, returns)[0, 1]
        ax.set_title(f'Probe Accuracy vs Agent Return (r={corr:.3f})')
    else:
        ax.set_title('Probe Accuracy vs Agent Return')

    ax.set_xlabel('Probe Accuracy')
    ax.set_ylabel('Agent Return')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


# =============================================================================
# INFORMATION CONTENT DASHBOARD (CRITICAL)
# =============================================================================

def create_information_content_dashboard(
    probe_metrics: dict,
    probe_tracking: ProbeTrackingState,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> np.ndarray:
    """
    CRITICAL: 4-panel information content dashboard.

    Shows:
    - Panel 1: Information table (accuracies vs random baseline)
    - Panel 2: Information gain by branch
    - Panel 3: Loss over training
    - Panel 4: Component breakdown

    Args:
        probe_metrics: Dict with current probe metrics
        probe_tracking: ProbeTrackingState with history
        env_height: Environment height
        env_width: Environment width

    Returns:
        RGB image as numpy array
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get baselines
    baselines = compute_random_baselines(env_height, env_width)

    # Panel 1: Information table
    ax = axes[0, 0]
    ax.axis('off')

    # Create table data
    table_data = []
    headers = ['Component', 'Probe', 'Random', 'Info Gain', 'Relative Gain']

    components = [
        ('Wall Accuracy', 'wall_accuracy', baselines['wall_accuracy']),
        ('Goal Top-1', 'goal_accuracy', baselines['goal_top1']),
        ('Agent Pos', 'agent_pos_accuracy', baselines['agent_pos_top1']),
        ('Agent Dir', 'agent_dir_accuracy', baselines['agent_dir']),
    ]

    for name, key, baseline in components:
        probe_val = probe_metrics.get(key, 0.0)
        info_gain = probe_val - baseline
        relative = info_gain / (1.0 - baseline) if baseline < 1.0 else 0.0
        table_data.append([name, f'{probe_val:.3f}', f'{baseline:.3f}', f'{info_gain:+.3f}', f'{relative:.1%}'])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Information Content Summary', fontsize=12, fontweight='bold', pad=20)

    # Panel 2: Information gain by branch
    ax = axes[0, 1]
    branch_names = ['Random', 'Replay', 'Mutate']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    buffer_size = probe_tracking.branch_loss_history.shape[1]
    gains = []
    for i in range(3):
        branch_ptr = int(probe_tracking.branch_ptrs[i])
        if branch_ptr > 0:
            branch_valid = min(branch_ptr, buffer_size)
            branch_loss = float(probe_tracking.branch_loss_history[i, :branch_valid].mean())
            # Info gain = random_loss - probe_loss
            gain = baselines['total_loss'] - branch_loss
            gains.append(gain)
        else:
            gains.append(0.0)

    bars = ax.bar(branch_names, gains, color=colors, alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Information Gain (bits)')
    ax.set_title('Information Gain by Branch')

    for bar, gain in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{gain:.2f}', ha='center', va='bottom', fontsize=10)

    # Panel 3: Loss over training
    ax = axes[1, 0]
    valid_samples = min(probe_tracking.total_samples, buffer_size)
    if valid_samples > 10:
        loss_history = np.array(probe_tracking.loss_history[:valid_samples])
        step_history = np.array(probe_tracking.training_step_history[:valid_samples])

        # Sort by step
        sort_idx = np.argsort(step_history)
        loss_sorted = loss_history[sort_idx]
        step_sorted = step_history[sort_idx]

        ax.plot(step_sorted, loss_sorted, 'b-', alpha=0.3, linewidth=1, label='Raw')

        # Smoothed version
        window = max(1, len(loss_sorted) // 20)
        if window > 1:
            loss_smooth = np.convolve(loss_sorted, np.ones(window)/window, mode='valid')
            step_smooth = step_sorted[window//2:window//2 + len(loss_smooth)]
            ax.plot(step_smooth, loss_smooth, 'b-', linewidth=2, label='Smoothed')

        ax.axhline(y=baselines['total_loss'], color='r', linestyle='--', alpha=0.7, label='Random Baseline')
    else:
        ax.text(0.5, 0.5, 'Not enough samples', ha='center', va='center', transform=ax.transAxes)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Probe Loss')
    ax.set_title('Loss Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Component breakdown
    ax = axes[1, 1]
    components = ['Wall', 'Goal', 'Agent Pos', 'Agent Dir']
    probe_losses = [
        probe_metrics.get('wall_loss', baselines['wall_loss']),
        probe_metrics.get('goal_loss', baselines['goal_loss']),
        probe_metrics.get('agent_pos_loss', baselines['agent_pos_loss']),
        probe_metrics.get('agent_dir_loss', baselines['agent_dir_loss']),
    ]
    random_losses = [
        baselines['wall_loss'],
        baselines['goal_loss'],
        baselines['agent_pos_loss'],
        baselines['agent_dir_loss'],
    ]

    x = np.arange(len(components))
    width = 0.35
    bars1 = ax.bar(x - width/2, probe_losses, width, label='Probe', alpha=0.8)
    bars2 = ax.bar(x + width/2, random_losses, width, label='Random', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.set_ylabel('Loss')
    ax.set_title('Component-wise Loss Comparison')
    ax.legend()

    plt.suptitle('Information Content Dashboard', y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


# =============================================================================
# PAIRED-SPECIFIC VISUALIZATIONS
# =============================================================================

def create_regret_dynamics_plot(
    pro_returns_history: np.ndarray,
    ant_returns_history: np.ndarray,
    training_steps: np.ndarray,
) -> np.ndarray:
    """
    3-panel PAIRED regret visualization.

    Shows:
    - Top: Protagonist vs Antagonist returns over time
    - Middle: Regret (ant_max - pro_mean) over time
    - Bottom: Rolling regret variance (stability)

    Args:
        pro_returns_history: Protagonist mean returns history
        ant_returns_history: Antagonist max returns history
        training_steps: Training step history

    Returns:
        RGB image as numpy array
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    n_points = len(pro_returns_history)
    if n_points < 2:
        for ax in axes:
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return img

    # Compute regret
    regret_history = ant_returns_history - pro_returns_history

    # Panel 1: Returns comparison
    ax = axes[0]
    ax.plot(training_steps, pro_returns_history, label='Protagonist', color='blue', linewidth=2)
    ax.plot(training_steps, ant_returns_history, label='Antagonist', color='red', linewidth=2)
    ax.fill_between(training_steps, pro_returns_history, ant_returns_history,
                    alpha=0.3, color='gray', label='Regret Gap')
    ax.legend()
    ax.set_ylabel('Returns')
    ax.set_title('Protagonist vs Antagonist Returns')
    ax.grid(True, alpha=0.3)

    # Panel 2: Regret over time
    ax = axes[1]
    ax.plot(training_steps, regret_history, color='purple', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(training_steps, 0, regret_history, alpha=0.3, color='purple')
    ax.set_ylabel('Regret')
    ax.set_title('Estimated Regret (Ant - Pro)')
    ax.grid(True, alpha=0.3)

    # Panel 3: Rolling variance
    ax = axes[2]
    window = max(5, n_points // 20)
    if window > 1 and n_points > window:
        # Compute rolling variance manually
        rolling_var = []
        for i in range(window, n_points):
            window_data = regret_history[i-window:i]
            rolling_var.append(np.var(window_data))
        rolling_var = np.array(rolling_var)
        ax.plot(training_steps[window:], rolling_var, color='orange', linewidth=2)
        ax.fill_between(training_steps[window:], 0, rolling_var, alpha=0.3, color='orange')
    else:
        ax.text(0.5, 0.5, 'Not enough data for rolling variance', ha='center', va='center', transform=ax.transAxes)
    ax.set_ylabel('Regret Variance')
    ax.set_xlabel('Training Steps')
    ax.set_title(f'Rolling Regret Variance (window={window})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_paired_dashboard(
    pro_metrics: dict,
    ant_metrics: dict,
    adv_metrics: dict,
    level_features: dict,
    regret_history: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    6-panel PAIRED comprehensive dashboard.

    Shows:
    - (0,0): Pro vs Ant returns scatter (per-level)
    - (0,1): Regret distribution histogram
    - (0,2): Level difficulty (wall density) distribution
    - (1,0): Pro loss components
    - (1,1): Ant loss components
    - (1,2): Regret statistics summary

    Args:
        pro_metrics: Protagonist metrics dict
        ant_metrics: Antagonist metrics dict
        adv_metrics: Adversary metrics dict
        level_features: Level feature statistics
        regret_history: Optional regret history array

    Returns:
        RGB image as numpy array
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Panel (0,0): Pro vs Ant returns per level
    ax = axes[0, 0]
    pro_returns = np.array(pro_metrics.get('per_level_returns', [0]))
    ant_returns = np.array(ant_metrics.get('per_level_returns', [0]))
    if len(pro_returns) > 1:
        ax.scatter(pro_returns, ant_returns, alpha=0.5, s=30)
        # Diagonal line (equal performance)
        max_val = max(pro_returns.max(), ant_returns.max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal')
        ax.set_xlabel('Protagonist Return')
        ax.set_ylabel('Antagonist Return')
    else:
        ax.text(0.5, 0.5, 'No per-level data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Per-Level Returns')
    ax.legend()

    # Panel (0,1): Regret distribution
    ax = axes[0, 1]
    if regret_history is not None and len(regret_history) > 1:
        ax.hist(regret_history, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(regret_history.mean(), color='r', linestyle='--',
                   label=f'Mean: {regret_history.mean():.3f}')
        ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
    else:
        ax.text(0.5, 0.5, 'No regret history', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Regret')
    ax.set_ylabel('Count')
    ax.set_title('Regret Distribution')
    ax.legend()

    # Panel (0,2): Wall density distribution
    ax = axes[0, 2]
    wall_density = level_features.get('wall_density', 0)
    if isinstance(wall_density, (list, np.ndarray)) and len(wall_density) > 1:
        ax.hist(wall_density, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(wall_density), color='r', linestyle='--',
                   label=f'Mean: {np.mean(wall_density):.3f}')
    else:
        ax.bar(['Mean Density'], [float(wall_density)], alpha=0.7)
    ax.set_xlabel('Wall Density')
    ax.set_ylabel('Count')
    ax.set_title('Level Difficulty Distribution')
    ax.legend()

    # Panel (1,0): Protagonist loss components
    ax = axes[1, 0]
    pro_loss_components = ['Actor', 'Critic', 'Entropy', 'Total']
    pro_loss_values = [
        float(pro_metrics.get('actor_loss', 0)),
        float(pro_metrics.get('critic_loss', 0)),
        float(pro_metrics.get('entropy_loss', 0)),
        float(pro_metrics.get('total_loss', 0)),
    ]
    ax.bar(pro_loss_components, pro_loss_values, color='blue', alpha=0.7)
    ax.set_ylabel('Loss')
    ax.set_title('Protagonist Loss Components')

    # Panel (1,1): Antagonist loss components
    ax = axes[1, 1]
    ant_loss_values = [
        float(ant_metrics.get('actor_loss', 0)),
        float(ant_metrics.get('critic_loss', 0)),
        float(ant_metrics.get('entropy_loss', 0)),
        float(ant_metrics.get('total_loss', 0)),
    ]
    ax.bar(pro_loss_components, ant_loss_values, color='red', alpha=0.7)
    ax.set_ylabel('Loss')
    ax.set_title('Antagonist Loss Components')

    # Panel (1,2): Regret statistics summary
    ax = axes[1, 2]
    ax.axis('off')

    # Create summary table
    if regret_history is not None and len(regret_history) > 1:
        summary_data = [
            ['Mean Regret', f'{regret_history.mean():.4f}'],
            ['Std Regret', f'{regret_history.std():.4f}'],
            ['Max Regret', f'{regret_history.max():.4f}'],
            ['Min Regret', f'{regret_history.min():.4f}'],
            ['Adversary Success %', f'{(regret_history > 0).mean():.1%}'],
        ]
    else:
        summary_data = [
            ['Est Regret', f'{pro_metrics.get("est_regret", 0):.4f}'],
            ['Pro Solve Rate', f'{pro_metrics.get("solve_rate", 0):.1%}'],
            ['Ant Solve Rate', f'{ant_metrics.get("solve_rate", 0):.1%}'],
        ]

    table = ax.table(
        cellText=summary_data,
        colLabels=['Metric', 'Value'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    ax.set_title('Regret Statistics', fontsize=12, pad=20)

    plt.suptitle('PAIRED Training Dashboard', y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_adversary_level_analysis(
    levels,
    pro_solve_rates: np.ndarray,
    ant_solve_rates: np.ndarray,
    env_height: int = DEFAULT_ENV_HEIGHT,
    env_width: int = DEFAULT_ENV_WIDTH,
) -> np.ndarray:
    """
    4-panel adversary level analysis.

    Shows:
    - Level difficulty distribution (wall density histogram)
    - Solvability gap (ant_rate - pro_rate per level)
    - Goal position heatmap
    - Difficulty vs solvability scatter

    Args:
        levels: Batch of generated levels
        pro_solve_rates: Protagonist solve rates per level
        ant_solve_rates: Antagonist solve rates per level
        env_height: Environment height
        env_width: Environment width

    Returns:
        RGB image as numpy array
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    batch_size = levels.wall_map.shape[0]

    # Panel (0,0): Wall density histogram
    ax = axes[0, 0]
    wall_counts = levels.wall_map.sum(axis=(1, 2))
    grid_size = env_height * env_width
    wall_densities = np.array(wall_counts / grid_size)
    ax.hist(wall_densities, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(wall_densities.mean(), color='r', linestyle='--',
               label=f'Mean: {wall_densities.mean():.3f}')
    ax.set_xlabel('Wall Density')
    ax.set_ylabel('Count')
    ax.set_title('Level Difficulty Distribution')
    ax.legend()

    # Panel (0,1): Solvability gap
    ax = axes[0, 1]
    solvability_gap = np.array(ant_solve_rates) - np.array(pro_solve_rates)
    ax.hist(solvability_gap, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(solvability_gap.mean(), color='r', linestyle='--',
               label=f'Mean: {solvability_gap.mean():.3f}')
    ax.set_xlabel('Solvability Gap (Ant - Pro)')
    ax.set_ylabel('Count')
    ax.set_title('Per-Level Solvability Gap')
    ax.legend()

    # Panel (1,0): Goal position heatmap
    ax = axes[1, 0]
    goal_heatmap = np.zeros((env_height, env_width))
    for i in range(batch_size):
        y, x = int(levels.goal_pos[i, 1]), int(levels.goal_pos[i, 0])
        if 0 <= x < env_width and 0 <= y < env_height:
            goal_heatmap[y, x] += 1
    goal_heatmap /= max(1, batch_size)
    im = ax.imshow(goal_heatmap, cmap='hot')
    ax.set_title('Goal Position Distribution')
    plt.colorbar(im, ax=ax, label='Frequency')

    # Panel (1,1): Difficulty vs solvability scatter
    ax = axes[1, 1]
    ax.scatter(wall_densities, pro_solve_rates, alpha=0.5, label='Protagonist', color='blue', s=20)
    ax.scatter(wall_densities, ant_solve_rates, alpha=0.5, label='Antagonist', color='red', s=20)
    ax.set_xlabel('Wall Density (Difficulty)')
    ax.set_ylabel('Solve Rate')
    ax.set_title('Difficulty vs Solvability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Adversary Level Analysis', y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img


def create_paired_openendedness_plot(
    novelty_history: np.ndarray,
    learnability_history: np.ndarray,
    regret_history: np.ndarray,
) -> np.ndarray:
    """
    PAIRED-specific Pareto plot with regret coloring.

    Shows:
    - X: Novelty (level diversity)
    - Y: Learnability (protagonist improvement)
    - Color: Regret magnitude
    - Quadrants marked

    Args:
        novelty_history: Array of novelty values
        learnability_history: Array of learnability values
        regret_history: Array of regret values for coloring

    Returns:
        RGB image as numpy array
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    n_points = len(novelty_history)
    if n_points < 2:
        ax.text(0.5, 0.5, 'Not enough data points', ha='center', va='center', transform=ax.transAxes)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return img

    # Scatter plot colored by regret
    scatter = ax.scatter(
        novelty_history, learnability_history,
        c=regret_history, cmap='viridis', alpha=0.7, s=50
    )
    plt.colorbar(scatter, ax=ax, label='Regret')

    # Connect points with trajectory line
    ax.plot(novelty_history, learnability_history, 'k-', alpha=0.3, linewidth=1)

    # Mark start and end
    ax.scatter([novelty_history[0]], [learnability_history[0]], c='green', s=200,
               marker='o', label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax.scatter([novelty_history[-1]], [learnability_history[-1]], c='red', s=200,
               marker='s', label='End', zorder=5, edgecolors='black', linewidths=2)

    # Quadrant lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Quadrant labels
    ax.text(0.75, 0.85, 'Open-ended', ha='center', va='center',
            fontsize=10, alpha=0.5, transform=ax.transAxes, color='green')
    ax.text(0.25, 0.85, 'Chaotic', ha='center', va='center',
            fontsize=10, alpha=0.5, transform=ax.transAxes, color='red')
    ax.text(0.75, 0.15, 'Converging', ha='center', va='center',
            fontsize=10, alpha=0.5, transform=ax.transAxes, color='blue')
    ax.text(0.25, 0.15, 'Stagnant', ha='center', va='center',
            fontsize=10, alpha=0.5, transform=ax.transAxes, color='gray')

    ax.set_xlabel('Novelty (level diversity)')
    ax.set_ylabel('Learnability (protagonist improvement)')
    ax.set_title('PAIRED Open-Endedness Space')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img