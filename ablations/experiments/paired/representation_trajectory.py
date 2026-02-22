"""
D1: Representation Trajectory.

Decompose representation drift into adversary-driven vs autonomous components.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment


@dataclass
class TrajectoryPoint:
    """Single point in representation trajectory."""
    step: int
    hstate_mean: np.ndarray
    hstate_cov: np.ndarray
    adversary_features: Dict[str, float]
    value_mean: float
    policy_entropy: float


class RepresentationTrajectoryExperiment(CheckpointExperiment):
    """
    Decompose representation drift into adversary-driven vs autonomous.

    Protocol:
    1. Track h-state trajectory over training
    2. Fit VAR model: h_{t+1} = A * h_t + B * adv_t + ε
    3. Decompose variance into adversary-driven and autonomous
    4. Analyze steering efficiency
    """

    @property
    def name(self) -> str:
        return "representation_trajectory"

    def __init__(
        self,
        n_samples_per_step: int = 50,
        trajectory_length: int = 100,
        hidden_dim: int = 256,
        reduced_dim: int = 20,  # For tractable VAR
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_samples_per_step = n_samples_per_step
        self.trajectory_length = trajectory_length
        self.hidden_dim = hidden_dim
        self.reduced_dim = reduced_dim
        self._trajectory: List[TrajectoryPoint] = []
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"RepresentationTrajectoryExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> List[TrajectoryPoint]:
        """Collect representation trajectory data."""
        for t in range(self.trajectory_length):
            rng, step_rng = jax.random.split(rng)
            point = self._collect_trajectory_point(step_rng, t)
            self._trajectory.append(point)

        return self._trajectory

    def _collect_trajectory_point(
        self,
        rng: chex.PRNGKey,
        step: int,
    ) -> TrajectoryPoint:
        """Collect data for a single trajectory point."""
        hstates = []
        values = []
        entropies = []

        # Simulate adversary features at this step
        adversary_features = {
            'difficulty': 0.3 + 0.4 * (step / self.trajectory_length) + float(jax.random.uniform(rng)) * 0.1,
            'wall_density_target': 0.1 + 0.2 * (step / self.trajectory_length),
        }

        for i in range(self.n_samples_per_step):
            rng, sample_rng, h_rng = jax.random.split(rng, 3)

            # Sample hidden state with trajectory-dependent structure
            h = np.array(jax.random.normal(h_rng, (self.hidden_dim,)))

            # Add drift over training
            h[:50] += step * 0.01  # Autonomous drift
            # Add adversary influence
            h[50:100] += adversary_features['difficulty'] * 2.0

            hstates.append(h)

            # Simulated value and entropy
            value = 0.7 - adversary_features['difficulty'] * 0.3 + float(jax.random.uniform(sample_rng)) * 0.1
            values.append(value)

            entropy = 1.5 + adversary_features['difficulty'] * 0.5
            entropies.append(entropy)

        hstates = np.array(hstates)

        return TrajectoryPoint(
            step=step,
            hstate_mean=hstates.mean(axis=0),
            hstate_cov=np.cov(hstates.T),
            adversary_features=adversary_features,
            value_mean=float(np.mean(values)),
            policy_entropy=float(np.mean(entropies)),
        )

    def analyze(self) -> Dict[str, Any]:
        """Analyze representation trajectory."""
        if not self._trajectory:
            raise ValueError("Must call collect_data first")

        results = {}

        # Fit VAR model
        var_results = self._fit_var_model()
        results['var_model'] = var_results

        # Decompose variance
        results['adversary_driven_fraction'] = self._compute_adversary_fraction(var_results)
        results['autonomous_fraction'] = 1.0 - results['adversary_driven_fraction']

        # Representation velocity
        results['representation_velocity'] = self._compute_velocity()

        # Directedness
        results['directedness'] = self._compute_directedness()

        # Steering efficiency
        results['adversary_steering_efficiency'] = self._compute_steering_efficiency(var_results)

        # Trajectory summary
        results['trajectory_summary'] = self._summarize_trajectory()

        return results

    def _fit_var_model(self) -> Dict[str, Any]:
        """Fit VAR model: h_{t+1} = A * h_t + B * adv_t + ε."""
        # Reduce dimensionality for tractable fitting
        from sklearn.decomposition import PCA

        # Get trajectory of mean h-states
        H = np.array([p.hstate_mean for p in self._trajectory])

        # PCA reduction
        pca = PCA(n_components=self.reduced_dim)
        H_reduced = pca.fit_transform(H)

        # Get adversary features
        adv_features = np.array([
            [p.adversary_features['difficulty'], p.adversary_features['wall_density_target']]
            for p in self._trajectory
        ])

        # Fit: h_{t+1} = A * h_t + B * adv_t + c
        n = len(H_reduced) - 1
        if n < 10:
            return {'error': 'Insufficient data'}

        # Prepare data
        Y = H_reduced[1:]  # h_{t+1}
        X_h = H_reduced[:-1]  # h_t
        X_adv = adv_features[:-1]  # adv_t

        # Full design matrix
        X = np.hstack([X_h, X_adv, np.ones((n, 1))])

        # Fit via least squares
        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {'error': 'Fit failed'}

        # Extract A, B, c
        A = beta[:self.reduced_dim, :]  # Autoregressive
        B = beta[self.reduced_dim:self.reduced_dim + 2, :]  # Adversary effect
        c = beta[-1, :]  # Intercept

        # Compute residuals
        Y_pred = X @ beta
        residuals = Y - Y_pred
        residual_var = residuals.var(axis=0).mean()

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

        return {
            'A_norm': float(np.linalg.norm(A)),
            'B_norm': float(np.linalg.norm(B)),
            'residual_variance': float(residual_var),
            'r2': float(r2),
            'pca_explained_variance': float(pca.explained_variance_ratio_.sum()),
        }

    def _compute_adversary_fraction(self, var_results: Dict[str, Any]) -> float:
        """Compute fraction of variance driven by adversary."""
        if 'error' in var_results:
            return 0.5

        # Rough decomposition: |B|^2 / (|A|^2 + |B|^2)
        a_norm = var_results['A_norm']
        b_norm = var_results['B_norm']

        total = a_norm ** 2 + b_norm ** 2
        if total < 1e-10:
            return 0.5

        return float(b_norm ** 2 / total)

    def _compute_velocity(self) -> Dict[str, float]:
        """Compute representation velocity over trajectory."""
        velocities = []

        for i in range(1, len(self._trajectory)):
            h_curr = self._trajectory[i].hstate_mean
            h_prev = self._trajectory[i - 1].hstate_mean
            velocity = np.linalg.norm(h_curr - h_prev)
            velocities.append(velocity)

        return {
            'mean_velocity': float(np.mean(velocities)),
            'velocity_std': float(np.std(velocities)),
            'max_velocity': float(np.max(velocities)),
            'velocity_trend': float(np.polyfit(range(len(velocities)), velocities, 1)[0]),
        }

    def _compute_directedness(self) -> float:
        """Compute how directed the trajectory is (vs random walk)."""
        if len(self._trajectory) < 3:
            return 0.0

        # Directedness = displacement / path length
        h_start = self._trajectory[0].hstate_mean
        h_end = self._trajectory[-1].hstate_mean
        displacement = np.linalg.norm(h_end - h_start)

        path_length = 0.0
        for i in range(1, len(self._trajectory)):
            h_curr = self._trajectory[i].hstate_mean
            h_prev = self._trajectory[i - 1].hstate_mean
            path_length += np.linalg.norm(h_curr - h_prev)

        if path_length < 1e-10:
            return 0.0

        return float(displacement / path_length)

    def _compute_steering_efficiency(self, var_results: Dict[str, Any]) -> float:
        """Compute how efficiently adversary steers representations."""
        if 'error' in var_results:
            return 0.0

        # Steering efficiency = adversary_fraction * directedness * (1 - residual_var)
        adv_fraction = self._compute_adversary_fraction(var_results)
        directedness = self._compute_directedness()
        residual_var = var_results['residual_variance']

        # Normalize residual variance (assuming typical range)
        residual_factor = max(0, 1 - residual_var / 10.0)

        return float(adv_fraction * directedness * residual_factor)

    def _summarize_trajectory(self) -> Dict[str, Any]:
        """Summarize trajectory statistics."""
        values = [p.value_mean for p in self._trajectory]
        entropies = [p.policy_entropy for p in self._trajectory]
        difficulties = [p.adversary_features['difficulty'] for p in self._trajectory]

        return {
            'value_trajectory': {
                'start': float(values[0]),
                'end': float(values[-1]),
                'mean': float(np.mean(values)),
                'trend': float(np.polyfit(range(len(values)), values, 1)[0]),
            },
            'entropy_trajectory': {
                'start': float(entropies[0]),
                'end': float(entropies[-1]),
                'mean': float(np.mean(entropies)),
            },
            'difficulty_trajectory': {
                'start': float(difficulties[0]),
                'end': float(difficulties[-1]),
                'range': float(max(difficulties) - min(difficulties)),
            },
        }

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize representation trajectory."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._trajectory:
            return figures

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        steps = [p.step for p in self._trajectory]

        # Velocity trajectory
        ax = axes[0, 0]
        velocities = []
        for i in range(1, len(self._trajectory)):
            h_curr = self._trajectory[i].hstate_mean
            h_prev = self._trajectory[i - 1].hstate_mean
            velocities.append(np.linalg.norm(h_curr - h_prev))
        ax.plot(steps[1:], velocities, 'b-', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Representation Velocity')
        ax.set_title('Representation Drift Speed')
        ax.grid(True, alpha=0.3)

        # Value and difficulty
        ax = axes[0, 1]
        values = [p.value_mean for p in self._trajectory]
        difficulties = [p.adversary_features['difficulty'] for p in self._trajectory]
        ax.plot(steps, values, 'g-', label='Value', linewidth=2)
        ax.plot(steps, difficulties, 'r-', label='Difficulty', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title('Value vs Adversary Difficulty')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2D PCA trajectory
        ax = axes[1, 0]
        from sklearn.decomposition import PCA
        H = np.array([p.hstate_mean for p in self._trajectory])
        pca = PCA(n_components=2)
        H_2d = pca.fit_transform(H)
        # Color by step
        scatter = ax.scatter(H_2d[:, 0], H_2d[:, 1], c=steps, cmap='viridis', s=50)
        ax.plot(H_2d[:, 0], H_2d[:, 1], 'k-', alpha=0.3, linewidth=1)
        ax.scatter(H_2d[0, 0], H_2d[0, 1], c='g', s=200, marker='o', label='Start', zorder=5)
        ax.scatter(H_2d[-1, 0], H_2d[-1, 1], c='r', s=200, marker='*', label='End', zorder=5)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Representation Trajectory (PCA)')
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Step')

        # Cumulative distance
        ax = axes[1, 1]
        cum_dist = [0.0]
        displacement = [0.0]
        h_start = self._trajectory[0].hstate_mean
        for i in range(1, len(self._trajectory)):
            h_curr = self._trajectory[i].hstate_mean
            h_prev = self._trajectory[i - 1].hstate_mean
            cum_dist.append(cum_dist[-1] + np.linalg.norm(h_curr - h_prev))
            displacement.append(np.linalg.norm(h_curr - h_start))
        ax.plot(steps, cum_dist, 'b-', label='Path Length', linewidth=2)
        ax.plot(steps, displacement, 'r-', label='Displacement', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Distance')
        ax.set_title(f'Path vs Displacement (Directedness: {self._compute_directedness():.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["representation_trajectory"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
