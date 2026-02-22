"""
Training Dynamics Characterization (Phase Transition Analysis).

EXPLORATORY diagnostics for understanding how representations and behavior
change during training. NOT confirmatory phase detection.

WARNING: This is exploratory visualization, not validated phase transition
detection. Any apparent transitions require independent validation.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import chex

from .base import TrainingTimeExperiment
from .utils.transition_metrics import (
    estimate_fisher_information,
    compute_effective_dimensionality,
    compute_synergy_measure,
    track_training_dynamics,
    compute_loss_curvature,
)


@dataclass
class DynamicsData:
    """Container for training dynamics data."""
    steps: List[int] = field(default_factory=list)

    # Loss and gradient metrics
    training_losses: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)

    # Representation metrics
    effective_dims: List[float] = field(default_factory=list)
    hidden_state_norms: List[float] = field(default_factory=list)

    # Performance metrics
    solve_rates: List[float] = field(default_factory=list)
    mean_returns: List[float] = field(default_factory=list)

    # Prediction/probe loss (actual curriculum awareness measure)
    prediction_losses: List[float] = field(default_factory=list)

    # Gradient samples for Fisher estimation
    gradient_samples: List[np.ndarray] = field(default_factory=list)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert to numpy arrays."""
        return {
            'steps': np.array(self.steps),
            'training_loss': np.array(self.training_losses),
            'gradient_norm': np.array(self.gradient_norms),
            'value_loss': np.array(self.value_losses),
            'policy_loss': np.array(self.policy_losses),
            'effective_dim': np.array(self.effective_dims),
            'hidden_norm': np.array(self.hidden_state_norms),
            'solve_rate': np.array(self.solve_rates),
            'mean_return': np.array(self.mean_returns),
            'prediction_loss': np.array(self.prediction_losses),
        }


class PhaseTransitionExperiment(TrainingTimeExperiment):
    """
    Track training dynamics for exploratory analysis.

    Collects:
    - Fisher information approximation
    - Effective dimensionality
    - Loss curvature
    - Synergy measures

    IMPORTANT CAVEATS:
    - These are EXPLORATORY diagnostics, not confirmed phase transitions
    - Do NOT claim "grokking" without controlled comparison
    - Transitions are smoothing artifacts until independently validated
    """

    @property
    def name(self) -> str:
        return "phase_transition"

    def __init__(
        self,
        collection_interval: int = 100,
        n_gradient_samples: int = 10,
        n_representation_samples: int = 50,
        **kwargs,
    ):
        """
        Initialize phase transition experiment.

        Args:
            collection_interval: Steps between data collection
            n_gradient_samples: Samples for Fisher estimation
            n_representation_samples: Samples for dimensionality estimation
        """
        super().__init__(**kwargs)
        self.collection_interval = collection_interval
        self.n_gradient_samples = n_gradient_samples
        self.n_representation_samples = n_representation_samples

        self._data = DynamicsData()
        self._results: Dict[str, Any] = {}

    def training_hook(
        self,
        train_state: Any,
        metrics: Dict[str, Any],
        step: int,
    ) -> Dict[str, Any]:
        """
        Hook called during training to collect dynamics data.
        """
        if step % self.collection_interval != 0:
            return {}

        self.train_state = train_state

        # Extract metrics
        training_loss = metrics.get('total_loss', metrics.get('loss', 0.0))
        grad_norm = metrics.get('grad_norm', metrics.get('gradient_norm', 0.0))
        value_loss = metrics.get('value_loss', metrics.get('v_loss', 0.0))
        policy_loss = metrics.get('policy_loss', metrics.get('pi_loss', 0.0))
        solve_rate = metrics.get('solve_rate', metrics.get('success_rate', 0.0))
        mean_return = metrics.get('mean_return', metrics.get('episode_return', 0.0))

        # Collect representation samples and estimate dimensionality
        effective_dim, hidden_norm = self._estimate_representation_metrics()

        # Compute actual prediction/probe loss
        prediction_loss = self._compute_prediction_loss()

        # Store data
        self._data.steps.append(step)
        self._data.training_losses.append(float(training_loss))
        self._data.gradient_norms.append(float(grad_norm))
        self._data.value_losses.append(float(value_loss))
        self._data.policy_losses.append(float(policy_loss))
        self._data.solve_rates.append(float(solve_rate))
        self._data.mean_returns.append(float(mean_return))
        self._data.effective_dims.append(float(effective_dim))
        self._data.hidden_state_norms.append(float(hidden_norm))
        self._data.prediction_losses.append(float(prediction_loss))

        return {
            'phase_transition/effective_dim': effective_dim,
            'phase_transition/hidden_norm': hidden_norm,
            'phase_transition/prediction_loss': prediction_loss,
        }

    def _estimate_representation_metrics(self) -> Tuple[float, float]:
        """Estimate effective dimensionality and hidden state norm."""
        import jax
        import jax.numpy as jnp

        if not hasattr(self, 'train_state') or self.train_state is None:
            return 0.0, 0.0

        try:
            rng = jax.random.PRNGKey(0)
            hidden_states = []

            for i in range(self.n_representation_samples):
                rng, sample_rng = jax.random.split(rng)

                # Create random observation
                obs_image = jax.random.uniform(sample_rng, (13, 13, 3))

                class Obs:
                    def __init__(self, img, direction):
                        self.image = img
                        self.agent_dir = direction

                obs = Obs(obs_image, jnp.array([0]))

                # Get hidden state
                rng, h_rng = jax.random.split(rng)
                hstate = self.agent.initialize_carry(h_rng, batch_dims=(1,))

                # Forward pass
                obs_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs)
                done_batch = jnp.zeros((1, 1), dtype=bool)

                new_hstate, _, _ = self.train_state.apply_fn(
                    self.train_state.params,
                    (obs_batch, done_batch),
                    hstate
                )

                # Flatten hidden state
                h_c, h_h = new_hstate
                h_flat = np.concatenate([
                    np.array(h_c).flatten(),
                    np.array(h_h).flatten()
                ])
                hidden_states.append(h_flat)

            hidden_states = np.stack(hidden_states)

            # Compute effective dimensionality
            dim_result = compute_effective_dimensionality(hidden_states)
            effective_dim = dim_result.get('effective_dimensionality', 0.0)

            # Compute mean hidden state norm
            hidden_norm = float(np.linalg.norm(hidden_states, axis=1).mean())

            return effective_dim, hidden_norm

        except Exception:
            return 0.0, 0.0

    def _compute_prediction_loss(self) -> float:
        """
        Compute actual prediction/probe loss for curriculum awareness tracking.

        Uses agent-aware loss computation to get the actual prediction loss
        (from probe for probe-based agents, from prediction head for
        next_env_prediction agent).
        """
        import jax

        if not hasattr(self, 'train_state') or self.train_state is None:
            return 1.0

        try:
            from .utils.agent_aware_loss import compute_agent_prediction_loss

            rng = jax.random.PRNGKey(0)
            losses = []

            # Sample a few levels and compute average prediction loss
            n_samples = min(10, self.n_representation_samples)
            for i in range(n_samples):
                rng, sample_rng, level_rng = jax.random.split(rng, 3)

                # Generate random level
                level = self._generate_random_level(level_rng)

                # Compute prediction loss
                loss, _ = compute_agent_prediction_loss(
                    self.agent,
                    self.train_state,
                    level,
                    sample_rng,
                )
                losses.append(loss)

            return float(np.mean(losses)) if losses else 1.0

        except Exception:
            return 1.0

    def _generate_random_level(self, rng) -> Dict[str, Any]:
        """Generate a random level for prediction loss evaluation."""
        import jax

        height, width = 13, 13
        wall_prob = 0.1 + float(jax.random.uniform(rng)) * 0.2

        wall_map = np.array(jax.random.bernoulli(rng, wall_prob, (height, width)))
        wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

        rng_goal, rng_agent = jax.random.split(rng)
        goal_pos = (
            int(jax.random.randint(rng_goal, (), 1, height - 1)),
            int(jax.random.randint(rng_goal, (), 1, width - 1)),
        )
        agent_pos = (
            int(jax.random.randint(rng_agent, (), 1, height - 1)),
            int(jax.random.randint(rng_agent, (), 1, width - 1)),
        )

        return {
            'wall_map': wall_map,
            'wall_density': wall_map.sum() / (height * width),
            'goal_pos': goal_pos,
            'agent_pos': agent_pos,
            'agent_dir': 0,
        }

    def collect_data(self, rng: chex.PRNGKey) -> Dict[str, np.ndarray]:
        """Return collected data."""
        return self._data.to_arrays()

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze training dynamics.

        WARNING: All analyses are EXPLORATORY. Do not claim phase
        transitions without independent validation.
        """
        data = self._data.to_arrays()

        if len(data['steps']) < 10:
            return {
                'error': 'Insufficient data points',
                'n_points': len(data['steps']),
            }

        results = {}

        # 1. Loss dynamics
        results['loss_dynamics'] = self._analyze_loss_dynamics(data)

        # 2. Representation dynamics
        results['representation_dynamics'] = self._analyze_representation_dynamics(data)

        # 3. Exploratory change point detection
        results['exploratory_transitions'] = self._detect_exploratory_transitions(data)

        # 4. Correlation between metrics
        results['metric_correlations'] = self._analyze_correlations(data)

        # 5. Strong caveats
        results['caveats'] = self._get_caveats()

        self._results = results
        return results

    def _analyze_loss_dynamics(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze loss dynamics over training."""
        loss = data['training_loss']
        steps = data['steps']

        # Smoothed loss
        window = min(10, len(loss) // 3)
        if window > 1:
            loss_smooth = np.convolve(loss, np.ones(window)/window, mode='valid')
            steps_smooth = steps[window-1:]
        else:
            loss_smooth = loss
            steps_smooth = steps

        # Loss statistics by training phase
        n_total = len(loss)
        early_idx = n_total // 3
        late_idx = 2 * n_total // 3

        return {
            'early_loss': float(np.mean(loss[:early_idx])),
            'mid_loss': float(np.mean(loss[early_idx:late_idx])),
            'late_loss': float(np.mean(loss[late_idx:])),
            'loss_decrease': float(loss[0] - loss[-1]) if len(loss) > 0 else 0.0,
            'final_loss': float(loss[-1]) if len(loss) > 0 else 0.0,
            'smoothed_loss': loss_smooth.tolist(),
            'smoothed_steps': steps_smooth.tolist(),
        }

    def _analyze_representation_dynamics(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze representation dynamics."""
        eff_dim = data['effective_dim']
        hidden_norm = data['hidden_norm']

        valid = np.isfinite(eff_dim) & (eff_dim > 0)
        if valid.sum() < 5:
            return {'error': 'Insufficient valid dimensionality samples'}

        eff_dim_valid = eff_dim[valid]

        return {
            'mean_effective_dim': float(np.mean(eff_dim_valid)),
            'std_effective_dim': float(np.std(eff_dim_valid)),
            'dim_trend': 'increasing' if eff_dim_valid[-1] > eff_dim_valid[0] else 'decreasing',
            'mean_hidden_norm': float(np.mean(hidden_norm)),
            'hidden_norm_trend': 'increasing' if hidden_norm[-1] > hidden_norm[0] else 'stable',
        }

    def _detect_exploratory_transitions(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Detect potential transitions (EXPLORATORY ONLY).

        WARNING: These are visualization aids, not validated transitions.
        """
        from .utils.time_series_analysis import detect_change_points

        results = {
            'warning': 'These are EXPLORATORY diagnostics, not confirmed transitions',
        }

        # Detect potential change points in loss
        try:
            loss_cps = detect_change_points(data['training_loss'], penalty=1.0)
            results['loss_change_points'] = [
                int(data['steps'][min(cp, len(data['steps'])-1)])
                for cp in loss_cps
            ]
        except Exception as e:
            results['loss_change_points_error'] = str(e)

        # Detect potential change points in solve rate
        try:
            perf_cps = detect_change_points(data['solve_rate'], penalty=1.0)
            results['performance_change_points'] = [
                int(data['steps'][min(cp, len(data['steps'])-1)])
                for cp in perf_cps
            ]
        except Exception as e:
            results['performance_change_points_error'] = str(e)

        return results

    def _analyze_correlations(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze correlations between training metrics."""
        correlations = {}

        metrics = [
            ('loss', data['training_loss']),
            ('solve_rate', data['solve_rate']),
            ('effective_dim', data['effective_dim']),
            ('grad_norm', data['gradient_norm']),
            ('prediction_loss', data['prediction_loss']),
        ]

        for i, (name_i, data_i) in enumerate(metrics):
            for j, (name_j, data_j) in enumerate(metrics):
                if i >= j:
                    continue

                valid = np.isfinite(data_i) & np.isfinite(data_j)
                if valid.sum() > 10:
                    corr = np.corrcoef(data_i[valid], data_j[valid])[0, 1]
                    correlations[f'{name_i}_vs_{name_j}'] = float(corr) if np.isfinite(corr) else 0.0

        return correlations

    def _get_caveats(self) -> List[str]:
        """Return strong caveats for interpretation."""
        return [
            "These are EXPLORATORY diagnostics, not confirmatory analysis",
            "Do NOT claim 'phase transitions' without independent validation",
            "Do NOT claim 'grokking' without controlled comparison",
            "Change points are smoothing artifacts until validated",
            "Correlations do not imply causation or phases",
            "Frame as 'exploratory characterization of training dynamics'",
        ]

    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data."""
        if not self._results:
            raise ValueError("Must call analyze before visualize")

        data = self._data.to_arrays()

        viz_data = {
            'time_series': {
                'steps': data['steps'].tolist(),
                'training_loss': data['training_loss'].tolist(),
                'solve_rate': data['solve_rate'].tolist(),
                'effective_dim': data['effective_dim'].tolist(),
                'grad_norm': data['gradient_norm'].tolist(),
                'prediction_loss': data['prediction_loss'].tolist(),
            },
            'loss_dynamics': self._results.get('loss_dynamics', {}),
            'representation_dynamics': self._results.get('representation_dynamics', {}),
            'transitions': self._results.get('exploratory_transitions', {}),
        }

        return viz_data
