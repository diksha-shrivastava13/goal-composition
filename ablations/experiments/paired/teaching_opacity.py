"""
F5: Teaching Opacity.

Measure adversary strategy visibility to protagonist.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment
from ..utils.paired_helpers import (
    generate_levels, extract_level_features_batch, get_pro_hstates,
    get_action_distribution, get_values_from_rollout,
)


@dataclass
class OpacityMeasurement:
    """Measurement of teaching opacity at a single point."""
    step: int
    adversary_strategy: Dict[str, float]
    protagonist_encoding: np.ndarray
    adversary_predictability: float  # How well protagonist can predict adversary
    strategy_visibility: float  # How visible strategy is in h-state
    curriculum_transparency: float  # Overall curriculum legibility


class TeachingOpacityExperiment(CheckpointExperiment):
    """
    Measure adversary strategy visibility to protagonist.

    Protocol:
    1. Generate levels (representing adversary's teaching strategy)
    2. Run protagonist on those levels to get real hidden states
    3. Extract level features as proxy for adversary strategy
    4. Train probes to predict strategy from protagonist h-state
    5. Measure strategy visibility and predictability
    """

    @property
    def name(self) -> str:
        return "teaching_opacity"

    def __init__(
        self,
        n_samples: int = 500,
        n_adversary_strategies: int = 5,
        hidden_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.n_adversary_strategies = n_adversary_strategies
        self.hidden_dim = hidden_dim
        self._measurements: List[OpacityMeasurement] = []
        self._strategy_predictor_weights: Optional[Dict] = None
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"TeachingOpacityExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> List[OpacityMeasurement]:
        """Collect opacity measurements using real network data."""
        # First train strategy predictor on real data
        rng, train_rng = jax.random.split(rng)
        self._train_strategy_predictor(train_rng)

        # Generate real levels and get real protagonist hidden states
        rng, level_rng, hstate_rng = jax.random.split(rng, 3)
        levels = generate_levels(self.agent, level_rng, self.n_samples)
        hstates = get_pro_hstates(hstate_rng, levels, self)
        self.hidden_dim = hstates.shape[1]

        # Extract level features as adversary strategy proxy
        features_batch = extract_level_features_batch(levels)

        for i in range(self.n_samples):
            wall_density = float(features_batch['wall_density'][i])
            goal_distance = float(features_batch['goal_distance'][i])

            # Adversary "strategy" = the level properties it chose
            strategy = {
                'difficulty': wall_density * 0.5 + goal_distance * 0.05,
                'wall_focus': wall_density,
                'distance_focus': goal_distance / 18.4,  # Normalize by max grid diagonal
                'variation': float(np.std(np.array(levels.wall_map)[i])),
                'strategy_type': float(
                    self._classify_strategy(wall_density, goal_distance)
                ),
            }

            h = hstates[i]

            # Predict strategy from protagonist hidden state
            predicted_strategy = self._predict_strategy(h)
            true_strategy = np.array([
                strategy['difficulty'],
                strategy['wall_focus'],
                strategy['distance_focus'],
            ])

            # Predictability = 1 - normalized prediction error
            prediction_error = np.linalg.norm(predicted_strategy - true_strategy)
            max_possible_error = np.sqrt(3)
            predictability = max(0, 1 - prediction_error / max_possible_error)

            # Strategy visibility = correlation-based metric
            visibility_score = self._compute_visibility(h, strategy)

            # Curriculum transparency = overall legibility
            transparency = (predictability + visibility_score) / 2

            self._measurements.append(OpacityMeasurement(
                step=i,
                adversary_strategy=strategy,
                protagonist_encoding=h,
                adversary_predictability=float(predictability),
                strategy_visibility=float(visibility_score),
                curriculum_transparency=float(transparency),
            ))

        return self._measurements

    def _classify_strategy(self, wall_density: float, goal_distance: float) -> int:
        """Classify level into discrete strategy type based on features."""
        if wall_density > 0.3 and goal_distance > 8.0:
            return 0  # Hard: dense walls + far goal
        elif wall_density > 0.3:
            return 1  # Wall-heavy
        elif goal_distance > 8.0:
            return 2  # Distance-heavy
        elif wall_density < 0.1:
            return 3  # Open/easy
        else:
            return 4  # Moderate

    def _train_strategy_predictor(self, rng: chex.PRNGKey):
        """Train a probe to predict adversary strategy from real protagonist h-state."""
        rng, level_rng, hstate_rng = jax.random.split(rng, 3)

        n_train = 200
        levels = generate_levels(self.agent, level_rng, n_train)
        training_hstates = get_pro_hstates(hstate_rng, levels, self)
        self.hidden_dim = training_hstates.shape[1]

        features_batch = extract_level_features_batch(levels)

        # Build strategy targets from real level features
        training_strategies = []
        for i in range(n_train):
            wd = float(features_batch['wall_density'][i])
            gd = float(features_batch['goal_distance'][i])
            training_strategies.append([
                wd * 0.5 + gd * 0.05,  # difficulty
                wd,                      # wall_focus
                gd / 18.4,              # distance_focus (normalized)
            ])

        training_strategies = np.array(training_strategies)

        from sklearn.linear_model import Ridge
        model = Ridge(alpha=0.1)
        model.fit(training_hstates, training_strategies)

        self._strategy_predictor_weights = {
            'coef': model.coef_,
            'intercept': model.intercept_,
        }

    def _predict_strategy(self, hstate: np.ndarray) -> np.ndarray:
        """Predict adversary strategy from protagonist h-state."""
        if self._strategy_predictor_weights is None:
            return np.zeros(3)

        prediction = hstate @ self._strategy_predictor_weights['coef'].T + self._strategy_predictor_weights['intercept']
        return prediction

    def _compute_visibility(self, hstate: np.ndarray, strategy: Dict[str, float]) -> float:
        """Compute how visible strategy is in h-state using probe predictions."""
        predicted = self._predict_strategy(hstate)
        true_vals = np.array([
            strategy['difficulty'],
            strategy['wall_focus'],
            strategy['distance_focus'],
        ])

        # Per-dimension accuracy (1 - relative error, clipped to [0, 1])
        rel_errors = np.abs(predicted - true_vals) / (np.abs(true_vals) + 1e-6)
        accuracies = np.clip(1.0 - rel_errors, 0, 1)

        return float(np.mean(accuracies))

    def analyze(self) -> Dict[str, Any]:
        """Analyze teaching opacity."""
        if not self._measurements:
            raise ValueError("Must call collect_data first")

        results = {}

        # Overall opacity statistics
        predictabilities = [m.adversary_predictability for m in self._measurements]
        visibilities = [m.strategy_visibility for m in self._measurements]
        transparencies = [m.curriculum_transparency for m in self._measurements]

        results['mean_predictability'] = float(np.mean(predictabilities))
        results['std_predictability'] = float(np.std(predictabilities))
        results['mean_visibility'] = float(np.mean(visibilities))
        results['std_visibility'] = float(np.std(visibilities))
        results['mean_transparency'] = float(np.mean(transparencies))

        # Opacity = 1 - transparency
        results['mean_opacity'] = float(1.0 - np.mean(transparencies))

        # Temporal evolution
        results['opacity_trajectory'] = self._analyze_trajectory()

        # Per-strategy opacity
        results['opacity_by_strategy'] = self._analyze_by_strategy()

        # Opacity's effect on learning (proxy)
        results['opacity_learning_effect'] = self._analyze_learning_effect()

        # Strategy predictor quality
        results['strategy_predictor_quality'] = self._evaluate_predictor()

        return results

    def _analyze_trajectory(self) -> Dict[str, Any]:
        """Analyze how opacity changes over training."""
        steps = [m.step for m in self._measurements]
        transparencies = [m.curriculum_transparency for m in self._measurements]

        # Split into thirds
        n = len(self._measurements)
        early = self._measurements[:n // 3]
        mid = self._measurements[n // 3: 2 * n // 3]
        late = self._measurements[2 * n // 3:]

        early_opacity = 1.0 - np.mean([m.curriculum_transparency for m in early])
        mid_opacity = 1.0 - np.mean([m.curriculum_transparency for m in mid])
        late_opacity = 1.0 - np.mean([m.curriculum_transparency for m in late])

        # Trend
        slope = np.polyfit(steps, transparencies, 1)[0]

        return {
            'early_opacity': float(early_opacity),
            'mid_opacity': float(mid_opacity),
            'late_opacity': float(late_opacity),
            'transparency_trend_slope': float(slope),
            'opacity_decreases_over_training': slope > 0,
        }

    def _analyze_by_strategy(self) -> Dict[int, Dict[str, float]]:
        """Analyze opacity by strategy type."""
        strategy_groups = {}

        for m in self._measurements:
            strategy_type = int(m.adversary_strategy['strategy_type'])
            if strategy_type not in strategy_groups:
                strategy_groups[strategy_type] = []
            strategy_groups[strategy_type].append(m)

        results = {}
        for strategy_type, measurements in strategy_groups.items():
            results[strategy_type] = {
                'mean_opacity': float(1.0 - np.mean([m.curriculum_transparency for m in measurements])),
                'mean_predictability': float(np.mean([m.adversary_predictability for m in measurements])),
                'count': len(measurements),
            }

        return results

    def _analyze_learning_effect(self) -> Dict[str, float]:
        """Analyze how opacity affects learning (proxy metrics)."""
        opacities = [1.0 - m.curriculum_transparency for m in self._measurements]

        # Encoding variance
        encoding_vars = [m.protagonist_encoding.var() for m in self._measurements]

        # Correlation
        opacity_variance_corr = np.corrcoef(opacities, encoding_vars)[0, 1]

        # Strategy difficulty and opacity
        difficulties = [m.adversary_strategy['difficulty'] for m in self._measurements]
        opacity_difficulty_corr = np.corrcoef(opacities, difficulties)[0, 1]

        return {
            'opacity_encoding_variance_correlation': float(opacity_variance_corr) if not np.isnan(opacity_variance_corr) else 0.0,
            'opacity_difficulty_correlation': float(opacity_difficulty_corr) if not np.isnan(opacity_difficulty_corr) else 0.0,
            'high_opacity_hurts_learning': float(opacity_variance_corr) > 0.2,
        }

    def _evaluate_predictor(self) -> Dict[str, float]:
        """Evaluate quality of strategy predictor."""
        if not self._measurements:
            return {}

        true_difficulties = [m.adversary_strategy['difficulty'] for m in self._measurements]
        true_wall_focus = [m.adversary_strategy['wall_focus'] for m in self._measurements]
        true_dist_focus = [m.adversary_strategy['distance_focus'] for m in self._measurements]

        pred_strategies = [self._predict_strategy(m.protagonist_encoding) for m in self._measurements]
        pred_difficulties = [p[0] for p in pred_strategies]
        pred_wall_focus = [p[1] for p in pred_strategies]
        pred_dist_focus = [p[2] for p in pred_strategies]

        def compute_r2(true, pred):
            ss_res = sum((t - p) ** 2 for t, p in zip(true, pred))
            ss_tot = sum((t - np.mean(true)) ** 2 for t in true)
            return float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

        return {
            'difficulty_r2': compute_r2(true_difficulties, pred_difficulties),
            'wall_focus_r2': compute_r2(true_wall_focus, pred_wall_focus),
            'distance_focus_r2': compute_r2(true_dist_focus, pred_dist_focus),
            'mean_r2': (compute_r2(true_difficulties, pred_difficulties) +
                       compute_r2(true_wall_focus, pred_wall_focus) +
                       compute_r2(true_dist_focus, pred_dist_focus)) / 3,
        }

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize teaching opacity."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._measurements:
            return figures

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Opacity trajectory
        ax = axes[0, 0]
        steps = [m.step for m in self._measurements]
        opacities = [1.0 - m.curriculum_transparency for m in self._measurements]

        # Rolling mean
        window = 20
        if len(opacities) > window:
            rolling_mean = np.convolve(opacities, np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], rolling_mean, 'r-', linewidth=2, label='Rolling Mean')

        ax.scatter(steps, opacities, alpha=0.3, s=10)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Opacity (1 - Transparency)')
        ax.set_title('Teaching Opacity Over Training')
        ax.legend()
        ax.set_ylim(0, 1)

        # Predictability vs Visibility
        ax = axes[0, 1]
        predictabilities = [m.adversary_predictability for m in self._measurements]
        visibilities = [m.strategy_visibility for m in self._measurements]
        ax.scatter(predictabilities, visibilities, alpha=0.5, s=20, c=steps, cmap='viridis')
        ax.set_xlabel('Strategy Predictability')
        ax.set_ylabel('Strategy Visibility')
        ax.set_title('Predictability vs Visibility')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)

        # Opacity by strategy type
        ax = axes[1, 0]
        by_strategy = self._analyze_by_strategy()
        if by_strategy:
            strategies = sorted(by_strategy.keys())
            opacities_by_type = [by_strategy[s]['mean_opacity'] for s in strategies]
            ax.bar([f'Type {s}' for s in strategies], opacities_by_type, alpha=0.7)
            ax.set_ylabel('Mean Opacity')
            ax.set_title('Opacity by Strategy Type')
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, 'No strategy data', ha='center', va='center', transform=ax.transAxes)

        # Predictor quality
        ax = axes[1, 1]
        pred_quality = self._evaluate_predictor()
        if pred_quality:
            dims = ['difficulty_r2', 'wall_focus_r2', 'distance_focus_r2']
            values = [pred_quality.get(d, 0) for d in dims]
            ax.bar(['Difficulty', 'Wall Focus', 'Dist Focus'], values, alpha=0.7)
            ax.set_ylabel('R² Score')
            ax.set_title('Strategy Predictor Quality')
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, 'No predictor data', ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["teaching_opacity"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
