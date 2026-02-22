"""
Goal Representation Extraction Experiment (Task-Relevant Activation Analysis).

Identifies which activations influence task-relevant behavior through:
- Gradient-based saliency maps
- Activation patching
- Task-controlling subspace identification

NOTE: This identifies "task-controlling activation subspaces", NOT "goal circuits".
We avoid claiming to have found goal representations.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
import chex

from .base import CheckpointExperiment
from .utils.activation_patching import (
    compute_saliency_map,
    compute_saliency_statistics,
    patch_activations,
    identify_task_controlling_subspace,
    compute_feature_attribution,
)


@dataclass
class GoalExtractionData:
    """Container for goal extraction data."""
    saliency_maps: List[Dict[str, np.ndarray]] = field(default_factory=list)
    patching_results: List[Dict[str, float]] = field(default_factory=list)
    attribution_scores: List[np.ndarray] = field(default_factory=list)

    # Metadata
    goal_positions: List[Tuple[int, int]] = field(default_factory=list)
    agent_positions: List[Tuple[int, int]] = field(default_factory=list)
    wall_densities: List[float] = field(default_factory=list)


class GoalExtractionExperiment(CheckpointExperiment):
    """
    Extract task-relevant activation patterns.

    Methods:
    1. Saliency maps: ∂V/∂obs to find influential features
    2. Activation patching: Test causal role of activations
    3. Task-controlling subspace: PCA on saliency-weighted activations

    Caveats:
    - Saliency ≠ importance (attribution methods have known issues)
    - "Goal circuits" is an overclaim - we find "task-controlling subspaces"
    """

    @property
    def name(self) -> str:
        return "goal_extraction"

    def __init__(
        self,
        n_samples: int = 100,
        n_patching_pairs: int = 50,
        n_attribution_steps: int = 50,
        **kwargs,
    ):
        """
        Initialize goal extraction experiment.

        Args:
            n_samples: Number of samples for saliency analysis
            n_patching_pairs: Number of level pairs for patching
            n_attribution_steps: Steps for integrated gradients
        """
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.n_patching_pairs = n_patching_pairs
        self.n_attribution_steps = n_attribution_steps

        self._data: Optional[GoalExtractionData] = None
        self._results: Dict[str, Any] = {}

    def collect_data(self, rng: chex.PRNGKey) -> GoalExtractionData:
        """
        Collect saliency and patching data.
        """
        self._data = GoalExtractionData()

        # 1. Collect saliency maps
        for i in range(self.n_samples):
            rng, level_rng, sample_rng = jax.random.split(rng, 3)

            level = self._generate_level(level_rng)
            saliency = self._compute_saliency(sample_rng, level)

            self._data.saliency_maps.append(saliency)
            self._data.goal_positions.append(level['goal_pos'])
            self._data.agent_positions.append(level['agent_pos'])
            self._data.wall_densities.append(level['wall_density'])

        # 2. Collect patching results
        for i in range(self.n_patching_pairs):
            rng, src_rng, tgt_rng, patch_rng = jax.random.split(rng, 4)

            source_level = self._generate_level(src_rng)
            target_level = self._generate_level(tgt_rng)

            patching_result = self._compute_patching(patch_rng, source_level, target_level)
            self._data.patching_results.append(patching_result)

        return self._data

    def _generate_level(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """Generate a test level."""
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
        }

    def _create_observation(self, level: Dict[str, Any]) -> Any:
        """Create observation from level."""
        height, width = level['wall_map'].shape

        image = np.zeros((height, width, 3), dtype=np.float32)
        image[:, :, 0] = level['wall_map'].astype(np.float32)
        image[level['goal_pos']] = [0, 1, 0]
        image[level['agent_pos'][0], level['agent_pos'][1], 2] = 1.0

        class Obs:
            def __init__(self, img, direction):
                self.image = img
                self.agent_dir = direction

        return Obs(jnp.array(image), jnp.array([0]))

    def _compute_saliency(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Compute saliency map for a level."""
        obs = self._create_observation(level)
        hstate = self.agent.initialize_carry(rng, batch_dims=(1,))

        try:
            saliency = compute_saliency_map(
                self.train_state,
                obs,
                hstate,
                target="value"
            )
            return saliency
        except Exception as e:
            # Return empty saliency on error
            height, width = level['wall_map'].shape
            return {
                'image_saliency': np.zeros((height, width, 3)),
                'image_saliency_spatial': np.zeros((height, width)),
                'direction_saliency': np.zeros(1),
                'error': str(e),
            }

    def _compute_patching(
        self,
        rng: chex.PRNGKey,
        source_level: Dict[str, Any],
        target_level: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute patching effect between two levels."""
        source_obs = self._create_observation(source_level)
        target_obs = self._create_observation(target_level)
        hstate = self.agent.initialize_carry(rng, batch_dims=(1,))

        try:
            result = patch_activations(
                self.train_state,
                source_obs,
                target_obs,
                hstate,
                patch_layer="lstm_hidden"
            )
            return result
        except Exception as e:
            return {
                'error': str(e),
                'kl_patched_vs_target': 0.0,
                'value_shift': 0.0,
            }

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze goal-relevant activations.
        """
        if self._data is None:
            raise ValueError("Must call collect_data before analyze")

        results = {}

        # 1. Saliency analysis
        results['saliency'] = self._analyze_saliency()

        # 2. Patching analysis
        results['patching'] = self._analyze_patching()

        # 3. Goal-distance correlation
        results['goal_distance_correlation'] = self._analyze_goal_correlation()

        # 4. Prediction context for saliency interpretation
        results['prediction_context'] = self._compute_prediction_context()

        # 5. Caveats
        results['caveats'] = self._get_caveats()

        self._results = results
        return results

    def _analyze_saliency(self) -> Dict[str, Any]:
        """Analyze saliency patterns."""
        valid_maps = [s for s in self._data.saliency_maps if 'error' not in s]

        if not valid_maps:
            return {'error': 'No valid saliency maps'}

        # Compute aggregate statistics
        stats = compute_saliency_statistics(valid_maps)

        # Analyze spatial distribution
        all_spatial = [s['image_saliency_spatial'] for s in valid_maps]
        stacked = np.stack(all_spatial)
        mean_saliency = stacked.mean(axis=0)

        # Find if saliency focuses on goal region
        goal_saliency = []
        non_goal_saliency = []

        for i, smap in enumerate(valid_maps):
            goal_pos = self._data.goal_positions[i]
            spatial = smap['image_saliency_spatial']

            # 3x3 region around goal
            y, x = goal_pos
            goal_region = spatial[max(0, y-1):y+2, max(0, x-1):x+2]
            goal_saliency.append(goal_region.mean())

            # Everything else
            mask = np.ones_like(spatial, dtype=bool)
            mask[max(0, y-1):y+2, max(0, x-1):x+2] = False
            non_goal_saliency.append(spatial[mask].mean())

        goal_focus = np.mean(goal_saliency) / (np.mean(non_goal_saliency) + 1e-6)

        return {
            'saliency_statistics': stats,
            'goal_focus_ratio': float(goal_focus),
            'focuses_on_goal': goal_focus > 1.5,
            'n_valid_maps': len(valid_maps),
        }

    def _analyze_patching(self) -> Dict[str, Any]:
        """Analyze patching effects."""
        valid_results = [r for r in self._data.patching_results if 'error' not in r]

        if not valid_results:
            return {'error': 'No valid patching results'}

        kl_shifts = [r.get('kl_patched_vs_target', 0) for r in valid_results]
        value_shifts = [r.get('value_shift', 0) for r in valid_results]

        return {
            'mean_kl_shift': float(np.mean(kl_shifts)),
            'std_kl_shift': float(np.std(kl_shifts)),
            'mean_value_shift': float(np.mean(value_shifts)),
            'std_value_shift': float(np.std(value_shifts)),
            'patching_has_effect': float(np.mean(kl_shifts)) > 0.1,
            'n_valid_pairs': len(valid_results),
        }

    def _analyze_goal_correlation(self) -> Dict[str, Any]:
        """Analyze correlation between saliency and goal distance."""
        valid_maps = [s for s in self._data.saliency_maps if 'error' not in s]

        if len(valid_maps) < 10:
            return {'error': 'Insufficient valid samples'}

        # Compute goal distances
        goal_distances = []
        max_saliencies = []

        for i, smap in enumerate(valid_maps):
            goal_pos = self._data.goal_positions[i]
            agent_pos = self._data.agent_positions[i]

            dist = abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])
            goal_distances.append(dist)

            max_saliencies.append(smap['image_saliency_spatial'].max())

        corr = np.corrcoef(goal_distances, max_saliencies)[0, 1]

        return {
            'correlation': float(corr) if np.isfinite(corr) else 0.0,
            'interpretation': (
                'Saliency increases with goal distance' if corr > 0.2
                else ('Saliency decreases with goal distance' if corr < -0.2
                      else 'No clear relationship')
            ),
        }

    def _compute_prediction_context(self) -> Dict[str, Any]:
        """
        Compute prediction loss for saliency samples to connect
        saliency findings to prediction ability.
        """
        try:
            from .utils.agent_aware_loss import (
                compute_agent_prediction_loss,
                compute_random_baseline_loss,
            )

            valid_maps = [s for s in self._data.saliency_maps if 'error' not in s]
            if not valid_maps:
                return {'error': 'No valid saliency maps for prediction context'}

            import jax

            rng = jax.random.PRNGKey(42)
            prediction_losses = []
            max_saliencies = []

            n_samples = min(len(valid_maps), len(self._data.goal_positions))
            for i in range(n_samples):
                rng, loss_rng = jax.random.split(rng)

                # Reconstruct level from stored data
                level = {
                    'wall_map': np.zeros((13, 13)),  # Placeholder
                    'wall_density': self._data.wall_densities[i],
                    'goal_pos': self._data.goal_positions[i],
                    'agent_pos': self._data.agent_positions[i],
                    'agent_dir': 0,
                }

                # Generate wall map from density
                wall_prob = self._data.wall_densities[i]
                wall_map = np.array(jax.random.bernoulli(loss_rng, wall_prob, (13, 13)))
                wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False
                level['wall_map'] = wall_map

                loss, _ = compute_agent_prediction_loss(
                    self.agent,
                    self.train_state,
                    level,
                    loss_rng,
                )
                prediction_losses.append(loss)
                max_saliencies.append(valid_maps[i]['image_saliency_spatial'].max())

            prediction_losses = np.array(prediction_losses)
            max_saliencies = np.array(max_saliencies)
            random_baseline = compute_random_baseline_loss()

            # Compute correlation between saliency and prediction loss
            corr = 0.0
            if len(prediction_losses) > 10:
                corr_val = np.corrcoef(max_saliencies, prediction_losses)[0, 1]
                corr = float(corr_val) if np.isfinite(corr_val) else 0.0

            return {
                'mean_prediction_loss': float(np.mean(prediction_losses)),
                'random_baseline': random_baseline,
                'information_gain': float(random_baseline - np.mean(prediction_losses)),
                'saliency_vs_prediction_correlation': corr,
                'n_samples': len(prediction_losses),
                'interpretation': (
                    "High saliency regions should correlate with prediction-relevant features. "
                    f"Correlation coefficient ({corr:.3f}) provides supporting evidence for causal role."
                ),
            }

        except Exception as e:
            return {'error': str(e)}

    def _get_caveats(self) -> List[str]:
        """Return analysis caveats."""
        return [
            "Saliency does NOT prove agent 'represents' the goal as a goal",
            "High saliency ≠ importance (attribution problems well-documented)",
            "Patching shows functional dependence, not 'goal circuits'",
            "Better framing: 'task-controlling activation subspaces'",
            "These are correlational findings, not mechanistic explanations",
        ]

    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data."""
        if not self._results:
            raise ValueError("Must call analyze before visualize")

        viz_data = {
            'saliency': self._results.get('saliency', {}),
            'patching': self._results.get('patching', {}),
            'goal_correlation': self._results.get('goal_distance_correlation', {}),
        }

        return viz_data
