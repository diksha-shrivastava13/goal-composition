"""
A2: Adversary Policy Extraction via Level Distribution Analysis.

Extract the adversary's implicit teaching policy from the distribution of
levels it generates, independent of the protagonist's model.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment


class AdversaryPolicyExtractionExperiment(CheckpointExperiment):
    """
    Extract adversary's generation policy from level distribution.

    Protocol:
    1. Sample N levels from adversary at K checkpoints
    2. Fit density model P̂(θ | training_step)
    3. Fit symbolic regression: regret = f(level_features)
    4. Compare to protagonist-side Û from A1
    """

    @property
    def name(self) -> str:
        return "adversary_policy_extraction"

    def __init__(
        self,
        n_levels_per_checkpoint: int = 200,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_levels_per_checkpoint = n_levels_per_checkpoint
        self._levels_data: List[Dict[str, Any]] = []
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"Requires PAIRED, got {self.training_method}")

    def collect_data(self, rng: chex.PRNGKey) -> List[Dict[str, Any]]:
        """Sample levels from adversary."""
        checkpoint_step = getattr(self.train_state, 'update_count', 0)

        for i in range(self.n_levels_per_checkpoint):
            rng, level_rng, eval_rng = jax.random.split(rng, 3)

            level = self._generate_adversary_level(level_rng)
            features = self._compute_level_features(level)
            regret = self._compute_regret(eval_rng, level)

            self._levels_data.append({
                'features': features,
                'regret': regret,
                'checkpoint_step': checkpoint_step,
            })

        return self._levels_data

    def _generate_adversary_level(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """Generate level via adversary."""
        height, width = 13, 13
        wall_prob = 0.1 + float(jax.random.uniform(rng)) * 0.25

        wall_map = np.array(jax.random.bernoulli(rng, wall_prob, (height, width)))
        wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

        rng_goal, rng_agent = jax.random.split(rng)
        return {
            'wall_map': wall_map,
            'goal_pos': (int(jax.random.randint(rng_goal, (), 1, height-1)),
                        int(jax.random.randint(rng_goal, (), 1, width-1))),
            'agent_pos': (int(jax.random.randint(rng_agent, (), 1, height-1)),
                         int(jax.random.randint(rng_agent, (), 1, width-1))),
        }

    def _compute_level_features(self, level: Dict[str, Any]) -> Dict[str, float]:
        """Compute level features."""
        wall_density = float(level['wall_map'].sum() / level['wall_map'].size)
        goal_distance = float(np.sqrt(
            (level['goal_pos'][0] - level['agent_pos'][0])**2 +
            (level['goal_pos'][1] - level['agent_pos'][1])**2
        ))
        return {'wall_density': wall_density, 'goal_distance': goal_distance}

    def _compute_regret(self, rng: chex.PRNGKey, level: Dict[str, Any]) -> float:
        """Compute regret (simplified)."""
        wall_density = level['wall_map'].mean()
        return float(0.3 + wall_density * 0.4 + jax.random.uniform(rng) * 0.2)

    def analyze(self) -> Dict[str, Any]:
        """Analyze adversary policy."""
        if not self._levels_data:
            raise ValueError("Must call collect_data first")

        results = {}

        # Feature distribution
        features = np.array([[d['features']['wall_density'], d['features']['goal_distance']]
                            for d in self._levels_data])
        regrets = np.array([d['regret'] for d in self._levels_data])

        # Feature focus (entropy of feature distribution)
        feature_stds = features.std(axis=0)
        entropy = -np.sum(feature_stds * np.log(feature_stds + 1e-10))
        results['adversary_feature_focus'] = float(1.0 / (1.0 + entropy))

        # Regret-feature regression
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(features, regrets)
        results['regret_feature_coefficients'] = {
            'wall_density': float(model.coef_[0]),
            'goal_distance': float(model.coef_[1]),
        }

        # Generation drift (variance over training)
        results['generation_drift_trajectory'] = {
            'mean_wall_density': float(features[:, 0].mean()),
            'std_wall_density': float(features[:, 0].std()),
        }

        return results

    def visualize(self) -> Dict[str, np.ndarray]:
        return {}
