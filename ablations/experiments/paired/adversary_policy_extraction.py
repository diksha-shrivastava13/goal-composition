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
from ..utils.paired_helpers import (
    generate_levels,
    extract_level_features_batch,
    get_pro_ant_returns,
)


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
        """Sample levels from adversary and evaluate with real rollouts."""
        checkpoint_step = getattr(self.train_state, 'update_count', 0)

        # Generate all levels in a single batched call
        rng, gen_rng, eval_rng = jax.random.split(rng, 3)
        levels = generate_levels(self.agent, gen_rng, self.n_levels_per_checkpoint)

        # Extract features for all levels at once
        batch_features = extract_level_features_batch(levels)

        # Get real protagonist and antagonist returns via rollouts
        pro_returns, ant_returns, regrets = get_pro_ant_returns(
            eval_rng, levels, self
        )

        # Build per-level records
        for i in range(self.n_levels_per_checkpoint):
            features = {
                'wall_density': float(batch_features['wall_density'][i]),
                'goal_distance': float(batch_features['goal_distance'][i]),
            }
            self._levels_data.append({
                'features': features,
                'regret': float(regrets[i]),
                'checkpoint_step': checkpoint_step,
            })

        return self._levels_data

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
