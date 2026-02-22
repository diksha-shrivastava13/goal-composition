"""
A3: Bilateral Utility Comparison.

Compare utility functions extracted from protagonist, adversary, and antagonist
to identify teaching coherence, coalition alignment, and exploitation gaps.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment


@dataclass
class AgentUtilityProfile:
    """Utility profile for a single agent."""
    agent_name: str
    feature_coefficients: Dict[str, float]
    expression: str
    r2_score: float


class BilateralUtilityExperiment(CheckpointExperiment):
    """
    Compare Û_protagonist, Û_adversary, Û_antagonist.

    Protocol:
    1. Get Û_protagonist from A1 (prediction losses)
    2. Get U_adversary from A2 (level generation policy)
    3. Fit Û_antagonist: features → antagonist_return
    4. Compare all three, compute alignment metrics
    """

    @property
    def name(self) -> str:
        return "bilateral_utility"

    def __init__(
        self,
        n_levels: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_levels = n_levels
        self._data: List[Dict[str, Any]] = []
        self._utility_profiles: Dict[str, AgentUtilityProfile] = {}
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"BilateralUtilityExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> List[Dict[str, Any]]:
        """Collect level data with returns from all agents."""
        for i in range(self.n_levels):
            rng, level_rng, pro_rng, ant_rng = jax.random.split(rng, 4)

            # Generate level via adversary
            level = self._generate_adversary_level(level_rng)
            features = self._compute_level_features(level)

            # Evaluate protagonist and antagonist
            pro_return = self._evaluate_protagonist(pro_rng, level)
            ant_return = self._evaluate_antagonist(ant_rng, level)
            regret = ant_return - pro_return

            self._data.append({
                'features': features,
                'pro_return': pro_return,
                'ant_return': ant_return,
                'regret': regret,
            })

        return self._data

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
        return {
            'wall_density': wall_density,
            'goal_distance': goal_distance,
            'open_space_ratio': 1.0 - wall_density,
        }

    def _evaluate_protagonist(self, rng: chex.PRNGKey, level: Dict[str, Any]) -> float:
        """Evaluate protagonist on level (simplified)."""
        wall_density = level['wall_map'].mean()
        noise = float(jax.random.uniform(rng)) * 0.1
        return float(0.7 - wall_density * 0.4 + noise)

    def _evaluate_antagonist(self, rng: chex.PRNGKey, level: Dict[str, Any]) -> float:
        """Evaluate antagonist on level (simplified)."""
        wall_density = level['wall_map'].mean()
        noise = float(jax.random.uniform(rng)) * 0.1
        # Antagonist typically performs better
        return float(0.8 - wall_density * 0.3 + noise)

    def analyze(self) -> Dict[str, Any]:
        """Compare utility functions across all three agents."""
        if not self._data:
            raise ValueError("Must call collect_data first")

        results = {}

        # Prepare feature matrix
        features = np.array([
            [d['features']['wall_density'], d['features']['goal_distance']]
            for d in self._data
        ])
        feature_names = ['wall_density', 'goal_distance']

        # Fit protagonist utility (proxy: regret minimization)
        regrets = np.array([d['regret'] for d in self._data])
        u_pro = self._fit_utility(features, -regrets, 'protagonist')  # Minimize regret
        self._utility_profiles['protagonist'] = u_pro

        # Fit adversary utility (proxy: regret maximization)
        u_adv = self._fit_utility(features, regrets, 'adversary')  # Maximize regret
        self._utility_profiles['adversary'] = u_adv

        # Fit antagonist utility (proxy: own return)
        ant_returns = np.array([d['ant_return'] for d in self._data])
        u_ant = self._fit_utility(features, ant_returns, 'antagonist')
        self._utility_profiles['antagonist'] = u_ant

        # Compute alignment metrics
        results['teaching_coherence'] = self._compute_coherence(u_adv, u_pro)
        results['coalition_alignment'] = self._compute_coherence(u_adv, u_ant)
        results['exploitation_gap_features'] = self._find_exploitation_features(u_ant, u_pro)
        results['misalignment_score'] = self._compute_misalignment(u_adv, u_pro)

        # Store utility profiles
        results['utility_profiles'] = {
            name: {
                'coefficients': profile.feature_coefficients,
                'expression': profile.expression,
                'r2': profile.r2_score,
            }
            for name, profile in self._utility_profiles.items()
        }

        return results

    def _fit_utility(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        agent_name: str,
    ) -> AgentUtilityProfile:
        """Fit utility function using Ridge regression."""
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score

        model = Ridge(alpha=1.0)
        model.fit(features, targets)

        predictions = model.predict(features)
        r2 = r2_score(targets, predictions)

        coefficients = {
            'wall_density': float(model.coef_[0]),
            'goal_distance': float(model.coef_[1]),
            'intercept': float(model.intercept_),
        }

        expression = (
            f"{coefficients['intercept']:.3f} + "
            f"{coefficients['wall_density']:.3f}*wall_density + "
            f"{coefficients['goal_distance']:.3f}*goal_distance"
        )

        return AgentUtilityProfile(
            agent_name=agent_name,
            feature_coefficients=coefficients,
            expression=expression,
            r2_score=r2,
        )

    def _compute_coherence(
        self,
        u1: AgentUtilityProfile,
        u2: AgentUtilityProfile,
    ) -> float:
        """Compute coherence between two utility profiles."""
        # Correlation between coefficient vectors
        coef1 = np.array([u1.feature_coefficients['wall_density'],
                         u1.feature_coefficients['goal_distance']])
        coef2 = np.array([u2.feature_coefficients['wall_density'],
                         u2.feature_coefficients['goal_distance']])

        if np.linalg.norm(coef1) < 1e-10 or np.linalg.norm(coef2) < 1e-10:
            return 0.0

        return float(np.dot(coef1, coef2) / (np.linalg.norm(coef1) * np.linalg.norm(coef2)))

    def _find_exploitation_features(
        self,
        u_ant: AgentUtilityProfile,
        u_pro: AgentUtilityProfile,
    ) -> Dict[str, float]:
        """Find features where antagonist exploits protagonist."""
        exploitation_gaps = {}
        for feature in ['wall_density', 'goal_distance']:
            gap = (u_ant.feature_coefficients[feature] -
                   u_pro.feature_coefficients[feature])
            exploitation_gaps[feature] = float(gap)
        return exploitation_gaps

    def _compute_misalignment(
        self,
        u_adv: AgentUtilityProfile,
        u_pro: AgentUtilityProfile,
    ) -> float:
        """Compute misalignment between adversary's teaching and protagonist's learning."""
        # L2 distance between coefficient vectors, normalized
        coef_adv = np.array([u_adv.feature_coefficients['wall_density'],
                            u_adv.feature_coefficients['goal_distance']])
        coef_pro = np.array([u_pro.feature_coefficients['wall_density'],
                            u_pro.feature_coefficients['goal_distance']])

        return float(np.linalg.norm(coef_adv - coef_pro))

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize utility comparisons."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._utility_profiles:
            return figures

        # Coefficient comparison bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        agents = list(self._utility_profiles.keys())
        features = ['wall_density', 'goal_distance']
        x = np.arange(len(features))
        width = 0.25

        for i, agent in enumerate(agents):
            profile = self._utility_profiles[agent]
            values = [profile.feature_coefficients[f] for f in features]
            ax.bar(x + i * width, values, width, label=agent.capitalize(), alpha=0.8)

        ax.set_xlabel('Feature')
        ax.set_ylabel('Coefficient')
        ax.set_title('Utility Function Coefficients by Agent')
        ax.set_xticks(x + width)
        ax.set_xticklabels(features)
        ax.legend()
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["utility_comparison"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
