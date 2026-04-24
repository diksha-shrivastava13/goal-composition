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
from ..utils.paired_helpers import (
    generate_levels,
    extract_level_features_batch,
    get_pro_ant_returns,
    get_pro_hstates,
)


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_levels = self.exp_config("n_levels")
        self.max_steps = self.exp_config("max_steps", 256)
        self._data: List[Dict[str, Any]] = []
        self._utility_profiles: Dict[str, AgentUtilityProfile] = {}
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"BilateralUtilityExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> List[Dict[str, Any]]:
        """Collect level data with real returns and hidden state features."""
        # Generate all levels in a single batched call
        rng, gen_rng, eval_rng, hstate_rng = jax.random.split(rng, 4)
        levels = generate_levels(self.agent, gen_rng, self.n_levels)

        # Extract features for all levels at once
        batch_features = extract_level_features_batch(levels)

        # Get real protagonist and antagonist returns via rollouts
        pro_returns, ant_returns, regrets = get_pro_ant_returns(
            eval_rng, levels, self, self.max_steps
        )

        # Get protagonist hidden states for richer feature extraction
        pro_hstates = get_pro_hstates(hstate_rng, levels, self, self.max_steps)
        pro_hstates_np = np.array(pro_hstates)

        # Build per-level records
        for i in range(self.n_levels):
            features = {k: float(v[i]) for k, v in batch_features.items()}

            # Add hidden state statistics as features
            if pro_hstates_np.ndim >= 2:
                h_i = pro_hstates_np[i]
                features['hstate_mean'] = float(np.mean(h_i))
                features['hstate_std'] = float(np.std(h_i))
                features['hstate_norm'] = float(np.linalg.norm(h_i))

            self._data.append({
                'features': features,
                'pro_return': float(pro_returns[i]),
                'ant_return': float(ant_returns[i]),
                'regret': float(regrets[i]),
                'pro_hstate': pro_hstates_np[i].copy() if pro_hstates_np.ndim >= 2 else None,
            })

        return self._data

    def analyze(self) -> Dict[str, Any]:
        """Compare utility functions across all three agents."""
        if not self._data:
            raise ValueError("Must call collect_data first")

        results = {}

        # Prepare feature matrix from all available features
        feature_names = list(self._data[0]['features'].keys())
        features = np.array([
            [d['features'].get(fn, 0.0) for fn in feature_names]
            for d in self._data
        ])

        # Fit protagonist utility (proxy: regret minimization)
        regrets = np.array([d['regret'] for d in self._data])
        u_pro = self._fit_utility(features, -regrets, 'protagonist')
        self._utility_profiles['protagonist'] = u_pro

        # Fit adversary utility (proxy: regret maximization = U* = R_ant - R_pro)
        u_adv = self._fit_utility(features, regrets, 'adversary')
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

        # Direct h-state → utility probing
        results['hstate_utility_probing'] = self._fit_utility_from_hstates()

        # Feature importance ranking
        results['feature_importance'] = {
            name: {
                feat: abs(coef)
                for feat, coef in profile.feature_coefficients.items()
                if feat not in ('intercept', 'selected_alpha')
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
        """Fit utility function using cross-validated Ridge regression.

        Uses all available features and selects regularization strength
        via cross-validation rather than a hardcoded alpha.
        """
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score

        # Standardize features for comparable coefficients
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Cross-validated Ridge with multiple alpha candidates
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(features_scaled, targets)

        predictions = model.predict(features_scaled)
        r2 = r2_score(targets, predictions)

        # Map coefficients back to feature names
        feature_names = list(self._data[0]['features'].keys()) if self._data else []
        coefficients = {}
        for j, name in enumerate(feature_names):
            if j < len(model.coef_):
                coefficients[name] = float(model.coef_[j])
        coefficients['intercept'] = float(model.intercept_)
        coefficients['selected_alpha'] = float(model.alpha_)

        # Build expression from top features (by absolute coefficient)
        sorted_feats = sorted(
            [(name, coef) for name, coef in coefficients.items()
             if name not in ('intercept', 'selected_alpha')],
            key=lambda x: abs(x[1]), reverse=True
        )
        expr_parts = [f"{coefficients['intercept']:.3f}"]
        for name, coef in sorted_feats[:5]:  # Top 5 features
            expr_parts.append(f"{coef:+.3f}*{name}")
        expression = " ".join(expr_parts)

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

    def _fit_utility_from_hstates(self) -> Dict[str, Any]:
        """Fit utility directly from full h-state vectors using Ridge regression."""
        from sklearn.linear_model import RidgeCV

        hstates = [d.get('pro_hstate') for d in self._data]
        if not hstates or hstates[0] is None:
            return {'error': 'No h-state vectors available'}

        X_h = np.array([h for h in hstates if h is not None])
        if len(X_h) < 20:
            return {'error': 'Insufficient h-state samples'}

        valid_data = [d for d in self._data if d.get('pro_hstate') is not None]
        regrets = np.array([d['regret'] for d in valid_data])
        pro_returns = np.array([d['pro_return'] for d in valid_data])
        ant_returns = np.array([d['ant_return'] for d in valid_data])

        results = {}
        for target_name, y in [('regret', regrets), ('pro_return', pro_returns), ('ant_return', ant_returns)]:
            try:
                ridge = RidgeCV(cv=5)
                ridge.fit(X_h, y)
                r2 = float(ridge.score(X_h, y))
                results[f'{target_name}_r2'] = r2
            except Exception:
                results[f'{target_name}_r2'] = 0.0

        return results

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
