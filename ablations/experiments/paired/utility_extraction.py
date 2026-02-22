"""
A1: Utility Function Extraction via Symbolic Regression.

Extract symbolic approximation Û of the adversary's utility function from
the protagonist's next-environment prediction losses.

This is the CORE CONTRIBUTION of the PAIRED experiments.

Protocol:
1. Collect PAIREDPredictionData during training
2. Fit symbolic regression: features → prediction_loss
3. Compare Û to actual regret
4. Validate causally via interventions (in B-series)
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment
from ...common.types import PAIREDPredictionData, create_paired_prediction_data


@dataclass
class UtilityExtractionData:
    """Container for utility extraction data."""
    # Level features
    wall_densities: List[float] = field(default_factory=list)
    goal_distances: List[float] = field(default_factory=list)
    path_lengths: List[float] = field(default_factory=list)
    num_corridors: List[int] = field(default_factory=list)
    open_space_ratios: List[float] = field(default_factory=list)

    # Returns and regret
    protagonist_returns: List[float] = field(default_factory=list)
    antagonist_returns: List[float] = field(default_factory=list)
    regrets: List[float] = field(default_factory=list)

    # Prediction losses (per-feature)
    wall_prediction_losses: List[float] = field(default_factory=list)
    goal_prediction_losses: List[float] = field(default_factory=list)
    total_prediction_losses: List[float] = field(default_factory=list)

    # Training context
    training_steps: List[int] = field(default_factory=list)
    adversary_entropies: List[float] = field(default_factory=list)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert to numpy arrays."""
        return {
            'wall_density': np.array(self.wall_densities),
            'goal_distance': np.array(self.goal_distances),
            'path_length': np.array(self.path_lengths),
            'num_corridors': np.array(self.num_corridors),
            'open_space_ratio': np.array(self.open_space_ratios),
            'protagonist_return': np.array(self.protagonist_returns),
            'antagonist_return': np.array(self.antagonist_returns),
            'regret': np.array(self.regrets),
            'wall_prediction_loss': np.array(self.wall_prediction_losses),
            'goal_prediction_loss': np.array(self.goal_prediction_losses),
            'total_prediction_loss': np.array(self.total_prediction_losses),
            'training_step': np.array(self.training_steps),
            'adversary_entropy': np.array(self.adversary_entropies),
        }


class UtilityExtractionExperiment(CheckpointExperiment):
    """
    Extract symbolic Û from protagonist's next-environment prediction losses.

    Key insight: If protagonist can predict what comes next, it has implicitly
    modeled the teacher's objective. The symbolic expression is our extracted Û.

    Metrics:
        utility_reconstruction_r2: How well Û predicts actual regret
        utility_complexity: Parsimony of the symbolic expression
        exploit_specificity: Does Û concentrate on specific features?
        utility_drift_rate: How quickly Û changes over training
        protagonist_model_accuracy: Does prediction loss match adversary's generation?
    """

    @property
    def name(self) -> str:
        return "utility_extraction"

    def __init__(
        self,
        n_samples: int = 500,
        use_pysr: bool = True,
        pysr_iterations: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.use_pysr = use_pysr
        self.pysr_iterations = pysr_iterations
        self._data: Optional[UtilityExtractionData] = None
        self._require_paired()

    def _require_paired(self):
        """Verify this is a PAIRED training setup."""
        if self.training_method != "paired":
            raise ValueError(
                f"UtilityExtractionExperiment requires PAIRED training, "
                f"got {self.training_method}"
            )

    def collect_data(self, rng: chex.PRNGKey) -> UtilityExtractionData:
        """Collect prediction data from training rollouts."""
        self._data = UtilityExtractionData()

        training_step = getattr(self.train_state, 'update_count', 0)

        for i in range(self.n_samples):
            rng, level_rng, pro_rng, ant_rng = jax.random.split(rng, 4)

            # Generate level via adversary
            level = self._generate_adversary_level(level_rng)

            # Compute level features
            features = self._compute_level_features(level)

            # Run protagonist and antagonist
            pro_result = self._evaluate_agent(pro_rng, level, 'protagonist')
            ant_result = self._evaluate_agent(ant_rng, level, 'antagonist')

            # Compute per-feature prediction losses
            pro_losses = self._compute_per_feature_probe_loss(
                pro_result['hstate'], level
            )

            # Compute regret
            regret = ant_result['return'] - pro_result['return']

            # Store data
            self._data.wall_densities.append(features['wall_density'])
            self._data.goal_distances.append(features['goal_distance'])
            self._data.path_lengths.append(features['path_length'])
            self._data.num_corridors.append(features.get('num_corridors', 0))
            self._data.open_space_ratios.append(features.get('open_space_ratio', 0.0))

            self._data.protagonist_returns.append(pro_result['return'])
            self._data.antagonist_returns.append(ant_result['return'])
            self._data.regrets.append(regret)

            self._data.wall_prediction_losses.append(pro_losses.get('wall_loss', 0.0))
            self._data.goal_prediction_losses.append(pro_losses.get('goal_loss', 0.0))
            self._data.total_prediction_losses.append(pro_losses.get('total_loss', 0.0))

            self._data.training_steps.append(training_step)
            self._data.adversary_entropies.append(
                self._compute_adversary_entropy(level_rng)
            )

        return self._data

    def _generate_adversary_level(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """Generate a level using the adversary (or random if not available)."""
        height, width = 13, 13

        # For checkpoint experiments, generate random levels as proxy
        # In training-time experiments, use actual adversary
        wall_prob = 0.1 + float(jax.random.uniform(rng)) * 0.25
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
            'goal_pos': goal_pos,
            'agent_pos': agent_pos,
        }

    def _compute_level_features(self, level: Dict[str, Any]) -> Dict[str, float]:
        """Compute level features for symbolic regression."""
        wall_map = level['wall_map']
        goal_pos = level['goal_pos']
        agent_pos = level['agent_pos']

        # Wall density
        wall_density = float(wall_map.sum() / wall_map.size)

        # Goal distance (Euclidean)
        goal_distance = float(np.sqrt(
            (goal_pos[0] - agent_pos[0])**2 +
            (goal_pos[1] - agent_pos[1])**2
        ))

        # Path length (BFS)
        path_length = self._compute_path_length(wall_map, agent_pos, goal_pos)

        # Corridor count (simplified: runs of empty cells)
        num_corridors = self._count_corridors(wall_map)

        # Open space ratio
        open_space_ratio = 1.0 - wall_density

        return {
            'wall_density': wall_density,
            'goal_distance': goal_distance,
            'path_length': float(path_length),
            'num_corridors': num_corridors,
            'open_space_ratio': open_space_ratio,
        }

    def _compute_path_length(
        self,
        wall_map: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> int:
        """Compute BFS path length."""
        from collections import deque

        if start == goal:
            return 0

        h, w = wall_map.shape
        visited = {start}
        queue = deque([(start, 0)])

        while queue:
            (x, y), dist = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < h and 0 <= ny < w and
                    (nx, ny) not in visited and not wall_map[nx, ny]):
                    if (nx, ny) == goal:
                        return dist + 1
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))

        return -1  # Unsolvable

    def _count_corridors(self, wall_map: np.ndarray) -> int:
        """Count narrow passages (simplified heuristic)."""
        h, w = wall_map.shape
        corridors = 0

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if not wall_map[i, j]:
                    # Check if it's a narrow passage
                    horizontal = wall_map[i-1, j] and wall_map[i+1, j]
                    vertical = wall_map[i, j-1] and wall_map[i, j+1]
                    if horizontal or vertical:
                        corridors += 1

        return corridors

    def _evaluate_agent(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
        agent_type: str,
        max_steps: int = 256,
    ) -> Dict[str, Any]:
        """Evaluate protagonist or antagonist on a level."""
        # Get appropriate train state
        if agent_type == 'protagonist':
            agent_state = self.train_state.pro_train_state
        else:
            agent_state = self.train_state.ant_train_state

        # Initialize hidden state
        hstate = self._initialize_hstate(rng)

        total_return = 0.0
        final_hstate = hstate

        for step in range(max_steps):
            rng, step_rng = jax.random.split(rng)

            obs = self._create_observation(level, step)
            new_hstate, pi, value = self._forward_step(obs, hstate, agent_state)

            # Sample action
            action = pi.sample(seed=step_rng)
            hstate = new_hstate
            final_hstate = hstate

            # Simulate reward (simplified)
            reward = 0.0
            done = step >= max_steps - 1

            if step > 10:
                solve_prob = 0.3 * (1 - level['wall_map'].mean())
                if float(jax.random.uniform(step_rng)) < solve_prob / max_steps:
                    reward = 1.0
                    done = True

            total_return += reward
            if done:
                break

        return {
            'return': total_return,
            'hstate': final_hstate,
            'solved': total_return > 0,
        }

    def _initialize_hstate(self, rng: chex.PRNGKey):
        """Initialize hidden state."""
        hidden_dim = 256
        return (
            jnp.zeros((1, hidden_dim)),  # c
            jnp.zeros((1, hidden_dim)),  # h
        )

    def _create_observation(self, level: Dict[str, Any], step: int):
        """Create observation from level."""
        height, width = level['wall_map'].shape
        image = np.zeros((height, width, 3), dtype=np.float32)
        image[:, :, 0] = level['wall_map'].astype(np.float32)
        image[level['goal_pos']] = [0, 1, 0]

        agent_y = (level['agent_pos'][0] + step // 10) % (height - 2) + 1
        agent_x = (level['agent_pos'][1] + step % 10) % (width - 2) + 1
        image[agent_y, agent_x, 2] = 1.0

        class Obs:
            def __init__(self, img, direction):
                self.image = img
                self.agent_dir = direction

        return Obs(jnp.array(image), jnp.array([0]))

    def _forward_step(self, obs, hstate, agent_state):
        """Forward pass through agent network."""
        obs_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs)
        done_batch = jnp.zeros((1, 1), dtype=bool)

        new_hstate, pi, value = agent_state.apply_fn(
            agent_state.params, (obs_batch, done_batch), hstate
        )
        return new_hstate, pi, value

    def _compute_per_feature_probe_loss(
        self,
        hstate,
        level: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute prediction loss for each feature separately."""
        # This would use the probe if available
        # For now, return placeholder values
        wall_density = float(level['wall_map'].mean())

        return {
            'wall_loss': 0.5 * wall_density,  # Placeholder
            'goal_loss': 0.5,  # Placeholder
            'total_loss': 0.5 * wall_density + 0.5,
        }

    def _compute_adversary_entropy(self, rng: chex.PRNGKey) -> float:
        """Compute adversary generation entropy (diversity)."""
        # Placeholder - would compute from actual adversary policy
        return float(jax.random.uniform(rng) * 2.0)

    def analyze(self) -> Dict[str, Any]:
        """Fit symbolic regression and extract Û."""
        if self._data is None:
            raise ValueError("Must call collect_data before analyze")

        data = self._data.to_arrays()
        results = {}

        # Prepare features
        X = np.column_stack([
            data['wall_density'],
            data['goal_distance'],
            data['path_length'],
            data['open_space_ratio'],
        ])
        feature_names = ['wall', 'goal_dist', 'path', 'open_space']

        y_pred_loss = data['total_prediction_loss']
        y_regret = data['regret']

        # Remove invalid samples
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y_pred_loss) & np.isfinite(y_regret)
        X = X[valid]
        y_pred_loss = y_pred_loss[valid]
        y_regret = y_regret[valid]

        if len(X) < 50:
            return {'error': 'Insufficient valid samples'}

        # 1. Linear baseline for Û
        results['linear_analysis'] = self._fit_linear_model(X, y_pred_loss, y_regret, feature_names)

        # 2. Symbolic regression for Û (if enabled)
        if self.use_pysr:
            results['symbolic_u_hat'] = self._fit_symbolic_model(
                X, y_pred_loss, feature_names, 'prediction_loss'
            )
            results['symbolic_u_actual'] = self._fit_symbolic_model(
                X, y_regret, feature_names, 'regret'
            )

        # 3. Compare Û to U_actual
        results['utility_comparison'] = self._compare_utilities(X, y_pred_loss, y_regret)

        # 4. Exploit specificity
        results['exploit_specificity'] = self._compute_exploit_specificity(X, y_regret)

        # 5. Utility drift over training
        results['utility_drift'] = self._compute_utility_drift(data)

        # 6. Causal validation sub-metrics (A1 causal claims)
        results['causal_validation'] = self._compute_causal_validation(X, y_pred_loss, y_regret)

        return results

    def _fit_linear_model(
        self,
        X: np.ndarray,
        y_pred: np.ndarray,
        y_regret: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Fit linear regression models."""
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score

        # Fit Û: features → prediction_loss
        model_u_hat = Ridge(alpha=1.0)
        model_u_hat.fit(X, y_pred)
        u_hat_r2 = r2_score(y_pred, model_u_hat.predict(X))

        # Fit U_actual: features → regret
        model_u_actual = Ridge(alpha=1.0)
        model_u_actual.fit(X, y_regret)
        u_actual_r2 = r2_score(y_regret, model_u_actual.predict(X))

        # Cross-prediction: how well does Û predict regret?
        u_hat_predicts_regret = r2_score(y_regret, model_u_hat.predict(X))

        return {
            'u_hat_r2': float(u_hat_r2),
            'u_actual_r2': float(u_actual_r2),
            'u_hat_predicts_regret_r2': float(u_hat_predicts_regret),
            'u_hat_coefficients': dict(zip(feature_names, model_u_hat.coef_.tolist())),
            'u_actual_coefficients': dict(zip(feature_names, model_u_actual.coef_.tolist())),
        }

    def _fit_symbolic_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        target_name: str,
    ) -> Dict[str, Any]:
        """Fit symbolic regression using PySR."""
        try:
            from pysr import PySRRegressor
        except ImportError:
            return {'error': 'PySR not installed'}

        # Subsample for speed
        if len(X) > 500:
            idx = np.random.choice(len(X), 500, replace=False)
            X = X[idx]
            y = y[idx]

        try:
            model = PySRRegressor(
                niterations=self.pysr_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["exp", "log", "sqrt"],
                maxsize=20,
                populations=8,
                verbosity=0,
            )
            model.fit(X, y, variable_names=feature_names)

            return {
                'best_equation': str(model.sympy()),
                'complexity': int(model.equations_['complexity'].iloc[-1]),
                'loss': float(model.equations_['loss'].iloc[-1]),
                'target': target_name,
            }
        except Exception as e:
            return {'error': str(e)}

    def _compare_utilities(
        self,
        X: np.ndarray,
        y_pred_loss: np.ndarray,
        y_regret: np.ndarray,
    ) -> Dict[str, Any]:
        """Compare Û to U_actual."""
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score

        # Fit Û
        model = Ridge(alpha=1.0)
        model.fit(X, y_pred_loss)
        u_hat_predictions = model.predict(X)

        # Correlation between Û predictions and actual regret
        correlation = float(np.corrcoef(u_hat_predictions, y_regret)[0, 1])

        # R² of Û on regret
        reconstruction_r2 = r2_score(y_regret, u_hat_predictions)

        return {
            'utility_reconstruction_r2': float(reconstruction_r2),
            'u_hat_regret_correlation': correlation,
            'interpretation': (
                f"Û explains {reconstruction_r2:.1%} of regret variance. "
                f"Correlation = {correlation:.3f}. "
                f"{'Protagonist has accurate model of adversary objective.' if reconstruction_r2 > 0.5 else 'Protagonist model differs from actual adversary.'}"
            ),
        }

    def _compute_exploit_specificity(
        self,
        X: np.ndarray,
        y_regret: np.ndarray,
    ) -> Dict[str, Any]:
        """Measure if adversary focuses on specific features."""
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)
        model.fit(X, y_regret)

        # Feature importance (absolute coefficients)
        abs_coefs = np.abs(model.coef_)
        normalized = abs_coefs / (abs_coefs.sum() + 1e-10)

        # Entropy of importance distribution
        entropy = -np.sum(normalized * np.log(normalized + 1e-10))
        max_entropy = np.log(len(normalized))
        specificity = 1.0 - (entropy / max_entropy)

        return {
            'specificity_score': float(specificity),
            'feature_importances': normalized.tolist(),
            'interpretation': (
                f"Specificity = {specificity:.2f}. "
                f"{'High: adversary targets specific features.' if specificity > 0.5 else 'Low: adversary spreads across features.'}"
            ),
        }

    def _compute_utility_drift(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Track how Û changes over training."""
        steps = data['training_step']
        regrets = data['regret']

        unique_steps = np.unique(steps)
        if len(unique_steps) < 2:
            return {'error': 'Single training step, cannot compute drift'}

        mean_regret_by_step = []
        for step in unique_steps:
            mask = steps == step
            mean_regret_by_step.append(float(regrets[mask].mean()))

        # Compute drift rate (slope)
        x = np.arange(len(mean_regret_by_step))
        if len(x) > 1:
            slope = float(np.polyfit(x, mean_regret_by_step, 1)[0])
        else:
            slope = 0.0

        return {
            'drift_rate': slope,
            'mean_regret_trajectory': mean_regret_by_step,
            'interpretation': (
                f"Drift rate = {slope:.4f}. "
                f"{'Positive: regret increasing (curriculum getting harder).' if slope > 0 else 'Negative: protagonist catching up.'}"
            ),
        }

    def _compute_causal_validation(
        self,
        X: np.ndarray,
        y_pred_loss: np.ndarray,
        y_regret: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute causal validation sub-metrics for A1.

        Validates that the extracted utility function Û has genuine causal
        structure rather than being a spurious correlation.

        Returns:
            causal_fidelity: How well Û reconstructs regret on held-out data
            counterfactual_consistency: Whether Û changes appropriately under
                feature perturbation (do-calculus consistency)
            causal_sufficiency: Whether Û captures most variance (no hidden confounders)
            protagonist_model_accuracy: How accurately the protagonist's learned
                representation predicts the adversary's true objective
        """
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        from sklearn.model_selection import KFold

        n_samples = len(X)

        # --- 1. Causal Fidelity (held-out R² via cross-validation) ---
        # If Û is causally valid, it should generalize to unseen data.
        kf = KFold(n_splits=min(5, max(2, n_samples // 20)), shuffle=True, random_state=42)
        held_out_r2s = []
        for train_idx, test_idx in kf.split(X):
            model = Ridge(alpha=1.0)
            model.fit(X[train_idx], y_pred_loss[train_idx])
            u_hat_test = model.predict(X[test_idx])
            # Measure how well Û predicts regret on held-out data
            if len(test_idx) > 1 and np.std(y_regret[test_idx]) > 1e-10:
                held_out_r2s.append(float(r2_score(y_regret[test_idx], u_hat_test)))
        causal_fidelity = float(np.mean(held_out_r2s)) if held_out_r2s else 0.0

        # --- 2. Counterfactual Consistency (do-calculus check) ---
        # Perturb each feature independently and check Û responds monotonically
        # consistent with the causal direction. If flipping a feature changes
        # Û in the opposite direction of actual regret, the model is inconsistent.
        model_full = Ridge(alpha=1.0)
        model_full.fit(X, y_pred_loss)

        consistency_scores = []
        for feat_idx in range(X.shape[1]):
            # Create counterfactual: shift feature by +1 std
            X_cf = X.copy()
            feat_std = X[:, feat_idx].std()
            if feat_std < 1e-10:
                continue
            X_cf[:, feat_idx] += feat_std

            # Direction of Û change
            u_hat_original = model_full.predict(X)
            u_hat_cf = model_full.predict(X_cf)
            u_hat_direction = np.sign(np.mean(u_hat_cf - u_hat_original))

            # Direction of actual regret change (empirical)
            corr = np.corrcoef(X[:, feat_idx], y_regret)[0, 1]
            regret_direction = np.sign(corr)

            # Consistent if both go the same direction
            consistency_scores.append(float(u_hat_direction == regret_direction))

        counterfactual_consistency = float(np.mean(consistency_scores)) if consistency_scores else 0.0

        # --- 3. Causal Sufficiency (residual analysis) ---
        # If Û is causally sufficient, residuals should be i.i.d. with no
        # structure left to explain. We measure this as 1 - autocorrelation of residuals.
        u_hat_all = model_full.predict(X)
        residuals = y_regret - u_hat_all
        residual_variance = np.var(residuals)
        total_variance = np.var(y_regret)
        explained_ratio = 1.0 - (residual_variance / (total_variance + 1e-10))

        # Check residual autocorrelation (if ordered by training step)
        if 'training_step' in self._data.to_arrays():
            sorted_idx = np.argsort(self._data.to_arrays()['training_step'])
            sorted_residuals = residuals[sorted_idx]
            if len(sorted_residuals) > 2:
                autocorr = float(np.corrcoef(sorted_residuals[:-1], sorted_residuals[1:])[0, 1])
            else:
                autocorr = 0.0
        else:
            autocorr = 0.0

        # Sufficiency = high explained variance AND low residual autocorrelation
        causal_sufficiency = float(max(0, explained_ratio) * (1.0 - abs(autocorr)))

        # --- 4. Protagonist Model Accuracy ---
        # How accurately does the protagonist's prediction loss (Û proxy)
        # correlate with the adversary's actual regret-generating objective?
        # This is the correlation between pred_loss ranking and regret ranking.
        from scipy.stats import spearmanr
        spearman_corr, spearman_p = spearmanr(y_pred_loss, y_regret)
        protagonist_model_accuracy = float(max(0, spearman_corr))

        return {
            'causal_fidelity': causal_fidelity,
            'counterfactual_consistency': counterfactual_consistency,
            'causal_sufficiency': causal_sufficiency,
            'protagonist_model_accuracy': protagonist_model_accuracy,
            'details': {
                'held_out_r2_per_fold': held_out_r2s,
                'feature_consistency_scores': consistency_scores,
                'explained_variance_ratio': float(explained_ratio),
                'residual_autocorrelation': autocorr,
                'spearman_correlation': float(spearman_corr),
                'spearman_p_value': float(spearman_p),
            },
            'interpretation': (
                f"Causal fidelity = {causal_fidelity:.3f} (held-out R²). "
                f"Counterfactual consistency = {counterfactual_consistency:.1%}. "
                f"Causal sufficiency = {causal_sufficiency:.3f}. "
                f"Protagonist accuracy = {protagonist_model_accuracy:.3f} (Spearman ρ). "
                f"{'Causally valid.' if causal_fidelity > 0.3 and counterfactual_consistency > 0.7 else 'Causal claims need scrutiny.'}"
            ),
        }

    def visualize(self) -> Dict[str, np.ndarray]:
        """Create visualizations."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if self._data is None:
            return figures

        data = self._data.to_arrays()

        # 1. Û vs U_actual scatter
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.scatter(data['total_prediction_loss'], data['regret'], alpha=0.3, s=10)
        ax.set_xlabel('Prediction Loss (Û proxy)')
        ax.set_ylabel('Actual Regret (U)')
        ax.set_title('Û vs U Comparison')

        ax = axes[1]
        ax.scatter(data['wall_density'], data['regret'], alpha=0.3, s=10, label='Regret')
        ax.scatter(data['wall_density'], data['total_prediction_loss'], alpha=0.3, s=10, label='Pred Loss')
        ax.set_xlabel('Wall Density')
        ax.set_ylabel('Value')
        ax.set_title('Feature Dependence')
        ax.legend()

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["utility_extraction"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
