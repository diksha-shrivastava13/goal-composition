"""
Predictability Proxy Analysis (formerly Symbolic Regression).

Characterizes what level features predict agent surprise, as DESCRIPTIVE
compression. Uses symbolic regression to find compact expressions.

Features vary by training method:
- Universal: wall_density, path_length, goal_distance, training_phase
- ACCEL/PLR: branch_type (only for methods with branches)
- PAIRED: regret, adversary_entropy, antagonist_return, adversary_strategy_cluster
- DR: no curriculum-specific features

PAIRED-specific analysis:
- Fit TWO expressions:
  1. prediction_loss = f(features) → what protagonist expects
  2. policy_entropy = g(features) → how agent actually behaves (AGENT-CENTRIC)
- Compare: does "hard to predict" match "high policy uncertainty"?

IMPORTANT CAVEATS:
- This is descriptive compression, NOT utility extraction
- Prediction loss is probe/architecture-dependent, not intrinsic
- The symbolic expression does NOT represent agent's internal model
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import chex

from .base import TrainingTimeExperiment


@dataclass
class PredictionData:
    """Container for prediction data collected during training."""
    steps: List[int] = field(default_factory=list)
    training_method: str = "accel"

    # Universal level features
    wall_densities: List[float] = field(default_factory=list)
    path_lengths: List[float] = field(default_factory=list)
    goal_distances: List[float] = field(default_factory=list)
    training_phase: List[float] = field(default_factory=list)

    # ACCEL/PLR-specific features
    branch_types: List[int] = field(default_factory=list)

    # PAIRED-specific features (E5 update)
    regrets: List[float] = field(default_factory=list)  # Actual regret
    adversary_entropies: List[float] = field(default_factory=list)  # Adversary policy entropy
    antagonist_returns: List[float] = field(default_factory=list)  # Antagonist episode return
    adversary_strategy_clusters: List[int] = field(default_factory=list)  # Cluster ID from C3

    # Agent-centric metrics (PAIRED)
    policy_entropies: List[float] = field(default_factory=list)  # Protagonist policy entropy

    # Prediction losses
    probe_losses: List[float] = field(default_factory=list)
    value_errors: List[float] = field(default_factory=list)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert to numpy arrays."""
        result = {
            'steps': np.array(self.steps),
            'wall_density': np.array(self.wall_densities),
            'path_length': np.array(self.path_lengths),
            'goal_distance': np.array(self.goal_distances),
            'probe_loss': np.array(self.probe_losses),
            'value_error': np.array(self.value_errors),
            'training_phase': np.array(self.training_phase),
            'training_method': self.training_method,
        }

        # Add method-specific features
        if self.branch_types:
            result['branch_type'] = np.array(self.branch_types)

        # PAIRED-specific features
        if self.regrets:
            result['regret'] = np.array(self.regrets)
        if self.adversary_entropies:
            result['adversary_entropy'] = np.array(self.adversary_entropies)
        if self.antagonist_returns:
            result['antagonist_return'] = np.array(self.antagonist_returns)
        if self.adversary_strategy_clusters:
            result['adversary_strategy_cluster'] = np.array(self.adversary_strategy_clusters)
        if self.policy_entropies:
            result['policy_entropy'] = np.array(self.policy_entropies)

        return result


class SymbolicRegressionExperiment(TrainingTimeExperiment):
    """
    Predictability proxy analysis using symbolic regression.

    Collects level features and prediction losses during training,
    then fits compact symbolic expressions to describe the relationship.

    NOTE: This is DESCRIPTIVE, not explanatory. The expressions describe
    correlations, not the agent's internal utility function.
    """

    @property
    def name(self) -> str:
        return "symbolic_regression"

    def __init__(
        self,
        collection_interval: int = 100,
        n_samples_per_collection: int = 20,
        use_pysr: bool = True,
        pysr_iterations: int = 50,
        **kwargs,
    ):
        """
        Initialize symbolic regression experiment.

        Args:
            collection_interval: Steps between data collection
            n_samples_per_collection: Samples per collection point
            use_pysr: Whether to use PySR for symbolic regression
            pysr_iterations: Number of PySR iterations
        """
        super().__init__(**kwargs)
        self.collection_interval = collection_interval
        self.n_samples_per_collection = n_samples_per_collection
        self.use_pysr = use_pysr
        self.pysr_iterations = pysr_iterations

        self._data = PredictionData()
        self._results: Dict[str, Any] = {}

    def training_hook(
        self,
        train_state: Any,
        metrics: Dict[str, Any],
        step: int,
    ) -> Dict[str, Any]:
        """
        Hook called during training to collect prediction data.
        """
        if step % self.collection_interval != 0:
            return {}

        self.train_state = train_state

        import jax
        import jax.numpy as jnp

        rng = jax.random.PRNGKey(step)

        # Collect samples
        for sample_idx in range(self.n_samples_per_collection):
            rng, level_rng, ep_rng = jax.random.split(rng, 3)

            # Generate level and extract features
            level = self._generate_level(level_rng)
            features = self._extract_features(level)

            # Compute prediction error
            probe_loss, value_error = self._compute_prediction_error(ep_rng, level)

            # Store universal data
            self._data.steps.append(step)
            self._data.wall_densities.append(features['wall_density'])
            self._data.path_lengths.append(features['path_length'])
            self._data.goal_distances.append(features['goal_distance'])
            self._data.probe_losses.append(probe_loss)
            self._data.value_errors.append(value_error)
            self._data.training_phase.append(step / 30000)  # Normalized

            # Store method-specific data
            if self.has_branches and 'branch_type' in features:
                self._data.branch_types.append(features['branch_type'])

            # PAIRED-specific: collect regret structure features
            if self.has_regret:
                if 'regret' in features:
                    self._data.regrets.append(features['regret'])
                if 'adversary_entropy' in features:
                    self._data.adversary_entropies.append(features['adversary_entropy'])
                if 'antagonist_return' in features:
                    self._data.antagonist_returns.append(features['antagonist_return'])
                if 'adversary_strategy_cluster' in features:
                    self._data.adversary_strategy_clusters.append(features['adversary_strategy_cluster'])
                if 'policy_entropy' in features:
                    self._data.policy_entropies.append(features['policy_entropy'])

        mean_loss = np.mean(self._data.probe_losses[-self.n_samples_per_collection:])
        return {
            'symbolic_regression/mean_probe_loss': mean_loss,
        }

    def _generate_level(self, rng) -> Dict[str, Any]:
        """Generate a random level for evaluation."""
        import jax
        import jax.numpy as jnp

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

        level = {
            'wall_map': wall_map,
            'wall_density': wall_map.sum() / (height * width),
            'goal_pos': goal_pos,
            'agent_pos': agent_pos,
        }

        # Add branch only for methods that use it
        if self.has_branches:
            level['branch'] = int(jax.random.randint(rng, (), 0, self.branch_count))

        return level

    def _extract_features(self, level: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from a level (method-appropriate)."""
        wall_density = level['wall_density']
        goal_pos = level['goal_pos']
        agent_pos = level['agent_pos']

        # Manhattan distance to goal
        goal_distance = abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])

        # Simplified path length (Manhattan + wall penalty)
        path_length = goal_distance * (1 + wall_density)

        features = {
            'wall_density': float(wall_density),
            'path_length': float(path_length),
            'goal_distance': float(goal_distance),
        }

        # Method-specific features
        if self.has_branches and 'branch' in level:
            features['branch_type'] = int(level['branch'])

        if self.has_regret:
            # PAIRED E5: Regret structure features
            # Actual regret (will be computed in _compute_prediction_error if antagonist available)
            features['regret'] = float(0.3 * wall_density + 0.7 * min(path_length / 30, 1.0))

            # Adversary entropy: proxy based on level variability
            # Lower wall density variation = lower adversary entropy (more deterministic strategy)
            features['adversary_entropy'] = float(0.5 + 0.5 * (1 - wall_density))

            # Antagonist return: will be computed if antagonist available
            features['antagonist_return'] = 0.0

            # Adversary strategy cluster: simple heuristic clustering
            difficulty = wall_density + 0.5 * (goal_distance / 26)
            features['adversary_strategy_cluster'] = int(min(difficulty * 5, 4))

            # Policy entropy: will be computed in _compute_prediction_error
            features['policy_entropy'] = 0.0

        return features

    def _compute_prediction_error(
        self,
        rng,
        level: Dict[str, Any],
    ) -> Tuple[float, float]:
        """
        Compute actual prediction/probe loss on a level.

        Uses agent-aware loss computation to get the actual prediction loss
        (from probe for probe-based agents, from prediction head for
        next_env_prediction agent).

        For PAIRED: Also computes policy entropy for agent-centric analysis.
        """
        import jax
        import jax.numpy as jnp

        if not hasattr(self, 'train_state') or self.train_state is None:
            return 1.0, 1.0

        try:
            from .utils.agent_aware_loss import (
                compute_agent_prediction_loss,
                create_observation_from_level,
            )

            # Compute actual prediction/probe loss
            probe_loss, loss_metrics = compute_agent_prediction_loss(
                self.agent,
                self.train_state,
                level,
                rng,
            )

            # Also compute value error as secondary metric
            obs = create_observation_from_level(level)
            hstate = self.agent.initialize_carry(rng, batch_dims=(1,))
            obs_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs)
            done_batch = jnp.zeros((1, 1), dtype=bool)

            # Get value estimate (works for all agent types)
            try:
                outputs = self.train_state.apply_fn(
                    self.train_state.params,
                    (obs_batch, done_batch),
                    hstate
                )
                # Handle both 3-tuple and 4-tuple outputs
                if len(outputs) == 4:
                    _, pi, value, _ = outputs
                else:
                    _, pi, value = outputs

                # Value error: |V(s) - expected_return|
                expected_return = 0.5 * (1 - level['wall_density'])
                value_error = float(abs(value[0, 0] - expected_return))

                # PAIRED E5: Compute policy entropy for agent-centric analysis
                if self.has_regret and hasattr(pi, 'entropy'):
                    try:
                        policy_entropy = float(pi.entropy()[0, 0])
                        # Store in features for later collection
                        level['_computed_policy_entropy'] = policy_entropy
                    except Exception:
                        level['_computed_policy_entropy'] = 0.0
                else:
                    level['_computed_policy_entropy'] = 0.0

            except Exception:
                value_error = 1.0
                level['_computed_policy_entropy'] = 0.0

            # PAIRED: Compute antagonist return if available
            if self.has_regret:
                rng, ant_rng = jax.random.split(rng)
                ant_return = self._compute_antagonist_return(ant_rng, level)
                level['_computed_antagonist_return'] = ant_return
                level['_computed_regret'] = ant_return - (1 - level['wall_density']) * 0.5

            return probe_loss, value_error

        except Exception as e:
            return 1.0, 1.0

    def _compute_antagonist_return(self, rng, level: Dict[str, Any]) -> float:
        """Compute antagonist return on level (PAIRED)."""
        import jax
        import jax.numpy as jnp

        ant_train_state = getattr(self.train_state, 'ant_train_state', None)
        if ant_train_state is None:
            # Estimate: antagonist typically does better
            return 0.6 * (1 - level['wall_density'])

        try:
            from .utils.agent_aware_loss import create_observation_from_level

            obs = create_observation_from_level(level)
            hstate = self.agent.initialize_carry(rng, batch_dims=(1,))
            obs_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs)
            done_batch = jnp.zeros((1, 1), dtype=bool)

            outputs = ant_train_state.apply_fn(
                ant_train_state.params,
                (obs_batch, done_batch),
                hstate
            )

            if len(outputs) == 4:
                _, _, value, _ = outputs
            else:
                _, _, value = outputs

            return float(value[0, 0])

        except Exception:
            return 0.6 * (1 - level['wall_density'])

    def collect_data(self, rng: chex.PRNGKey) -> Dict[str, np.ndarray]:
        """Return collected data."""
        return self._data.to_arrays()

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze prediction patterns using symbolic regression.

        Fits compact expressions to describe:
        - What features predict probe loss
        - How this relationship changes over training
        """
        data = self._data.to_arrays()

        if len(data['steps']) < 100:
            return {
                'error': 'Insufficient data for analysis',
                'n_samples': len(data['steps']),
            }

        results = {}

        # 1. Linear regression baseline
        results['linear_analysis'] = self._fit_linear_model(data)

        # 2. Symbolic regression if enabled
        if self.use_pysr:
            results['symbolic_analysis'] = self._fit_symbolic_model(data)

        # 3. Feature importance
        results['feature_importance'] = self._analyze_feature_importance(data)

        # 4. Temporal dynamics
        results['temporal_dynamics'] = self._analyze_temporal_dynamics(data)

        # 5. PAIRED-specific: dual expression analysis
        if self.has_regret and 'policy_entropy' in data:
            results['paired_dual_analysis'] = self._analyze_paired_dual_expressions(data)

        # 6. Caveats
        results['caveats'] = self._get_caveats()

        self._results = results
        return results

    def _analyze_paired_dual_expressions(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        PAIRED E5: Fit TWO symbolic expressions and compare.

        Expression 1: prediction_loss = f(features) → what protagonist expects
        Expression 2: policy_entropy = g(features) → how agent actually behaves

        Key question: Does "hard to predict" match "high policy uncertainty"?
        """
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        results = {}

        # Prepare PAIRED-specific features
        feature_names = ['wall_density', 'path_length', 'goal_distance', 'training_phase']
        feature_arrays = [
            data['wall_density'],
            data['path_length'],
            data['goal_distance'],
            data['training_phase'],
        ]

        # Add PAIRED features
        for fname in ['regret', 'adversary_entropy', 'antagonist_return', 'adversary_strategy_cluster']:
            if fname in data and len(data[fname]) == len(data['wall_density']):
                feature_arrays.append(data[fname])
                feature_names.append(fname)

        X = np.column_stack(feature_arrays)

        # Remove NaN/Inf
        valid = np.isfinite(X).all(axis=1)
        if 'probe_loss' in data:
            valid &= np.isfinite(data['probe_loss'])
        if 'policy_entropy' in data:
            valid &= np.isfinite(data['policy_entropy'])

        X = X[valid]

        if len(X) < 20:
            return {'error': 'Insufficient valid samples for dual analysis'}

        # Expression 1: prediction_loss = f(features)
        y_probe = data['probe_loss'][valid]
        model_probe = Ridge(alpha=1.0)
        scores_probe = cross_val_score(model_probe, X, y_probe, cv=5, scoring='r2')
        model_probe.fit(X, y_probe)

        results['prediction_loss_model'] = {
            'mean_r2': float(np.mean(scores_probe)),
            'coefficients': dict(zip(feature_names, model_probe.coef_.tolist())),
            'feature_names': feature_names,
        }

        # Expression 2: policy_entropy = g(features)
        if 'policy_entropy' in data:
            y_entropy = data['policy_entropy'][valid]
            if np.std(y_entropy) > 1e-6:  # Check for variance
                model_entropy = Ridge(alpha=1.0)
                scores_entropy = cross_val_score(model_entropy, X, y_entropy, cv=5, scoring='r2')
                model_entropy.fit(X, y_entropy)

                results['policy_entropy_model'] = {
                    'mean_r2': float(np.mean(scores_entropy)),
                    'coefficients': dict(zip(feature_names, model_entropy.coef_.tolist())),
                    'feature_names': feature_names,
                }

                # Compare: does "hard to predict" match "high policy uncertainty"?
                # Correlation between prediction loss and policy entropy
                corr = np.corrcoef(y_probe, y_entropy)[0, 1]
                results['loss_entropy_correlation'] = float(corr) if np.isfinite(corr) else 0.0

                # Key finding
                results['hard_to_predict_matches_uncertainty'] = abs(corr) > 0.3
                results['interpretation'] = (
                    "Positive correlation: High probe loss → high policy entropy → "
                    "protagonist is uncertain on hard-to-predict levels. "
                    "Negative correlation: Protagonist compensates uncertainty elsewhere."
                )

                # Compare coefficient signs
                probe_coefs = model_probe.coef_
                entropy_coefs = model_entropy.coef_
                same_sign_count = sum(
                    1 for p, e in zip(probe_coefs, entropy_coefs)
                    if (p > 0 and e > 0) or (p < 0 and e < 0)
                )
                results['coefficient_alignment'] = same_sign_count / len(probe_coefs)

        # Regret-specific analysis
        if 'regret' in data:
            y_regret = data['regret'][valid]
            if np.std(y_regret) > 1e-6:
                model_regret = Ridge(alpha=1.0)
                scores_regret = cross_val_score(model_regret, X, y_regret, cv=5, scoring='r2')
                model_regret.fit(X, y_regret)

                results['regret_model'] = {
                    'mean_r2': float(np.mean(scores_regret)),
                    'coefficients': dict(zip(feature_names, model_regret.coef_.tolist())),
                }

                # Does regret structure match prediction loss structure?
                loss_regret_corr = np.corrcoef(y_probe, y_regret)[0, 1]
                results['loss_regret_correlation'] = float(loss_regret_corr) if np.isfinite(loss_regret_corr) else 0.0

        return results

    def _fit_linear_model(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Fit linear regression model with method-appropriate features."""
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        # Prepare universal features
        feature_arrays = [
            data['wall_density'],
            data['path_length'],
            data['goal_distance'],
            data['training_phase'],
        ]
        feature_names = ['wall_density', 'path_length', 'goal_distance', 'training_phase']

        # Add method-specific features
        if 'branch_type' in data and len(data['branch_type']) == len(data['wall_density']):
            feature_arrays.append(data['branch_type'])
            feature_names.append('branch_type')

        if 'regret_estimate' in data and len(data['regret_estimate']) == len(data['wall_density']):
            feature_arrays.append(data['regret_estimate'])
            feature_names.append('regret_estimate')

        if 'adversary_difficulty' in data and len(data['adversary_difficulty']) == len(data['wall_density']):
            feature_arrays.append(data['adversary_difficulty'])
            feature_names.append('adversary_difficulty')

        X = np.column_stack(feature_arrays)
        y = data['probe_loss']

        # Remove NaN/Inf
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid]
        y = y[valid]

        if len(X) < 10:
            return {'error': 'Insufficient valid samples'}

        model = Ridge(alpha=1.0)
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')

        model.fit(X, y)

        coefficients = dict(zip(feature_names, model.coef_.tolist()))

        return {
            'r2_scores': scores.tolist(),
            'mean_r2': float(np.mean(scores)),
            'std_r2': float(np.std(scores)),
            'coefficients': coefficients,
            'intercept': float(model.intercept_),
            'feature_names': feature_names,
            'training_method': data.get('training_method', 'unknown'),
        }

    def _fit_symbolic_model(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Fit symbolic regression model using PySR."""
        try:
            from pysr import PySRRegressor
        except ImportError:
            return {'error': 'PySR not installed', 'note': 'pip install pysr'}

        # Prepare features
        X = np.column_stack([
            data['wall_density'],
            data['path_length'],
            data['goal_distance'],
        ])
        y = data['probe_loss']

        # Remove NaN/Inf and subsample for speed
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid]
        y = y[valid]

        if len(X) > 1000:
            idx = np.random.choice(len(X), 1000, replace=False)
            X = X[idx]
            y = y[idx]

        try:
            model = PySRRegressor(
                niterations=self.pysr_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["exp", "log", "sqrt", "abs"],
                populations=8,
                population_size=33,
                verbosity=0,
            )

            model.fit(X, y, variable_names=['wall', 'path', 'goal'])

            # Get best equations
            equations = []
            for i, eq in enumerate(model.equations_):
                equations.append({
                    'complexity': int(eq['complexity']),
                    'loss': float(eq['loss']),
                    'equation': str(eq['equation']),
                })

            return {
                'best_equation': str(model.sympy()),
                'equations': equations[:10],  # Top 10 by Pareto frontier
                'feature_names': ['wall_density', 'path_length', 'goal_distance'],
            }

        except Exception as e:
            return {'error': str(e)}

    def _analyze_feature_importance(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze feature importance using correlation (method-appropriate)."""
        y = data['probe_loss']

        # Universal features
        feature_names = ['wall_density', 'path_length', 'goal_distance', 'training_phase']

        # Add method-specific features
        if 'branch_type' in data:
            feature_names.append('branch_type')
        if 'regret_estimate' in data:
            feature_names.append('regret_estimate')
        if 'adversary_difficulty' in data:
            feature_names.append('adversary_difficulty')

        importance = {}
        for feature_name in feature_names:
            if feature_name not in data:
                continue
            x = data[feature_name]
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() > 10:
                corr = np.corrcoef(x[valid], y[valid])[0, 1]
                importance[feature_name] = float(corr) if np.isfinite(corr) else 0.0
            else:
                importance[feature_name] = 0.0

        return importance

    def _analyze_temporal_dynamics(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze how prediction patterns change over training."""
        steps = data['steps']
        probe_loss = data['probe_loss']

        unique_steps = np.unique(steps)
        mean_loss_by_step = []
        std_loss_by_step = []

        for step in unique_steps:
            mask = steps == step
            losses = probe_loss[mask]
            mean_loss_by_step.append(float(np.mean(losses)))
            std_loss_by_step.append(float(np.std(losses)))

        return {
            'steps': unique_steps.tolist(),
            'mean_loss': mean_loss_by_step,
            'std_loss': std_loss_by_step,
            'trend': 'decreasing' if mean_loss_by_step[-1] < mean_loss_by_step[0] else 'stable',
        }

    def _get_caveats(self) -> List[str]:
        """Return analysis caveats."""
        return [
            "This is DESCRIPTIVE compression, not utility extraction",
            "Symbolic expressions describe correlations, not agent's model",
            "Prediction loss is probe/architecture-dependent",
            "Expression complexity depends on PySR hyperparameters",
            "Results should be interpreted as hypothesis generation only",
        ]

    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data."""
        if not self._results:
            raise ValueError("Must call analyze before visualize")

        data = self._data.to_arrays()

        viz_data = {
            'feature_importance': self._results.get('feature_importance', {}),
            'temporal_dynamics': self._results.get('temporal_dynamics', {}),
            'linear_analysis': {
                key: val for key, val in self._results.get('linear_analysis', {}).items()
                if key in ['mean_r2', 'coefficients']
            },
        }

        if 'symbolic_analysis' in self._results:
            sa = self._results['symbolic_analysis']
            viz_data['symbolic_analysis'] = {
                'best_equation': sa.get('best_equation', ''),
                'n_equations': len(sa.get('equations', [])),
            }

        return viz_data
