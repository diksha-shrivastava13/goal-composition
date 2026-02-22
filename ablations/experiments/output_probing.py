"""
Output Probing Experiment.

Analyzes policy and value outputs to determine if they reveal curriculum
information. Tests whether behavioral outputs encode curriculum structure
beyond what's needed for task performance.

PAIRED-specific:
- Can policy/value outputs predict adversary's next level features?
- Compare protagonist vs antagonist output entropy on high-regret levels
- AGENT-CENTRIC: Measure policy entropy by regret condition
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
import chex

from .base import CheckpointExperiment
from .probes.output_probe import (
    PolicyEntropyAnalyzer,
    BranchClassifier,
    OutputProbe,
)


@dataclass
class OutputData:
    """Container for output data."""
    # Policy outputs
    policy_logits: List[np.ndarray] = field(default_factory=list)
    policy_entropies: List[float] = field(default_factory=list)

    # Value outputs
    value_estimates: List[float] = field(default_factory=list)

    # Metadata
    branch_types: List[int] = field(default_factory=list)
    wall_densities: List[float] = field(default_factory=list)
    episode_outcomes: List[int] = field(default_factory=list)  # 0=timeout, 1=solved

    # Actual prediction losses (PRIMARY METRIC for causal analysis)
    prediction_losses: List[float] = field(default_factory=list)
    levels: List[Dict[str, Any]] = field(default_factory=list)  # Store levels for loss computation

    # PAIRED-specific data
    regrets: List[float] = field(default_factory=list)
    adversary_difficulty: List[float] = field(default_factory=list)
    antagonist_entropies: List[float] = field(default_factory=list)  # Bilateral comparison
    antagonist_values: List[float] = field(default_factory=list)


class OutputProbingExperiment(CheckpointExperiment):
    """
    Probe policy and value outputs for curriculum encoding.

    Tests:
    1. Policy entropy by branch type
    2. Branch prediction from (policy, value, entropy)
    3. Actor-critic divergence
    4. Value head curriculum encoding
    """

    @property
    def name(self) -> str:
        return "output_probing"

    def __init__(
        self,
        n_episodes: int = 200,
        n_steps_per_episode: int = 50,
        **kwargs,
    ):
        """
        Initialize output probing experiment.

        Args:
            n_episodes: Number of episodes to collect
            n_steps_per_episode: Steps per episode to analyze
        """
        super().__init__(**kwargs)
        self.n_episodes = n_episodes
        self.n_steps_per_episode = n_steps_per_episode

        self._data: Optional[OutputData] = None
        self._results: Dict[str, Any] = {}

    def collect_data(self, rng: chex.PRNGKey) -> OutputData:
        """
        Collect policy and value outputs across episodes.
        """
        self._data = OutputData()

        for ep_idx in range(self.n_episodes):
            rng, ep_rng, level_rng = jax.random.split(rng, 3)

            # Generate level with known branch
            branch = int(jax.random.randint(rng, (), 0, 3))
            level = self._generate_level(level_rng, branch)

            # Run episode and collect outputs
            episode_data = self._run_episode(ep_rng, level)

            # Store data
            self._data.policy_logits.extend(episode_data['policy_logits'])
            self._data.policy_entropies.extend(episode_data['entropies'])
            self._data.value_estimates.extend(episode_data['values'])
            self._data.branch_types.extend([branch] * len(episode_data['values']))
            self._data.wall_densities.extend([level['wall_density']] * len(episode_data['values']))
            self._data.episode_outcomes.extend([episode_data['solved']] * len(episode_data['values']))

            # Store prediction loss (same for all steps in episode)
            self._data.prediction_losses.extend([episode_data['prediction_loss']] * len(episode_data['values']))
            self._data.levels.append(level)

            # PAIRED-specific: collect regret and antagonist data
            if self.has_regret:
                rng, ant_rng = jax.random.split(rng)
                paired_data = self._collect_paired_episode_data(ant_rng, level, episode_data)

                self._data.regrets.extend([paired_data['regret']] * len(episode_data['values']))
                self._data.adversary_difficulty.extend([paired_data['adversary_difficulty']] * len(episode_data['values']))
                self._data.antagonist_entropies.extend(paired_data['antagonist_entropies'])
                self._data.antagonist_values.extend(paired_data['antagonist_values'])

        return self._data

    def _collect_paired_episode_data(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
        protagonist_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Collect PAIRED-specific episode data including antagonist outputs."""
        # Estimate regret based on protagonist performance
        pro_return = sum(1 if protagonist_data['solved'] else 0 for _ in range(1))

        # Compute adversary difficulty
        wall_density = level['wall_density']
        goal_pos = level['goal_pos']
        agent_pos = level['agent_pos']
        goal_dist = abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])
        adversary_difficulty = wall_density * 0.5 + (goal_dist / 26) * 0.5

        # Get antagonist outputs on same level
        ant_train_state = getattr(self.train_state, 'ant_train_state', None)

        antagonist_entropies = []
        antagonist_values = []

        if ant_train_state is not None:
            hstate = self.agent.initialize_carry(rng, batch_dims=(1,))

            for step in range(min(self.n_steps_per_episode, 10)):  # Sample fewer steps for antagonist
                rng, step_rng = jax.random.split(rng)

                obs = self._create_observation(level, step)
                obs_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs)
                done_batch = jnp.zeros((1, 1), dtype=bool)

                hstate, pi, value = ant_train_state.apply_fn(
                    ant_train_state.params, (obs_batch, done_batch), hstate
                )

                antagonist_entropies.append(float(pi.entropy()[0, 0]))
                antagonist_values.append(float(value[0, 0]))
        else:
            # No antagonist - use placeholder
            antagonist_entropies = [0.0] * len(protagonist_data['values'])
            antagonist_values = [0.0] * len(protagonist_data['values'])

        # Pad to match protagonist data length
        while len(antagonist_entropies) < len(protagonist_data['values']):
            antagonist_entropies.append(antagonist_entropies[-1] if antagonist_entropies else 0.0)
            antagonist_values.append(antagonist_values[-1] if antagonist_values else 0.0)

        # Estimate regret from value difference
        pro_mean_value = np.mean(protagonist_data['values'])
        ant_mean_value = np.mean(antagonist_values) if antagonist_values else 0.0
        regret = max(0, ant_mean_value - pro_mean_value)

        return {
            'regret': regret,
            'adversary_difficulty': adversary_difficulty,
            'antagonist_entropies': antagonist_entropies[:len(protagonist_data['values'])],
            'antagonist_values': antagonist_values[:len(protagonist_data['values'])],
        }

    def _generate_level(self, rng: chex.PRNGKey, branch: int) -> Dict[str, Any]:
        """Generate a level with specified branch characteristics."""
        height, width = 13, 13

        if branch == 0:  # DR
            wall_prob = float(jax.random.uniform(rng)) * 0.3
        elif branch == 1:  # Replay
            wall_prob = 0.15 + float(jax.random.uniform(rng)) * 0.1
        else:  # Mutate
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
            'branch': branch,
        }

    def _run_episode(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run episode and collect outputs."""
        hstate = self.agent.initialize_carry(rng, batch_dims=(1,))

        policy_logits = []
        entropies = []
        values = []
        solved = False

        # Compute prediction loss at episode start (PRIMARY METRIC)
        from .utils.agent_aware_loss import compute_agent_prediction_loss
        rng, loss_rng = jax.random.split(rng)
        prediction_loss, _ = compute_agent_prediction_loss(
            self.agent, self.train_state, level, loss_rng
        )

        for step in range(self.n_steps_per_episode):
            rng, step_rng = jax.random.split(rng)

            obs = self._create_observation(level, step)
            new_hstate, pi, value = self._forward_step(obs, hstate)

            policy_logits.append(np.array(pi.logits[0, 0]))
            entropies.append(float(pi.entropy()[0, 0]))
            values.append(float(value[0, 0]))

            hstate = new_hstate

            # Check if solved (simplified)
            if step > 10:
                solve_prob = 0.3 * (1 - level['wall_density'])
                if float(jax.random.uniform(step_rng)) < solve_prob / self.n_steps_per_episode:
                    solved = True
                    break

        return {
            'policy_logits': policy_logits,
            'entropies': entropies,
            'values': values,
            'solved': int(solved),
            'prediction_loss': prediction_loss,
        }

    def _create_observation(self, level: Dict[str, Any], step: int) -> Any:
        """Create observation from level state."""
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

    def _forward_step(self, obs: Any, hstate: Any) -> Tuple[Any, Any, Any]:
        """Run single forward step."""
        params = self.train_state.params
        apply_fn = self.train_state.apply_fn

        obs_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs)
        done_batch = jnp.zeros((1, 1), dtype=bool)

        new_hstate, pi, value = apply_fn(params, (obs_batch, done_batch), hstate)

        return new_hstate, pi, value

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze output patterns for curriculum encoding.
        """
        if self._data is None:
            raise ValueError("Must call collect_data before analyze")

        results = {}

        # PRIMARY: Prediction loss by branch (actual causal measure)
        results['prediction_loss_by_branch'] = self._analyze_prediction_loss_by_branch()

        # 1. Policy entropy by branch
        results['entropy_by_branch'] = self._analyze_entropy_by_branch()

        # 2. Branch prediction from outputs
        results['branch_prediction'] = self._analyze_branch_prediction()

        # 3. Value by branch
        results['value_by_branch'] = self._analyze_value_by_branch()

        # 4. Output correlation with difficulty
        results['difficulty_correlation'] = self._analyze_difficulty_correlation()

        # 5. Actor-critic divergence
        results['actor_critic_divergence'] = self._analyze_actor_critic_divergence()

        # 6. PAIRED-specific: regret-conditioned analysis
        if self.has_regret and self._data.regrets:
            results['regret_conditioned'] = self._analyze_regret_conditioned_outputs()
            results['bilateral_entropy'] = self._analyze_bilateral_entropy()
            results['adversary_prediction'] = self._analyze_adversary_prediction()

        self._results = results
        return results

    def _analyze_regret_conditioned_outputs(self) -> Dict[str, Any]:
        """
        Analyze policy entropy by regret condition (PAIRED AGENT-CENTRIC).

        Key question: Does protagonist have higher entropy on high-regret levels?
        """
        entropies = np.array(self._data.policy_entropies)
        values = np.array(self._data.value_estimates)
        regrets = np.array(self._data.regrets)

        # Split into regret terciles
        regret_33 = np.percentile(regrets, 33)
        regret_66 = np.percentile(regrets, 66)

        results = {}

        for name, mask in [
            ('low_regret', regrets <= regret_33),
            ('medium_regret', (regrets > regret_33) & (regrets <= regret_66)),
            ('high_regret', regrets > regret_66),
        ]:
            if mask.sum() < 10:
                continue

            results[name] = {
                'mean_entropy': float(np.mean(entropies[mask])),
                'std_entropy': float(np.std(entropies[mask])),
                'mean_value': float(np.mean(values[mask])),
                'mean_regret': float(np.mean(regrets[mask])),
                'n_samples': int(mask.sum()),
            }

        # Key finding: Does protagonist have higher entropy on high-regret?
        if 'high_regret' in results and 'low_regret' in results:
            high_entropy = results['high_regret']['mean_entropy']
            low_entropy = results['low_regret']['mean_entropy']

            results['higher_entropy_on_high_regret'] = high_entropy > low_entropy
            results['entropy_gap'] = float(high_entropy - low_entropy)
            results['interpretation'] = (
                "Higher entropy on high-regret levels suggests protagonist is "
                "less certain on levels where antagonist outperforms."
                if high_entropy > low_entropy else
                "Protagonist maintains similar confidence across regret levels."
            )

        # Correlation between regret and entropy
        valid = np.isfinite(regrets) & np.isfinite(entropies)
        if valid.sum() > 10:
            corr = np.corrcoef(regrets[valid], entropies[valid])[0, 1]
            results['regret_entropy_correlation'] = float(corr) if np.isfinite(corr) else 0.0

        return results

    def _analyze_bilateral_entropy(self) -> Dict[str, Any]:
        """
        Compare protagonist vs antagonist entropy on high-regret levels (PAIRED).
        """
        pro_entropies = np.array(self._data.policy_entropies)
        ant_entropies = np.array(self._data.antagonist_entropies)
        regrets = np.array(self._data.regrets)

        if len(ant_entropies) == 0 or len(ant_entropies) != len(pro_entropies):
            return {'error': 'Insufficient antagonist data'}

        results = {}

        # Overall comparison
        results['overall'] = {
            'protagonist_mean_entropy': float(np.mean(pro_entropies)),
            'antagonist_mean_entropy': float(np.mean(ant_entropies)),
            'antagonist_more_confident': np.mean(ant_entropies) < np.mean(pro_entropies),
        }

        # High-regret comparison
        regret_66 = np.percentile(regrets, 66)
        high_regret_mask = regrets > regret_66

        if high_regret_mask.sum() >= 10:
            results['high_regret'] = {
                'protagonist_entropy': float(np.mean(pro_entropies[high_regret_mask])),
                'antagonist_entropy': float(np.mean(ant_entropies[high_regret_mask])),
                'entropy_gap': float(
                    np.mean(pro_entropies[high_regret_mask]) -
                    np.mean(ant_entropies[high_regret_mask])
                ),
                'antagonist_more_confident_on_hard': (
                    np.mean(ant_entropies[high_regret_mask]) <
                    np.mean(pro_entropies[high_regret_mask])
                ),
            }

            # Key finding
            results['interpretation'] = (
                "Antagonist has lower entropy on high-regret levels, suggesting "
                "it is more confident on levels where protagonist struggles."
                if results['high_regret']['antagonist_more_confident_on_hard'] else
                "Protagonist and antagonist have similar confidence on high-regret levels."
            )

        return results

    def _analyze_adversary_prediction(self) -> Dict[str, Any]:
        """
        Test if policy/value outputs can predict adversary's next level features (PAIRED).
        """
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        # Prepare features: policy logits + value + entropy
        policy_logits = np.array(self._data.policy_logits)
        values = np.array(self._data.value_estimates).reshape(-1, 1)
        entropies = np.array(self._data.policy_entropies).reshape(-1, 1)

        X = np.concatenate([policy_logits, values, entropies], axis=1)

        # Target: adversary difficulty (proxy for next level features)
        y = np.array(self._data.adversary_difficulty)

        # Remove invalid samples
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid]
        y = y[valid]

        if len(X) < 50:
            return {'error': 'Insufficient samples'}

        # Cross-validated R²
        model = Ridge(alpha=1.0)
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')

        model.fit(X, y)

        results = {
            'adversary_difficulty_prediction_r2': float(np.mean(scores)),
            'r2_std': float(np.std(scores)),
            'can_predict_adversary': np.mean(scores) > 0.1,
            'interpretation': (
                "Policy/value outputs encode information about adversary strategy, "
                "suggesting protagonist may have a model of adversary behavior."
                if np.mean(scores) > 0.1 else
                "Policy/value outputs do not strongly predict adversary strategy."
            ),
        }

        # Also test prediction of regret
        y_regret = np.array(self._data.regrets)[valid]
        if np.std(y_regret) > 1e-6:
            scores_regret = cross_val_score(model, X, y_regret, cv=5, scoring='r2')
            results['regret_prediction_r2'] = float(np.mean(scores_regret))
            results['can_predict_regret'] = np.mean(scores_regret) > 0.1

        return results

    def _analyze_entropy_by_branch(self) -> Dict[str, Any]:
        """Analyze policy entropy by branch type."""
        entropies = np.array(self._data.policy_entropies)
        branches = np.array(self._data.branch_types)

        results = {}
        for branch in [0, 1, 2]:
            mask = branches == branch
            if mask.sum() > 0:
                branch_name = ['DR', 'Replay', 'Mutate'][branch]
                results[branch_name] = {
                    'mean_entropy': float(np.mean(entropies[mask])),
                    'std_entropy': float(np.std(entropies[mask])),
                    'n_samples': int(mask.sum()),
                }

        # Compare: Is Replay entropy lower? (hypothesis: more confident on practiced levels)
        if 'Replay' in results and 'DR' in results:
            results['replay_lower_than_dr'] = (
                results['Replay']['mean_entropy'] < results['DR']['mean_entropy']
            )

        return results

    def _analyze_prediction_loss_by_branch(self) -> Dict[str, Any]:
        """
        Analyze prediction loss by branch type.

        This is the PRIMARY metric for curriculum awareness - it measures
        actual prediction ability, not proxy measures like classification accuracy.
        """
        from .utils.agent_aware_loss import compute_random_baseline_loss

        losses = np.array(self._data.prediction_losses)
        branches = np.array(self._data.branch_types)

        random_baseline = compute_random_baseline_loss()

        results = {}
        for branch in [0, 1, 2]:
            mask = branches == branch
            if mask.sum() > 0:
                branch_name = ['DR', 'Replay', 'Mutate'][branch]
                branch_losses = losses[mask]
                results[branch_name] = {
                    'mean_prediction_loss': float(np.mean(branch_losses)),
                    'std_prediction_loss': float(np.std(branch_losses)),
                    'information_gain': float(random_baseline - np.mean(branch_losses)),
                    'n_samples': int(mask.sum()),
                }

        # Overall metrics
        results['overall'] = {
            'mean_prediction_loss': float(np.mean(losses)),
            'random_baseline': random_baseline,
            'total_information_gain': float(random_baseline - np.mean(losses)),
        }

        # Correlation between outputs and prediction loss
        entropies = np.array(self._data.policy_entropies)
        values = np.array(self._data.value_estimates)

        valid = np.isfinite(losses) & np.isfinite(entropies) & np.isfinite(values)
        if valid.sum() > 10:
            entropy_corr = np.corrcoef(losses[valid], entropies[valid])[0, 1]
            value_corr = np.corrcoef(losses[valid], values[valid])[0, 1]
            results['correlations'] = {
                'entropy_vs_loss': float(entropy_corr) if np.isfinite(entropy_corr) else 0.0,
                'value_vs_loss': float(value_corr) if np.isfinite(value_corr) else 0.0,
                'interpretation': (
                    "Negative entropy_vs_loss means lower entropy (more confident) correlates with better prediction. "
                    "Negative value_vs_loss means higher value correlates with better prediction."
                ),
            }

        return results

    def _analyze_branch_prediction(self) -> Dict[str, Any]:
        """Train classifier to predict branch from outputs."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        # Prepare features: policy logits + value + entropy
        policy_logits = np.array(self._data.policy_logits)
        values = np.array(self._data.value_estimates).reshape(-1, 1)
        entropies = np.array(self._data.policy_entropies).reshape(-1, 1)

        X = np.concatenate([policy_logits, values, entropies], axis=1)
        y = np.array(self._data.branch_types)

        # Remove invalid samples
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid]
        y = y[valid]

        if len(X) < 50:
            return {'error': 'Insufficient samples'}

        # Cross-validated accuracy
        model = LogisticRegression(max_iter=1000, multi_class='multinomial')
        scores = cross_val_score(model, X, y, cv=5)

        # Fit final model for feature importance
        model.fit(X, y)

        # Chance level for 3 classes
        chance_level = 1.0 / 3.0

        return {
            'cv_accuracy': float(np.mean(scores)),
            'cv_std': float(np.std(scores)),
            'chance_level': chance_level,
            'above_chance': float(np.mean(scores)) > chance_level + 0.05,
            'n_samples': len(X),
        }

    def _analyze_value_by_branch(self) -> Dict[str, Any]:
        """Analyze value estimates by branch type."""
        values = np.array(self._data.value_estimates)
        branches = np.array(self._data.branch_types)

        results = {}
        for branch in [0, 1, 2]:
            mask = branches == branch
            if mask.sum() > 0:
                branch_name = ['DR', 'Replay', 'Mutate'][branch]
                results[branch_name] = {
                    'mean_value': float(np.mean(values[mask])),
                    'std_value': float(np.std(values[mask])),
                    'n_samples': int(mask.sum()),
                }

        # ANOVA-like test: do values differ by branch?
        from scipy import stats
        branch_values = [values[branches == b] for b in [0, 1, 2] if (branches == b).sum() > 0]
        if len(branch_values) >= 2:
            f_stat, p_value = stats.f_oneway(*branch_values)
            results['anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
            }

        return results

    def _analyze_difficulty_correlation(self) -> Dict[str, Any]:
        """Analyze correlation between outputs and difficulty (wall density)."""
        entropies = np.array(self._data.policy_entropies)
        values = np.array(self._data.value_estimates)
        densities = np.array(self._data.wall_densities)

        valid = np.isfinite(densities) & np.isfinite(entropies) & np.isfinite(values)

        if valid.sum() < 10:
            return {'error': 'Insufficient valid samples'}

        # Correlations
        entropy_corr = np.corrcoef(densities[valid], entropies[valid])[0, 1]
        value_corr = np.corrcoef(densities[valid], values[valid])[0, 1]

        return {
            'entropy_density_correlation': float(entropy_corr) if np.isfinite(entropy_corr) else 0.0,
            'value_density_correlation': float(value_corr) if np.isfinite(value_corr) else 0.0,
            'interpretation': {
                'entropy': 'Higher density → higher entropy (more uncertain)' if entropy_corr > 0.1 else 'No clear relationship',
                'value': 'Higher density → lower value (harder)' if value_corr < -0.1 else 'No clear relationship',
            },
        }

    def _analyze_actor_critic_divergence(self) -> Dict[str, Any]:
        """Analyze divergence between policy and value patterns."""
        policy_logits = np.array(self._data.policy_logits)
        values = np.array(self._data.value_estimates)
        branches = np.array(self._data.branch_types)

        # Compute how well policy and value predict branch
        from sklearn.linear_model import LogisticRegression

        valid = np.isfinite(policy_logits).all(axis=1) & np.isfinite(values)
        policy_logits = policy_logits[valid]
        values = values[valid].reshape(-1, 1)
        branches = branches[valid]

        if len(branches) < 50:
            return {'error': 'Insufficient samples'}

        # Policy-only prediction
        policy_model = LogisticRegression(max_iter=1000, multi_class='multinomial')
        policy_model.fit(policy_logits, branches)
        policy_acc = policy_model.score(policy_logits, branches)

        # Value-only prediction
        value_model = LogisticRegression(max_iter=1000, multi_class='multinomial')
        value_model.fit(values, branches)
        value_acc = value_model.score(values, branches)

        return {
            'policy_branch_accuracy': float(policy_acc),
            'value_branch_accuracy': float(value_acc),
            'divergence': float(abs(policy_acc - value_acc)),
            'interpretation': (
                'Policy encodes more branch info' if policy_acc > value_acc + 0.05
                else ('Value encodes more branch info' if value_acc > policy_acc + 0.05
                      else 'Similar encoding')
            ),
        }

    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data."""
        if not self._results:
            raise ValueError("Must call analyze before visualize")

        viz_data = {
            'entropy_by_branch': self._results.get('entropy_by_branch', {}),
            'branch_prediction': {
                'accuracy': self._results.get('branch_prediction', {}).get('cv_accuracy', 0),
                'chance': self._results.get('branch_prediction', {}).get('chance_level', 0.33),
            },
            'value_by_branch': self._results.get('value_by_branch', {}),
        }

        return viz_data
