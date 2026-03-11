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
        Collect policy and value outputs across episodes (GPU-batched).
        """
        import time
        import logging
        from tqdm import tqdm
        from .utils.batched_rollout import batched_rollout

        logger = logging.getLogger(__name__)
        timings = {}

        try:
            import wandb
            _wandb_active = wandb.run is not None
        except ImportError:
            _wandb_active = False

        def _log(phase, elapsed=None, msg=None):
            if elapsed is not None:
                timings[phase] = elapsed
                logger.info(f"[{self.name}] {phase}: {elapsed:.2f}s")
            if msg:
                logger.info(f"[{self.name}] {msg}")
            if _wandb_active:
                log_dict = {}
                if elapsed is not None:
                    log_dict[f"{self.name}/timing/{phase}"] = elapsed
                if msg:
                    log_dict[f"{self.name}/status"] = msg
                if log_dict:
                    wandb.log(log_dict)

        self._data = OutputData()
        n_episodes = self.n_episodes
        max_steps = self.n_steps_per_episode

        # --- 1. Generate all levels in batch ---
        _log("generate_levels", msg="Generating levels...")
        t0 = time.time()
        rng, rng_levels = jax.random.split(rng)
        level_rngs = jax.random.split(rng_levels, n_episodes)
        levels = jax.vmap(self.agent.sample_random_level)(level_rngs)
        jax.block_until_ready(levels)
        _log("generate_levels", time.time() - t0, "Level generation complete")

        # --- 2. CPU-side level properties ---
        t0 = time.time()
        wall_maps = np.array(levels.wall_map)
        goal_positions = np.array(levels.goal_pos)
        agent_positions = np.array(levels.agent_pos)
        wall_density = wall_maps.mean(axis=(1, 2))
        branches = np.arange(n_episodes) % 3
        _log("cpu_level_properties", time.time() - t0)

        # --- 3. Batched protagonist rollout with per-step values, entropies, and logits ---
        _log("protagonist_rollout", msg="Running batched protagonist rollout...")
        t0 = time.time()
        rng, rng_pro = jax.random.split(rng)
        pro_result = batched_rollout(
            rng_pro, levels, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n_episodes),
            collect_values=True, collect_entropies=True,
            collect_logits=True, collect_actions=True,
        )
        _log("protagonist_rollout", time.time() - t0, "Protagonist rollout complete")

        # --- 4. Compute prediction losses ---
        _log("prediction_losses", msg="Computing prediction losses...")
        t0 = time.time()
        from .utils.agent_aware_loss import compute_agent_prediction_loss
        prediction_losses = []
        for i in tqdm(range(n_episodes), desc="Prediction losses", leave=False):
            rng, loss_rng = jax.random.split(rng)
            level_i = jax.tree_util.tree_map(lambda x: x[i], levels)
            pred_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state, level_i, loss_rng
            )
            prediction_losses.append(pred_loss)
        _log("prediction_losses", time.time() - t0, "Prediction losses complete")

        # --- 5. Assemble per-step data ---
        for i in range(n_episodes):
            ep_len = int(pro_result.episode_lengths[i])
            n_steps = min(ep_len, max_steps)

            ep_values = pro_result.values[i, :n_steps].tolist()
            ep_entropies = pro_result.entropies[i, :n_steps].tolist()

            self._data.value_estimates.extend(ep_values)
            self._data.policy_entropies.extend(ep_entropies)
            for t in range(n_steps):
                self._data.policy_logits.append(pro_result.logits[i, t])
            self._data.branch_types.extend([int(branches[i])] * n_steps)
            self._data.wall_densities.extend([float(wall_density[i])] * n_steps)
            self._data.episode_outcomes.extend([int(pro_result.episode_solved[i])] * n_steps)
            self._data.prediction_losses.extend([prediction_losses[i]] * n_steps)

        # --- 6. PAIRED bilateral ---
        if self.has_regret:
            ant_train_state = getattr(self.train_state, 'ant_train_state', None)
            if ant_train_state is not None:
                _log("antagonist_rollout", msg="Running batched antagonist rollout...")
                t0 = time.time()
                rng, rng_ant = jax.random.split(rng)
                ant_result = batched_rollout(
                    rng_ant, levels, max_steps,
                    ant_train_state.apply_fn, ant_train_state.params,
                    self.agent.env, self.agent.env_params,
                    self.agent.initialize_hidden_state(n_episodes),
                    collect_values=True, collect_entropies=True,
                    collect_logits=True, collect_actions=True,
                )
                _log("antagonist_rollout", time.time() - t0, "Antagonist rollout complete")

                for i in range(n_episodes):
                    ep_len = min(int(pro_result.episode_lengths[i]), max_steps)
                    ant_ep_len = min(int(ant_result.episode_lengths[i]), max_steps)
                    n_steps = min(ep_len, ant_ep_len)

                    regret = max(0, float(ant_result.episode_returns[i] - pro_result.episode_returns[i]))
                    goal_dist = abs(float(goal_positions[i][0] - agent_positions[i][0])) + \
                                abs(float(goal_positions[i][1] - agent_positions[i][1]))
                    adv_difficulty = float(wall_density[i]) * 0.5 + (goal_dist / 26) * 0.5

                    self._data.regrets.extend([regret] * ep_len)
                    self._data.adversary_difficulty.extend([adv_difficulty] * ep_len)
                    self._data.antagonist_entropies.extend(
                        ant_result.entropies[i, :ep_len].tolist()
                    )
                    self._data.antagonist_values.extend(
                        ant_result.values[i, :ep_len].tolist()
                    )
            else:
                for i in range(n_episodes):
                    ep_len = min(int(pro_result.episode_lengths[i]), max_steps)
                    self._data.regrets.extend([0.0] * ep_len)
                    self._data.adversary_difficulty.extend([0.0] * ep_len)
                    self._data.antagonist_entropies.extend([0.0] * ep_len)
                    self._data.antagonist_values.extend([0.0] * ep_len)

        if _wandb_active:
            wandb.log({
                f"{self.name}/mean_return": float(pro_result.episode_returns.mean()),
                f"{self.name}/solve_rate": float(pro_result.episode_solved.mean()),
            })

        total_time = sum(timings.values())
        _log("total", total_time, f"TOTAL collect_data: {total_time:.2f}s | breakdown: {timings}")

        return self._data

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
        model = LogisticRegression(max_iter=1000)
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
        policy_model = LogisticRegression(max_iter=1000)
        policy_model.fit(policy_logits, branches)
        policy_acc = policy_model.score(policy_logits, branches)

        # Value-only prediction
        value_model = LogisticRegression(max_iter=1000)
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
