"""
Counterfactual Memory Injection Experiment.

Tests whether internal state causally influences predictions and behavior
by injecting false history patterns into memory mechanisms.

For each agent type:
- persistent_lstm: Manipulate hidden state
- context_vector: Manipulate context embedding
- episodic_memory: Inject false episodes into buffer

PAIRED-specific:
- Bilateral injection: Inject antagonist h-state into protagonist
- Key test: Does injecting antagonist state improve protagonist performance?
  If yes → protagonist lacks information antagonist has
- Inject from high-regret to low-regret contexts
- Primary metric: Policy/value change (AGENT-CENTRIC)
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import jax
import jax.numpy as jnp
import chex

from .base import CheckpointExperiment
from .utils.history_injection import (
    create_failure_history,
    create_success_history,
    inject_hidden_state,
    inject_context_vector,
    inject_episodic_memory,
)


class InjectionType(Enum):
    """Types of history injection."""
    NONE = "none"
    SUCCESS_HISTORY = "success"
    FAILURE_HISTORY = "failure"
    RANDOM_HISTORY = "random"
    # PAIRED-specific injection types
    ANTAGONIST_STATE = "antagonist"  # Inject antagonist h-state into protagonist
    HIGH_REGRET_TO_LOW = "high_to_low"  # Inject state from high-regret context
    LOW_REGRET_TO_HIGH = "low_to_high"  # Inject state from low-regret context


@dataclass
class InjectionResult:
    """Results from a single injection experiment."""
    injection_type: InjectionType
    n_episodes: int
    solve_rate: float
    mean_return: float
    mean_value_estimate: float
    mean_policy_entropy: float
    action_distribution: np.ndarray
    # Prediction loss tracking (PRIMARY CAUSAL METRIC)
    mean_prediction_loss: float = 0.0
    prediction_losses: List[float] = field(default_factory=list)


class CounterfactualExperiment(CheckpointExperiment):
    """
    Test causal influence of memory content on behavior.

    Protocol:
    1. Inject false history (success/failure patterns)
    2. Run episodes and measure behavior change
    3. Compare to baseline (no injection)

    What this tests:
    - Whether memory content causally affects predictions
    - Whether behavior changes in expected direction
    - Whether agent uses memory for decision-making

    What this does NOT prove:
    - Agent "models the generator"
    - Agent has "Theory of Mind"
    - Agent "understands" curriculum
    """

    @property
    def name(self) -> str:
        return "counterfactual"

    def __init__(
        self,
        n_episodes_per_condition: int = 100,
        injection_strength: float = 1.0,
        n_injection_episodes: int = 10,
        **kwargs,
    ):
        """
        Initialize counterfactual experiment.

        Args:
            n_episodes_per_condition: Episodes per injection condition
            injection_strength: How strongly to inject false history
            n_injection_episodes: Number of fake episodes to simulate
        """
        super().__init__(**kwargs)
        self.n_episodes_per_condition = n_episodes_per_condition
        self.injection_strength = injection_strength
        self.n_injection_episodes = n_injection_episodes

        self._baseline_result: Optional[InjectionResult] = None
        self._injection_results: Dict[InjectionType, InjectionResult] = {}
        self._results: Dict[str, Any] = {}

    def collect_data(self, rng: chex.PRNGKey) -> Dict[str, InjectionResult]:
        """
        Collect data under each injection condition.
        """
        results = {}

        # Determine which injection types to use based on training method
        if self.has_regret:
            # PAIRED: Use bilateral injection types
            injection_types = [
                InjectionType.NONE,
                InjectionType.SUCCESS_HISTORY,
                InjectionType.FAILURE_HISTORY,
                InjectionType.ANTAGONIST_STATE,
                InjectionType.HIGH_REGRET_TO_LOW,
                InjectionType.LOW_REGRET_TO_HIGH,
            ]
        else:
            # Standard injection types
            injection_types = [
                InjectionType.NONE,
                InjectionType.SUCCESS_HISTORY,
                InjectionType.FAILURE_HISTORY,
                InjectionType.RANDOM_HISTORY,
            ]

        for injection_type in injection_types:
            rng, cond_rng = jax.random.split(rng)

            result = self._run_condition(cond_rng, injection_type)
            results[injection_type] = result

            if injection_type == InjectionType.NONE:
                self._baseline_result = result

        self._injection_results = results
        return results

    def _run_condition(
        self,
        rng: chex.PRNGKey,
        injection_type: InjectionType,
    ) -> InjectionResult:
        """Run episodes under a specific injection condition."""
        all_solved = []
        all_returns = []
        all_values = []
        all_entropies = []
        all_actions = []
        all_prediction_losses = []

        for ep_idx in range(self.n_episodes_per_condition):
            rng, ep_rng, level_rng, inject_rng, loss_rng = jax.random.split(rng, 5)

            # Generate test level
            level = self._generate_level(level_rng)

            # Create injected hidden state
            injected_hstate = self._create_injected_state(inject_rng, injection_type)

            # Run episode with injected state
            result = self._run_episode(ep_rng, level, injected_hstate)

            all_solved.append(result['solved'])
            all_returns.append(result['total_return'])
            all_values.append(result.get('mean_value', 0.0))
            all_entropies.append(result.get('mean_entropy', 0.0))
            all_actions.extend(result.get('actions', []))

            # Compute prediction loss for this episode (PRIMARY CAUSAL METRIC)
            from .utils.agent_aware_loss import compute_agent_prediction_loss
            pred_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state, level, loss_rng
            )
            all_prediction_losses.append(pred_loss)

        # Compute action distribution
        if all_actions:
            action_counts = np.bincount(all_actions, minlength=7)
            action_dist = action_counts / (len(all_actions) + 1e-6)
        else:
            action_dist = np.zeros(7)

        return InjectionResult(
            injection_type=injection_type,
            n_episodes=self.n_episodes_per_condition,
            solve_rate=float(np.mean(all_solved)),
            mean_return=float(np.mean(all_returns)),
            mean_value_estimate=float(np.mean(all_values)),
            mean_policy_entropy=float(np.mean(all_entropies)),
            action_distribution=action_dist,
            mean_prediction_loss=float(np.mean(all_prediction_losses)),
            prediction_losses=all_prediction_losses,
        )

    def _create_injected_state(
        self,
        rng: chex.PRNGKey,
        injection_type: InjectionType,
    ) -> Any:
        """Create hidden state with injected history."""
        # Initialize base hidden state
        base_hstate = self.agent.initialize_carry(rng, batch_dims=(1,))

        if injection_type == InjectionType.NONE:
            return base_hstate

        # Get hidden state components
        h_c, h_h = base_hstate

        if injection_type == InjectionType.SUCCESS_HISTORY:
            # Inject pattern associated with success
            pattern = create_success_history(self.n_injection_episodes)
            h_c_new, h_h_new = inject_hidden_state(
                h_c, h_h, pattern, self.injection_strength
            )

        elif injection_type == InjectionType.FAILURE_HISTORY:
            # Inject pattern associated with failure
            pattern = create_failure_history(self.n_injection_episodes)
            h_c_new, h_h_new = inject_hidden_state(
                h_c, h_h, pattern, self.injection_strength
            )

        elif injection_type == InjectionType.RANDOM_HISTORY:
            # Inject random pattern
            h_c_shape = h_c.shape
            h_h_shape = h_h.shape

            rng_c, rng_h = jax.random.split(rng)
            noise_c = jax.random.normal(rng_c, h_c_shape) * self.injection_strength * 0.5
            noise_h = jax.random.normal(rng_h, h_h_shape) * self.injection_strength * 0.5

            h_c_new = h_c + noise_c
            h_h_new = h_h + noise_h

        elif injection_type == InjectionType.ANTAGONIST_STATE:
            # PAIRED bilateral: inject antagonist h-state into protagonist
            h_c_new, h_h_new = self._get_antagonist_state(rng)

        elif injection_type == InjectionType.HIGH_REGRET_TO_LOW:
            # PAIRED: inject state from high-regret context into low-regret evaluation
            h_c_new, h_h_new = self._get_regret_conditioned_state(rng, source_regret='high')

        elif injection_type == InjectionType.LOW_REGRET_TO_HIGH:
            # PAIRED: inject state from low-regret context into high-regret evaluation
            h_c_new, h_h_new = self._get_regret_conditioned_state(rng, source_regret='low')

        else:
            return base_hstate

        return (h_c_new, h_h_new)

    def _get_antagonist_state(self, rng: chex.PRNGKey) -> Tuple[Any, Any]:
        """
        Get antagonist hidden state for bilateral injection (PAIRED).

        Key test: Does injecting antagonist state improve protagonist performance?
        If yes → protagonist lacks information antagonist has.
        """
        ant_train_state = getattr(self.train_state, 'ant_train_state', None)

        if ant_train_state is None:
            # No antagonist available - return random initialization
            base_hstate = self.agent.initialize_carry(rng, batch_dims=(1,))
            return base_hstate

        # Run antagonist on a sample level to get its hidden state
        rng, level_rng, rollout_rng = jax.random.split(rng, 3)
        level = self._generate_level(level_rng)

        # Initialize antagonist hidden state
        ant_hstate = self.agent.initialize_carry(rng, batch_dims=(1,))

        # Run a few steps to populate hidden state
        for step in range(10):
            rng, step_rng = jax.random.split(rng)
            obs = self._create_observation(level, step)

            obs_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs)
            done_batch = jnp.zeros((1, 1), dtype=bool)

            ant_hstate, pi, value = ant_train_state.apply_fn(
                ant_train_state.params, (obs_batch, done_batch), ant_hstate
            )

        return ant_hstate

    def _get_regret_conditioned_state(
        self,
        rng: chex.PRNGKey,
        source_regret: str,  # 'high' or 'low'
    ) -> Tuple[Any, Any]:
        """
        Get hidden state from a regret-conditioned context (PAIRED).

        Used to test whether states from high-regret contexts transfer
        differently than states from low-regret contexts.
        """
        rng, level_rng, rollout_rng = jax.random.split(rng, 3)

        # Generate level with appropriate difficulty
        if source_regret == 'high':
            # High regret: hard level (high wall density)
            level = self._generate_level(level_rng)
            level['wall_density'] = 0.35  # Higher difficulty
        else:
            # Low regret: easy level (low wall density)
            level = self._generate_level(level_rng)
            level['wall_density'] = 0.1  # Lower difficulty

        # Run protagonist on this level to get conditioned hidden state
        hstate = self.agent.initialize_carry(rng, batch_dims=(1,))

        for step in range(20):  # More steps for better context
            rng, step_rng = jax.random.split(rng)
            obs = self._create_observation(level, step)

            obs_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs)
            done_batch = jnp.zeros((1, 1), dtype=bool)

            hstate, pi, value = self.train_state.apply_fn(
                self.train_state.params, (obs_batch, done_batch), hstate
            )

        return hstate

    def _generate_level(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """Generate a test level."""
        height, width = 13, 13

        wall_prob = 0.15 + float(jax.random.uniform(rng)) * 0.1
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

    def _run_episode(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
        initial_hstate: Any,
        max_steps: int = 256,
    ) -> Dict[str, Any]:
        """Run a single episode with given initial state."""
        hstate = initial_hstate

        total_return = 0.0
        solved = False
        values = []
        entropies = []
        actions = []

        for step in range(max_steps):
            rng, step_rng = jax.random.split(rng)

            obs = self._create_observation(level, step)
            new_hstate, pi, value = self._forward_step(obs, hstate)

            values.append(float(value[0, 0]))
            entropies.append(float(pi.entropy()[0, 0]))

            action = pi.sample(seed=step_rng)
            actions.append(int(action[0, 0]))

            hstate = new_hstate

            # Simulate step
            reward = 0.0
            done = step >= max_steps - 1

            if step > 10:
                solve_prob = 0.3 * (1 - level['wall_density'])
                if float(jax.random.uniform(step_rng)) < solve_prob / max_steps:
                    solved = True
                    reward = 1.0
                    done = True

            total_return += reward

            if done:
                break

        return {
            'total_return': total_return,
            'solved': solved,
            'n_steps': step + 1,
            'mean_value': float(np.mean(values)),
            'mean_entropy': float(np.mean(entropies)),
            'actions': actions,
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
        Analyze counterfactual injection effects.

        Measures:
        - Behavioral shift (performance, value, entropy)
        - Action distribution changes
        - Whether effects are in expected direction
        """
        if self._baseline_result is None:
            raise ValueError("Must call collect_data before analyze")

        results = {}

        # 1. Compare injection conditions to baseline
        results['condition_comparison'] = self._compare_conditions()

        # 2. Behavioral shift analysis
        results['behavioral_shifts'] = self._analyze_behavioral_shifts()

        # 3. Action distribution analysis
        results['action_distribution'] = self._analyze_action_distributions()

        # 4. Effect size analysis
        results['effect_sizes'] = self._compute_effect_sizes()

        # 5. Interpretation
        results['interpretation'] = self._interpret_results()

        # 6. PAIRED-specific: bilateral injection analysis
        if self.has_regret:
            results['bilateral_injection'] = self._analyze_bilateral_injection()
            results['regret_transfer'] = self._analyze_regret_transfer()

        # 7. Caveats
        results['caveats'] = self._get_caveats()

        self._results = results
        return results

    def _analyze_bilateral_injection(self) -> Dict[str, Any]:
        """
        Analyze bilateral injection: antagonist state into protagonist (PAIRED).

        Key question: Does injecting antagonist state improve protagonist performance?
        If yes → protagonist lacks information antagonist has.
        """
        baseline = self._baseline_result
        ant_result = self._injection_results.get(InjectionType.ANTAGONIST_STATE)

        if ant_result is None:
            return {'error': 'No antagonist injection data'}

        results = {}

        # Compare protagonist performance with antagonist state
        solve_rate_delta = ant_result.solve_rate - baseline.solve_rate
        return_delta = ant_result.mean_return - baseline.mean_return
        value_delta = ant_result.mean_value_estimate - baseline.mean_value_estimate

        results['protagonist_with_antagonist_state'] = {
            'solve_rate_delta': float(solve_rate_delta),
            'return_delta': float(return_delta),
            'value_delta': float(value_delta),
            'antagonist_state_helps': solve_rate_delta > 0.05 or return_delta > 0.1,
        }

        # Key finding: Does protagonist improve with antagonist state?
        results['protagonist_lacks_antagonist_info'] = (
            solve_rate_delta > 0.05 or return_delta > 0.1
        )

        # Policy change analysis (AGENT-CENTRIC)
        baseline_dist = baseline.action_distribution
        ant_dist = ant_result.action_distribution

        # KL divergence
        eps = 1e-10
        kl_div = float(np.sum(
            ant_dist * np.log((ant_dist + eps) / (baseline_dist + eps))
        ))

        results['policy_change'] = {
            'kl_divergence': kl_div,
            'significant_change': kl_div > 0.1,
            'interpretation': (
                "Large KL divergence suggests antagonist state carries "
                "different policy-relevant information than protagonist state."
            ),
        }

        # Entropy change (uncertainty)
        entropy_delta = ant_result.mean_policy_entropy - baseline.mean_policy_entropy
        results['uncertainty_change'] = {
            'entropy_delta': float(entropy_delta),
            'antagonist_state_increases_confidence': entropy_delta < 0,
        }

        return results

    def _analyze_regret_transfer(self) -> Dict[str, Any]:
        """
        Analyze regret-conditioned state transfer (PAIRED).

        Tests whether states from high-regret contexts transfer
        differently than states from low-regret contexts.
        """
        baseline = self._baseline_result
        high_to_low = self._injection_results.get(InjectionType.HIGH_REGRET_TO_LOW)
        low_to_high = self._injection_results.get(InjectionType.LOW_REGRET_TO_HIGH)

        results = {}

        if high_to_low:
            results['high_regret_state_transfer'] = {
                'solve_rate_delta': float(high_to_low.solve_rate - baseline.solve_rate),
                'value_delta': float(high_to_low.mean_value_estimate - baseline.mean_value_estimate),
                'entropy_delta': float(high_to_low.mean_policy_entropy - baseline.mean_policy_entropy),
            }

        if low_to_high:
            results['low_regret_state_transfer'] = {
                'solve_rate_delta': float(low_to_high.solve_rate - baseline.solve_rate),
                'value_delta': float(low_to_high.mean_value_estimate - baseline.mean_value_estimate),
                'entropy_delta': float(low_to_high.mean_policy_entropy - baseline.mean_policy_entropy),
            }

        # Compare transfer effects
        if high_to_low and low_to_high:
            results['transfer_asymmetry'] = {
                'high_to_low_solve_delta': float(high_to_low.solve_rate - baseline.solve_rate),
                'low_to_high_solve_delta': float(low_to_high.solve_rate - baseline.solve_rate),
                'asymmetric_transfer': abs(
                    (high_to_low.solve_rate - baseline.solve_rate) -
                    (low_to_high.solve_rate - baseline.solve_rate)
                ) > 0.1,
                'interpretation': (
                    "Asymmetric transfer suggests regret-conditioned states "
                    "carry context-specific information that doesn't transfer "
                    "equally across difficulty levels."
                ),
            }

        return results

    def _compare_conditions(self) -> Dict[str, Any]:
        """Compare each injection condition to baseline."""
        baseline = self._baseline_result
        comparisons = {}

        for injection_type, result in self._injection_results.items():
            if injection_type == InjectionType.NONE:
                continue

            comparisons[injection_type.value] = {
                'solve_rate_delta': float(result.solve_rate - baseline.solve_rate),
                'return_delta': float(result.mean_return - baseline.mean_return),
                'value_delta': float(result.mean_value_estimate - baseline.mean_value_estimate),
                'entropy_delta': float(result.mean_policy_entropy - baseline.mean_policy_entropy),
                # Prediction loss comparison (PRIMARY CAUSAL METRIC)
                'prediction_loss_delta': float(result.mean_prediction_loss - baseline.mean_prediction_loss),
                'prediction_loss': float(result.mean_prediction_loss),
            }

        comparisons['baseline'] = {
            'solve_rate': baseline.solve_rate,
            'mean_return': baseline.mean_return,
            'mean_value_estimate': baseline.mean_value_estimate,
            'mean_policy_entropy': baseline.mean_policy_entropy,
            'mean_prediction_loss': baseline.mean_prediction_loss,
        }

        return comparisons

    def _analyze_behavioral_shifts(self) -> Dict[str, Any]:
        """Analyze behavioral shifts from injections."""
        success_result = self._injection_results.get(InjectionType.SUCCESS_HISTORY)
        failure_result = self._injection_results.get(InjectionType.FAILURE_HISTORY)
        baseline = self._baseline_result

        shifts = {}

        if success_result and failure_result:
            # Compute difference between success and failure injection
            shifts['success_vs_failure'] = {
                'solve_rate_diff': float(success_result.solve_rate - failure_result.solve_rate),
                'return_diff': float(success_result.mean_return - failure_result.mean_return),
                'value_diff': float(success_result.mean_value_estimate - failure_result.mean_value_estimate),
                'expected_direction': success_result.mean_value_estimate > failure_result.mean_value_estimate,
            }

        # Does injection cause any effect?
        if success_result:
            shifts['success_injection_effect'] = {
                'has_effect': abs(success_result.solve_rate - baseline.solve_rate) > 0.05,
                'magnitude': float(abs(success_result.solve_rate - baseline.solve_rate)),
            }

        if failure_result:
            shifts['failure_injection_effect'] = {
                'has_effect': abs(failure_result.solve_rate - baseline.solve_rate) > 0.05,
                'magnitude': float(abs(failure_result.solve_rate - baseline.solve_rate)),
            }

        return shifts

    def _analyze_action_distributions(self) -> Dict[str, Any]:
        """Analyze how action distributions change with injection."""
        baseline_dist = self._baseline_result.action_distribution

        distributions = {}
        for injection_type, result in self._injection_results.items():
            if injection_type == InjectionType.NONE:
                continue

            # KL divergence from baseline
            eps = 1e-10
            kl_div = float(np.sum(
                result.action_distribution * np.log(
                    (result.action_distribution + eps) / (baseline_dist + eps)
                )
            ))

            # L1 distance
            l1_dist = float(np.sum(np.abs(result.action_distribution - baseline_dist)))

            distributions[injection_type.value] = {
                'action_distribution': result.action_distribution.tolist(),
                'kl_from_baseline': kl_div,
                'l1_from_baseline': l1_dist,
            }

        distributions['baseline'] = baseline_dist.tolist()

        return distributions

    def _compute_effect_sizes(self) -> Dict[str, Any]:
        """Compute effect sizes for injections."""
        baseline = self._baseline_result
        effect_sizes = {}

        for injection_type, result in self._injection_results.items():
            if injection_type == InjectionType.NONE:
                continue

            # Cohen's d for solve rate (treating as continuous)
            solve_diff = result.solve_rate - baseline.solve_rate
            # Approximate pooled std (binary outcome)
            pooled_std = np.sqrt(
                (baseline.solve_rate * (1 - baseline.solve_rate) +
                 result.solve_rate * (1 - result.solve_rate)) / 2
            ) + 1e-6

            cohens_d = solve_diff / pooled_std

            effect_sizes[injection_type.value] = {
                'cohens_d_solve_rate': float(cohens_d),
                'raw_difference': float(solve_diff),
                'effect_magnitude': 'large' if abs(cohens_d) > 0.8 else ('medium' if abs(cohens_d) > 0.5 else 'small'),
            }

        return effect_sizes

    def _interpret_results(self) -> Dict[str, Any]:
        """Interpret the counterfactual results."""
        success_result = self._injection_results.get(InjectionType.SUCCESS_HISTORY)
        failure_result = self._injection_results.get(InjectionType.FAILURE_HISTORY)
        baseline = self._baseline_result

        interpretation = {
            'memory_influences_behavior': False,
            'memory_influences_prediction': False,
            'direction_is_expected': False,
            'evidence_strength': 'none',
        }

        if success_result and failure_result:
            # Check behavioral effects
            any_behavioral_effect = (
                abs(success_result.solve_rate - baseline.solve_rate) > 0.05 or
                abs(failure_result.solve_rate - baseline.solve_rate) > 0.05
            )

            # Check prediction loss effects (PRIMARY CAUSAL METRIC)
            any_prediction_effect = (
                abs(success_result.mean_prediction_loss - baseline.mean_prediction_loss) > 0.1 or
                abs(failure_result.mean_prediction_loss - baseline.mean_prediction_loss) > 0.1
            )

            # Expected: failure injection should increase prediction loss
            prediction_direction_correct = (
                failure_result.mean_prediction_loss > baseline.mean_prediction_loss
            )

            # Check if success injection leads to better outcomes than failure
            success_better = success_result.mean_value_estimate > failure_result.mean_value_estimate

            interpretation['memory_influences_behavior'] = any_behavioral_effect
            interpretation['memory_influences_prediction'] = any_prediction_effect
            interpretation['direction_is_expected'] = prediction_direction_correct
            interpretation['evidence_strength'] = (
                'strong' if any_prediction_effect and prediction_direction_correct else
                ('weak' if any_behavioral_effect else 'none')
            )

            # Add prediction loss specific interpretation
            interpretation['prediction_loss_analysis'] = {
                'baseline_loss': baseline.mean_prediction_loss,
                'success_loss': success_result.mean_prediction_loss,
                'failure_loss': failure_result.mean_prediction_loss,
                'failure_increases_loss': failure_result.mean_prediction_loss > baseline.mean_prediction_loss,
                'success_decreases_loss': success_result.mean_prediction_loss < baseline.mean_prediction_loss,
            }

        return interpretation

    def _get_caveats(self) -> List[str]:
        """Return caveats for interpretation."""
        return [
            "Response indicates learned associations, not mental modeling",
            "Does NOT prove agent 'models the generator'",
            "Does NOT prove agent has 'Theory of Mind'",
            "Effects may be due to simple state-action correlations",
            "Injection may disrupt normal processing rather than convey information",
        ]

    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data."""
        if not self._results:
            raise ValueError("Must call analyze before visualize")

        viz_data = {
            'condition_comparison': self._results.get('condition_comparison', {}),
            'behavioral_shifts': self._results.get('behavioral_shifts', {}),
            'action_distributions': {},
        }

        # Action distribution bar chart
        if 'action_distribution' in self._results:
            ad = self._results['action_distribution']
            viz_data['action_distributions'] = {
                'baseline': ad.get('baseline', []),
                'success': ad.get('success', {}).get('action_distribution', []),
                'failure': ad.get('failure', {}).get('action_distribution', []),
            }

        return viz_data
