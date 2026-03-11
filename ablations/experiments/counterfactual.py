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
from .utils.batched_rollout import batched_rollout


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
        import time, logging
        from tqdm import tqdm
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

        results = {}
        n = self.n_episodes_per_condition
        max_steps = 256

        # Determine injection types based on training method
        if self.has_regret:
            injection_types = [
                InjectionType.NONE,
                InjectionType.SUCCESS_HISTORY,
                InjectionType.FAILURE_HISTORY,
                InjectionType.ANTAGONIST_STATE,
                InjectionType.HIGH_REGRET_TO_LOW,
                InjectionType.LOW_REGRET_TO_HIGH,
            ]
        else:
            injection_types = [
                InjectionType.NONE,
                InjectionType.SUCCESS_HISTORY,
                InjectionType.FAILURE_HISTORY,
                InjectionType.RANDOM_HISTORY,
            ]

        for injection_type in tqdm(injection_types, desc="Injection conditions"):
            rng, cond_rng = jax.random.split(rng)
            cond_key = injection_type.value
            _log(f"{cond_key}/start", msg=f"Running condition: {cond_key}")

            # --- Generate all levels for this condition ---
            _log(f"{cond_key}/generate_levels", msg="Generating levels...")
            t0 = time.time()
            rng_levels, rng_inject, rng_rollout = jax.random.split(cond_rng, 3)
            level_rngs = jax.random.split(rng_levels, n)
            levels = jax.vmap(self.agent.sample_random_level)(level_rngs)
            jax.block_until_ready(levels)
            _log(f"{cond_key}/generate_levels", time.time() - t0)

            # --- Create injected hidden states for batch ---
            _log(f"{cond_key}/inject_hstate", msg="Creating injected hidden states...")
            t0 = time.time()
            init_hstate = self._create_batched_injected_state(rng_inject, injection_type, n)
            _log(f"{cond_key}/inject_hstate", time.time() - t0)

            # --- Batched rollout ---
            _log(f"{cond_key}/rollout", msg="Running batched rollout...")
            t0 = time.time()
            result = batched_rollout(
                rng_rollout, levels, max_steps,
                self.train_state.apply_fn, self.train_state.params,
                self.agent.env, self.agent.env_params,
                init_hstate,
                collect_values=True,
                collect_entropies=True,
                collect_actions=True,
            )
            jax.block_until_ready(result.episode_returns)
            _log(f"{cond_key}/rollout", time.time() - t0)

            # --- Compute prediction losses (CPU loop) ---
            _log(f"{cond_key}/pred_losses", msg="Computing prediction losses...")
            t0 = time.time()
            from .utils.agent_aware_loss import compute_agent_prediction_loss
            all_prediction_losses = []
            for i in tqdm(range(n), desc=f"Pred losses ({cond_key})", leave=False):
                rng, loss_rng = jax.random.split(rng)
                pred_loss, _ = compute_agent_prediction_loss(
                    self.agent, self.train_state,
                    jax.tree_util.tree_map(lambda x: x[i], levels),
                    loss_rng
                )
                all_prediction_losses.append(pred_loss)
            _log(f"{cond_key}/pred_losses", time.time() - t0)

            # --- Assemble action distribution ---
            all_actions = []
            for i in range(n):
                ep_len = int(result.episode_lengths[i])
                all_actions.extend(np.array(result.actions[i, :ep_len]).tolist())
            if all_actions:
                action_counts = np.bincount(all_actions, minlength=7)
                action_dist = action_counts / (len(all_actions) + 1e-6)
            else:
                action_dist = np.zeros(7)

            # --- Compute mean values/entropies per episode, then average ---
            mean_values = []
            mean_entropies = []
            for i in range(n):
                ep_len = int(result.episode_lengths[i])
                if ep_len > 0:
                    vals = np.array(result.values[i, :ep_len])
                    ents = np.array(result.entropies[i, :ep_len])
                    mean_values.append(float(np.nanmean(vals)))
                    mean_entropies.append(float(np.nanmean(ents)))
                else:
                    mean_values.append(0.0)
                    mean_entropies.append(0.0)

            injection_result = InjectionResult(
                injection_type=injection_type,
                n_episodes=n,
                solve_rate=float(np.mean(result.episode_solved)),
                mean_return=float(np.mean(result.episode_returns)),
                mean_value_estimate=float(np.mean(mean_values)),
                mean_policy_entropy=float(np.mean(mean_entropies)),
                action_distribution=action_dist,
                mean_prediction_loss=float(np.mean(all_prediction_losses)),
                prediction_losses=all_prediction_losses,
            )
            results[injection_type] = injection_result

            if injection_type == InjectionType.NONE:
                self._baseline_result = injection_result

            _log(f"{cond_key}/done", msg=f"Condition complete: solve_rate={injection_result.solve_rate:.3f}")

        self._injection_results = results
        return results

    def _create_batched_injected_state(
        self,
        rng: chex.PRNGKey,
        injection_type: InjectionType,
        n: int,
    ) -> Any:
        """Create batched hidden states with injected history for n environments."""
        # Base hidden state for full batch
        base_hstate = self.agent.initialize_hidden_state(n)

        if injection_type == InjectionType.NONE:
            return base_hstate

        if injection_type == InjectionType.SUCCESS_HISTORY:
            pattern = create_success_history(self.n_injection_episodes)
            return inject_hidden_state(
                base_hstate, pattern, self.injection_strength
            )

        elif injection_type == InjectionType.FAILURE_HISTORY:
            pattern = create_failure_history(self.n_injection_episodes)
            return inject_hidden_state(
                base_hstate, pattern, self.injection_strength
            )

        elif injection_type == InjectionType.RANDOM_HISTORY:
            return inject_hidden_state(
                base_hstate, "random", self.injection_strength
            )

        elif injection_type == InjectionType.ANTAGONIST_STATE:
            # Run antagonist on n levels for 10 steps to populate hidden state
            return self._get_batched_antagonist_state(rng, n)

        elif injection_type == InjectionType.HIGH_REGRET_TO_LOW:
            return self._get_batched_regret_conditioned_state(rng, n, source_regret='high')

        elif injection_type == InjectionType.LOW_REGRET_TO_HIGH:
            return self._get_batched_regret_conditioned_state(rng, n, source_regret='low')

        else:
            return base_hstate

    def _get_batched_antagonist_state(
        self,
        rng: chex.PRNGKey,
        n: int,
    ) -> Any:
        """
        Get antagonist hidden states for bilateral injection (PAIRED).
        Runs antagonist for 10 steps on n levels to populate hidden states.
        """
        ant_train_state = getattr(self.train_state, 'ant_train_state', None)
        if ant_train_state is None:
            return self.agent.initialize_hidden_state(n)

        rng, rng_levels, rng_rollout = jax.random.split(rng, 3)
        level_rngs = jax.random.split(rng_levels, n)
        conditioning_levels = jax.vmap(self.agent.sample_random_level)(level_rngs)

        # Short rollout (10 steps) to populate antagonist hidden state
        result = batched_rollout(
            rng_rollout, conditioning_levels, 10,
            ant_train_state.apply_fn,
            ant_train_state.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n),
            return_final_hstate=True,
        )
        return result.final_hstate

    def _get_batched_regret_conditioned_state(
        self,
        rng: chex.PRNGKey,
        n: int,
        source_regret: str,
    ) -> Any:
        """
        Get hidden states from regret-conditioned context (PAIRED).
        Runs protagonist for 20 steps on n conditioning levels.
        """
        rng, rng_levels, rng_rollout = jax.random.split(rng, 3)
        level_rngs = jax.random.split(rng_levels, n)
        conditioning_levels = jax.vmap(self.agent.sample_random_level)(level_rngs)

        # Short rollout (20 steps) to build conditioned hidden state
        result = batched_rollout(
            rng_rollout, conditioning_levels, 20,
            self.train_state.apply_fn,
            self.train_state.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n),
            return_final_hstate=True,
        )
        return result.final_hstate

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
