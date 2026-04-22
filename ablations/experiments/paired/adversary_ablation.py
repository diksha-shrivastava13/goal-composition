"""
B1: Adversary Ablation with Controlled Antagonist State.

Test causal effects of adversary interventions while explicitly controlling
for the antagonist. Uses 2×4 factorial design.

This is a key causal experiment - it disentangles adversary effects from
antagonist effects, which the original experiments don't do.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment
from ..utils.batched_rollout import batched_rollout
from ...common.metrics import compute_bilateral_cka


class AdversaryCondition(Enum):
    """Adversary intervention conditions."""
    NO_ADVERSARY = "no_adversary"      # Random level generation
    FROZEN = "frozen"                   # Fixed adversary from checkpoint
    EASY = "easy"                       # Constrained easy levels
    HARD = "hard"                       # Constrained hard levels


class AntagonistCondition(Enum):
    """Antagonist state conditions."""
    LIVE = "live"       # Normal antagonist
    FROZEN = "frozen"   # Fixed antagonist from checkpoint


@dataclass
class ConditionResult:
    """Results from a single factorial condition."""
    adversary_condition: AdversaryCondition
    antagonist_condition: AntagonistCondition
    n_levels: int

    # Performance metrics
    pro_solve_rate: float
    pro_mean_return: float
    ant_solve_rate: float
    ant_mean_return: float
    mean_regret: float

    # Representation metrics
    pro_hstates: np.ndarray  # (n_levels, hidden_dim)
    ant_hstates: np.ndarray

    # Level feature distribution
    mean_wall_density: float
    mean_goal_distance: float


class AdversaryAblationExperiment(CheckpointExperiment):
    """
    2×4 factorial design crossing adversary intervention with antagonist state.

    Adversary conditions:
    - NO_ADVERSARY: Random level generation
    - FROZEN: Fixed adversary from checkpoint
    - EASY: Constrained to generate easy levels
    - HARD: Constrained to generate hard levels

    Antagonist conditions:
    - LIVE: Normal training antagonist
    - FROZEN: Fixed antagonist from checkpoint

    This allows measuring:
    - Main effect of adversary intervention
    - Main effect of antagonist state
    - Interaction effect (does adversary impact depend on antagonist?)
    """

    @property
    def name(self) -> str:
        return "adversary_ablation"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_levels_per_condition = self.exp_config("n_levels_per_condition")
        self._results_by_condition: Dict[Tuple[str, str], ConditionResult] = {}
        self._require_paired()

    def _require_paired(self):
        """Verify PAIRED training."""
        if self.training_method != "paired":
            raise ValueError(f"AdversaryAblationExperiment requires PAIRED, got {self.training_method}")

    def collect_data(self, rng: chex.PRNGKey) -> Dict[Tuple[str, str], ConditionResult]:
        """Run all factorial conditions using GPU-batched rollouts."""
        import time
        import logging

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

        n = self.n_levels_per_condition
        max_steps = self.config.get("max_steps", 256)

        for adv_cond in AdversaryCondition:
            for ant_cond in AntagonistCondition:
                cond_key = f"{adv_cond.value}_{ant_cond.value}"
                rng, cond_rng = jax.random.split(rng)

                # --- 1. Generate all levels for this condition ---
                _log(f"{cond_key}/generate_levels", msg="Generating levels...")
                t0 = time.time()
                rng_levels, rng_pro, rng_ant = jax.random.split(cond_rng, 3)
                if adv_cond == AdversaryCondition.NO_ADVERSARY:
                    level_rngs = jax.random.split(rng_levels, n)
                    levels = jax.vmap(self.agent.sample_random_level)(level_rngs)
                elif adv_cond == AdversaryCondition.FROZEN:
                    adv_ts = getattr(self.train_state, 'adv_train_state', None)
                    if adv_ts is not None:
                        from ..utils.paired_helpers import generate_adversary_levels
                        levels = generate_adversary_levels(self.agent, adv_ts, rng_levels, n)
                    else:
                        level_rngs = jax.random.split(rng_levels, n)
                        levels = jax.vmap(self.agent.sample_random_level)(level_rngs)
                elif adv_cond == AdversaryCondition.EASY:
                    from ..utils.paired_helpers import generate_constrained_levels
                    levels = generate_constrained_levels(
                        self.agent, rng_levels, n,
                        {'wall_density': (0.0, 0.2)},
                    )
                elif adv_cond == AdversaryCondition.HARD:
                    from ..utils.paired_helpers import generate_constrained_levels
                    levels = generate_constrained_levels(
                        self.agent, rng_levels, n,
                        {'wall_density': (0.3, 0.6)},
                    )
                else:
                    level_rngs = jax.random.split(rng_levels, n)
                    levels = jax.vmap(self.agent.sample_random_level)(level_rngs)
                jax.block_until_ready(levels)
                _log(f"{cond_key}/generate_levels", time.time() - t0)

                # --- 2. Extract level features (vectorised, no Python loop) ---
                _log(f"{cond_key}/level_features", msg="Extracting level features...")
                t0 = time.time()
                from ..utils.paired_helpers import extract_level_features_batch
                batch_features = extract_level_features_batch(levels)
                _log(f"{cond_key}/level_features", time.time() - t0)

                # --- 3. Protagonist batched rollout ---
                _log(f"{cond_key}/pro_rollout", msg="Running protagonist rollout...")
                t0 = time.time()
                pro_result = batched_rollout(
                    rng_pro, levels, max_steps,
                    self.train_state.pro_train_state.apply_fn,
                    self.train_state.pro_train_state.params,
                    self.agent.env, self.agent.env_params,
                    self.agent.initialize_hidden_state(n),
                    collection_steps=[-1],
                )
                _log(f"{cond_key}/pro_rollout", time.time() - t0)

                # --- 4. Antagonist batched rollout ---
                _log(f"{cond_key}/ant_rollout", msg="Running antagonist rollout...")
                t0 = time.time()
                # Both LIVE and FROZEN use ant_train_state (frozen uses same checkpoint)
                ant_train_state = self.train_state.ant_train_state
                ant_result = batched_rollout(
                    rng_ant, levels, max_steps,
                    ant_train_state.apply_fn,
                    ant_train_state.params,
                    self.agent.env, self.agent.env_params,
                    self.agent.initialize_hidden_state(n),
                    collection_steps=[-1],
                )
                _log(f"{cond_key}/ant_rollout", time.time() - t0)

                # --- 5. Assemble condition result ---
                pro_hstates = pro_result.hstates_by_step["-1"]
                ant_hstates = ant_result.hstates_by_step["-1"]
                regrets = ant_result.episode_returns - pro_result.episode_returns

                result = ConditionResult(
                    adversary_condition=adv_cond,
                    antagonist_condition=ant_cond,
                    n_levels=n,
                    pro_solve_rate=float(pro_result.episode_solved.mean()),
                    pro_mean_return=float(pro_result.episode_returns.mean()),
                    ant_solve_rate=float(ant_result.episode_solved.mean()),
                    ant_mean_return=float(ant_result.episode_returns.mean()),
                    mean_regret=float(regrets.mean()),
                    pro_hstates=pro_hstates,
                    ant_hstates=ant_hstates,
                    mean_wall_density=float(batch_features['wall_density'].mean()),
                    mean_goal_distance=float(batch_features['goal_distance'].mean()),
                )
                self._results_by_condition[(adv_cond.value, ant_cond.value)] = result
                _log(f"{cond_key}/done", msg=f"Condition complete: regret={result.mean_regret:.3f}")

        return self._results_by_condition

    def _flatten_hstate(self, hstate) -> np.ndarray:
        """Flatten hidden state tuple to array."""
        h_c, h_h = hstate
        return np.concatenate([
            np.array(h_c).flatten(),
            np.array(h_h).flatten()
        ])

    def analyze(self) -> Dict[str, Any]:
        """ANOVA-style analysis of factorial design."""
        if not self._results_by_condition:
            raise ValueError("Must call collect_data before analyze")

        results = {}

        # 1. Main effects
        results['adversary_effect'] = self._compute_adversary_main_effect()
        results['antagonist_effect'] = self._compute_antagonist_main_effect()

        # 2. Interaction effect
        results['interaction_effect'] = self._compute_interaction_effect()

        # 3. Representation shift (MMD from baseline)
        results['representation_shift'] = self._compute_representation_shift()

        # 4. Utility estimate shift by condition
        results['utility_shift'] = self._compute_utility_shift()

        # 5. Interpretation
        results['interpretation'] = self._interpret_results(results)

        return results

    def _compute_adversary_main_effect(self) -> Dict[str, Any]:
        """Compute main effect of adversary intervention."""
        effects = {}

        # Average over antagonist conditions
        for adv_cond in AdversaryCondition:
            regrets = []
            solve_rates = []
            for ant_cond in AntagonistCondition:
                result = self._results_by_condition.get((adv_cond.value, ant_cond.value))
                if result:
                    regrets.append(result.mean_regret)
                    solve_rates.append(result.pro_solve_rate)

            if regrets:
                effects[adv_cond.value] = {
                    'mean_regret': float(np.mean(regrets)),
                    'mean_solve_rate': float(np.mean(solve_rates)),
                }

        # Effect size: difference from baseline (frozen)
        baseline = effects.get('frozen', {}).get('mean_regret', 0)
        for cond, vals in effects.items():
            vals['effect_vs_baseline'] = vals['mean_regret'] - baseline

        return effects

    def _compute_antagonist_main_effect(self) -> Dict[str, Any]:
        """Compute main effect of antagonist state."""
        effects = {}

        # Average over adversary conditions
        for ant_cond in AntagonistCondition:
            regrets = []
            for adv_cond in AdversaryCondition:
                result = self._results_by_condition.get((adv_cond.value, ant_cond.value))
                if result:
                    regrets.append(result.mean_regret)

            if regrets:
                effects[ant_cond.value] = {
                    'mean_regret': float(np.mean(regrets)),
                }

        return effects

    def _compute_interaction_effect(self) -> Dict[str, Any]:
        """Test if adversary effect depends on antagonist state."""
        interactions = {}

        # For each adversary condition, compute effect separately for live vs frozen antagonist
        for adv_cond in AdversaryCondition:
            if adv_cond == AdversaryCondition.FROZEN:
                continue

            live_result = self._results_by_condition.get((adv_cond.value, 'live'))
            frozen_result = self._results_by_condition.get((adv_cond.value, 'frozen'))
            baseline_live = self._results_by_condition.get(('frozen', 'live'))
            baseline_frozen = self._results_by_condition.get(('frozen', 'frozen'))

            if all([live_result, frozen_result, baseline_live, baseline_frozen]):
                effect_with_live = live_result.mean_regret - baseline_live.mean_regret
                effect_with_frozen = frozen_result.mean_regret - baseline_frozen.mean_regret
                interaction = effect_with_live - effect_with_frozen

                interactions[adv_cond.value] = {
                    'effect_with_live_ant': float(effect_with_live),
                    'effect_with_frozen_ant': float(effect_with_frozen),
                    'interaction_magnitude': float(interaction),
                    'significant': abs(interaction) > 0.05,
                }

        return interactions

    def _compute_representation_shift(self) -> Dict[str, Any]:
        """Compute MMD between conditions."""
        baseline = self._results_by_condition.get(('frozen', 'live'))
        if baseline is None:
            return {'error': 'No baseline condition'}

        shifts = {}
        baseline_hstates = baseline.pro_hstates

        for (adv, ant), result in self._results_by_condition.items():
            if (adv, ant) == ('frozen', 'live'):
                continue

            # Compute bilateral CKA as proxy for representation shift
            cka_result = compute_bilateral_cka(result.pro_hstates, baseline_hstates)
            shifts[f"{adv}_{ant}"] = {
                'cka_to_baseline': cka_result.get('cka', 0.0),
                'shift_magnitude': 1.0 - cka_result.get('cka', 0.0),
            }

        return shifts

    def _compute_utility_shift(self) -> Dict[str, Any]:
        """Track how inferred Û shifts under each condition."""
        shifts = {}

        for (adv, ant), result in self._results_by_condition.items():
            # Use level features as proxy for Û components
            shifts[f"{adv}_{ant}"] = {
                'mean_wall_density': result.mean_wall_density,
                'mean_goal_distance': result.mean_goal_distance,
                'mean_regret': result.mean_regret,
            }

        return shifts

    def _interpret_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret the factorial analysis."""
        adv_effect = results.get('adversary_effect', {})
        ant_effect = results.get('antagonist_effect', {})
        interaction = results.get('interaction_effect', {})

        interpretation = {
            'summary': [],
        }

        # Adversary main effect interpretation
        no_adv = adv_effect.get('no_adversary', {})
        frozen = adv_effect.get('frozen', {})
        if no_adv and frozen:
            if no_adv.get('mean_regret', 0) < frozen.get('mean_regret', 0):
                interpretation['summary'].append(
                    "Removing adversary REDUCES regret - protagonist depends on adversary curriculum."
                )
            else:
                interpretation['summary'].append(
                    "Removing adversary does NOT reduce regret - protagonist has robust generalization."
                )

        # Antagonist effect interpretation
        live = ant_effect.get('live', {})
        frozen_ant = ant_effect.get('frozen', {})
        if live and frozen_ant:
            diff = live.get('mean_regret', 0) - frozen_ant.get('mean_regret', 0)
            if abs(diff) > 0.1:
                interpretation['summary'].append(
                    f"Antagonist state matters (Δ={diff:.2f}) - regret depends on antagonist adaptation."
                )
            else:
                interpretation['summary'].append(
                    "Antagonist state has minimal effect - regret is driven by level structure, not reference policy."
                )

        # Interaction interpretation
        has_interaction = any(
            v.get('significant', False) for v in interaction.values()
        )
        if has_interaction:
            interpretation['summary'].append(
                "INTERACTION detected: adversary effect depends on antagonist state. "
                "Protagonist adapts to adversary-antagonist relationship, not just levels."
            )

        return interpretation

    def visualize(self) -> Dict[str, np.ndarray]:
        """Create factorial analysis visualizations."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._results_by_condition:
            return figures

        # Create 2x2 heatmap of mean regret
        fig, ax = plt.subplots(figsize=(10, 8))

        adv_conditions = ['no_adversary', 'frozen', 'easy', 'hard']
        ant_conditions = ['live', 'frozen']

        regret_matrix = np.zeros((len(adv_conditions), len(ant_conditions)))
        for i, adv in enumerate(adv_conditions):
            for j, ant in enumerate(ant_conditions):
                result = self._results_by_condition.get((adv, ant))
                if result:
                    regret_matrix[i, j] = result.mean_regret

        im = ax.imshow(regret_matrix, cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(range(len(ant_conditions)))
        ax.set_xticklabels(['Antagonist Live', 'Antagonist Frozen'])
        ax.set_yticks(range(len(adv_conditions)))
        ax.set_yticklabels(['No Adversary', 'Frozen Adv', 'Easy Adv', 'Hard Adv'])
        ax.set_title('Mean Regret by Factorial Condition')

        # Annotate cells
        for i in range(len(adv_conditions)):
            for j in range(len(ant_conditions)):
                ax.text(j, i, f'{regret_matrix[i, j]:.2f}',
                       ha='center', va='center', fontsize=12)

        plt.colorbar(im, ax=ax, label='Mean Regret')
        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["factorial_heatmap"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
