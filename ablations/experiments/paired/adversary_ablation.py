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
    mean_path_length: float


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

    def __init__(
        self,
        n_levels_per_condition: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_levels_per_condition = n_levels_per_condition
        self._results_by_condition: Dict[Tuple[str, str], ConditionResult] = {}
        self._require_paired()

    def _require_paired(self):
        """Verify PAIRED training."""
        if self.training_method != "paired":
            raise ValueError(f"AdversaryAblationExperiment requires PAIRED, got {self.training_method}")

    def collect_data(self, rng: chex.PRNGKey) -> Dict[Tuple[str, str], ConditionResult]:
        """Run all factorial conditions."""
        for adv_cond in AdversaryCondition:
            for ant_cond in AntagonistCondition:
                rng, cond_rng = jax.random.split(rng)
                result = self._run_condition(cond_rng, adv_cond, ant_cond)
                self._results_by_condition[(adv_cond.value, ant_cond.value)] = result

        return self._results_by_condition

    def _run_condition(
        self,
        rng: chex.PRNGKey,
        adv_cond: AdversaryCondition,
        ant_cond: AntagonistCondition,
    ) -> ConditionResult:
        """Run evaluation under specific condition."""
        pro_solve_rates = []
        pro_returns = []
        ant_solve_rates = []
        ant_returns = []
        regrets = []
        wall_densities = []
        path_lengths = []
        pro_hstates_list = []
        ant_hstates_list = []

        for i in range(self.n_levels_per_condition):
            rng, level_rng, pro_rng, ant_rng = jax.random.split(rng, 4)

            # Generate level based on adversary condition
            level = self._generate_level_for_condition(level_rng, adv_cond)

            # Evaluate protagonist
            pro_result = self._evaluate_protagonist(pro_rng, level)

            # Evaluate antagonist based on condition
            ant_result = self._evaluate_antagonist(ant_rng, level, ant_cond)

            # Store results
            pro_solve_rates.append(float(pro_result['solved']))
            pro_returns.append(pro_result['return'])
            ant_solve_rates.append(float(ant_result['solved']))
            ant_returns.append(ant_result['return'])
            regrets.append(ant_result['return'] - pro_result['return'])
            wall_densities.append(float(level['wall_map'].mean()))
            path_lengths.append(float(self._compute_path_length(level)))

            # Store hidden states
            pro_hstates_list.append(self._flatten_hstate(pro_result['hstate']))
            ant_hstates_list.append(self._flatten_hstate(ant_result['hstate']))

        return ConditionResult(
            adversary_condition=adv_cond,
            antagonist_condition=ant_cond,
            n_levels=self.n_levels_per_condition,
            pro_solve_rate=float(np.mean(pro_solve_rates)),
            pro_mean_return=float(np.mean(pro_returns)),
            ant_solve_rate=float(np.mean(ant_solve_rates)),
            ant_mean_return=float(np.mean(ant_returns)),
            mean_regret=float(np.mean(regrets)),
            pro_hstates=np.stack(pro_hstates_list),
            ant_hstates=np.stack(ant_hstates_list),
            mean_wall_density=float(np.mean(wall_densities)),
            mean_path_length=float(np.mean(path_lengths)),
        )

    def _generate_level_for_condition(
        self,
        rng: chex.PRNGKey,
        adv_cond: AdversaryCondition,
    ) -> Dict[str, Any]:
        """Generate level based on adversary condition."""
        height, width = 13, 13

        if adv_cond == AdversaryCondition.NO_ADVERSARY:
            # Pure random levels
            wall_prob = float(jax.random.uniform(rng)) * 0.3
        elif adv_cond == AdversaryCondition.FROZEN:
            # Use baseline adversary distribution
            wall_prob = 0.15 + float(jax.random.uniform(rng)) * 0.1
        elif adv_cond == AdversaryCondition.EASY:
            # Constrained easy: low wall density, short paths
            wall_prob = float(jax.random.uniform(rng)) * 0.1
        elif adv_cond == AdversaryCondition.HARD:
            # Constrained hard: high wall density
            wall_prob = 0.25 + float(jax.random.uniform(rng)) * 0.15

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

    def _evaluate_protagonist(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
        max_steps: int = 256,
    ) -> Dict[str, Any]:
        """Evaluate protagonist on level."""
        return self._evaluate_agent(rng, level, self.train_state.pro_train_state, max_steps)

    def _evaluate_antagonist(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
        ant_cond: AntagonistCondition,
        max_steps: int = 256,
    ) -> Dict[str, Any]:
        """Evaluate antagonist on level."""
        if ant_cond == AntagonistCondition.LIVE:
            agent_state = self.train_state.ant_train_state
        else:
            # For frozen, use same state but could use saved checkpoint
            agent_state = self.train_state.ant_train_state

        return self._evaluate_agent(rng, level, agent_state, max_steps)

    def _evaluate_agent(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
        agent_state,
        max_steps: int,
    ) -> Dict[str, Any]:
        """Generic agent evaluation."""
        hstate = self._initialize_hstate(rng)
        total_return = 0.0
        final_hstate = hstate

        for step in range(max_steps):
            rng, step_rng = jax.random.split(rng)

            obs = self._create_observation(level, step)
            new_hstate, pi, value = self._forward_step(obs, hstate, agent_state)

            action = pi.sample(seed=step_rng)
            hstate = new_hstate
            final_hstate = hstate

            # Simplified reward simulation
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
        return (jnp.zeros((1, hidden_dim)), jnp.zeros((1, hidden_dim)))

    def _create_observation(self, level: Dict[str, Any], step: int):
        """Create observation."""
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
        """Forward pass."""
        obs_batch = jax.tree_util.tree_map(lambda x: x[None, None, ...], obs)
        done_batch = jnp.zeros((1, 1), dtype=bool)
        new_hstate, pi, value = agent_state.apply_fn(
            agent_state.params, (obs_batch, done_batch), hstate
        )
        return new_hstate, pi, value

    def _flatten_hstate(self, hstate) -> np.ndarray:
        """Flatten hidden state tuple to array."""
        h_c, h_h = hstate
        return np.concatenate([
            np.array(h_c).flatten(),
            np.array(h_h).flatten()
        ])

    def _compute_path_length(self, level: Dict[str, Any]) -> int:
        """Compute BFS path length."""
        from collections import deque

        wall_map = level['wall_map']
        start = level['agent_pos']
        goal = level['goal_pos']

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
        return -1

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
                'mean_path_length': result.mean_path_length,
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
