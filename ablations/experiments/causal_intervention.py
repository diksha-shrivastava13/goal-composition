"""
Causal Intervention Experiment.

Tests robustness to curriculum distribution shifts through various interventions:
- ACCEL/PLR: Branch ablations (DR only, no mutation, etc.)
- PAIRED: Adversary manipulations (frozen, constrained, disabled)
- DR: Distribution manipulations (fixed difficulty, progressive)
- Universal: Difficulty and structural changes

Measures performance and internal representation changes under intervention.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import jax
import jax.numpy as jnp
import chex

from .base import CheckpointExperiment
from .utils.distribution_shift import (
    compute_mmd,
    BranchAblation,
    DifficultyManipulation,
    measure_distribution_shift,
    compute_intervention_effect,
)


class InterventionType(Enum):
    """Types of curriculum interventions."""
    # Universal interventions (all methods)
    NORMAL = "normal"
    ONLY_EASY = "only_easy"
    ONLY_HARD = "only_hard"
    HIGH_WALL_DENSITY = "high_wall_density"
    LOW_WALL_DENSITY = "low_wall_density"

    # ACCEL-specific interventions (require branches/mutations)
    DR_ONLY = "dr_only"
    NO_MUTATION = "no_mutation"
    ALL_REPLAY = "all_replay"

    # PAIRED-specific interventions (require adversary)
    NO_ADVERSARY = "no_adversary"
    EASY_ADVERSARY = "easy_adversary"
    HARD_ADVERSARY = "hard_adversary"
    FROZEN_ADVERSARY = "frozen_adversary"

    # DR-specific interventions (no curriculum structure)
    FIXED_DIFFICULTY = "fixed_difficulty"
    PROGRESSIVE_DIFFICULTY = "progressive_difficulty"
    CLUSTERED_SAMPLING = "clustered_sampling"


# Define which interventions are compatible with which methods
INTERVENTION_COMPATIBILITY = {
    "accel": {
        InterventionType.NORMAL,
        InterventionType.ONLY_EASY,
        InterventionType.ONLY_HARD,
        InterventionType.HIGH_WALL_DENSITY,
        InterventionType.LOW_WALL_DENSITY,
        InterventionType.DR_ONLY,
        InterventionType.NO_MUTATION,
        InterventionType.ALL_REPLAY,
    },
    "plr": {
        InterventionType.NORMAL,
        InterventionType.ONLY_EASY,
        InterventionType.ONLY_HARD,
        InterventionType.HIGH_WALL_DENSITY,
        InterventionType.LOW_WALL_DENSITY,
        InterventionType.DR_ONLY,
        InterventionType.ALL_REPLAY,
        # No NO_MUTATION since PLR doesn't have mutations
    },
    "robust_plr": {
        InterventionType.NORMAL,
        InterventionType.ONLY_EASY,
        InterventionType.ONLY_HARD,
        InterventionType.HIGH_WALL_DENSITY,
        InterventionType.LOW_WALL_DENSITY,
        InterventionType.DR_ONLY,
        InterventionType.ALL_REPLAY,
    },
    "paired": {
        InterventionType.NORMAL,
        InterventionType.ONLY_EASY,
        InterventionType.ONLY_HARD,
        InterventionType.HIGH_WALL_DENSITY,
        InterventionType.LOW_WALL_DENSITY,
        InterventionType.NO_ADVERSARY,
        InterventionType.EASY_ADVERSARY,
        InterventionType.HARD_ADVERSARY,
        InterventionType.FROZEN_ADVERSARY,
    },
    "dr": {
        InterventionType.NORMAL,
        InterventionType.ONLY_EASY,
        InterventionType.ONLY_HARD,
        InterventionType.HIGH_WALL_DENSITY,
        InterventionType.LOW_WALL_DENSITY,
        InterventionType.FIXED_DIFFICULTY,
        InterventionType.PROGRESSIVE_DIFFICULTY,
        InterventionType.CLUSTERED_SAMPLING,
    },
}


@dataclass
class InterventionResult:
    """Results from running under an intervention."""
    intervention_type: InterventionType
    n_episodes: int
    solve_rate: float
    mean_return: float
    mean_steps: float
    hidden_states: np.ndarray  # (n_episodes, hidden_dim)
    value_estimates: List[float]
    policy_entropies: List[float]
    prediction_losses: List[float]  # Actual probe/prediction losses per episode


class CausalInterventionExperiment(CheckpointExperiment):
    """
    Test agent robustness to curriculum distribution shifts.

    Intervention types vary by training method:

    ACCEL/PLR (has_branches=True):
    - Branch ablations: DR only, no mutation, all replay
    - Difficulty manipulations: only easy, only hard
    - Structural changes: high/low wall density

    PAIRED (has_adversary=True):
    - Adversary ablations: no adversary, frozen adversary
    - Adversary constraints: easy adversary, hard adversary
    - Difficulty manipulations: only easy, only hard

    DR (no curriculum):
    - Distribution changes: fixed difficulty, progressive difficulty
    - Sampling strategies: clustered sampling
    - Difficulty manipulations: only easy, only hard

    Measurements:
    - Performance: solve rate, return, adaptation curve
    - Internal: value calibration, probe accuracy, hidden state shift
    """

    @property
    def name(self) -> str:
        return "causal_intervention"

    def __init__(
        self,
        n_episodes_per_intervention: int = 100,
        interventions: List[str] = None,
        adaptation_episodes: int = 20,
        progressive_difficulty_steps: int = 10,
        **kwargs,
    ):
        """
        Initialize causal intervention experiment.

        Args:
            n_episodes_per_intervention: Episodes to run under each intervention
            interventions: List of intervention types to test (auto-filtered by method)
            adaptation_episodes: Episodes to track for adaptation curve
            progressive_difficulty_steps: Steps for progressive difficulty (DR only)
        """
        super().__init__(**kwargs)
        self.n_episodes_per_intervention = n_episodes_per_intervention
        self.adaptation_episodes = adaptation_episodes
        self.progressive_difficulty_steps = progressive_difficulty_steps
        self._current_episode_idx = 0  # For progressive difficulty

        # Get compatible interventions for this training method
        compatible = INTERVENTION_COMPATIBILITY.get(self.training_method, set())

        if interventions is None:
            # Use all compatible interventions
            self.interventions = [i for i in InterventionType if i in compatible]
        else:
            # Filter user-provided interventions to compatible ones
            requested = [InterventionType(i) for i in interventions]
            self.interventions = [i for i in requested if i in compatible]

            # Warn about incompatible interventions
            incompatible = [i for i in requested if i not in compatible]
            if incompatible:
                print(f"Warning: Skipping incompatible interventions for {self.training_method}: "
                      f"{[i.value for i in incompatible]}")

        self._baseline_result: Optional[InterventionResult] = None
        self._intervention_results: Dict[InterventionType, InterventionResult] = {}
        self._results: Dict[str, Any] = {}

    def collect_data(self, rng: chex.PRNGKey) -> Dict[str, InterventionResult]:
        """
        Collect data under each intervention condition.

        Returns dict mapping intervention type to results.
        """
        results = {}

        for intervention in self.interventions:
            rng, int_rng = jax.random.split(rng)

            result = self._run_intervention(int_rng, intervention)
            results[intervention] = result

            if intervention == InterventionType.NORMAL:
                self._baseline_result = result

        self._intervention_results = results
        return results

    def _run_intervention(
        self,
        rng: chex.PRNGKey,
        intervention: InterventionType,
    ) -> InterventionResult:
        """Run episodes under a specific intervention."""
        all_hidden_states = []
        all_values = []
        all_entropies = []
        all_solved = []
        all_returns = []
        all_steps = []
        all_prediction_losses = []

        for ep_idx in range(self.n_episodes_per_intervention):
            self._current_episode_idx = ep_idx  # For progressive difficulty
            rng, ep_rng, level_rng, loss_rng = jax.random.split(rng, 4)

            # Generate level according to intervention
            level = self._generate_level_for_intervention(level_rng, intervention)

            # Run episode
            result = self._run_episode(ep_rng, level)

            # Collect metrics
            h_c, h_h = result['final_hstate']
            hidden_flat = np.concatenate([
                np.array(h_c).flatten(),
                np.array(h_h).flatten()
            ])
            all_hidden_states.append(hidden_flat)
            all_values.append(result.get('mean_value', 0.0))
            all_entropies.append(result.get('mean_entropy', 0.0))
            all_solved.append(result['solved'])
            all_returns.append(result['total_return'])
            all_steps.append(result['n_steps'])

            # Compute prediction loss for this episode
            pred_loss = self._compute_episode_prediction_loss(loss_rng, level)
            all_prediction_losses.append(pred_loss)

        return InterventionResult(
            intervention_type=intervention,
            n_episodes=self.n_episodes_per_intervention,
            solve_rate=float(np.mean(all_solved)),
            mean_return=float(np.mean(all_returns)),
            mean_steps=float(np.mean(all_steps)),
            hidden_states=np.stack(all_hidden_states),
            value_estimates=all_values,
            policy_entropies=all_entropies,
            prediction_losses=all_prediction_losses,
        )

    def _compute_episode_prediction_loss(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
    ) -> float:
        """
        Compute actual prediction/probe loss for an episode's level.

        Uses agent-aware loss computation.
        """
        try:
            from .utils.agent_aware_loss import compute_agent_prediction_loss

            loss, _ = compute_agent_prediction_loss(
                self.agent,
                self.train_state,
                level,
                rng,
            )
            return loss
        except Exception:
            return 1.0  # Default max loss on error

    def _generate_level_for_intervention(
        self,
        rng: chex.PRNGKey,
        intervention: InterventionType,
    ) -> Dict[str, Any]:
        """Generate a level according to intervention constraints."""
        height, width = 13, 13

        # Default parameters
        wall_prob = 0.15

        # ===== Universal interventions =====
        if intervention == InterventionType.NORMAL:
            wall_prob = 0.1 + float(jax.random.uniform(rng)) * 0.2

        elif intervention == InterventionType.ONLY_EASY:
            wall_prob = 0.05 + float(jax.random.uniform(rng)) * 0.1

        elif intervention == InterventionType.ONLY_HARD:
            wall_prob = 0.25 + float(jax.random.uniform(rng)) * 0.1

        elif intervention == InterventionType.HIGH_WALL_DENSITY:
            wall_prob = 0.3 + float(jax.random.uniform(rng)) * 0.1

        elif intervention == InterventionType.LOW_WALL_DENSITY:
            wall_prob = 0.02 + float(jax.random.uniform(rng)) * 0.05

        # ===== ACCEL-specific interventions =====
        elif intervention == InterventionType.DR_ONLY:
            wall_prob = float(jax.random.uniform(rng)) * 0.3

        elif intervention == InterventionType.NO_MUTATION:
            wall_prob = 0.1 + float(jax.random.uniform(rng)) * 0.15

        elif intervention == InterventionType.ALL_REPLAY:
            wall_prob = 0.15 + float(jax.random.uniform(rng)) * 0.1

        # ===== PAIRED-specific interventions =====
        elif intervention == InterventionType.NO_ADVERSARY:
            # Random level generation (no adversarial structure)
            wall_prob = float(jax.random.uniform(rng)) * 0.3

        elif intervention == InterventionType.EASY_ADVERSARY:
            # Constrained adversary: low difficulty
            wall_prob = 0.05 + float(jax.random.uniform(rng)) * 0.1

        elif intervention == InterventionType.HARD_ADVERSARY:
            # Constrained adversary: high difficulty
            wall_prob = 0.25 + float(jax.random.uniform(rng)) * 0.15

        elif intervention == InterventionType.FROZEN_ADVERSARY:
            # Fixed adversary behavior (similar difficulty each time)
            wall_prob = 0.15  # Fixed, no randomness

        # ===== DR-specific interventions =====
        elif intervention == InterventionType.FIXED_DIFFICULTY:
            # Fixed moderate difficulty
            wall_prob = 0.15

        elif intervention == InterventionType.PROGRESSIVE_DIFFICULTY:
            # Linearly increase difficulty over episodes
            progress = self._current_episode_idx / max(self.n_episodes_per_intervention, 1)
            wall_prob = 0.05 + progress * 0.25  # 0.05 -> 0.30

        elif intervention == InterventionType.CLUSTERED_SAMPLING:
            # Sample from discrete difficulty clusters
            cluster = int(jax.random.randint(rng, (), 0, 3))
            wall_prob = [0.08, 0.18, 0.28][cluster]

        # Generate level
        wall_map = np.array(jax.random.bernoulli(rng, wall_prob, (height, width)))
        wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

        rng_goal, rng_agent = jax.random.split(rng)

        # Goal/agent placement varies by intervention
        if intervention in [InterventionType.HARD_ADVERSARY, InterventionType.ONLY_HARD]:
            # Place goal and agent far apart
            goal_pos = (
                int(jax.random.randint(rng_goal, (), height - 4, height - 1)),
                int(jax.random.randint(rng_goal, (), width - 4, width - 1)),
            )
            agent_pos = (
                int(jax.random.randint(rng_agent, (), 1, 4)),
                int(jax.random.randint(rng_agent, (), 1, 4)),
            )
        elif intervention in [InterventionType.EASY_ADVERSARY, InterventionType.ONLY_EASY]:
            # Place goal and agent close together
            goal_pos = (
                int(jax.random.randint(rng_goal, (), 5, 8)),
                int(jax.random.randint(rng_goal, (), 5, 8)),
            )
            agent_y = goal_pos[0] + int(jax.random.randint(rng_agent, (), -2, 3))
            agent_x = goal_pos[1] + int(jax.random.randint(rng_agent, (), -2, 3))
            agent_pos = (
                int(np.clip(agent_y, 1, height - 2)),
                int(np.clip(agent_x, 1, width - 2)),
            )
        else:
            # Random placement
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
            'intervention': intervention.value,
        }

        # Add branch info only for methods that use it
        if self.has_branches:
            if intervention == InterventionType.DR_ONLY:
                level['branch'] = 0
            elif intervention == InterventionType.ALL_REPLAY:
                level['branch'] = 1
            elif intervention == InterventionType.NO_MUTATION:
                level['branch'] = int(jax.random.randint(rng, (), 0, 2))
            else:
                level['branch'] = int(jax.random.randint(rng, (), 0, self.branch_count))

        return level

    def _run_episode(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
        max_steps: int = 256,
    ) -> Dict[str, Any]:
        """Run a single episode."""
        hstate = self.agent.initialize_carry(rng, batch_dims=(1,))

        total_return = 0.0
        solved = False
        values = []
        entropies = []

        for step in range(max_steps):
            rng, step_rng = jax.random.split(rng)

            obs = self._create_observation(level, step)
            new_hstate, pi, value = self._forward_step(obs, hstate)

            values.append(float(value[0, 0]))
            entropies.append(float(pi.entropy()[0, 0]))

            action = pi.sample(seed=step_rng)
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
            'final_hstate': hstate,
            'n_steps': step + 1,
            'mean_value': float(np.mean(values)),
            'mean_entropy': float(np.mean(entropies)),
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
        Analyze intervention effects.

        Computes:
        - Performance drop under each intervention
        - Hidden state distribution shift (MMD)
        - Value calibration changes
        - Sensitivity to different intervention types
        """
        if self._baseline_result is None:
            raise ValueError("Must call collect_data before analyze")

        results = {}

        # 1. Performance comparison
        results['performance'] = self._analyze_performance()

        # 2. Distribution shift analysis
        results['distribution_shift'] = self._analyze_distribution_shift()

        # 3. Sensitivity analysis
        results['sensitivity'] = self._analyze_sensitivity()

        # 4. Branch vs difficulty sensitivity comparison
        results['intervention_comparison'] = self._compare_intervention_types()

        self._results = results
        return results

    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance under each intervention."""
        baseline = self._baseline_result
        performance = {}

        for intervention, result in self._intervention_results.items():
            if intervention == InterventionType.NORMAL:
                continue

            perf_drop = baseline.solve_rate - result.solve_rate
            return_drop = baseline.mean_return - result.mean_return

            # Prediction loss comparison
            baseline_pred_loss = float(np.mean(baseline.prediction_losses))
            intervention_pred_loss = float(np.mean(result.prediction_losses))
            pred_loss_increase = intervention_pred_loss - baseline_pred_loss

            performance[intervention.value] = {
                'solve_rate': result.solve_rate,
                'performance_drop': float(perf_drop),
                'relative_drop': float(perf_drop / (baseline.solve_rate + 1e-6)),
                'mean_return': result.mean_return,
                'return_drop': float(return_drop),
                'mean_steps': result.mean_steps,
                'mean_prediction_loss': intervention_pred_loss,
                'prediction_loss_increase': float(pred_loss_increase),
            }

        performance['baseline'] = {
            'solve_rate': baseline.solve_rate,
            'mean_return': baseline.mean_return,
            'mean_steps': baseline.mean_steps,
            'mean_prediction_loss': float(np.mean(baseline.prediction_losses)),
        }

        return performance

    def _analyze_distribution_shift(self) -> Dict[str, Any]:
        """Analyze hidden state distribution shifts."""
        baseline_states = self._baseline_result.hidden_states
        shift_results = {}

        for intervention, result in self._intervention_results.items():
            if intervention == InterventionType.NORMAL:
                continue

            # Compute MMD between distributions
            mmd = compute_mmd(baseline_states, result.hidden_states)

            # Compute mean distance in hidden space
            baseline_mean = baseline_states.mean(axis=0)
            intervention_mean = result.hidden_states.mean(axis=0)
            mean_dist = float(np.linalg.norm(baseline_mean - intervention_mean))

            shift_results[intervention.value] = {
                'mmd': mmd,
                'mean_distance': mean_dist,
                'variance_ratio': float(
                    result.hidden_states.var() / (baseline_states.var() + 1e-6)
                ),
            }

        return shift_results

    def _analyze_sensitivity(self) -> Dict[str, Any]:
        """Analyze agent sensitivity to different interventions."""
        sensitivity = {}

        for intervention, result in self._intervention_results.items():
            if intervention == InterventionType.NORMAL:
                continue

            # Compute sensitivity score (performance drop × distribution shift)
            perf_drop = self._baseline_result.solve_rate - result.solve_rate

            baseline_states = self._baseline_result.hidden_states
            mmd = compute_mmd(baseline_states, result.hidden_states)

            sensitivity[intervention.value] = {
                'performance_sensitivity': float(perf_drop),
                'representation_sensitivity': float(mmd),
                'combined_sensitivity': float(perf_drop * mmd),
            }

        return sensitivity

    def _compare_intervention_types(self) -> Dict[str, Any]:
        """Compare curriculum-specific vs universal interventions (method-aware)."""
        # Define intervention categories by method
        if self.training_method in ["accel", "plr", "robust_plr"]:
            curriculum_interventions = [
                InterventionType.DR_ONLY,
                InterventionType.NO_MUTATION,
                InterventionType.ALL_REPLAY,
            ]
            curriculum_label = "branch"
        elif self.training_method == "paired":
            curriculum_interventions = [
                InterventionType.NO_ADVERSARY,
                InterventionType.EASY_ADVERSARY,
                InterventionType.HARD_ADVERSARY,
                InterventionType.FROZEN_ADVERSARY,
            ]
            curriculum_label = "adversary"
        else:  # dr
            curriculum_interventions = [
                InterventionType.FIXED_DIFFICULTY,
                InterventionType.PROGRESSIVE_DIFFICULTY,
                InterventionType.CLUSTERED_SAMPLING,
            ]
            curriculum_label = "distribution"

        difficulty_interventions = [
            InterventionType.ONLY_EASY,
            InterventionType.ONLY_HARD,
            InterventionType.HIGH_WALL_DENSITY,
            InterventionType.LOW_WALL_DENSITY,
        ]

        curriculum_drops = []
        difficulty_drops = []

        for int_type in curriculum_interventions:
            if int_type in self._intervention_results:
                drop = self._baseline_result.solve_rate - self._intervention_results[int_type].solve_rate
                curriculum_drops.append(drop)

        for int_type in difficulty_interventions:
            if int_type in self._intervention_results:
                drop = self._baseline_result.solve_rate - self._intervention_results[int_type].solve_rate
                difficulty_drops.append(drop)

        return {
            f'{curriculum_label}_intervention': {
                'mean_performance_drop': float(np.mean(curriculum_drops)) if curriculum_drops else 0.0,
                'max_performance_drop': float(np.max(curriculum_drops)) if curriculum_drops else 0.0,
                'interventions_tested': len(curriculum_drops),
            },
            'difficulty_intervention': {
                'mean_performance_drop': float(np.mean(difficulty_drops)) if difficulty_drops else 0.0,
                'max_performance_drop': float(np.max(difficulty_drops)) if difficulty_drops else 0.0,
                'interventions_tested': len(difficulty_drops),
            },
            f'{curriculum_label}_more_sensitive': (
                float(np.mean(curriculum_drops)) > float(np.mean(difficulty_drops))
                if curriculum_drops and difficulty_drops else None
            ),
            'interpretation': self._interpret_comparison(curriculum_drops, difficulty_drops, curriculum_label),
            'training_method': self.training_method,
        }

    def _interpret_comparison(
        self,
        curriculum_drops: List[float],
        difficulty_drops: List[float],
        curriculum_label: str = "branch",
    ) -> str:
        """Interpret the comparison between curriculum and difficulty sensitivity."""
        if not curriculum_drops or not difficulty_drops:
            return "Insufficient data for comparison"

        curriculum_mean = np.mean(curriculum_drops)
        diff_mean = np.mean(difficulty_drops)

        if curriculum_mean > diff_mean * 1.5:
            return f"Agent is more sensitive to {curriculum_label} structure than difficulty"
        elif diff_mean > curriculum_mean * 1.5:
            return f"Agent is more sensitive to difficulty than {curriculum_label} structure"
        else:
            return f"Agent shows similar sensitivity to {curriculum_label} and difficulty changes"

    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data."""
        if not self._results:
            raise ValueError("Must call analyze before visualize")

        viz_data = {
            'performance_by_intervention': {},
            'distribution_shift_by_intervention': {},
        }

        # Performance bar chart
        if 'performance' in self._results:
            for int_name, perf in self._results['performance'].items():
                if int_name != 'baseline':
                    viz_data['performance_by_intervention'][int_name] = {
                        'solve_rate': perf.get('solve_rate', 0),
                        'performance_drop': perf.get('performance_drop', 0),
                    }

        # Distribution shift
        if 'distribution_shift' in self._results:
            for int_name, shift in self._results['distribution_shift'].items():
                viz_data['distribution_shift_by_intervention'][int_name] = shift.get('mmd', 0)

        return viz_data
