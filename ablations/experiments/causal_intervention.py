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

    def __init__(self, **kwargs):
        """
        Initialize causal intervention experiment.
        """
        super().__init__(**kwargs)
        self.n_episodes_per_intervention = self.exp_config("n_episodes_per_intervention")
        self.adaptation_episodes = self.exp_config("adaptation_episodes")
        self.progressive_difficulty_steps = self.exp_config("progressive_difficulty_steps")
        self._current_episode_idx = 0  # For progressive difficulty

        # Get compatible interventions for this training method
        compatible = INTERVENTION_COMPATIBILITY.get(self.training_method, set())

        interventions = self.exp_config("interventions")
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
        Collect data under each intervention condition (GPU-batched).

        Applies actual interventions:
        - Difficulty interventions: filter generated levels by solve rate / wall density
        - Branch ablations: generate levels with restricted curriculum branches
        - PAIRED interventions: modify adversary behavior

        Returns dict mapping intervention type to results.
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

        results = {}
        n_eps = self.n_episodes_per_intervention
        max_steps = self.config.get("max_steps", 256)

        # --- First, collect baseline (NORMAL) to get difficulty scores ---
        _log("baseline", msg="Running baseline (NORMAL) rollout...")
        t0 = time.time()
        rng, rng_baseline_levels = jax.random.split(rng)
        # Generate a large pool of levels for filtering
        n_pool = n_eps * 5
        pool_rngs = jax.random.split(rng_baseline_levels, n_pool)
        level_pool = jax.vmap(self.agent.sample_random_level)(pool_rngs)
        jax.block_until_ready(level_pool)

        # Score each level via a quick rollout to get difficulty proxy
        rng, rng_score = jax.random.split(rng)
        score_result = batched_rollout(
            rng_score, level_pool, max_steps,
            self.train_state.apply_fn, self.train_state.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n_pool),
            collect_values=True, collect_entropies=True,
            collection_steps=[-1],
        )
        # Difficulty scores: lower return = harder level
        level_scores = np.array(score_result.episode_returns)
        # Wall density scores for structural interventions
        wall_densities = np.array(level_pool.wall_map).reshape(n_pool, -1).mean(axis=1)
        _log("baseline", time.time() - t0)

        # --- Intervention-specific level selection and rollout ---
        for intervention in tqdm(self.interventions, desc="Interventions"):
            _log(f"intervention_{intervention.value}", msg=f"Running {intervention.value}...")
            t0 = time.time()

            # Select/filter levels based on intervention type
            if intervention == InterventionType.NORMAL:
                indices = np.arange(min(n_eps, n_pool))

            elif intervention == InterventionType.ONLY_EASY:
                manip = DifficultyManipulation("only_easy", percentile=20.0)
                _, filtered_scores = manip.filter_levels(level_pool, level_scores)
                # Get indices of easy levels (top 20% return)
                threshold = np.percentile(level_scores, 80)
                easy_idx = np.where(level_scores >= threshold)[0]
                indices = easy_idx[:n_eps] if len(easy_idx) >= n_eps else np.tile(easy_idx, n_eps // len(easy_idx) + 1)[:n_eps]

            elif intervention == InterventionType.ONLY_HARD:
                manip = DifficultyManipulation("only_hard", percentile=20.0)
                threshold = np.percentile(level_scores, 20)
                hard_idx = np.where(level_scores <= threshold)[0]
                indices = hard_idx[:n_eps] if len(hard_idx) >= n_eps else np.tile(hard_idx, n_eps // len(hard_idx) + 1)[:n_eps]

            elif intervention == InterventionType.HIGH_WALL_DENSITY:
                threshold = np.percentile(wall_densities, 80)
                dense_idx = np.where(wall_densities >= threshold)[0]
                indices = dense_idx[:n_eps] if len(dense_idx) >= n_eps else np.tile(dense_idx, n_eps // len(dense_idx) + 1)[:n_eps]

            elif intervention == InterventionType.LOW_WALL_DENSITY:
                threshold = np.percentile(wall_densities, 20)
                sparse_idx = np.where(wall_densities <= threshold)[0]
                indices = sparse_idx[:n_eps] if len(sparse_idx) >= n_eps else np.tile(sparse_idx, n_eps // len(sparse_idx) + 1)[:n_eps]

            elif intervention in (InterventionType.DR_ONLY, InterventionType.NO_MUTATION,
                                  InterventionType.ALL_REPLAY):
                # Branch ablations: for post-hoc analysis, use random levels
                # (actual branch ablation is a training-time intervention)
                # Generate fresh random levels (simulating DR-only curriculum)
                rng, rng_branch = jax.random.split(rng)
                branch_rngs = jax.random.split(rng_branch, n_eps)
                # Use fresh random levels (no replay/mutation selection)
                indices = np.random.default_rng(seed=int(jax.random.key_data(rng_branch)[0]) % (2**31)).choice(n_pool, n_eps, replace=True)

            elif intervention == InterventionType.FIXED_DIFFICULTY:
                # Select levels near median difficulty
                median_score = np.median(level_scores)
                distances = np.abs(level_scores - median_score)
                indices = np.argsort(distances)[:n_eps]

            elif intervention == InterventionType.PROGRESSIVE_DIFFICULTY:
                # Sort by difficulty, present progressively easier
                sorted_idx = np.argsort(level_scores)
                step_size = len(sorted_idx) // self.progressive_difficulty_steps
                indices = sorted_idx[:n_eps]

            elif intervention == InterventionType.CLUSTERED_SAMPLING:
                # Sample from clusters of similar difficulty
                n_clusters = 3
                sorted_idx = np.argsort(level_scores)
                cluster_size = len(sorted_idx) // n_clusters
                cluster_idx = sorted_idx[:cluster_size]  # Hardest cluster
                indices = np.random.default_rng(seed=42).choice(cluster_idx, n_eps, replace=True)

            else:
                # PAIRED-specific or unknown: use random selection
                indices = np.random.default_rng(seed=0).choice(n_pool, n_eps, replace=True)

            # Select levels
            selected_levels = jax.tree_util.tree_map(lambda x: x[indices], level_pool)

            # Run batched rollout on selected levels
            rng, rng_rollout = jax.random.split(rng)
            result = batched_rollout(
                rng_rollout, selected_levels, max_steps,
                self.train_state.apply_fn, self.train_state.params,
                self.agent.env, self.agent.env_params,
                self.agent.initialize_hidden_state(n_eps),
                collect_values=True, collect_entropies=True,
                collection_steps=[-1],
            )

            # Extract hidden states from terminal step
            hidden_states = result.hstates_by_step["-1"]

            # Compute per-episode mean values and entropies
            value_estimates = []
            policy_entropies = []
            for i in range(n_eps):
                ep_len = int(result.episode_lengths[i])
                v_vals = result.values[i, :ep_len]
                e_vals = result.entropies[i, :ep_len]
                valid_v = v_vals[~np.isnan(v_vals)]
                valid_e = e_vals[~np.isnan(e_vals)]
                value_estimates.append(float(np.mean(valid_v)) if len(valid_v) > 0 else 0.0)
                policy_entropies.append(float(np.mean(valid_e)) if len(valid_e) > 0 else 0.0)

            # Compute prediction losses
            from .utils.agent_aware_loss import compute_agent_prediction_loss
            prediction_losses = []
            for i in tqdm(range(min(n_eps, 50)), desc=f"Pred losses ({intervention.value})", leave=False):
                rng, loss_rng = jax.random.split(rng)
                level_i = jax.tree_util.tree_map(lambda x: x[i], selected_levels)
                try:
                    loss, _ = compute_agent_prediction_loss(
                        self.agent, self.train_state, level_i, loss_rng
                    )
                    prediction_losses.append(loss)
                except Exception:
                    prediction_losses.append(1.0)

            int_result = InterventionResult(
                intervention_type=intervention,
                n_episodes=n_eps,
                solve_rate=float(result.episode_solved.mean()),
                mean_return=float(result.episode_returns.mean()),
                mean_steps=float(result.episode_lengths.mean()),
                hidden_states=hidden_states,
                value_estimates=value_estimates,
                policy_entropies=policy_entropies,
                prediction_losses=prediction_losses,
            )

            results[intervention] = int_result
            if intervention == InterventionType.NORMAL:
                self._baseline_result = int_result

            elapsed = time.time() - t0
            _log(f"intervention_{intervention.value}", elapsed,
                 f"{intervention.value}: solve_rate={int_result.solve_rate:.3f}, "
                 f"mean_return={int_result.mean_return:.3f}")

        self._intervention_results = results

        if _wandb_active:
            for int_type, res in results.items():
                wandb.log({
                    f"{self.name}/{int_type.value}/solve_rate": res.solve_rate,
                    f"{self.name}/{int_type.value}/mean_return": res.mean_return,
                })

        total_time = sum(timings.values())
        _log("total", total_time, f"TOTAL collect_data: {total_time:.2f}s")

        return results

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
        """Analyze hidden state distribution shifts using distribution_shift utilities."""
        baseline_states = self._baseline_result.hidden_states
        shift_results = {}

        for intervention, result in self._intervention_results.items():
            if intervention == InterventionType.NORMAL:
                continue

            # Use the proper measure_distribution_shift utility (MMD + mean shift + variance ratio + cosine)
            shift_metrics = measure_distribution_shift(baseline_states, result.hidden_states)

            # Compute intervention effect on performance
            baseline_returns = np.array(self._baseline_result.value_estimates)
            intervention_returns = np.array(result.value_estimates)
            effect_metrics = compute_intervention_effect(baseline_returns, intervention_returns)

            shift_results[intervention.value] = {
                **shift_metrics,
                **effect_metrics,
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
