"""
B4: Counterfactual Curriculum via History Injection.

Tests whether injecting synthetic episode histories into the protagonist's
hidden state changes its behavior on the SAME levels with the SAME policy
parameters. All conditions share pro_ts; only the initial h-state differs.
This tests h-state sensitivity, not policy retraining.

Protocol:
1. Generate evaluation levels, run protagonist for baseline h-states + returns
2. For each condition {baseline, success_injection, failure_injection}:
   a. Create history via create_success_history() / create_failure_history()
   b. Inject into h-state via inject_hidden_state()
   c. Run protagonist with injected h-state, collect returns + policy logits
3. Measure behavioral change (return diff, action KL) and probe accuracy change
"""

from typing import Dict, Any, Optional
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment
from ..utils.paired_helpers import (
    generate_levels,
    extract_level_features_batch,
    get_pro_hstates,
    run_batched_rollout,
)
from ..utils.history_injection import (
    create_success_history,
    create_failure_history,
    inject_hidden_state,
    measure_injection_effect,
)
from ..probes.property_probe import train_probe


class CounterfactualCurriculumExperiment(CheckpointExperiment):
    """
    Counterfactual curriculum test via history injection.

    Evaluates whether injecting synthetic success/failure histories into
    the agent's hidden state causally affects behavior and representations.
    """

    @property
    def name(self) -> str:
        return "counterfactual_curriculum"

    CONDITIONS = ['baseline', 'success_injection', 'failure_injection']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_eval_levels = self.exp_config("n_eval_levels")
        self.max_steps = self.exp_config("max_steps", 256)
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"CounterfactualCurriculumExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """Collect data for all injection conditions."""
        n = self.n_eval_levels
        pro_ts = getattr(self.train_state, 'pro_train_state', self.train_state)

        # Generate evaluation levels
        rng, gen_rng = jax.random.split(rng)
        levels = generate_levels(self.agent, gen_rng, n)
        level_features = extract_level_features_batch(levels)

        # --- Baseline: run protagonist with fresh h-state ---
        rng, bl_rng = jax.random.split(rng)
        baseline_result = run_batched_rollout(
            bl_rng, levels, pro_ts, self.agent,
            max_steps=self.max_steps,
            collect_logits=True,
            return_final_hstate=True,
        )

        # Get baseline h-states
        rng, h_rng = jax.random.split(rng)
        baseline_hstates = get_pro_hstates(h_rng, levels, self, self.max_steps)

        condition_data = {
            'baseline': {
                'returns': np.array(baseline_result.episode_returns),
                'logits': baseline_result.logits,
                'hstates': baseline_hstates,
            }
        }

        # --- Success injection ---
        rng, succ_rng = jax.random.split(rng)
        success_history = create_success_history(
            self.agent, pro_ts, succ_rng,
            n_episodes=5,
        )
        init_hstate_succ = self.agent.initialize_hidden_state(n)
        injected_hstate_succ = inject_hidden_state(
            init_hstate_succ, success_history
        )

        rng, succ_roll_rng = jax.random.split(rng)
        from ..utils.batched_rollout import batched_rollout
        succ_result = batched_rollout(
            succ_roll_rng, levels, self.max_steps,
            pro_ts.apply_fn, pro_ts.params,
            self.agent.env, self.agent.env_params,
            injected_hstate_succ,
            collect_logits=True,
            collection_steps=[-1],
        )

        succ_hstates = succ_result.hstates_by_step.get("-1") if succ_result.hstates_by_step else baseline_hstates

        condition_data['success_injection'] = {
            'returns': np.array(succ_result.episode_returns),
            'logits': succ_result.logits,
            'hstates': succ_hstates,
        }

        # --- Failure injection ---
        rng, fail_rng = jax.random.split(rng)
        failure_history = create_failure_history(
            self.agent, pro_ts, fail_rng,
            n_episodes=5,
        )
        init_hstate_fail = self.agent.initialize_hidden_state(n)
        injected_hstate_fail = inject_hidden_state(
            init_hstate_fail, failure_history
        )

        rng, fail_roll_rng = jax.random.split(rng)
        fail_result = batched_rollout(
            fail_roll_rng, levels, self.max_steps,
            pro_ts.apply_fn, pro_ts.params,
            self.agent.env, self.agent.env_params,
            injected_hstate_fail,
            collect_logits=True,
            collection_steps=[-1],
        )

        fail_hstates = fail_result.hstates_by_step.get("-1") if fail_result.hstates_by_step else baseline_hstates

        condition_data['failure_injection'] = {
            'returns': np.array(fail_result.episode_returns),
            'logits': fail_result.logits,
            'hstates': fail_hstates,
        }

        # --- Measure injection effects ---
        injection_effects = {}
        for condition in ['success_injection', 'failure_injection']:
            cond = condition_data[condition]
            baseline = condition_data['baseline']
            effect = measure_injection_effect(
                baseline_predictions={"hstates": baseline_hstates},
                injected_predictions={"hstates": cond['hstates']},
                baseline_behavior={"values": baseline['returns']},
                injected_behavior={"values": cond['returns']},
            )
            injection_effects[condition] = effect

        self.data = {
            'condition_data': condition_data,
            'injection_effects': injection_effects,
            'level_features': level_features,
            'n_levels': n,
        }
        return self.data

    def analyze(self) -> Dict[str, Any]:
        """Analyze behavioral and representational changes per condition."""
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("Must call collect_data first")

        condition_data = self.data['condition_data']
        level_features = self.data['level_features']
        results = {}

        # --- Per-condition behavioral change ---
        baseline_returns = condition_data['baseline']['returns']
        behavioral = {}
        for condition in ['success_injection', 'failure_injection']:
            cond_returns = condition_data[condition]['returns']
            return_diff = float(np.mean(cond_returns) - np.mean(baseline_returns))

            # Action distribution KL divergence
            bl_logits = condition_data['baseline']['logits']
            cond_logits = condition_data[condition]['logits']
            kl = 0.0
            if bl_logits is not None and cond_logits is not None:
                # Mean KL across levels and timesteps
                bl_probs = _softmax(bl_logits)
                cond_probs = _softmax(cond_logits)
                eps = 1e-10
                kl_per = np.sum(cond_probs * np.log((cond_probs + eps) / (bl_probs + eps)), axis=-1)
                kl = float(np.nanmean(kl_per))

            behavioral[condition] = {
                'mean_return_diff': return_diff,
                'mean_return_baseline': float(np.mean(baseline_returns)),
                'mean_return_injected': float(np.mean(cond_returns)),
                'action_kl_divergence': kl,
            }
        results['behavioral_change'] = behavioral

        # --- Probe accuracy change ---
        # Train probes on baseline h-states, evaluate on injected
        baseline_hstates = condition_data['baseline']['hstates']
        probe_transfer = {}

        for feat_name in ['wall_density', 'goal_distance']:
            if feat_name not in level_features:
                continue
            targets = level_features[feat_name]

            # Train on baseline
            probe, bl_metrics = train_probe(
                baseline_hstates, targets,
                probe_type="linear", task="regression",
            )
            bl_r2 = bl_metrics.get('mean_score', 0.0)

            per_cond = {'baseline_r2': float(bl_r2)}
            for condition in ['success_injection', 'failure_injection']:
                cond_hstates = condition_data[condition]['hstates']
                preds = probe.predict(cond_hstates)
                from sklearn.metrics import r2_score
                try:
                    cond_r2 = float(r2_score(targets, preds))
                except ValueError:
                    cond_r2 = 0.0
                per_cond[f'{condition}_r2'] = cond_r2
                per_cond[f'{condition}_r2_drop'] = float(bl_r2 - cond_r2)
            probe_transfer[feat_name] = per_cond

        results['probe_accuracy_change'] = probe_transfer

        # --- Injection effects (from collect_data) ---
        results['injection_effects'] = self.data['injection_effects']

        # --- Correlation: injection magnitude vs behavioral change ---
        for condition in ['success_injection', 'failure_injection']:
            effect = self.data['injection_effects'].get(condition, {})
            beh = behavioral.get(condition, {})
            results[f'{condition}_magnitude_vs_behavior'] = {
                'hstate_change': effect.get('mean_hstate_change', 0.0),
                'return_change': beh.get('mean_return_diff', 0.0),
            }

        self.results = results
        return results

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize injection effects."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not hasattr(self, 'results') or not self.results:
            return figures

        try:
            behavioral = self.results.get('behavioral_change', {})

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Return comparison across conditions
            ax = axes[0]
            conditions = ['baseline', 'success_injection', 'failure_injection']
            means = []
            for c in conditions:
                if c == 'baseline':
                    means.append(behavioral.get('success_injection', {}).get(
                        'mean_return_baseline', 0.0))
                else:
                    means.append(behavioral.get(c, {}).get(
                        'mean_return_injected', 0.0))
            colors = ['gray', 'green', 'red']
            ax.bar(range(len(conditions)), means, color=colors, alpha=0.8)
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, rotation=30, ha='right')
            ax.set_ylabel("Mean Return")
            ax.set_title("Returns by Injection Condition")

            # Probe R² comparison
            ax = axes[1]
            probe_data = self.results.get('probe_accuracy_change', {})
            feat_names = list(probe_data.keys())
            x = np.arange(len(feat_names))
            width = 0.25
            for i, condition in enumerate(['baseline', 'success_injection', 'failure_injection']):
                key = f'{condition}_r2' if condition != 'baseline' else 'baseline_r2'
                vals = [probe_data[f].get(key, 0.0) for f in feat_names]
                ax.bar(x + i * width, vals, width, label=condition,
                       color=colors[i], alpha=0.8)
            ax.set_xticks(x + width)
            ax.set_xticklabels(feat_names, rotation=30, ha='right')
            ax.set_ylabel("Probe R²")
            ax.set_title("Probe Accuracy by Condition")
            ax.legend()

            plt.tight_layout()
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            figures["counterfactual_curriculum"] = np.asarray(buf)[:, :, :3]
            plt.close(fig)
        except Exception:
            pass

        return figures


def _softmax(logits):
    """Numerically stable softmax."""
    if logits is None:
        return None
    x = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-10)
