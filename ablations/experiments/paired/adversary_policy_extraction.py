"""
A2: Adversary Policy Extraction via H-State Probing.

Probe adversary hidden states to extract its implicit teaching policy.
Uses 4-tier probing framework (mirroring level_probing.py):
  Tier 0: env variables (wall_density, goal_distance)
  Tier 1: curriculum dynamics (regret, difficulty)
  Tier 2: agent-curriculum interaction (pro_return, ant_return)
  Tier 3: adversary strategy (entropy, generation pattern)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment
from ..utils.paired_helpers import (
    generate_levels,
    extract_level_features_batch,
    get_pro_ant_returns,
    get_pro_hstates,
    compute_difficulty,
    run_adversary_rollout,
)
from ..probes.property_probe import (
    train_probe,
    compute_probe_comparison,
)


class AdversaryPolicyExtractionExperiment(CheckpointExperiment):
    """
    Extract adversary's generation policy via h-state probing.

    Protocol:
    1. Generate levels, run adversary through them to get adversary h-states
    2. Run protagonist + antagonist to get regret, returns
    3. 4-tier probing on adversary h-states
    4. Compare linear vs MLP probes
    """

    @property
    def name(self) -> str:
        return "adversary_policy_extraction"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_levels = self.exp_config("n_levels")
        self.max_steps = self.exp_config("max_steps", 256)
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"Requires PAIRED, got {self.training_method}")

    def collect_data(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """Collect adversary h-states and evaluation metrics.

        Uses run_adversary_rollout() to get real adversary hidden states and
        entropy from the MazeEditor environment, then evaluates protagonist
        and antagonist on the adversary-generated levels.
        """
        n = self.n_levels
        adv_ts = getattr(self.train_state, 'adv_train_state', None)

        if adv_ts is not None:
            # Run adversary through MazeEditor to get levels, h-states, entropy
            rng, adv_rng = jax.random.split(rng)
            adv_result = run_adversary_rollout(self.agent, adv_ts, adv_rng, n)
            levels = adv_result.levels
            adv_hstates = adv_result.final_hstates
            adv_entropy = adv_result.per_level_entropy
        else:
            # Fallback for non-PAIRED checkpoints
            import logging
            logging.getLogger(__name__).warning(
                "No adv_train_state found; falling back to random levels and protagonist h-states as proxy"
            )
            rng, gen_rng = jax.random.split(rng)
            levels = generate_levels(self.agent, gen_rng, n)
            rng, pro_rng = jax.random.split(rng)
            adv_hstates = get_pro_hstates(pro_rng, levels, self, self.max_steps)
            adv_entropy = np.zeros(n)

        # Extract level features
        batch_features = extract_level_features_batch(levels)

        # Get protagonist and antagonist returns + regret on adversary-generated levels
        rng, eval_rng = jax.random.split(rng)
        pro_returns, ant_returns, regrets = get_pro_ant_returns(
            eval_rng, levels, self, self.max_steps
        )

        # Compute difficulty
        rng, diff_rng = jax.random.split(rng)
        difficulties = compute_difficulty(levels, self, diff_rng, self.max_steps)

        self.data = {
            'adv_hstates': adv_hstates,
            'wall_density': np.array(batch_features['wall_density']),
            'goal_distance': np.array(batch_features['goal_distance']),
            'pro_returns': pro_returns,
            'ant_returns': ant_returns,
            'regrets': regrets,
            'difficulties': np.array(difficulties),
            'adv_entropy': np.array(adv_entropy),
            'n_levels': n,
        }
        return self.data

    def analyze(self) -> Dict[str, Any]:
        """4-tier probing on adversary h-states."""
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("Must call collect_data first")

        hstates = self.data['adv_hstates']
        results = {}

        # --- Tier 0: Environment variables ---
        tier0 = {}
        for feat_name in ['wall_density', 'goal_distance']:
            targets = self.data[feat_name]
            comparison = compute_probe_comparison(hstates, targets, task="regression")
            tier0[feat_name] = comparison
        results['tier0_env_variables'] = tier0

        # --- Tier 1: Curriculum dynamics ---
        tier1 = {}
        for feat_name in ['regrets', 'difficulties']:
            targets = self.data[feat_name]
            comparison = compute_probe_comparison(hstates, targets, task="regression")
            tier1[feat_name.rstrip('s')] = comparison
        results['tier1_curriculum_dynamics'] = tier1

        # --- Tier 2: Agent-curriculum interaction ---
        tier2 = {}
        for feat_name in ['pro_returns', 'ant_returns']:
            targets = self.data[feat_name]
            comparison = compute_probe_comparison(hstates, targets, task="regression")
            tier2[feat_name] = comparison
        results['tier2_agent_interaction'] = tier2

        # --- Tier 3: Adversary strategy ---
        tier3 = {}
        # Entropy probe
        targets = self.data['adv_entropy']
        comparison = compute_probe_comparison(hstates, targets, task="regression")
        tier3['adversary_entropy'] = comparison

        # Generation pattern: predict wall_density from adversary hstate
        # (this tells us if the adversary's internal state encodes what it generates)
        targets = self.data['wall_density']
        _, gen_metrics = train_probe(hstates, targets, probe_type="mlp", task="regression")
        tier3['generation_pattern_r2'] = gen_metrics.get('mean_score', 0.0)

        results['tier3_adversary_strategy'] = tier3

        # --- Summary across tiers ---
        summary = {}
        for tier_name, tier_data in results.items():
            tier_r2s = []
            for feat, data in tier_data.items():
                if isinstance(data, dict) and 'linear' in data:
                    tier_r2s.append(data['linear'].get('mean_score', 0.0))
                elif isinstance(data, (int, float)):
                    tier_r2s.append(float(data))
            if tier_r2s:
                summary[tier_name] = {
                    'mean_linear_r2': float(np.mean(tier_r2s)),
                    'max_linear_r2': float(np.max(tier_r2s)),
                }
        results['summary'] = summary

        # Feature regression coefficients (backward compat)
        from sklearn.linear_model import Ridge
        features = np.column_stack([
            self.data['wall_density'],
            self.data['goal_distance'],
        ])
        model = Ridge(alpha=1.0)
        model.fit(features, self.data['regrets'])
        results['regret_feature_coefficients'] = {
            'wall_density': float(model.coef_[0]),
            'goal_distance': float(model.coef_[1]),
        }

        self.results = results
        return results

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize tier-by-tier probe results."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not hasattr(self, 'results') or not self.results:
            return figures

        try:
            # Bar chart of linear R² across all tiers
            tier_names = []
            feat_names = []
            r2_values = []

            for tier_name in ['tier0_env_variables', 'tier1_curriculum_dynamics',
                              'tier2_agent_interaction', 'tier3_adversary_strategy']:
                tier_data = self.results.get(tier_name, {})
                for feat, data in tier_data.items():
                    if isinstance(data, dict) and 'linear' in data:
                        tier_names.append(tier_name.split('_', 1)[0])
                        feat_names.append(feat)
                        r2_values.append(data['linear'].get('mean_score', 0.0))

            if r2_values:
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                x = np.arange(len(feat_names))
                colors = {'tier0': 'green', 'tier1': 'blue',
                          'tier2': 'orange', 'tier3': 'red'}
                bar_colors = [colors.get(t, 'gray') for t in tier_names]
                ax.bar(x, r2_values, color=bar_colors, alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(feat_names, rotation=45, ha='right')
                ax.set_ylabel("Linear Probe R²")
                ax.set_title("Adversary Policy Extraction: 4-Tier Probing")
                ax.axhline(y=0, color='k', linewidth=0.5)
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()

                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                figures["adversary_probing"] = np.asarray(buf)[:, :, :3]
                plt.close(fig)
        except Exception:
            pass

        return figures
