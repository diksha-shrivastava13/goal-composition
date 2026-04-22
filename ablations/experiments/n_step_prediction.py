"""
N-STEP Prediction Experiment — Within-Episode N-Step Prediction.

Tests how far ahead within an episode the agent's hidden state can predict
future observations. Trains probes from h-state[t] → obs-features[t+n]
for varying n-step horizons.

Different from N-ENV:
- N-ENV: predicts features of future *levels* (across-episode, curriculum horizon)
- N-STEP: predicts features of future *timesteps* within a single episode

Key outputs:
- Per-horizon R² within episode
- Within-episode prediction decay curve
"""

import jax
import numpy as np
from typing import Dict, Any, Optional
from scipy.optimize import curve_fit

from .base import CheckpointExperiment
from .utils.paired_helpers import generate_levels
from .utils.batched_rollout import batched_rollout
from .probes.property_probe import train_probe


class NStepPredictionExperiment(CheckpointExperiment):
    """Within-episode n-step prediction.

    Protocol:
    1. Generate M levels, run full rollouts collecting h-states at all timesteps
    2. Collect per-step observation features (rewards, done flags)
    3. For each n in {1, 5, 10, 25}: pair h-state[t] with obs-features[t+n]
    4. Train probes and measure within-episode prediction decay
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_levels = self.exp_config("n_levels")
        self.max_steps = self.exp_config("max_steps")
        self.horizons = self.exp_config("horizons")

    @property
    def name(self) -> str:
        return "n_step_prediction"

    def collect_data(self, rng) -> Dict[str, Any]:
        """Run full rollouts collecting h-states and rewards at all timesteps."""
        n = self.n_levels
        max_steps = self.max_steps

        # Generate levels
        rng, gen_rng = jax.random.split(rng)
        levels = generate_levels(self.agent, gen_rng, n)

        # Run rollout collecting h-states at every timestep + rewards
        rng, rollout_rng = jax.random.split(rng)
        pro_ts = getattr(self.train_state, 'pro_train_state', self.train_state)

        collection_steps = list(range(1, max_steps + 1))
        result = batched_rollout(
            rollout_rng, levels, max_steps,
            pro_ts.apply_fn, pro_ts.params,
            self.agent.env, self.agent.env_params,
            self.agent.initialize_hidden_state(n),
            collect_values=True,
            collect_rewards=True,
            collection_steps=collection_steps,
        )

        # Build h-states matrix: (n_levels, max_steps, hidden_dim)
        # hstates_by_step has keys "1", "2", ..., "max_steps"
        hstates_list = []
        for t in range(1, max_steps + 1):
            key = str(t)
            if key in result.hstates_by_step:
                hstates_list.append(result.hstates_by_step[key])
            else:
                break
        T = len(hstates_list)

        if T == 0:
            # Fallback: use terminal only
            self.data = {
                'error': 'No per-step h-states collected',
                'horizons': self.horizons,
            }
            return self.data

        hstates_matrix = np.stack(hstates_list, axis=1)  # (n_levels, T, hidden_dim)

        # Obs features at each timestep: values and rewards
        # values shape: (n_levels, max_steps), rewards shape: (n_levels, max_steps)
        values = result.values if result.values is not None else np.zeros((n, max_steps))
        rewards = result.rewards if result.rewards is not None else np.zeros((n, max_steps))

        self.data = {
            'hstates_matrix': hstates_matrix,
            'values': values[:, :T],
            'rewards': rewards[:, :T],
            'episode_lengths': np.array(result.episode_lengths),
            'episode_returns': np.array(result.episode_returns),
            'n_levels': n,
            'T': T,
            'horizons': self.horizons,
        }
        return self.data

    def analyze(self) -> Dict[str, Any]:
        """Train per-horizon probes for within-episode prediction."""
        if 'error' in self.data:
            self.results = self.data
            return self.results

        hstates = self.data['hstates_matrix']  # (n_levels, T, hidden_dim)
        values = self.data['values']            # (n_levels, T)
        rewards = self.data['rewards']          # (n_levels, T)
        ep_lengths = self.data['episode_lengths']
        n_levels, T, hidden_dim = hstates.shape

        feature_targets = {
            'value': values,
            'reward': rewards,
        }

        per_horizon_r2 = {feat: {} for feat in feature_targets}

        for n_step in self.horizons:
            if n_step >= T:
                continue

            # Collect valid (h[t], target[t+n]) pairs across all levels
            h_all = []
            targets_all = {feat: [] for feat in feature_targets}

            for i in range(n_levels):
                ep_len = min(int(ep_lengths[i]), T)
                valid_t = ep_len - n_step
                if valid_t <= 0:
                    continue

                h_all.append(hstates[i, :valid_t, :])
                for feat, feat_matrix in feature_targets.items():
                    targets_all[feat].append(feat_matrix[i, n_step:n_step + valid_t])

            if not h_all:
                continue

            h_concat = np.concatenate(h_all, axis=0)
            for feat in feature_targets:
                t_concat = np.concatenate(targets_all[feat], axis=0)

                # Filter out NaN targets
                valid_mask = np.isfinite(t_concat)
                if valid_mask.sum() < 20:
                    continue

                probe, metrics = train_probe(
                    h_concat[valid_mask],
                    t_concat[valid_mask],
                    probe_type="linear",
                    task="regression",
                )
                per_horizon_r2[feat][n_step] = float(metrics.get('mean_score', 0.0))

        # Fit exponential decay for each feature
        decay_params = {}
        for feat in feature_targets:
            horizons_used = sorted(per_horizon_r2[feat].keys())
            if len(horizons_used) < 3:
                decay_params[feat] = {'b': 0.0, 'fit_success': False}
                continue

            x = np.array(horizons_used, dtype=float)
            y = np.array([per_horizon_r2[feat][h] for h in horizons_used])

            def exp_decay(n, a, b, c):
                return a * np.exp(-b * n) + c

            try:
                popt, _ = curve_fit(
                    exp_decay, x, y,
                    p0=[max(y), 0.1, min(y)],
                    bounds=([0, 0, -1], [2, 10, 1]),
                    maxfev=5000,
                )
                decay_params[feat] = {
                    'a': float(popt[0]),
                    'b': float(popt[1]),
                    'c': float(popt[2]),
                    'fit_success': True,
                }
            except (RuntimeError, ValueError):
                decay_params[feat] = {'b': 0.0, 'fit_success': False}

        analysis = {
            'per_horizon_r2': per_horizon_r2,
            'decay_params': decay_params,
            'horizons': self.horizons,
            'n_levels': self.data['n_levels'],
            'T': self.data['T'],
            'training_method': self.training_method,
        }

        # Summary
        for feat in feature_targets:
            if 1 in per_horizon_r2[feat]:
                analysis[f'{feat}_r2_horizon_1'] = per_horizon_r2[feat][1]
            if decay_params[feat].get('fit_success'):
                analysis[f'{feat}_decay_rate'] = decay_params[feat]['b']

        self.results = analysis
        return analysis

    def visualize(self) -> Dict[str, Any]:
        """Create within-episode prediction decay plot."""
        import matplotlib.pyplot as plt

        viz = {}

        try:
            per_horizon_r2 = self.results['per_horizon_r2']
            decay_params = self.results['decay_params']

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            colors = {'value': 'blue', 'reward': 'green'}
            markers = {'value': 'o', 'reward': 's'}

            for feat in per_horizon_r2:
                horizons_used = sorted(per_horizon_r2[feat].keys())
                if not horizons_used:
                    continue
                x = np.array(horizons_used)
                y = np.array([per_horizon_r2[feat][h] for h in horizons_used])

                ax.plot(x, y, f'-{markers.get(feat, "o")}',
                        color=colors.get(feat, 'gray'),
                        label=feat, markersize=6)

                dp = decay_params.get(feat, {})
                if dp.get('fit_success'):
                    x_fit = np.linspace(min(x), max(x), 50)
                    y_fit = dp['a'] * np.exp(-dp['b'] * x_fit) + dp['c']
                    ax.plot(x_fit, y_fit, '--',
                            color=colors.get(feat, 'gray'), alpha=0.5,
                            label=f"{feat} fit (b={dp['b']:.3f})")

            ax.set_xlabel("N-step lookahead (timesteps)")
            ax.set_ylabel("Probe R²")
            ax.set_title("N-STEP: Within-Episode Prediction Decay")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=-0.05)

            plt.tight_layout()
            viz["within_episode_decay"] = fig
            plt.close(fig)
        except Exception:
            pass

        self.figures = viz
        return viz
