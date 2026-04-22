"""
N-ENV Prediction Experiment — Sequential Horizon Prediction.

Tests how far into the future an agent's hidden state can predict level features.
Given a sequence of levels, trains probes to predict features of level[i+n]
from the hidden state after level[i], for varying horizons n.

Dual-path protocol:
- **prediction_head_agent**: Uses the integrated 4-tier prediction head. The head
  takes curriculum_features (cross-episode history) and outputs Tier 0 predictions
  (wall_logits, goal_logits, etc.) trained end-to-end with auxiliary loss during RL.
  We derive wall_density from sigmoid(wall_logits).mean() and goal_distance from
  argmax(goal_logits), then compute R² against actual level features at each horizon.
- **All other agents**: Train post-hoc linear probes from h-state[i] → features[i+n].

The comparison between head R² and probe R² reveals whether explicit prediction
training during RL captures more about level sequences than what probes extract
from h-states alone. The prediction_head_agent is expected to show slower decay.

Key outputs:
- Per-horizon R² for each feature (wall_density, goal_distance)
- Exponential decay fit: R²(n) = a·exp(-b·n) + c, reporting decay rate b
- Decay curve plot: R² vs horizon n
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional
from scipy.optimize import curve_fit

from .base import CheckpointExperiment
from .utils.paired_helpers import (
    generate_levels,
    extract_level_features_batch,
    run_batched_rollout,
)
from .probes.property_probe import LinearPropertyProbe, train_probe


class NEnvPredictionExperiment(CheckpointExperiment):
    """Sequential horizon prediction across level sequences.

    Protocol:
    1. Generate M sequences of L levels each
    2. Run agent through each sequence, collecting terminal h-states per level
    3a. (prediction_head_agent) Use integrated head to predict next-level features
    3b. (other agents) For each horizon n: train probe from h-state[i] → features[i+n]
    4. Report per-horizon R² and fit exponential decay
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_sequences = self.exp_config("n_sequences")
        self.sequence_length = self.exp_config("sequence_length")
        self.horizons = self.exp_config("horizons")
        self.max_steps = self.exp_config("max_steps")

        # Detect whether this agent has the integrated prediction head
        from .utils.agent_aware_loss import uses_prediction_head
        self._is_prediction_head_agent = uses_prediction_head(self.agent)

        # Verify curriculum_state is available for head path
        self._has_curriculum_state = hasattr(self.train_state, 'curriculum_state') and (
            self.train_state.curriculum_state is not None
        )

    @property
    def name(self) -> str:
        return "n_env_prediction"

    def collect_data(self, rng) -> Dict[str, Any]:
        """Generate level sequences and collect terminal h-states."""
        M = self.n_sequences
        L = self.sequence_length
        total_levels = M * L

        # Generate all levels at once
        rng, gen_rng = jax.random.split(rng)
        all_levels = generate_levels(self.agent, gen_rng, total_levels)

        # Extract features for all levels
        all_features = extract_level_features_batch(all_levels)

        # Run agent on all levels to get terminal h-states
        rng, rollout_rng = jax.random.split(rng)
        pro_ts = getattr(self.train_state, 'pro_train_state', self.train_state)
        result = run_batched_rollout(
            rollout_rng, all_levels, pro_ts, self.agent,
            max_steps=self.max_steps,
            return_final_hstate=True,
        )

        # Get terminal h-states
        from .utils.paired_helpers import get_real_hstates
        rng, hstate_rng = jax.random.split(rng)
        hstates = get_real_hstates(
            hstate_rng, all_levels, pro_ts, self.agent, self.max_steps
        )

        # Reshape into sequences: (M, L, ...)
        hstates_seq = hstates.reshape(M, L, -1)
        features_seq = {
            k: v.reshape(M, L) for k, v in all_features.items()
        }

        data = {
            'hstates_seq': hstates_seq,
            'features_seq': features_seq,
            'n_sequences': M,
            'sequence_length': L,
            'horizons': self.horizons,
            'returns': np.array(result.episode_returns).reshape(M, L),
        }

        # For prediction_head_agent, also collect head predictions
        if self._is_prediction_head_agent and self._has_curriculum_state:
            rng, head_rng = jax.random.split(rng)
            head_preds = self._collect_head_predictions(head_rng, all_levels, pro_ts)
            # head_preds: {'pred_wall_density': (total,), 'pred_goal_distance': (total,)}
            data['head_predictions'] = {
                k: v.reshape(M, L) for k, v in head_preds.items()
            }

        self.data = data
        return self.data

    def _collect_head_predictions(self, rng, all_levels, pro_ts):
        """Run prediction head to get Tier 0 predictions for each level.

        The head takes curriculum_features and outputs wall_logits/goal_logits.
        We derive scalar features matching extract_level_features_batch output:
        - wall_density from sigmoid(wall_logits).mean()
        - goal_distance from argmax(goal_logits) → euclidean to agent_pos
        """
        from ablations.common.curriculum_state import get_curriculum_features

        # Get curriculum features from the checkpoint's curriculum state
        curriculum_features = get_curriculum_features(
            self.train_state.curriculum_state,
            max_training_steps=self.config.get("num_updates", 30000),
            max_buffer_capacity=self.config.get("level_buffer_capacity", 4000),
        )

        # We need a single forward pass with predict_curriculum=True.
        # The head only needs curriculum_features (not per-level obs), so we run
        # one forward pass with dummy obs and extract the predictions.
        dummy_obs = jax.tree_util.tree_map(
            lambda x: x[None, None, ...],
            self.agent.env.reset_to_level(
                jax.random.PRNGKey(0),
                self.agent.sample_random_level(jax.random.PRNGKey(0)),
                self.agent.env_params,
            )[0]
        )
        dummy_dones = jnp.zeros((1, 1), dtype=bool)
        dummy_hidden = self.agent.initialize_hidden_state(1)

        _, _, _, predictions = pro_ts.apply_fn(
            pro_ts.params,
            (dummy_obs, dummy_dones),
            dummy_hidden,
            curriculum_features=curriculum_features,
            predict_curriculum=True,
        )

        if predictions is None:
            return {
                'pred_wall_density': np.zeros(len(np.array(all_levels.wall_map))),
                'pred_goal_distance': np.zeros(len(np.array(all_levels.wall_map))),
            }

        # Derive wall_density from wall_logits
        wall_logits = np.array(predictions['wall_logits'])  # (H, W)
        pred_wall_density = float((1.0 / (1.0 + np.exp(-wall_logits))).mean())

        # Derive goal_distance from goal_logits
        goal_logits = np.array(predictions['goal_logits'])  # (H*W,)
        H = wall_logits.shape[0]
        W = wall_logits.shape[1]
        pred_goal_idx = np.argmax(goal_logits)
        pred_goal_row, pred_goal_col = divmod(int(pred_goal_idx), W)

        # Compute predicted goal distance for each level's actual agent position
        agent_positions = np.array(all_levels.agent_pos)  # (n, 2) or (n,)
        n_levels = agent_positions.shape[0]

        if agent_positions.ndim > 1:
            pred_goal_distances = np.sqrt(
                (agent_positions[:, 0] - pred_goal_row) ** 2 +
                (agent_positions[:, 1] - pred_goal_col) ** 2
            )
        else:
            pred_goal_distances = np.abs(agent_positions - pred_goal_idx).astype(float)

        return {
            'pred_wall_density': np.full(n_levels, pred_wall_density),
            'pred_goal_distance': pred_goal_distances,
        }

    def analyze(self) -> Dict[str, Any]:
        """Analyze: head path for prediction_head_agent, probe path for others."""
        if self._is_prediction_head_agent and 'head_predictions' in self.data:
            return self._analyze_prediction_head()
        return self._analyze_probes()

    def _analyze_prediction_head(self) -> Dict[str, Any]:
        """Compute R² from the integrated prediction head's outputs vs actual features.

        Comparison fairness: The prediction head outputs constant predictions
        (from curriculum_features, not per-level obs), so we compare against
        INITIAL features at position 0 rather than future positions. Comparing
        against position i+n would be unfair since predictions don't vary with
        source index.

        For horizon > 0 we still report R², but the comparison target remains
        position 0 to measure how well the head captures the current curriculum
        distribution, not future levels.
        """
        features_seq = self.data['features_seq']  # {feat: (M, L)}
        head_preds = self.data['head_predictions']  # {pred_feat: (M, L)}
        M, L = features_seq['wall_density'].shape

        feature_map = {
            'wall_density': 'pred_wall_density',
            'goal_distance': 'pred_goal_distance',
        }
        per_horizon_r2 = {feat: {} for feat in feature_map}

        for n in self.horizons:
            if n >= L:
                continue

            source_indices = np.arange(L - n)

            for feat, pred_key in feature_map.items():
                if feat not in features_seq or pred_key not in head_preds:
                    continue

                # Head predictions are constant (same curriculum_features),
                # so compare against INITIAL features at position 0 for fairness.
                actual = features_seq[feat][:, 0:1].repeat(len(source_indices), axis=1).reshape(-1)
                predicted = head_preds[pred_key][:, source_indices].reshape(-1)

                # Compute R²
                ss_res = np.sum((actual - predicted) ** 2)
                ss_tot = np.sum((actual - actual.mean()) ** 2)
                r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
                per_horizon_r2[feat][n] = float(r2)

        # Fit exponential decay
        decay_params = self._fit_decay(per_horizon_r2)

        analysis = {
            'per_horizon_r2': per_horizon_r2,
            'decay_params': decay_params,
            'horizons': self.horizons,
            'n_sequences': self.data['n_sequences'],
            'sequence_length': self.data['sequence_length'],
            'training_method': self.training_method,
            'is_prediction_head_agent': True,
            'prediction_method': 'integrated_head',
        }

        for feat in per_horizon_r2:
            if 1 in per_horizon_r2[feat]:
                analysis[f'{feat}_r2_horizon_1'] = per_horizon_r2[feat][1]
            if decay_params[feat]['fit_success']:
                analysis[f'{feat}_decay_rate'] = decay_params[feat]['b']

        self.results = analysis
        return analysis

    def _analyze_probes(self) -> Dict[str, Any]:
        """Train per-horizon probes and fit decay curve (non-head agents)."""
        hstates_seq = self.data['hstates_seq']  # (M, L, hidden_dim)
        features_seq = self.data['features_seq']  # {feat: (M, L)}
        M, L, hidden_dim = hstates_seq.shape

        feature_names = ['wall_density', 'goal_distance']
        per_horizon_r2 = {feat: {} for feat in feature_names}

        for n in self.horizons:
            if n >= L:
                continue

            source_indices = np.arange(L - n)
            h_source = hstates_seq[:, source_indices, :].reshape(-1, hidden_dim)

            for feat in feature_names:
                if feat not in features_seq:
                    continue
                targets = features_seq[feat][:, source_indices + n].reshape(-1)

                probe, metrics = train_probe(
                    h_source, targets,
                    probe_type="linear",
                    task="regression",
                )
                r2 = metrics.get('mean_score', 0.0)
                per_horizon_r2[feat][n] = float(r2)

        decay_params = self._fit_decay(per_horizon_r2)

        analysis = {
            'per_horizon_r2': per_horizon_r2,
            'decay_params': decay_params,
            'horizons': self.horizons,
            'n_sequences': self.data['n_sequences'],
            'sequence_length': self.data['sequence_length'],
            'training_method': self.training_method,
            'is_prediction_head_agent': self._is_prediction_head_agent,
            'prediction_method': 'probes',
        }

        for feat in per_horizon_r2:
            if 1 in per_horizon_r2[feat]:
                analysis[f'{feat}_r2_horizon_1'] = per_horizon_r2[feat][1]
            if decay_params[feat]['fit_success']:
                analysis[f'{feat}_decay_rate'] = decay_params[feat]['b']

        self.results = analysis
        return analysis

    def _fit_decay(self, per_horizon_r2: dict) -> dict:
        """Fit exponential decay R²(n) = a·exp(-b·n) + c for each feature."""
        decay_params = {}
        for feat in per_horizon_r2:
            horizons_used = sorted(per_horizon_r2[feat].keys())
            if len(horizons_used) < 3:
                decay_params[feat] = {'a': 0.0, 'b': 0.0, 'c': 0.0, 'fit_success': False}
                continue

            x = np.array(horizons_used, dtype=float)
            y = np.array([per_horizon_r2[feat][h] for h in horizons_used])

            def exp_decay(n, a, b, c):
                return a * np.exp(-b * n) + c

            try:
                popt, _ = curve_fit(
                    exp_decay, x, y,
                    p0=[max(y), 0.3, min(y)],
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
                decay_params[feat] = {'a': 0.0, 'b': 0.0, 'c': 0.0, 'fit_success': False}

        return decay_params

    def visualize(self) -> Dict[str, Any]:
        """Create decay curve plot: R² vs horizon n."""
        import matplotlib.pyplot as plt

        viz = {}

        try:
            per_horizon_r2 = self.results['per_horizon_r2']
            decay_params = self.results['decay_params']
            is_head = self.results.get('is_prediction_head_agent', False)
            method = self.results.get('prediction_method', 'probes')

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            colors = {'wall_density': 'blue', 'goal_distance': 'red'}
            markers = {'wall_density': 'o', 'goal_distance': 's'}

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
                            label=f"{feat} fit (b={dp['b']:.2f})")

            method_label = "integrated head" if method == 'integrated_head' else "probes"
            agent_label = f" (prediction_head, {method_label})" if is_head else ""
            ax.set_xlabel("Horizon n (levels ahead)")
            ax.set_ylabel("R²")
            ax.set_title(f"N-ENV: Sequential Horizon Prediction Decay{agent_label}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=-0.05)

            plt.tight_layout()
            viz["decay_curve"] = fig
            plt.close(fig)
        except Exception:
            pass

        self.figures = viz
        return viz
