"""
Cross-Episode Information Flow Experiment.

Tracks how information persists across episode boundaries by:
- Injecting distinctive signals and testing retrieval
- Testing memory capacity (how many episodes back)
- Analyzing selective memory (what is preferentially retained)

PAIRED-specific:
- Test if hidden state encodes adversary generation patterns
- Test if high-regret episodes persist in memory longer
- For persistent LSTM: does h-state at episode K encode adversary features from K-1, K-2?
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
import chex

from .base import CheckpointExperiment
from .utils.memory_probing import (
    create_distinctive_level_pattern,
    test_memory_capacity,
    analyze_selective_memory,
    compute_memory_decay_curve,
)
from .utils.batched_rollout import batched_rollout


@dataclass
class CrossEpisodeData:
    """Container for cross-episode flow data."""
    # Memory capacity test results
    probe_accuracies_by_lag: Dict[int, List[float]] = field(default_factory=dict)

    # Selective memory features
    episode_features: List[Dict[str, float]] = field(default_factory=list)
    retained_in_memory: List[bool] = field(default_factory=list)

    # Hidden state trajectories (store as tuples for proper probe usage)
    hidden_states_over_episodes: List[np.ndarray] = field(default_factory=list)
    hidden_state_tuples: List[Any] = field(default_factory=list)  # Store (h_c, h_h) tuples

    # PAIRED-specific data
    adversary_patterns: List[Dict[str, float]] = field(default_factory=list)  # Adversary strategy features per episode
    regrets: List[float] = field(default_factory=list)  # Regret for each episode
    adversary_pattern_retention_by_lag: Dict[int, List[float]] = field(default_factory=dict)  # How well adversary patterns persist


class CrossEpisodeFlowExperiment(CheckpointExperiment):
    """
    Track information flow across episode boundaries.

    Tests:
    1. Information injection: Can we decode level info from later states?
    2. Memory capacity: How many episodes back can we decode?
    3. Selective memory: What types of episodes are retained?

    Expected differences by agent type:
    - accel_probe: No cross-episode memory (reset between episodes)
    - persistent_lstm: May retain some information
    - episodic_memory: Should show retrieval patterns
    """

    @property
    def name(self) -> str:
        return "cross_episode_flow"

    def __init__(self, **kwargs):
        """
        Initialize cross-episode flow experiment.
        """
        super().__init__(**kwargs)
        self.n_episode_sequences = self.exp_config("n_episode_sequences")
        self.sequence_length = self.exp_config("sequence_length")
        self.max_lag_to_test = min(
            self.exp_config("max_lag_to_test"),
            self.sequence_length - 1,
        )

        self._data: Optional[CrossEpisodeData] = None
        self._results: Dict[str, Any] = {}

    def collect_data(self, rng: chex.PRNGKey) -> CrossEpisodeData:
        """
        Collect cross-episode flow data.
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

        self._data = CrossEpisodeData()
        n_seqs = self.n_episode_sequences
        max_steps = 50  # Short episodes for cross-episode flow

        # Initialize probe accuracy tracking
        for lag in range(1, self.max_lag_to_test + 1):
            self._data.probe_accuracies_by_lag[lag] = []
            if self.has_regret:
                self._data.adversary_pattern_retention_by_lag[lag] = []

        # --- Run all sequences in parallel, episodes within each sequence are sequential ---
        _log("episode_sequences", msg=f"Running {n_seqs} sequences x {self.sequence_length} episodes...")
        t0_total = time.time()

        # Initialize hstate for all sequences in parallel
        hstate = self.agent.initialize_hidden_state(n_seqs)

        # Store per-sequence per-episode data
        all_sequence_data = [[] for _ in range(n_seqs)]

        for ep_idx in tqdm(range(self.sequence_length), desc="Sequential episodes"):
            rng, rng_levels, rng_rollout = jax.random.split(rng, 3)

            # Generate levels for all sequences at this episode index
            _log(f"ep_{ep_idx}/generate", msg=f"Generating levels for episode {ep_idx}...")
            t0 = time.time()
            level_rngs = jax.random.split(rng_levels, n_seqs)
            levels = jax.vmap(self.agent.sample_random_level)(level_rngs)
            jax.block_until_ready(levels)
            _log(f"ep_{ep_idx}/generate", time.time() - t0)

            # Run batched rollout for this episode, carrying hstate from previous episode
            _log(f"ep_{ep_idx}/rollout", msg=f"Running batched rollout for episode {ep_idx}...")
            t0 = time.time()
            result = batched_rollout(
                rng_rollout, levels, max_steps,
                self.train_state.apply_fn, self.train_state.params,
                self.agent.env, self.agent.env_params,
                hstate,  # Carry hstate from previous episode
                collection_steps=[-1],
                return_final_hstate=True,
            )
            jax.block_until_ready(result.episode_returns)
            _log(f"ep_{ep_idx}/rollout", time.time() - t0)

            # Update hstate for next episode (cross-episode persistence)
            hstate = result.final_hstate

            # Extract terminal hstates and episode data for all sequences
            terminal_hstates = result.hstates_by_step["-1"]  # (n_seqs, hidden_dim)
            wall_maps = np.array(levels.wall_map)

            for seq_idx in range(n_seqs):
                wall_density = float(wall_maps[seq_idx].mean())

                # Compute novelty as L2 distance of level features from running mean
                level_feat = np.array([wall_density, float(result.episode_returns[seq_idx])])
                if not hasattr(self, '_running_mean'):
                    self._running_mean = level_feat.copy()
                    self._running_count = 1
                    novelty = 0.0
                else:
                    novelty = float(np.linalg.norm(level_feat - self._running_mean))
                    self._running_count += 1
                    alpha = 1.0 / self._running_count
                    self._running_mean = (1 - alpha) * self._running_mean + alpha * level_feat

                features = {
                    'return': float(result.episode_returns[seq_idx]),
                    'solved': 1.0 if result.episode_solved[seq_idx] else 0.0,
                    'length': int(result.episode_lengths[seq_idx]),
                    'episode_idx': ep_idx,
                    'wall_density': wall_density,
                    'novelty_score': novelty,
                }

                ep_data = {
                    'level': {'wall_density': wall_density},
                    'result': {
                        'total_return': float(result.episode_returns[seq_idx]),
                        'solved': bool(result.episode_solved[seq_idx]),
                        'n_steps': int(result.episode_lengths[seq_idx]),
                    },
                    'features': features,
                    'hidden_state': terminal_hstates[seq_idx].copy(),
                    'pattern_signature': ep_idx,  # Use episode index as signature
                }

                # PAIRED-specific: compute adversary pattern and regret
                if self.has_regret:
                    rng, adv_rng = jax.random.split(rng)
                    adversary_pattern = self._compute_adversary_pattern(
                        {'wall_density': wall_density}, ep_idx, adv_rng
                    )
                    ep_data['adversary_pattern'] = adversary_pattern

                    # Real regret: run antagonist on this level if available
                    ant_ts = getattr(self.train_state, 'ant_train_state', None)
                    if ant_ts is not None:
                        try:
                            rng, ant_rng = jax.random.split(rng)
                            level_i = jax.tree_util.tree_map(lambda x: x[seq_idx:seq_idx+1], levels)
                            ant_r = batched_rollout(
                                ant_rng, level_i, max_steps,
                                ant_ts.apply_fn, ant_ts.params,
                                self.agent.env, self.agent.env_params,
                                self.agent.initialize_hidden_state(1),
                            )
                            regret = float(ant_r.episode_returns[0]) - float(result.episode_returns[seq_idx])
                        except Exception:
                            regret = float('nan')
                    else:
                        regret = float('nan')
                    ep_data['regret'] = regret
                    features['regret'] = regret

                all_sequence_data[seq_idx].append(ep_data)

        _log("episode_sequences", time.time() - t0_total)

        # --- Post-process: test memory at various lags ---
        _log("memory_probing", msg="Testing memory capacity at various lags...")
        t0 = time.time()
        for seq_idx in tqdm(range(n_seqs), desc="Memory probing", leave=False):
            sequence_data = all_sequence_data[seq_idx]

            # Compute lag-1 accuracies first and attach to episode data
            lag1_accuracies = self._test_lag_accuracy(sequence_data, 1)
            for i, acc in enumerate(lag1_accuracies):
                if i < len(sequence_data):
                    sequence_data[i]['probe_accuracy'] = acc

            for lag in range(1, self.max_lag_to_test + 1):
                if lag == 1:
                    self._data.probe_accuracies_by_lag[lag].extend(lag1_accuracies)
                else:
                    accuracies = self._test_lag_accuracy(sequence_data, lag)
                    self._data.probe_accuracies_by_lag[lag].extend(accuracies)

                if self.has_regret:
                    adv_retention = self._test_adversary_pattern_retention(sequence_data, lag)
                    self._data.adversary_pattern_retention_by_lag[lag].extend(adv_retention)

            # Collect episode features for selective memory analysis
            for ep_data in sequence_data:
                self._data.episode_features.append(ep_data['features'])
                retained = ep_data.get('probe_accuracy', 0) > 0.5
                self._data.retained_in_memory.append(retained)

                if self.has_regret:
                    self._data.adversary_patterns.append(ep_data.get('adversary_pattern', {}))
                    self._data.regrets.append(ep_data.get('regret', 0.0))

            # Store final hidden state of sequence
            self._data.hidden_states_over_episodes.append(sequence_data[-1]['hidden_state'])
        _log("memory_probing", time.time() - t0)

        _log("collect_data_done", msg=f"Data collection complete ({n_seqs} sequences x {self.sequence_length} episodes)")
        return self._data

    def _test_lag_accuracy(
        self,
        sequence_data: List[Dict[str, Any]],
        lag: int,
    ) -> List[float]:
        """
        Test probe accuracy at decoding episode info from lag episodes later.

        Uses actual probe/prediction loss to measure how well the SOURCE
        episode's level can be decoded from the PROBE episode's hidden state.
        """
        accuracies = []

        # Get random baseline for normalization
        if not hasattr(self, '_random_baseline_loss'):
            self._random_baseline_loss = self._compute_random_baseline()

        for i in range(len(sequence_data) - lag):
            source_ep = sequence_data[i]
            probe_ep = sequence_data[i + lag]

            # Get hidden state from probe episode
            probe_hstate = probe_ep.get('hidden_state_tuple')
            if probe_hstate is None:
                # Use dense hidden state with Ridge probe to predict source features
                source_hidden = source_ep['hidden_state']
                probe_hidden = probe_ep['hidden_state']

                # Build feature vector from source level
                source_features = source_ep.get('features', {})
                if source_features:
                    feat_vals = np.array([float(v) for v in source_features.values()])
                    # Quick Ridge probe: predict source features from probe hidden state
                    from sklearn.linear_model import Ridge
                    X_train = probe_hidden.reshape(1, -1)
                    y_train = feat_vals.reshape(1, -1)
                    # With single sample, R² isn't meaningful — use reconstruction error
                    pred = X_train @ np.linalg.lstsq(X_train, y_train, rcond=None)[0]
                    recon_error = np.mean((pred - y_train) ** 2)
                    max_var = np.var(feat_vals) + 1e-8
                    accuracy = float(np.clip(1.0 - recon_error / max_var, 0, 1))
                else:
                    accuracy = 0.0
                accuracies.append(accuracy)
                continue

            # Try to decode SOURCE episode's level from PROBE's hidden state
            source_level = source_ep['level']

            # Use actual probe/prediction loss as accuracy measure
            loss = self._compute_decoding_loss(probe_hstate, source_level)

            # Convert loss to accuracy (lower loss = higher accuracy)
            # Normalize relative to random baseline
            accuracy = max(0, 1 - (loss / self._random_baseline_loss))
            accuracies.append(float(accuracy))

        return accuracies

    def _compute_decoding_loss(self, hstate_tuple, level: Dict[str, Any]) -> float:
        """
        Compute probe/prediction loss for decoding level from hidden state.

        Uses the agent-aware dispatch to compute loss correctly for each agent type.
        """
        try:
            from .utils.agent_aware_loss import (
                detect_agent_type,
                create_level_object,
            )
            from ablations.common.metrics import (
                compute_probe_loss,
                compute_curriculum_prediction_loss,
            )

            agent_type = detect_agent_type(self.agent)

            # Flatten hidden state for probe
            h_c, h_h = hstate_tuple
            hstate_flat = jnp.concatenate([
                jnp.array(h_c).reshape(1, -1),
                jnp.array(h_h).reshape(1, -1)
            ], axis=-1)
            hstate_flat = jax.lax.stop_gradient(hstate_flat)

            if agent_type == "next_env_prediction":
                # For prediction head agents, we can't easily compute loss
                # without a forward pass, so fall back to probe-like behavior
                # if probe is available
                pass

            # Apply probe if available
            if hasattr(self.train_state, 'probe_params') and self.train_state.probe_params is not None:
                if hasattr(self.agent, 'probe'):
                    probe = self.agent.probe
                elif hasattr(self.agent, 'curriculum_probe'):
                    probe = self.agent.curriculum_probe
                else:
                    return self._random_baseline_loss

                predictions = probe.apply(
                    self.train_state.probe_params,
                    hstate_flat,
                    episode_return=jnp.zeros(1),
                    episode_solved=jnp.zeros(1),
                    episode_length=jnp.ones(1) * 50,
                )

                level_obj = create_level_object(level)
                loss, _ = compute_probe_loss(predictions, level_obj)
                return float(loss)
            else:
                return self._random_baseline_loss

        except Exception as e:
            return self._random_baseline_loss

    def _compute_random_baseline(self) -> float:
        """Compute random baseline loss for normalization."""
        try:
            from .utils.agent_aware_loss import compute_random_baseline_loss
            return compute_random_baseline_loss()
        except Exception:
            return 10.0  # Fallback default

    def _compute_adversary_pattern(
        self,
        level: Dict[str, Any],
        episode_idx: int,
        rng: chex.PRNGKey,
    ) -> Dict[str, float]:
        """Compute adversary generation pattern features from real network data (PAIRED).

        Uses adversary network forward pass to get action entropy and value estimate.
        Falls back to structural features if adversary is unavailable.
        """
        wall_density = level['wall_density']

        adv_ts = getattr(self.train_state, 'adv_train_state', None)
        if adv_ts is not None:
            try:
                from .utils.agent_aware_loss import create_observation_from_level
                obs = create_observation_from_level(level)
                hstate = self.agent.initialize_hidden_state(1)
                obs_batch = type(obs)(obs.image[None, None, ...], obs.agent_dir[None, None, ...])
                done_batch = jnp.zeros((1, 1), dtype=bool)

                outputs = adv_ts.apply_fn(adv_ts.params, (obs_batch, done_batch), hstate)
                if len(outputs) == 4:
                    _, pi, value, _ = outputs
                else:
                    _, pi, value = outputs

                action_entropy = float(pi.entropy()[0, 0]) if hasattr(pi, 'entropy') else 0.5
                value_est = float(value[0, 0])

                return {
                    'action_entropy': action_entropy,
                    'value_estimate': value_est,
                    'wall_density': wall_density,
                    'episode_in_curriculum': float(episode_idx),
                }
            except Exception:
                pass

        # Fallback: structural features only
        return {
            'action_entropy': 0.5,  # Unknown
            'value_estimate': wall_density,
            'wall_density': wall_density,
            'episode_in_curriculum': float(episode_idx),
        }

    def _test_adversary_pattern_retention(
        self,
        sequence_data: List[Dict[str, Any]],
        lag: int,
    ) -> List[float]:
        """
        Test how well adversary patterns are retained in hidden state (PAIRED).

        For each episode pair (source, probe) separated by lag episodes,
        test whether the probe episode's hidden state encodes information
        about the source episode's adversary pattern.
        """
        retention_scores = []

        for i in range(len(sequence_data) - lag):
            source_ep = sequence_data[i]
            probe_ep = sequence_data[i + lag]

            source_pattern = source_ep.get('adversary_pattern', {})
            probe_hstate = probe_ep['hidden_state']

            if not source_pattern:
                continue

            # Compute retention score
            # Method: correlation between hidden state dimensions and pattern features
            retention = self._compute_pattern_hstate_correlation(
                probe_hstate, source_pattern
            )
            retention_scores.append(retention)

        return retention_scores

    def _compute_pattern_hstate_correlation(
        self,
        hstate: np.ndarray,
        pattern: Dict[str, float],
    ) -> float:
        """
        Compute correlation between hidden state and adversary pattern features.

        Uses Ridge probe: fits a linear model predicting pattern features from
        the full hidden state, returns R^2 as the encoding score.
        """
        pattern_values = np.array(list(pattern.values()))

        # If we have collected enough hstate-pattern pairs, use Ridge probe
        if hasattr(self, '_pattern_hstates') and len(self._pattern_hstates) >= 5:
            from sklearn.linear_model import Ridge
            X = np.array(self._pattern_hstates)
            y = np.array(self._pattern_values)
            try:
                model = Ridge(alpha=1.0)
                model.fit(X, y)
                score = model.score(X, y)
                return float(max(score, 0.0))
            except Exception:
                pass

        # Collect hstate-pattern pairs for future Ridge probe
        if not hasattr(self, '_pattern_hstates'):
            self._pattern_hstates = []
            self._pattern_values = []
        self._pattern_hstates.append(hstate.copy())
        self._pattern_values.append(pattern_values.copy())

        # Fallback for first few samples: simple correlation
        self._used_fallback_scoring = True
        if np.std(hstate) < 1e-6 or np.std(pattern_values) < 1e-6:
            return 0.0

        # Use full hstate correlation with pattern as simple fallback
        n_dims = min(len(pattern_values), len(hstate))
        correlation = np.corrcoef(hstate[:n_dims], pattern_values[:n_dims])[0, 1]
        return float(np.abs(correlation)) if not np.isnan(correlation) else 0.0

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze cross-episode information flow.
        """
        if self._data is None:
            raise ValueError("Must call collect_data before analyze")

        results = {}

        # 1. Memory capacity analysis
        results['memory_capacity'] = self._analyze_memory_capacity()

        # 2. Memory decay curve
        results['decay_curve'] = self._analyze_decay_curve()

        # 3. Selective memory analysis
        results['selective_memory'] = self._analyze_selective_memory()

        # 4. Agent type predictions
        results['agent_type_predictions'] = self._predict_agent_behavior()

        # 5. PAIRED-specific: adversary pattern retention analysis
        if self.has_regret and self._data.adversary_pattern_retention_by_lag:
            results['adversary_pattern_retention'] = self._analyze_adversary_pattern_retention()
            results['regret_memory_relationship'] = self._analyze_regret_memory_relationship()

        # Flag if fallback correlation scoring was used instead of Ridge probe
        if getattr(self, '_used_fallback_scoring', False):
            results['encoding_score_is_fallback'] = True

        self._results = results
        return results

    def _analyze_adversary_pattern_retention(self) -> Dict[str, Any]:
        """
        Analyze how well adversary generation patterns are retained across episodes (PAIRED).

        Key question: Does the hidden state at episode K encode adversary features from K-1, K-2, etc.?
        """
        results = {}

        for lag, retention_scores in self._data.adversary_pattern_retention_by_lag.items():
            if not retention_scores:
                continue

            mean_retention = float(np.mean(retention_scores))
            std_retention = float(np.std(retention_scores))

            # Is retention above chance (random correlation ~0)?
            above_chance = mean_retention > 0.15

            results[f'lag_{lag}'] = {
                'mean_retention': mean_retention,
                'std_retention': std_retention,
                'above_chance': above_chance,
                'n_samples': len(retention_scores),
            }

        # Compute adversary pattern memory horizon
        adv_memory_horizon = 0
        for lag in sorted(self._data.adversary_pattern_retention_by_lag.keys()):
            scores = self._data.adversary_pattern_retention_by_lag[lag]
            if scores and np.mean(scores) > 0.15:
                adv_memory_horizon = lag
            else:
                break

        results['adversary_pattern_horizon'] = adv_memory_horizon

        # Compare to level feature retention
        level_memory_horizon = self._results.get('memory_capacity', {}).get('memory_horizon', 0)
        results['adversary_vs_level_horizon'] = {
            'adversary_horizon': adv_memory_horizon,
            'level_horizon': level_memory_horizon,
            'adversary_patterns_persist_longer': adv_memory_horizon > level_memory_horizon,
        }

        return results

    def _analyze_regret_memory_relationship(self) -> Dict[str, Any]:
        """
        Analyze whether high-regret episodes persist in memory longer (PAIRED).
        """
        if len(self._data.regrets) < 20:
            return {'error': 'Insufficient regret data'}

        regrets = np.array(self._data.regrets)
        retained = np.array(self._data.retained_in_memory)

        # Split by regret tercile
        regret_33 = np.percentile(regrets, 33)
        regret_66 = np.percentile(regrets, 66)

        results = {}

        for name, mask in [
            ('low_regret', regrets <= regret_33),
            ('medium_regret', (regrets > regret_33) & (regrets <= regret_66)),
            ('high_regret', regrets > regret_66),
        ]:
            if mask.sum() < 5:
                continue

            retention_rate = float(retained[mask].mean())
            results[name] = {
                'retention_rate': retention_rate,
                'mean_regret': float(regrets[mask].mean()),
                'n_episodes': int(mask.sum()),
            }

        # Key finding: Do high-regret episodes persist longer?
        if 'high_regret' in results and 'low_regret' in results:
            high_retention = results['high_regret']['retention_rate']
            low_retention = results['low_regret']['retention_rate']

            results['high_regret_persists_longer'] = high_retention > low_retention
            results['retention_difference'] = float(high_retention - low_retention)

            # Correlation between regret and retention
            if len(regrets) > 10:
                from scipy.stats import pearsonr
                corr, p_value = pearsonr(regrets, retained.astype(float))
                results['regret_retention_correlation'] = {
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                }

        return results

    def _analyze_memory_capacity(self) -> Dict[str, Any]:
        """Analyze memory capacity at different lags."""
        capacity_results = {}

        for lag, accuracies in self._data.probe_accuracies_by_lag.items():
            if accuracies:
                mean_acc = float(np.mean(accuracies))
                std_acc = float(np.std(accuracies))
                above_chance = mean_acc > 0.55  # Slightly above 0.5

                capacity_results[f'lag_{lag}'] = {
                    'mean_accuracy': mean_acc,
                    'std_accuracy': std_acc,
                    'above_chance': above_chance,
                    'n_samples': len(accuracies),
                }

        # Estimate memory horizon
        memory_horizon = 0
        for lag in sorted(self._data.probe_accuracies_by_lag.keys()):
            accuracies = self._data.probe_accuracies_by_lag[lag]
            if accuracies and np.mean(accuracies) > 0.55:
                memory_horizon = lag
            else:
                break

        capacity_results['memory_horizon'] = memory_horizon

        return capacity_results

    def _analyze_decay_curve(self) -> Dict[str, Any]:
        """Analyze memory decay over lag."""
        lags = []
        accuracies = []

        for lag, acc_list in sorted(self._data.probe_accuracies_by_lag.items()):
            if acc_list:
                lags.append(lag)
                accuracies.append(np.mean(acc_list))

        if len(lags) < 2:
            return {'error': 'Insufficient lags for decay curve'}

        decay_result = compute_memory_decay_curve(accuracies, lags)

        return {
            'lags': lags,
            'accuracies': accuracies,
            'decay_rate': decay_result.get('decay_rate', 0.0),
            'half_life': decay_result.get('half_life', float('inf')),
            'fit_r2': decay_result.get('fit_r2', 0.0),
        }

    def _analyze_selective_memory(self) -> Dict[str, Any]:
        """Analyze what types of episodes are preferentially retained."""
        if len(self._data.episode_features) < 10:
            return {'error': 'Insufficient episodes for selective memory analysis'}

        selective_result = analyze_selective_memory(
            self._data.episode_features,
            self._data.retained_in_memory,
        )

        return selective_result

    def _predict_agent_behavior(self) -> Dict[str, Any]:
        """Predict expected behavior by agent type."""
        # Get overall memory retention
        all_accuracies = []
        for acc_list in self._data.probe_accuracies_by_lag.values():
            all_accuracies.extend(acc_list)

        mean_retention = np.mean(all_accuracies) if all_accuracies else 0.5

        return {
            'mean_retention': float(mean_retention),
            'predictions': {
                'accel_probe': 'No cross-episode memory expected (accuracy ≈ 0.5)',
                'persistent_lstm': 'May show retention if memory accumulates',
                'episodic_memory': 'Should show retrieval if buffer accessed',
            },
            'observed_behavior': (
                'Strong retention' if mean_retention > 0.7
                else ('Weak retention' if mean_retention > 0.55
                      else 'No significant retention')
            ),
        }

    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data."""
        if not self._results:
            raise ValueError("Must call analyze before visualize")

        viz_data = {
            'memory_capacity': self._results.get('memory_capacity', {}),
            'decay_curve': self._results.get('decay_curve', {}),
            'selective_memory': self._results.get('selective_memory', {}),
        }

        return viz_data
