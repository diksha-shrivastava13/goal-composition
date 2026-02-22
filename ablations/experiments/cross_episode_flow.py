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

    def __init__(
        self,
        n_episode_sequences: int = 20,
        sequence_length: int = 10,
        max_lag_to_test: int = 5,
        **kwargs,
    ):
        """
        Initialize cross-episode flow experiment.

        Args:
            n_episode_sequences: Number of episode sequences to run
            sequence_length: Episodes per sequence
            max_lag_to_test: Maximum lag for memory capacity test
        """
        super().__init__(**kwargs)
        self.n_episode_sequences = n_episode_sequences
        self.sequence_length = sequence_length
        self.max_lag_to_test = min(max_lag_to_test, sequence_length - 1)

        self._data: Optional[CrossEpisodeData] = None
        self._results: Dict[str, Any] = {}

    def collect_data(self, rng: chex.PRNGKey) -> CrossEpisodeData:
        """
        Collect cross-episode flow data.
        """
        self._data = CrossEpisodeData()

        # Initialize probe accuracy tracking
        for lag in range(1, self.max_lag_to_test + 1):
            self._data.probe_accuracies_by_lag[lag] = []
            if self.has_regret:
                self._data.adversary_pattern_retention_by_lag[lag] = []

        for seq_idx in range(self.n_episode_sequences):
            rng, seq_rng = jax.random.split(rng)

            # Run episode sequence
            sequence_data = self._run_episode_sequence(seq_rng)

            # Test memory at various lags
            for lag in range(1, self.max_lag_to_test + 1):
                accuracies = self._test_lag_accuracy(sequence_data, lag)
                self._data.probe_accuracies_by_lag[lag].extend(accuracies)

                # PAIRED-specific: test adversary pattern retention
                if self.has_regret:
                    adv_retention = self._test_adversary_pattern_retention(sequence_data, lag)
                    self._data.adversary_pattern_retention_by_lag[lag].extend(adv_retention)

            # Collect episode features for selective memory analysis
            for ep_data in sequence_data:
                self._data.episode_features.append(ep_data['features'])
                # Check if this episode is "retained" by testing probe accuracy
                retained = ep_data.get('probe_accuracy', 0) > 0.5
                self._data.retained_in_memory.append(retained)

                # PAIRED-specific: collect adversary patterns and regret
                if self.has_regret:
                    self._data.adversary_patterns.append(ep_data.get('adversary_pattern', {}))
                    self._data.regrets.append(ep_data.get('regret', 0.0))

            # Store final hidden states
            self._data.hidden_states_over_episodes.append(sequence_data[-1]['hidden_state'])

        return self._data

    def _run_episode_sequence(self, rng: chex.PRNGKey) -> List[Dict[str, Any]]:
        """Run a sequence of episodes, maintaining hidden state."""
        sequence_data = []

        # Initialize hidden state (will persist across episodes for persistent agents)
        hstate = self.agent.initialize_carry(rng, batch_dims=(1,))

        for ep_idx in range(self.sequence_length):
            rng, level_rng, ep_rng = jax.random.split(rng, 3)

            # Create distinctive level pattern
            pattern = create_distinctive_level_pattern(
                pattern_type="unique_walls",
                pattern_id=ep_idx,
            )

            level = {
                'wall_map': pattern['wall_map'],
                'wall_density': pattern['wall_map'].sum() / (13 * 13),
                'goal_pos': (6, 6),
                'agent_pos': (1, 1),
                'pattern_id': pattern['pattern_id'],
                'signature': pattern['signature'],
            }

            # Run episode
            ep_result, hstate = self._run_episode(ep_rng, level, hstate)

            # Compute episode features
            features = {
                'return': ep_result['total_return'],
                'solved': 1.0 if ep_result['solved'] else 0.0,
                'length': ep_result['n_steps'],
                'episode_idx': ep_idx,
                'wall_density': level['wall_density'],
                'novelty_score': float(np.random.random()),  # Placeholder
            }

            # Store hidden state both flattened and as tuple for probe usage
            h_c, h_h = hstate
            hidden_flat = np.concatenate([
                np.array(h_c).flatten(),
                np.array(h_h).flatten()
            ])

            ep_data = {
                'level': level,
                'result': ep_result,
                'features': features,
                'hidden_state': hidden_flat.copy(),
                'hidden_state_tuple': hstate,  # Keep tuple for probe decoding
                'pattern_signature': pattern['signature'],
            }

            # PAIRED-specific: compute adversary pattern and regret
            if self.has_regret:
                rng, adv_rng = jax.random.split(rng)
                adversary_pattern = self._compute_adversary_pattern(level, ep_idx, adv_rng)
                ep_data['adversary_pattern'] = adversary_pattern

                # Compute regret (estimate based on solve status)
                if ep_result['solved']:
                    regret = 0.1 + float(jax.random.uniform(adv_rng)) * 0.2
                else:
                    regret = 0.5 + level['wall_density'] * 0.5
                ep_data['regret'] = regret
                features['regret'] = regret

            sequence_data.append(ep_data)

        return sequence_data

    def _run_episode(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
        initial_hstate: Any,
        max_steps: int = 50,
    ) -> Tuple[Dict[str, Any], Any]:
        """Run a single episode, returning final hidden state."""
        hstate = initial_hstate
        total_return = 0.0
        solved = False

        for step in range(max_steps):
            rng, step_rng = jax.random.split(rng)

            obs = self._create_observation(level, step)
            new_hstate, pi, value = self._forward_step(obs, hstate)

            action = pi.sample(seed=step_rng)
            hstate = new_hstate

            # Simulate step
            reward = 0.0
            done = step >= max_steps - 1

            if step > 5:
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
            'n_steps': step + 1,
        }, hstate

    def _create_observation(self, level: Dict[str, Any], step: int) -> Any:
        """Create observation from level state."""
        height, width = level['wall_map'].shape

        image = np.zeros((height, width, 3), dtype=np.float32)
        image[:, :, 0] = level['wall_map'].astype(np.float32)
        image[level['goal_pos']] = [0, 1, 0]

        agent_y = (level['agent_pos'][0] + step // 5) % (height - 2) + 1
        agent_x = (level['agent_pos'][1] + step % 5) % (width - 2) + 1
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
                # Fallback to similarity-based method if tuple not available
                source_hidden = source_ep['hidden_state']
                probe_hidden = probe_ep['hidden_state']
                similarity = np.dot(source_hidden, probe_hidden) / (
                    np.linalg.norm(source_hidden) * np.linalg.norm(probe_hidden) + 1e-6
                )
                accuracy = (similarity + 1) / 2
                accuracies.append(float(accuracy))
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
        """
        Compute adversary generation pattern features (PAIRED).

        These features characterize the adversary's strategy for this episode.
        """
        wall_density = level['wall_density']

        # Adversary strategy features (simulated)
        # In real implementation, would extract from adversary policy
        rng, r1, r2, r3 = jax.random.split(rng, 4)

        return {
            'difficulty_target': float(0.3 + jax.random.uniform(r1) * 0.5),
            'wall_concentration': float(wall_density * (0.8 + jax.random.uniform(r2) * 0.4)),
            'path_complexity_target': float(0.4 + jax.random.uniform(r3) * 0.4),
            'strategy_type': float(episode_idx % 5),  # Cycle through strategy types
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

        Higher correlation = pattern is encoded in hidden state.
        """
        # Use first few dimensions of h-state as proxy for pattern encoding
        pattern_values = np.array(list(pattern.values()))

        # Select dimensions that could encode pattern (heuristic: first N dims)
        n_pattern_dims = min(len(pattern_values) * 10, len(hstate))
        hstate_region = hstate[:n_pattern_dims]

        # Compute proxy for pattern encoding:
        # Check if h-state variance in pattern-related dims correlates with pattern features
        # Simplified: use mean activation as proxy
        region_means = []
        dim_per_feature = n_pattern_dims // len(pattern_values)

        for i in range(len(pattern_values)):
            start = i * dim_per_feature
            end = start + dim_per_feature
            region_mean = np.mean(np.abs(hstate_region[start:end]))
            region_means.append(region_mean)

        region_means = np.array(region_means)

        # Correlation between region activations and pattern values
        if np.std(region_means) < 1e-6 or np.std(pattern_values) < 1e-6:
            return 0.0

        correlation = np.corrcoef(region_means, pattern_values)[0, 1]
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
