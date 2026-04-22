"""
D2: Belief Revision Detection.

Detect discrete belief revision events and attribute them to adversary actions.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment
from ..utils.paired_helpers import (
    generate_levels,
    extract_level_features_batch,
    get_pro_hstates,
    get_values_from_rollout,
    get_action_distribution,
)


@dataclass
class BeliefRevisionEvent:
    """A detected belief revision event."""
    step: int
    value_change: float
    policy_change: float
    representation_change: float
    adversary_context: Dict[str, Any]
    persistence: float  # How long the change persists


class BeliefRevisionDetectionExperiment(CheckpointExperiment):
    """
    Detect discrete belief revision events and attribute to adversary.

    Protocol:
    1. Track V, π, and representation changes over training
    2. Detect steps where all three exceed 2σ
    3. Analyze adversary context at each event
    4. Measure persistence of changes
    """

    @property
    def name(self) -> str:
        return "belief_revision_detection"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_samples_per_step = self.exp_config("n_samples_per_step")
        self.trajectory_length = self.exp_config("trajectory_length")
        self.detection_window = self.exp_config("detection_window")
        self.sigma_threshold = self.exp_config("sigma_threshold")
        self.hidden_dim = self.exp_config("hidden_dim")
        self._trajectory_data: List[Dict[str, Any]] = []
        self._events: List[BeliefRevisionEvent] = []
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"BeliefRevisionDetectionExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> List[Dict[str, Any]]:
        """Collect trajectory data for event detection."""
        for t in range(self.trajectory_length):
            rng, step_rng = jax.random.split(rng)
            data = self._collect_step_data(step_rng, t)
            self._trajectory_data.append(data)

        return self._trajectory_data

    def _collect_step_data(self, rng: chex.PRNGKey, step: int) -> Dict[str, Any]:
        """Collect data for a single step using real network evaluations."""
        rng, level_rng, h_rng, val_rng, act_rng, diff_rng = jax.random.split(rng, 6)

        # Generate real levels for this step
        levels = generate_levels(self.agent, level_rng, self.n_samples_per_step)

        # Extract level features to characterize adversary context
        batch_features = extract_level_features_batch(levels)
        mean_wall_density = float(batch_features['wall_density'].mean())
        mean_goal_distance = float(batch_features['goal_distance'].mean())

        # Detect level type shifts by comparing current wall density distribution
        # to expected baseline (no artificial spike injection)
        wall_density_std = float(batch_features['wall_density'].std())
        level_type_shift = wall_density_std > 0.15  # High variance indicates mixed types

        from ..utils.paired_helpers import compute_difficulty
        difficulties = compute_difficulty(levels, self, diff_rng)
        adversary_features = {
            'difficulty': float(np.mean(difficulties)),
            'level_type_shift': level_type_shift,
            'wall_density': mean_wall_density,
        }

        # Get real protagonist hidden states
        hstates = get_pro_hstates(h_rng, levels, self)

        # Get real value estimates
        value_matrix = get_values_from_rollout(
            self.train_state, self.agent, levels, val_rng
        )
        values = value_matrix.mean(axis=1)  # Per-level mean value

        # Get real policy logits and entropies
        logits_matrix, entropy_matrix = get_action_distribution(
            self.train_state, self.agent, levels, act_rng
        )
        # Mean logits across timesteps per level, then mean across levels
        policy_logits_mean = logits_matrix.mean(axis=1).mean(axis=0)  # (n_actions,)
        mean_entropy = float(entropy_matrix.mean())

        return {
            'step': step,
            'hstate_mean': hstates.mean(axis=0),
            'hstate_std': hstates.std(axis=0),
            'value_mean': float(values.mean()),
            'value_std': float(values.std()),
            'policy_logits_mean': policy_logits_mean,
            'policy_entropy': mean_entropy,
            'adversary_features': adversary_features,
        }

    def _compute_entropy(self, logits: np.ndarray) -> float:
        """Compute entropy from logits."""
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        return float(-np.sum(probs * np.log(probs + 1e-10)))

    def detect_events(self) -> List[BeliefRevisionEvent]:
        """Detect belief revision events using probe-based detection.

        Instead of z-scores on raw changes, trains Ridge probes on sliding
        baseline windows and detects revision events where probe accuracy
        drops significantly, then measures recovery rate.
        """
        if len(self._trajectory_data) < self.detection_window * 2:
            return []

        from sklearn.linear_model import Ridge

        events = []
        n = len(self._trajectory_data)

        # Collect h-states and difficulty targets
        hstates = np.array([d['hstate_mean'] for d in self._trajectory_data])
        difficulties = np.array([
            d.get('wall_density', d.get('difficulty', 0.0))
            for d in self._trajectory_data
        ])

        # Sliding window probe-based detection
        window = self.detection_window
        accuracy_drop_threshold = 0.20  # Flag revision when accuracy drops > 20%

        for i in range(window, n - window):
            # Baseline window: [i - window, i)
            baseline_h = hstates[i - window:i]
            baseline_y = difficulties[i - window:i]

            # Test window: [i, i + window)
            test_h = hstates[i:i + window]
            test_y = difficulties[i:i + window]

            if np.std(baseline_y) < 1e-8 or np.std(test_y) < 1e-8:
                continue

            try:
                probe = Ridge(alpha=1.0)
                probe.fit(baseline_h, baseline_y)

                # Baseline R² (on training data)
                baseline_r2 = float(probe.score(baseline_h, baseline_y))

                # Test R² (on next window)
                test_r2 = float(probe.score(test_h, test_y))

                accuracy_drop = baseline_r2 - test_r2

                if accuracy_drop > accuracy_drop_threshold:
                    # Measure recovery: how quickly does probe accuracy recover?
                    recovery_steps = 0
                    for j in range(i + 1, min(n - window, i + window * 3)):
                        recovery_h = hstates[j:j + window]
                        recovery_y = difficulties[j:j + window]
                        if np.std(recovery_y) < 1e-8:
                            continue
                        recovery_r2 = float(probe.score(recovery_h, recovery_y))
                        if recovery_r2 >= baseline_r2 * 0.8:
                            recovery_steps = j - i
                            break

                    # Get adversary context
                    adversary_context = self._get_adversary_context(i, window=5)

                    # Measure persistence
                    persistence = self._measure_persistence(i)

                    # Compute change magnitudes for the event
                    curr = self._trajectory_data[i]
                    prev = self._trajectory_data[i - 1]
                    v_change = abs(curr['value_mean'] - prev['value_mean'])
                    p_change = abs(curr['policy_entropy'] - prev['policy_entropy'])
                    r_change = float(np.linalg.norm(curr['hstate_mean'] - prev['hstate_mean']))

                    events.append(BeliefRevisionEvent(
                        step=i,
                        value_change=float(v_change),
                        policy_change=float(p_change),
                        representation_change=float(r_change),
                        adversary_context=adversary_context,
                        persistence=persistence,
                    ))
            except Exception:
                continue

        self._events = events
        return events

    def _get_adversary_context(self, step: int, window: int = 5) -> Dict[str, Any]:
        """Get adversary context around an event."""
        start = max(0, step - window)
        end = min(len(self._trajectory_data), step + window)

        context_data = self._trajectory_data[start:end]

        difficulties = [d['adversary_features']['difficulty'] for d in context_data]
        wall_densities = [d['adversary_features']['wall_density'] for d in context_data]
        type_shifts = [d['adversary_features']['level_type_shift'] for d in context_data]

        return {
            'difficulty_before': float(np.mean(difficulties[:window])) if len(difficulties) > window else 0.0,
            'difficulty_after': float(np.mean(difficulties[window:])) if len(difficulties) > window else 0.0,
            'difficulty_change': float(np.mean(difficulties[window:]) - np.mean(difficulties[:window])) if len(difficulties) > window else 0.0,
            'wall_density_change': float(np.mean(wall_densities[window:]) - np.mean(wall_densities[:window])) if len(wall_densities) > window else 0.0,
            'had_level_type_shift': any(type_shifts),
        }

    def _measure_persistence(self, step: int, horizon: int = 20) -> float:
        """Measure how long a revision persists."""
        if step >= len(self._trajectory_data) - 1:
            return 0.0

        # Get state at event
        event_state = self._trajectory_data[step]['hstate_mean']

        # Track how long it stays different from pre-event
        pre_event_state = self._trajectory_data[max(0, step - 5)]['hstate_mean']
        event_distance = np.linalg.norm(event_state - pre_event_state)

        if event_distance < 1e-10:
            return 0.0

        # Count steps where representation stays at least 50% of event distance
        persistence_count = 0
        for t in range(step + 1, min(step + horizon, len(self._trajectory_data))):
            future_state = self._trajectory_data[t]['hstate_mean']
            distance_from_pre = np.linalg.norm(future_state - pre_event_state)
            if distance_from_pre > event_distance * 0.5:
                persistence_count += 1
            else:
                break

        return float(persistence_count / horizon)

    def analyze(self) -> Dict[str, Any]:
        """Analyze belief revision events."""
        if not self._trajectory_data:
            raise ValueError("Must call collect_data first")

        # Detect events
        events = self.detect_events()

        results = {}

        results['num_revision_events'] = len(events)

        if events:
            results['revision_persistence'] = float(np.mean([e.persistence for e in events]))
            results['mean_value_change'] = float(np.mean([e.value_change for e in events]))
            results['mean_policy_change'] = float(np.mean([e.policy_change for e in events]))
            results['mean_repr_change'] = float(np.mean([e.representation_change for e in events]))

            # Analyze triggers
            results['revision_trigger_features'] = self._analyze_triggers(events)

            # Analyze directions
            results['revision_direction'] = self._analyze_directions(events)

            # Test adversary intentionality
            results['adversary_intentionality'] = self._test_intentionality(events)

            # Event details
            results['events'] = [
                {
                    'step': e.step,
                    'value_change': e.value_change,
                    'policy_change': e.policy_change,
                    'repr_change': e.representation_change,
                    'persistence': e.persistence,
                    'adversary_context': e.adversary_context,
                }
                for e in events
            ]
        else:
            results['revision_persistence'] = 0.0
            results['revision_trigger_features'] = {}
            results['revision_direction'] = {}
            results['adversary_intentionality'] = 0.0
            results['events'] = []

        return results

    def _analyze_triggers(self, events: List[BeliefRevisionEvent]) -> Dict[str, float]:
        """Analyze what triggers revision events."""
        difficulty_changes = []
        wall_density_changes = []
        had_type_shift = []

        for event in events:
            ctx = event.adversary_context
            difficulty_changes.append(ctx['difficulty_change'])
            wall_density_changes.append(ctx['wall_density_change'])
            had_type_shift.append(1.0 if ctx['had_level_type_shift'] else 0.0)

        return {
            'mean_difficulty_change': float(np.mean(difficulty_changes)) if difficulty_changes else 0.0,
            'mean_wall_density_change': float(np.mean(wall_density_changes)) if wall_density_changes else 0.0,
            'type_shift_fraction': float(np.mean(had_type_shift)) if had_type_shift else 0.0,
        }

    def _analyze_directions(self, events: List[BeliefRevisionEvent]) -> Dict[str, float]:
        """Analyze revision directions."""
        value_directions = []
        for event in events:
            if event.step > 0 and event.step < len(self._trajectory_data):
                curr = self._trajectory_data[event.step]['value_mean']
                prev = self._trajectory_data[event.step - 1]['value_mean']
                value_directions.append(1.0 if curr > prev else -1.0)

        return {
            'value_increase_fraction': float(np.mean([1 for d in value_directions if d > 0]) / len(value_directions)) if value_directions else 0.5,
            'value_decrease_fraction': float(np.mean([1 for d in value_directions if d < 0]) / len(value_directions)) if value_directions else 0.5,
        }

    def _test_intentionality(self, events: List[BeliefRevisionEvent]) -> float:
        """Test if revisions are adversary-intentional (correlated with adversary actions)."""
        # Intentionality = fraction of events preceded by adversary difficulty spikes
        intentional_count = 0
        for event in events:
            ctx = event.adversary_context
            # Consider intentional if difficulty increased or type shifted
            if ctx['difficulty_change'] > 0.05 or ctx['had_level_type_shift']:
                intentional_count += 1

        return float(intentional_count / len(events)) if events else 0.0

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize belief revision events."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._trajectory_data:
            return figures

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        steps = [d['step'] for d in self._trajectory_data]
        values = [d['value_mean'] for d in self._trajectory_data]
        entropies = [d['policy_entropy'] for d in self._trajectory_data]
        difficulties = [d['adversary_features']['difficulty'] for d in self._trajectory_data]

        # Value trajectory with events
        ax = axes[0, 0]
        ax.plot(steps, values, 'b-', linewidth=2, label='Value')
        for event in self._events:
            ax.axvline(x=event.step, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title('Value Trajectory with Revision Events')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Policy entropy with events
        ax = axes[0, 1]
        ax.plot(steps, entropies, 'g-', linewidth=2, label='Policy Entropy')
        for event in self._events:
            ax.axvline(x=event.step, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Entropy')
        ax.set_title('Policy Entropy with Revision Events')
        ax.grid(True, alpha=0.3)

        # Adversary difficulty
        ax = axes[1, 0]
        ax.plot(steps, difficulties, 'm-', linewidth=2, label='Adversary Difficulty')
        for event in self._events:
            ax.axvline(x=event.step, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Difficulty')
        ax.set_title('Adversary Difficulty with Revision Events')
        ax.grid(True, alpha=0.3)

        # Event statistics
        ax = axes[1, 1]
        if self._events:
            event_steps = [e.step for e in self._events]
            persistences = [e.persistence for e in self._events]
            ax.bar(range(len(event_steps)), persistences, alpha=0.7)
            ax.set_xticks(range(len(event_steps)))
            ax.set_xticklabels([f'Step {s}' for s in event_steps], rotation=45, ha='right')
            ax.set_ylabel('Persistence')
            ax.set_title(f'Revision Event Persistence (n={len(self._events)})')
        else:
            ax.text(0.5, 0.5, 'No events detected', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Revision Event Persistence')

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["belief_revision"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
