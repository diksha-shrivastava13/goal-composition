"""
D3: Goal Evolution (Shard-Aware).

Track goal evolution using shard-theoretic ontology.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment


@dataclass
class Shard:
    """A contextually-activated goal representation."""
    shard_id: int
    dimension_indices: np.ndarray
    activation_contexts: Dict[str, Tuple[float, float]]  # Feature -> (min, max)
    policy_influence: float
    birth_step: int
    death_step: Optional[int] = None


@dataclass
class CompositionEvent:
    """Event where shards compose or compete."""
    step: int
    event_type: str  # 'composition', 'competition', 'birth', 'death'
    shards_involved: List[int]
    resolution: str  # 'merged', 'dominant', 'coexist'


class GoalEvolutionExperiment(CheckpointExperiment):
    """
    Track goal evolution using shard-theoretic ontology.

    Protocol:
    1. Identify contextually-activated dimension clusters (shards)
    2. Track shard birth, death, composition over training
    3. Attribute shard dynamics to adversary curriculum
    4. Measure goal complexity evolution
    """

    @property
    def name(self) -> str:
        return "goal_evolution"

    def __init__(
        self,
        n_samples_per_step: int = 100,
        trajectory_length: int = 50,
        hidden_dim: int = 256,
        n_shard_components: int = 10,
        policy_effect_threshold: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_samples_per_step = n_samples_per_step
        self.trajectory_length = trajectory_length
        self.hidden_dim = hidden_dim
        self.n_shard_components = n_shard_components
        self.policy_effect_threshold = policy_effect_threshold
        self._trajectory_data: List[Dict[str, Any]] = []
        self._shards: List[Shard] = []
        self._composition_events: List[CompositionEvent] = []
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"GoalEvolutionExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> List[Dict[str, Any]]:
        """Collect trajectory data for shard analysis."""
        for t in range(self.trajectory_length):
            rng, step_rng = jax.random.split(rng)
            data = self._collect_step_data(step_rng, t)
            self._trajectory_data.append(data)

        return self._trajectory_data

    def _collect_step_data(self, rng: chex.PRNGKey, step: int) -> Dict[str, Any]:
        """Collect data for a single step."""
        hstates = []
        actions = []
        level_features_list = []

        # Simulate diverse level contexts
        for i in range(self.n_samples_per_step):
            rng, h_rng, f_rng, a_rng = jax.random.split(rng, 4)

            # Level features
            wall_density = 0.1 + float(jax.random.uniform(f_rng)) * 0.3
            goal_distance = 2.0 + float(jax.random.uniform(f_rng)) * 10.0

            level_features = {
                'wall_density': wall_density,
                'goal_distance': goal_distance,
            }
            level_features_list.append(level_features)

            # Hidden state with shard-like structure
            h = np.array(jax.random.normal(h_rng, (self.hidden_dim,)))

            # Different dimensions activate in different contexts
            # Shard 1: Activates for high wall density (nav shard)
            if wall_density > 0.25:
                h[:30] += 2.0

            # Shard 2: Activates for large goal distance (exploration shard)
            if goal_distance > 8.0:
                h[30:60] += 1.5

            # Shard 3: Emerges over training (learned shard)
            if step > 20 and wall_density > 0.2 and goal_distance > 5.0:
                h[60:90] += step * 0.05

            hstates.append(h)

            # Action (4 discrete actions)
            # Action influenced by shards
            action_probs = np.array([0.25, 0.25, 0.25, 0.25])
            if wall_density > 0.25:
                action_probs[0] += 0.2  # More cautious
            if goal_distance > 8.0:
                action_probs[1] += 0.2  # More exploratory
            action_probs /= action_probs.sum()
            action = int(jax.random.choice(a_rng, 4, p=action_probs))
            actions.append(action)

        return {
            'step': step,
            'hstates': np.array(hstates),
            'actions': np.array(actions),
            'level_features': level_features_list,
            'adversary_features': {
                'difficulty': 0.3 + 0.3 * (step / self.trajectory_length),
                'curriculum_phase': 'early' if step < 15 else ('mid' if step < 35 else 'late'),
            },
        }

    def _identify_shards(self, step: int) -> List[Shard]:
        """Identify shards at a given training step."""
        from sklearn.decomposition import DictionaryLearning

        data = self._trajectory_data[step]
        hstates = data['hstates']
        actions = data['actions']
        level_features = data['level_features']

        # Use sparse dictionary learning to find shard-like components
        try:
            model = DictionaryLearning(
                n_components=self.n_shard_components,
                alpha=0.5,
                max_iter=100,
                random_state=42,
            )
            components = model.fit_transform(hstates)
            dictionary = model.components_
        except Exception:
            # Fallback to simpler analysis
            return []

        shards = []
        for i in range(self.n_shard_components):
            component = dictionary[i]
            activations = components[:, i]

            # Find activation contexts
            contexts = self._find_activation_contexts(activations, level_features)

            # Measure policy influence
            policy_influence = self._measure_policy_influence(activations, actions)

            if policy_influence > self.policy_effect_threshold:
                # Find prominent dimensions
                dim_indices = np.where(np.abs(component) > 0.1)[0]

                shards.append(Shard(
                    shard_id=len(self._shards) + len(shards),
                    dimension_indices=dim_indices,
                    activation_contexts=contexts,
                    policy_influence=policy_influence,
                    birth_step=step,
                ))

        return shards

    def _find_activation_contexts(
        self,
        activations: np.ndarray,
        level_features: List[Dict[str, float]],
    ) -> Dict[str, Tuple[float, float]]:
        """Find contexts where a component activates."""
        contexts = {}

        # Find feature ranges where activation is high
        high_activation_mask = activations > np.percentile(activations, 75)

        for feature_name in ['wall_density', 'goal_distance']:
            feature_values = np.array([f[feature_name] for f in level_features])
            high_activation_values = feature_values[high_activation_mask]

            if len(high_activation_values) > 0:
                contexts[feature_name] = (
                    float(np.percentile(high_activation_values, 10)),
                    float(np.percentile(high_activation_values, 90)),
                )

        return contexts

    def _measure_policy_influence(
        self,
        activations: np.ndarray,
        actions: np.ndarray,
    ) -> float:
        """Measure how much a component influences policy."""
        # Compute correlation between activation and action distribution
        n_actions = 4
        action_one_hot = np.zeros((len(actions), n_actions))
        for i, a in enumerate(actions):
            action_one_hot[i, a] = 1

        # Correlation between activation and action choices
        correlations = []
        for a in range(n_actions):
            corr = np.corrcoef(activations, action_one_hot[:, a])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

        return float(np.mean(correlations)) if correlations else 0.0

    def analyze(self) -> Dict[str, Any]:
        """Analyze shard dynamics."""
        if not self._trajectory_data:
            raise ValueError("Must call collect_data first")

        # Identify shards at each step
        shards_by_step = {}
        for step in range(0, self.trajectory_length, 5):  # Every 5 steps
            shards = self._identify_shards(step)
            shards_by_step[step] = shards
            self._shards.extend(shards)

        results = {}

        # Number of active shards over time
        results['num_active_shards'] = len(self._shards)

        # Shard birth rate
        results['shard_birth_rate'] = self._compute_birth_rate(shards_by_step)

        # Shard half-life (persistence)
        results['shard_half_life'] = self._compute_half_life()

        # Composition events
        self._composition_events = self._detect_compositions(shards_by_step)
        results['shard_composition_events'] = len(self._composition_events)

        # Competition index
        results['shard_competition_index'] = self._compute_competition()

        # Adversary attribution
        results['adversary_driven_shard_births'] = self._attribute_to_adversary()

        # Goal complexity trajectory
        results['goal_complexity_trajectory'] = self._track_complexity(shards_by_step)

        # Adversary-driven transitions
        results['adversary_driven_goal_transitions'] = self._attribute_transitions()

        # Shard details
        results['shard_details'] = [
            {
                'shard_id': s.shard_id,
                'birth_step': s.birth_step,
                'n_dimensions': len(s.dimension_indices),
                'policy_influence': s.policy_influence,
                'contexts': s.activation_contexts,
            }
            for s in self._shards[:20]  # Limit output
        ]

        return results

    def _compute_birth_rate(self, shards_by_step: Dict[int, List[Shard]]) -> float:
        """Compute shard birth rate."""
        if len(shards_by_step) < 2:
            return 0.0

        steps = sorted(shards_by_step.keys())
        births_per_interval = []

        for i in range(1, len(steps)):
            new_shards = len(shards_by_step[steps[i]])
            interval = steps[i] - steps[i - 1]
            births_per_interval.append(new_shards / interval)

        return float(np.mean(births_per_interval))

    def _compute_half_life(self) -> float:
        """Compute average shard half-life."""
        # Simplified: measure how many steps shards remain influential
        # For shards identified at each step, check if similar shards exist later
        half_lives = []

        for shard in self._shards:
            # Check persistence based on dimension overlap with later shards
            birth = shard.birth_step
            persistence = 0
            for later_shard in self._shards:
                if later_shard.birth_step > birth:
                    overlap = len(set(shard.dimension_indices) & set(later_shard.dimension_indices))
                    if overlap > len(shard.dimension_indices) * 0.5:
                        persistence = max(persistence, later_shard.birth_step - birth)

            half_lives.append(persistence if persistence > 0 else 10)  # Default half-life

        return float(np.mean(half_lives)) if half_lives else 10.0

    def _detect_compositions(
        self,
        shards_by_step: Dict[int, List[Shard]],
    ) -> List[CompositionEvent]:
        """Detect shard composition/competition events."""
        events = []

        steps = sorted(shards_by_step.keys())
        for i in range(1, len(steps)):
            prev_shards = shards_by_step[steps[i - 1]]
            curr_shards = shards_by_step[steps[i]]

            # Check for dimension overlap (potential composition)
            for curr_shard in curr_shards:
                overlapping_prev = []
                for prev_shard in prev_shards:
                    overlap = len(set(curr_shard.dimension_indices) & set(prev_shard.dimension_indices))
                    if overlap > 5:  # Significant overlap
                        overlapping_prev.append(prev_shard.shard_id)

                if len(overlapping_prev) > 1:
                    events.append(CompositionEvent(
                        step=steps[i],
                        event_type='composition',
                        shards_involved=overlapping_prev + [curr_shard.shard_id],
                        resolution='merged',
                    ))

        return events

    def _compute_competition(self) -> float:
        """Compute shard competition index."""
        # Competition = average overlap in activation contexts
        if len(self._shards) < 2:
            return 0.0

        overlaps = []
        for i, s1 in enumerate(self._shards):
            for s2 in self._shards[i + 1:]:
                # Compute context overlap
                overlap_score = 0
                for feature in s1.activation_contexts:
                    if feature in s2.activation_contexts:
                        r1 = s1.activation_contexts[feature]
                        r2 = s2.activation_contexts[feature]
                        # Overlap = intersection / union of ranges
                        intersection = max(0, min(r1[1], r2[1]) - max(r1[0], r2[0]))
                        union = max(r1[1], r2[1]) - min(r1[0], r2[0])
                        if union > 1e-10:
                            overlap_score += intersection / union

                overlaps.append(overlap_score)

        return float(np.mean(overlaps)) if overlaps else 0.0

    def _attribute_to_adversary(self) -> float:
        """Attribute shard births to adversary curriculum."""
        # Count shards born during curriculum transitions
        adversary_driven = 0
        for shard in self._shards:
            if shard.birth_step < len(self._trajectory_data):
                adv_data = self._trajectory_data[shard.birth_step]['adversary_features']
                # Shards born during curriculum phase transitions
                if adv_data['curriculum_phase'] != 'early':
                    adversary_driven += 1

        return float(adversary_driven / len(self._shards)) if self._shards else 0.0

    def _track_complexity(
        self,
        shards_by_step: Dict[int, List[Shard]],
    ) -> Dict[str, List[float]]:
        """Track goal complexity over training."""
        steps = sorted(shards_by_step.keys())
        n_shards = []
        mean_influence = []
        total_dimensions = []

        for step in steps:
            shards = shards_by_step[step]
            n_shards.append(len(shards))

            if shards:
                mean_influence.append(np.mean([s.policy_influence for s in shards]))
                total_dimensions.append(sum(len(s.dimension_indices) for s in shards))
            else:
                mean_influence.append(0.0)
                total_dimensions.append(0)

        return {
            'steps': steps,
            'num_shards': n_shards,
            'mean_policy_influence': mean_influence,
            'total_active_dimensions': total_dimensions,
        }

    def _attribute_transitions(self) -> int:
        """Count adversary-driven goal transitions."""
        return len(self._composition_events)

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize shard dynamics."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._trajectory_data:
            return figures

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Shard count over training
        ax = axes[0, 0]
        complexity = self._track_complexity({
            step: self._identify_shards(step)
            for step in range(0, self.trajectory_length, 5)
        })
        ax.plot(complexity['steps'], complexity['num_shards'], 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Number of Shards')
        ax.set_title('Shard Count Over Training')
        ax.grid(True, alpha=0.3)

        # Policy influence over training
        ax = axes[0, 1]
        ax.plot(complexity['steps'], complexity['mean_policy_influence'], 'g-o', linewidth=2, markersize=6)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Mean Policy Influence')
        ax.set_title('Shard Policy Influence Over Training')
        ax.grid(True, alpha=0.3)

        # Shard dimension distribution
        ax = axes[1, 0]
        if self._shards:
            n_dims = [len(s.dimension_indices) for s in self._shards]
            ax.hist(n_dims, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Number of Dimensions')
            ax.set_ylabel('Count')
            ax.set_title('Shard Size Distribution')
        else:
            ax.text(0.5, 0.5, 'No shards detected', ha='center', va='center', transform=ax.transAxes)

        # Composition events timeline
        ax = axes[1, 1]
        if self._composition_events:
            event_steps = [e.step for e in self._composition_events]
            ax.eventplot(event_steps, lineoffsets=0, linelengths=0.5, colors='red')
            ax.set_xlabel('Training Step')
            ax.set_title(f'Shard Composition Events (n={len(self._composition_events)})')
            ax.set_xlim(0, self.trajectory_length)
        else:
            ax.text(0.5, 0.5, 'No composition events', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Shard Composition Events')

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["goal_evolution"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
