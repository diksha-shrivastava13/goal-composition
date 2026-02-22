"""
F3: Shard Dynamics.

Track shard competition and coalitional resolution over training.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment


class ShardState(Enum):
    """State of a shard."""
    NASCENT = "nascent"  # Just emerged
    ACTIVE = "active"  # Currently influencing policy
    DORMANT = "dormant"  # Exists but not active
    MERGED = "merged"  # Combined with another shard
    DEAD = "dead"  # No longer present


@dataclass
class ShardClusterInfo:
    """Information about a shard cluster."""
    shard_id: int
    dimension_indices: np.ndarray
    activation_strength: float
    policy_influence: float
    context_specificity: Dict[str, Tuple[float, float]]
    birth_step: int
    state: ShardState
    competing_shards: List[int]
    allied_shards: List[int]


@dataclass
class CompetitionEvent:
    """Event where shards compete for behavioral control."""
    step: int
    shard_a: int
    shard_b: int
    context: Dict[str, float]
    winner: int
    resolution_type: str  # 'dominance', 'compromise', 'alternation'
    policy_effect: float


class ShardDynamicsExperiment(CheckpointExperiment):
    """
    Track shard competition and coalitional resolution.

    Protocol:
    1. Identify shard clusters via sparse dictionary learning
    2. Track competition events where shards conflict
    3. Analyze resolution mechanisms
    4. Link to adversary strategy changes
    """

    @property
    def name(self) -> str:
        return "shard_dynamics"

    def __init__(
        self,
        n_samples_per_step: int = 100,
        trajectory_length: int = 50,
        hidden_dim: int = 256,
        n_shard_components: int = 15,
        competition_threshold: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_samples_per_step = n_samples_per_step
        self.trajectory_length = trajectory_length
        self.hidden_dim = hidden_dim
        self.n_shard_components = n_shard_components
        self.competition_threshold = competition_threshold
        self._trajectory_data: List[Dict[str, Any]] = []
        self._shards: Dict[int, ShardClusterInfo] = {}
        self._competition_events: List[CompetitionEvent] = []
        self._shard_counter: int = 0
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"ShardDynamicsExperiment requires PAIRED")

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
        policy_logits_list = []
        level_features_list = []

        # Adversary curriculum phase
        curriculum_phase = 'early' if step < 15 else ('mid' if step < 35 else 'late')
        adversary_difficulty = 0.2 + 0.5 * (step / self.trajectory_length)

        for i in range(self.n_samples_per_step):
            rng, h_rng, f_rng, p_rng = jax.random.split(rng, 4)

            # Level features
            wall_density = 0.1 + float(jax.random.uniform(f_rng)) * 0.35
            goal_distance = 2.0 + float(jax.random.uniform(f_rng)) * 10.0
            open_space = 1.0 - wall_density

            level_features = {
                'wall_density': wall_density,
                'goal_distance': goal_distance,
                'open_space': open_space,
            }
            level_features_list.append(level_features)

            # Hidden state with competing shard structure
            h = np.array(jax.random.normal(h_rng, (self.hidden_dim,)))

            # Shard 1: Navigation shard - activates for high wall density
            nav_activation = 0.0
            if wall_density > 0.25:
                nav_activation = (wall_density - 0.25) * 4.0
                h[:40] += nav_activation * 1.5

            # Shard 2: Exploration shard - activates for large goal distance
            explore_activation = 0.0
            if goal_distance > 6.0:
                explore_activation = (goal_distance - 6.0) / 6.0
                h[40:80] += explore_activation * 1.2

            # Shard 3: Efficiency shard - activates for open spaces
            efficiency_activation = 0.0
            if open_space > 0.75:
                efficiency_activation = (open_space - 0.75) * 4.0
                h[80:120] += efficiency_activation * 1.0

            # Shard 4: Caution shard - emerges late in training
            caution_activation = 0.0
            if step > 25 and wall_density > 0.2:
                caution_activation = (step - 25) / 25.0 * wall_density
                h[120:160] += caution_activation * 0.8

            # Shard 5: Adversary-response shard - responds to curriculum difficulty
            adversary_activation = adversary_difficulty * 0.5
            h[160:200] += adversary_activation

            hstates.append(h)

            # Policy logits influenced by competing shards
            base_logits = np.zeros(4)

            # Different shards prefer different actions
            base_logits[0] += nav_activation * 0.5  # Careful movement
            base_logits[1] += explore_activation * 0.4  # Exploratory
            base_logits[2] += efficiency_activation * 0.3  # Direct path
            base_logits[3] += caution_activation * 0.6  # Wait/observe

            # Add noise
            logits = base_logits + np.array(jax.random.normal(p_rng, (4,))) * 0.3
            policy_logits_list.append(logits)

            # Sample action
            probs = np.exp(logits - np.max(logits))
            probs = probs / probs.sum()
            action = int(jax.random.choice(p_rng, 4, p=probs))
            actions.append(action)

        return {
            'step': step,
            'hstates': np.array(hstates),
            'actions': np.array(actions),
            'policy_logits': np.array(policy_logits_list),
            'level_features': level_features_list,
            'adversary_features': {
                'difficulty': adversary_difficulty,
                'curriculum_phase': curriculum_phase,
            },
        }

    def _identify_shards_at_step(self, step: int) -> List[ShardClusterInfo]:
        """Identify active shards at a given step."""
        from sklearn.decomposition import DictionaryLearning

        data = self._trajectory_data[step]
        hstates = data['hstates']
        actions = data['actions']
        policy_logits = data['policy_logits']
        level_features = data['level_features']

        try:
            model = DictionaryLearning(
                n_components=self.n_shard_components,
                alpha=0.5,
                max_iter=100,
                random_state=step,
            )
            activations = model.fit_transform(hstates)
            dictionary = model.components_
        except Exception:
            return []

        shards = []
        for i in range(self.n_shard_components):
            component = dictionary[i]
            component_activations = activations[:, i]

            # Measure activation strength
            activation_strength = float(np.abs(component_activations).mean())
            if activation_strength < 0.1:
                continue

            # Find context specificity
            context = self._find_context_specificity(component_activations, level_features)

            # Measure policy influence
            policy_influence = self._measure_policy_influence(
                component_activations, policy_logits, actions
            )

            if policy_influence < 0.05:
                continue

            # Get dimension indices
            dim_indices = np.where(np.abs(component) > 0.1)[0]

            shard_id = self._shard_counter
            self._shard_counter += 1

            shard = ShardClusterInfo(
                shard_id=shard_id,
                dimension_indices=dim_indices,
                activation_strength=activation_strength,
                policy_influence=policy_influence,
                context_specificity=context,
                birth_step=step,
                state=ShardState.ACTIVE,
                competing_shards=[],
                allied_shards=[],
            )
            shards.append(shard)

        return shards

    def _find_context_specificity(
        self,
        activations: np.ndarray,
        level_features: List[Dict[str, float]],
    ) -> Dict[str, Tuple[float, float]]:
        """Find contexts where shard activates."""
        context = {}
        high_mask = activations > np.percentile(activations, 75)

        for feature in ['wall_density', 'goal_distance', 'open_space']:
            values = np.array([f[feature] for f in level_features])
            high_values = values[high_mask]

            if len(high_values) > 5:
                context[feature] = (
                    float(np.percentile(high_values, 20)),
                    float(np.percentile(high_values, 80)),
                )

        return context

    def _measure_policy_influence(
        self,
        activations: np.ndarray,
        policy_logits: np.ndarray,
        actions: np.ndarray,
    ) -> float:
        """Measure how much a shard influences policy."""
        # Correlation between activation and action distribution
        influences = []
        for a in range(4):
            action_prob = np.exp(policy_logits[:, a]) / np.exp(policy_logits).sum(axis=1)
            corr = np.corrcoef(activations, action_prob)[0, 1]
            if not np.isnan(corr):
                influences.append(abs(corr))

        return float(np.max(influences)) if influences else 0.0

    def _detect_competition_events(self, step: int, shards: List[ShardClusterInfo]) -> List[CompetitionEvent]:
        """Detect competition events between shards."""
        events = []
        data = self._trajectory_data[step]

        for i, shard_a in enumerate(shards):
            for shard_b in shards[i + 1:]:
                # Check for overlapping contexts (competition condition)
                overlap = self._compute_context_overlap(
                    shard_a.context_specificity,
                    shard_b.context_specificity,
                )

                if overlap > self.competition_threshold:
                    # Determine winner based on policy influence
                    if shard_a.policy_influence > shard_b.policy_influence * 1.2:
                        winner = shard_a.shard_id
                        resolution = 'dominance'
                    elif shard_b.policy_influence > shard_a.policy_influence * 1.2:
                        winner = shard_b.shard_id
                        resolution = 'dominance'
                    else:
                        winner = shard_a.shard_id if shard_a.policy_influence > shard_b.policy_influence else shard_b.shard_id
                        resolution = 'compromise'

                    # Get example context
                    sample_idx = len(data['level_features']) // 2
                    context = data['level_features'][sample_idx]

                    events.append(CompetitionEvent(
                        step=step,
                        shard_a=shard_a.shard_id,
                        shard_b=shard_b.shard_id,
                        context=context,
                        winner=winner,
                        resolution_type=resolution,
                        policy_effect=abs(shard_a.policy_influence - shard_b.policy_influence),
                    ))

                    # Update competing shards lists
                    shard_a.competing_shards.append(shard_b.shard_id)
                    shard_b.competing_shards.append(shard_a.shard_id)

        return events

    def _compute_context_overlap(
        self,
        context_a: Dict[str, Tuple[float, float]],
        context_b: Dict[str, Tuple[float, float]],
    ) -> float:
        """Compute overlap between two context specifications."""
        overlaps = []

        for feature in context_a:
            if feature in context_b:
                range_a = context_a[feature]
                range_b = context_b[feature]

                intersection = max(0, min(range_a[1], range_b[1]) - max(range_a[0], range_b[0]))
                union = max(range_a[1], range_b[1]) - min(range_a[0], range_b[0])

                if union > 1e-10:
                    overlaps.append(intersection / union)

        return float(np.mean(overlaps)) if overlaps else 0.0

    def analyze(self) -> Dict[str, Any]:
        """Analyze shard dynamics."""
        if not self._trajectory_data:
            raise ValueError("Must call collect_data first")

        results = {}

        # Identify shards and track competition at each step
        all_shards = []
        for step in range(0, self.trajectory_length, 3):  # Every 3 steps
            shards = self._identify_shards_at_step(step)
            all_shards.extend(shards)

            # Store in dict
            for shard in shards:
                self._shards[shard.shard_id] = shard

            # Detect competition
            events = self._detect_competition_events(step, shards)
            self._competition_events.extend(events)

        # Shard count statistics
        results['total_shards_identified'] = len(all_shards)
        results['unique_shards'] = len(self._shards)

        # Competition statistics
        results['total_competition_events'] = len(self._competition_events)
        results['competition_rate'] = len(self._competition_events) / max(self.trajectory_length // 3, 1)

        # Resolution analysis
        results['resolution_statistics'] = self._analyze_resolutions()

        # Shard lifecycle
        results['shard_lifecycle'] = self._analyze_lifecycle()

        # Coalition detection
        results['coalition_structure'] = self._detect_coalitions()

        # Adversary influence on shard dynamics
        results['adversary_shard_influence'] = self._analyze_adversary_influence()

        # Competition by curriculum phase
        results['competition_by_phase'] = self._analyze_by_phase()

        return results

    def _analyze_resolutions(self) -> Dict[str, Any]:
        """Analyze how competitions are resolved."""
        if not self._competition_events:
            return {'dominance': 0.0, 'compromise': 0.0, 'alternation': 0.0}

        resolution_counts = {'dominance': 0, 'compromise': 0, 'alternation': 0}
        for event in self._competition_events:
            resolution_counts[event.resolution_type] = resolution_counts.get(event.resolution_type, 0) + 1

        total = len(self._competition_events)
        return {
            k: float(v / total) for k, v in resolution_counts.items()
        }

    def _analyze_lifecycle(self) -> Dict[str, float]:
        """Analyze shard lifecycle statistics."""
        if not self._shards:
            return {}

        birth_steps = [s.birth_step for s in self._shards.values()]
        activation_strengths = [s.activation_strength for s in self._shards.values()]
        policy_influences = [s.policy_influence for s in self._shards.values()]

        return {
            'mean_birth_step': float(np.mean(birth_steps)),
            'birth_step_std': float(np.std(birth_steps)),
            'mean_activation_strength': float(np.mean(activation_strengths)),
            'mean_policy_influence': float(np.mean(policy_influences)),
            'shards_with_competition': float(
                np.mean([len(s.competing_shards) > 0 for s in self._shards.values()])
            ),
        }

    def _detect_coalitions(self) -> Dict[str, Any]:
        """Detect shard coalitions (groups that don't compete)."""
        if len(self._shards) < 2:
            return {'num_coalitions': 0, 'coalition_sizes': []}

        # Build competition graph
        competing_pairs = set()
        for event in self._competition_events:
            competing_pairs.add((min(event.shard_a, event.shard_b), max(event.shard_a, event.shard_b)))

        # Find non-competing groups (naive clustering)
        shard_ids = list(self._shards.keys())
        coalitions = []
        remaining = set(shard_ids)

        while remaining:
            # Start new coalition
            start = remaining.pop()
            coalition = {start}

            # Add non-competing shards
            for shard_id in list(remaining):
                competes = False
                for member in coalition:
                    if (min(shard_id, member), max(shard_id, member)) in competing_pairs:
                        competes = True
                        break
                if not competes:
                    coalition.add(shard_id)
                    remaining.remove(shard_id)

            coalitions.append(coalition)

        # Update allied shards
        for coalition in coalitions:
            for shard_id in coalition:
                if shard_id in self._shards:
                    self._shards[shard_id].allied_shards = list(coalition - {shard_id})

        return {
            'num_coalitions': len(coalitions),
            'coalition_sizes': [len(c) for c in coalitions],
            'mean_coalition_size': float(np.mean([len(c) for c in coalitions])) if coalitions else 0.0,
        }

    def _analyze_adversary_influence(self) -> Dict[str, float]:
        """Analyze how adversary curriculum affects shard dynamics."""
        # Correlate adversary difficulty with competition events
        events_by_step = {}
        for event in self._competition_events:
            events_by_step[event.step] = events_by_step.get(event.step, 0) + 1

        difficulties = []
        event_counts = []
        for step in range(0, self.trajectory_length, 3):
            if step < len(self._trajectory_data):
                difficulties.append(self._trajectory_data[step]['adversary_features']['difficulty'])
                event_counts.append(events_by_step.get(step, 0))

        if len(difficulties) < 3:
            return {'difficulty_competition_correlation': 0.0}

        corr = np.corrcoef(difficulties, event_counts)[0, 1]

        return {
            'difficulty_competition_correlation': float(corr) if not np.isnan(corr) else 0.0,
            'mean_difficulty_at_competition': float(np.mean([
                self._trajectory_data[e.step]['adversary_features']['difficulty']
                for e in self._competition_events
                if e.step < len(self._trajectory_data)
            ])) if self._competition_events else 0.0,
        }

    def _analyze_by_phase(self) -> Dict[str, Dict[str, float]]:
        """Analyze competition by curriculum phase."""
        phase_events = {'early': [], 'mid': [], 'late': []}

        for event in self._competition_events:
            if event.step < len(self._trajectory_data):
                phase = self._trajectory_data[event.step]['adversary_features']['curriculum_phase']
                phase_events[phase].append(event)

        results = {}
        for phase, events in phase_events.items():
            results[phase] = {
                'num_events': len(events),
                'mean_policy_effect': float(np.mean([e.policy_effect for e in events])) if events else 0.0,
                'dominance_rate': float(np.mean([e.resolution_type == 'dominance' for e in events])) if events else 0.0,
            }

        return results

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize shard dynamics."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._trajectory_data:
            return figures

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Competition events over time
        ax = axes[0, 0]
        events_by_step = {}
        for event in self._competition_events:
            events_by_step[event.step] = events_by_step.get(event.step, 0) + 1

        steps = sorted(events_by_step.keys())
        counts = [events_by_step[s] for s in steps]
        ax.bar(steps, counts, alpha=0.7, width=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Competition Events')
        ax.set_title('Shard Competition Events Over Training')

        # Resolution type distribution
        ax = axes[0, 1]
        resolutions = self._analyze_resolutions()
        if resolutions and sum(resolutions.values()) > 0:
            labels = list(resolutions.keys())
            sizes = list(resolutions.values())
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.set_title('Competition Resolution Types')
        else:
            ax.text(0.5, 0.5, 'No competition events', ha='center', va='center', transform=ax.transAxes)

        # Shard policy influence distribution
        ax = axes[1, 0]
        if self._shards:
            influences = [s.policy_influence for s in self._shards.values()]
            ax.hist(influences, bins=15, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Policy Influence')
            ax.set_ylabel('Count')
            ax.set_title('Shard Policy Influence Distribution')
        else:
            ax.text(0.5, 0.5, 'No shards identified', ha='center', va='center', transform=ax.transAxes)

        # Competition by phase
        ax = axes[1, 1]
        by_phase = self._analyze_by_phase()
        phases = ['early', 'mid', 'late']
        event_counts = [by_phase[p]['num_events'] for p in phases]
        ax.bar(phases, event_counts, alpha=0.7)
        ax.set_xlabel('Curriculum Phase')
        ax.set_ylabel('Competition Events')
        ax.set_title('Competition by Curriculum Phase')

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["shard_dynamics"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
