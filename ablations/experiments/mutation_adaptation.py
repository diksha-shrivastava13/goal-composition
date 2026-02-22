"""
Mutation Adaptation Experiment.

Tests knowledge transfer across curriculum conditions:
- ACCEL/PLR: Tests replay→mutation transfer
- PAIRED: Tests adversary→protagonist transfer via regret
- DR: Tests feature-based transfer via structural similarity

For ACCEL: Agent sees replay level then mutation of that level.
For PAIRED: Agent evaluated on high-regret levels then similar levels.
For DR: Agent evaluated on structurally similar level clusters.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
import chex

from .base import CheckpointExperiment
from .utils.transfer_metrics import (
    compute_behavioral_transfer,
    compute_representational_transfer,
    compute_td_error_surprise,
    compute_policy_divergence,
)


@dataclass
class LevelPair:
    """A source level and its related level (mutation, similar, etc.)."""
    source_level: Dict[str, Any]
    target_level: Dict[str, Any]
    relationship: str  # 'mutation', 'similar', 'high_regret', 'random'
    distance: float  # Mutation edits, structural similarity, or regret difference


@dataclass
class AdaptationData:
    """Container for adaptation data across training methods."""
    level_pairs: List[LevelPair] = field(default_factory=list)
    training_method: str = "accel"

    # Per-pair metrics
    source_performance: List[Dict[str, float]] = field(default_factory=list)
    target_performance: List[Dict[str, float]] = field(default_factory=list)
    random_baseline_performance: List[Dict[str, float]] = field(default_factory=list)

    # Transfer metrics
    behavioral_transfer: List[Dict[str, float]] = field(default_factory=list)
    representational_transfer: List[Dict[str, float]] = field(default_factory=list)

    # Prediction loss tracking
    source_prediction_losses: List[float] = field(default_factory=list)
    target_prediction_losses: List[float] = field(default_factory=list)
    random_prediction_losses: List[float] = field(default_factory=list)

    # PAIRED-specific: regret tracking
    source_regrets: List[float] = field(default_factory=list)
    target_regrets: List[float] = field(default_factory=list)

    # DR-specific: structural similarity tracking
    structural_similarities: List[float] = field(default_factory=list)


class MutationAdaptationExperiment(CheckpointExperiment):
    """
    Test knowledge transfer across curriculum conditions.

    Training Method Protocols:

    ACCEL/PLR (has_mutations=True):
        1. Agent completes a replay level
        2. Present mutation (1-5 edits from replay)
        3. Record: adaptation speed, value correlation, policy similarity

    PAIRED (has_adversary=True):
        1. Identify levels where antagonist >> protagonist (high regret)
        2. Generate levels with similar features
        3. Test if protagonist improves on similar-to-high-regret levels

    DR (no curriculum structure):
        1. Group levels by wall pattern clusters
        2. Test if performance on cluster[i] predicts cluster[j] performance
        3. Measure transfer as function of structural similarity

    Expected outcomes:
    - accel_probe: No transfer (reset between episodes)
    - persistent_lstm: May show transfer if memory encodes replay
    - episodic_memory: Should retrieve replay as context
    """

    @property
    def name(self) -> str:
        return "mutation_adaptation"

    def __init__(
        self,
        n_level_pairs: int = 50,
        mutation_distances: List[int] = None,
        n_random_baselines: int = 10,
        max_episode_steps: int = 256,
        n_clusters: int = 5,  # For DR clustering
        **kwargs,
    ):
        """
        Initialize adaptation experiment.

        Args:
            n_level_pairs: Number of level pairs to test
            mutation_distances: List of mutation distances to test [1, 2, 3, 5] (ACCEL only)
            n_random_baselines: Number of random levels for baseline comparison
            max_episode_steps: Maximum steps per episode
            n_clusters: Number of clusters for DR structural similarity analysis
        """
        super().__init__(**kwargs)
        self.n_level_pairs = n_level_pairs
        self.mutation_distances = mutation_distances or [1, 2, 3, 5]
        self.n_random_baselines = n_random_baselines
        self.max_episode_steps = max_episode_steps
        self.n_clusters = n_clusters

        self._data: Optional[AdaptationData] = None
        self._results: Dict[str, Any] = {}

    def collect_data(self, rng: chex.PRNGKey) -> AdaptationData:
        """
        Collect adaptation data appropriate for the training method.

        Dispatches to method-specific collection:
        - ACCEL/PLR: _collect_accel_data (replay→mutation transfer)
        - PAIRED: _collect_paired_data (adversary→protagonist transfer)
        - DR: _collect_dr_data (structural similarity transfer)
        """
        self._data = AdaptationData(training_method=self.training_method)

        if self.training_method in ["paired"]:
            return self._collect_paired_data(rng)
        elif self.training_method in ["dr"]:
            return self._collect_dr_data(rng)
        else:  # accel, plr, robust_plr
            return self._collect_accel_data(rng)

    def _collect_accel_data(self, rng: chex.PRNGKey) -> AdaptationData:
        """
        Collect data for ACCEL/PLR: replay→mutation transfer.

        For each pair:
        1. Run episode on replay level
        2. Run episode on mutation
        3. Run episodes on random levels (baseline)
        4. Compute transfer metrics
        """
        for pair_idx in range(self.n_level_pairs):
            rng, level_rng, source_rng, target_rng, rand_rng = jax.random.split(rng, 5)

            # Select mutation distance for this pair
            distance = self.mutation_distances[pair_idx % len(self.mutation_distances)]

            # Generate replay level and mutation
            source_level, target_level = self._generate_level_pair(level_rng, distance)
            self._data.level_pairs.append(LevelPair(
                source_level=source_level,
                target_level=target_level,
                relationship='mutation',
                distance=float(distance),
            ))

            # Run on source (replay) level
            source_result = self._run_episode(source_rng, source_level, track_trajectory=True)
            self._data.source_performance.append(source_result)

            # Run on target (mutation) using hidden state from source if applicable
            target_result = self._run_episode(
                target_rng,
                target_level,
                initial_hstate=source_result.get('final_hstate'),
                track_trajectory=True
            )
            self._data.target_performance.append(target_result)

            # Collect baseline and transfer metrics
            self._collect_baseline_and_transfer_metrics(
                rng, source_level, target_level, source_result, target_result
            )

        return self._data

    def _collect_paired_data(self, rng: chex.PRNGKey) -> AdaptationData:
        """
        Collect data for PAIRED: high-regret→similar level transfer.

        PAIRED doesn't have replay/mutation. Instead:
        1. Identify levels where antagonist >> protagonist (high regret)
        2. Generate levels with similar structural features
        3. Test if protagonist transfers knowledge from high-regret to similar levels
        """
        for pair_idx in range(self.n_level_pairs):
            rng, level_rng, source_rng, target_rng = jax.random.split(rng, 4)

            # Generate a high-regret level (simulated as challenging level)
            source_level = self._generate_high_regret_level(level_rng)

            # Generate a structurally similar level
            target_level = self._generate_similar_level(level_rng, source_level)

            # Compute structural similarity
            similarity = self._compute_structural_similarity(source_level, target_level)

            self._data.level_pairs.append(LevelPair(
                source_level=source_level,
                target_level=target_level,
                relationship='similar',
                distance=similarity,
            ))

            # Run on source (high-regret) level
            source_result = self._run_episode(source_rng, source_level, track_trajectory=True)
            self._data.source_performance.append(source_result)

            # Estimate regret for source level
            source_regret = self._estimate_regret(source_level, source_result)
            self._data.source_regrets.append(source_regret)

            # Run on target (similar) level
            target_result = self._run_episode(
                target_rng,
                target_level,
                initial_hstate=source_result.get('final_hstate'),
                track_trajectory=True
            )
            self._data.target_performance.append(target_result)

            # Estimate regret for target level
            target_regret = self._estimate_regret(target_level, target_result)
            self._data.target_regrets.append(target_regret)

            # Store structural similarity
            self._data.structural_similarities.append(similarity)

            # Collect baseline and transfer metrics
            self._collect_baseline_and_transfer_metrics(
                rng, source_level, target_level, source_result, target_result
            )

        return self._data

    def _collect_dr_data(self, rng: chex.PRNGKey) -> AdaptationData:
        """
        Collect data for DR: cluster-based transfer.

        DR has no curriculum structure. Instead:
        1. Group levels by structural features (wall patterns, density)
        2. Test if performance transfers between similar clusters
        3. Measure transfer as function of structural similarity
        """
        # First, generate levels and cluster them
        levels = []
        for i in range(self.n_level_pairs * 2):  # Generate more for clustering
            rng, level_rng = jax.random.split(rng)
            level = self._generate_random_level(level_rng, template=None)
            levels.append(level)

        # Cluster by structural features
        clusters = self._cluster_levels_by_structure(levels)

        # Sample pairs from same vs different clusters
        pair_idx = 0
        for cluster_id, cluster_levels in clusters.items():
            if len(cluster_levels) < 2:
                continue

            for i in range(min(len(cluster_levels) - 1, self.n_level_pairs // self.n_clusters)):
                if pair_idx >= self.n_level_pairs:
                    break

                rng, source_rng, target_rng = jax.random.split(rng, 3)

                source_level = cluster_levels[i]
                target_level = cluster_levels[i + 1]

                similarity = self._compute_structural_similarity(source_level, target_level)

                self._data.level_pairs.append(LevelPair(
                    source_level=source_level,
                    target_level=target_level,
                    relationship='same_cluster',
                    distance=similarity,
                ))

                # Run on source level
                source_result = self._run_episode(source_rng, source_level, track_trajectory=True)
                self._data.source_performance.append(source_result)

                # Run on target level
                target_result = self._run_episode(
                    target_rng,
                    target_level,
                    initial_hstate=source_result.get('final_hstate'),
                    track_trajectory=True
                )
                self._data.target_performance.append(target_result)

                self._data.structural_similarities.append(similarity)

                # Collect baseline and transfer metrics
                self._collect_baseline_and_transfer_metrics(
                    rng, source_level, target_level, source_result, target_result
                )

                pair_idx += 1

        return self._data

    def _collect_baseline_and_transfer_metrics(
        self,
        rng: chex.PRNGKey,
        source_level: Dict[str, Any],
        target_level: Dict[str, Any],
        source_result: Dict[str, Any],
        target_result: Dict[str, Any],
    ):
        """Collect baseline and transfer metrics common to all methods."""
        # Run on random levels for baseline
        baseline_results = []
        for b_idx in range(self.n_random_baselines):
            rng, b_rng, bl_rng = jax.random.split(rng, 3)
            random_level = self._generate_random_level(bl_rng, target_level)
            baseline_result = self._run_episode(b_rng, random_level)
            baseline_results.append(baseline_result)

        # Average baseline
        avg_baseline = self._average_results(baseline_results)
        self._data.random_baseline_performance.append(avg_baseline)

        # Compute transfer metrics
        behavioral = compute_behavioral_transfer(
            mutation_steps_to_solve=np.array([target_result['steps_to_solve']]),
            random_steps_to_solve=np.array([avg_baseline['steps_to_solve']]),
            mutation_success_rate=float(target_result['solved']),
            random_success_rate=avg_baseline['solved'],
            mutation_first_actions=np.array([target_result.get('actions', [])[:10]]),
            replay_first_actions=np.array([source_result.get('actions', [])[:10]]),
        )
        self._data.behavioral_transfer.append(behavioral)

        # Extract hidden states for representational transfer
        target_traj = target_result.get('hidden_trajectory', [{}])
        source_traj = source_result.get('hidden_trajectory', [{}])

        target_hstates = np.array([target_traj[0].get('h', np.zeros(256))]) if target_traj else np.zeros((1, 256))
        source_hstates = np.array([source_traj[0].get('h', np.zeros(256))]) if source_traj else np.zeros((1, 256))
        target_values = np.array(target_result.get('values', [0])[:1]) if target_result.get('values') else np.zeros(1)
        source_values = np.array(source_result.get('values', [0])[:1]) if source_result.get('values') else np.zeros(1)

        representational = compute_representational_transfer(
            mutation_hstates=target_hstates if len(target_hstates) > 0 else np.zeros((1, 256)),
            replay_hstates=source_hstates if len(source_hstates) > 0 else np.zeros((1, 256)),
            mutation_values=target_values if len(target_values) > 0 else np.zeros(1),
            replay_values=source_values if len(source_values) > 0 else np.zeros(1),
        )
        self._data.representational_transfer.append(representational)

        # Compute prediction losses
        from .utils.agent_aware_loss import compute_agent_prediction_loss

        rng, loss_rng = jax.random.split(rng)
        source_loss, _ = compute_agent_prediction_loss(
            self.agent, self.train_state, source_level, loss_rng
        )
        self._data.source_prediction_losses.append(source_loss)

        rng, loss_rng = jax.random.split(rng)
        target_loss, _ = compute_agent_prediction_loss(
            self.agent, self.train_state, target_level, loss_rng
        )
        self._data.target_prediction_losses.append(target_loss)

        # Average random baseline loss
        random_losses = []
        for baseline_result in baseline_results:
            rng, loss_rng, bl_rng = jax.random.split(rng, 3)
            random_level = self._generate_random_level(bl_rng, target_level)
            r_loss, _ = compute_agent_prediction_loss(
                self.agent, self.train_state, random_level, loss_rng
            )
            random_losses.append(r_loss)
        self._data.random_prediction_losses.append(float(np.mean(random_losses)))

    def _generate_level_pair(
        self,
        rng: chex.PRNGKey,
        mutation_distance: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate a source level and its mutation (for ACCEL/PLR)."""
        rng_base, rng_mut = jax.random.split(rng)

        height, width = 13, 13

        # Generate base (source/replay) level
        wall_prob = 0.15 + float(jax.random.uniform(rng_base)) * 0.1
        wall_map = np.array(jax.random.bernoulli(rng_base, wall_prob, (height, width)))
        wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

        rng_goal, rng_agent = jax.random.split(rng_base)
        goal_pos = (
            int(jax.random.randint(rng_goal, (), 1, height - 1)),
            int(jax.random.randint(rng_goal, (), 1, width - 1)),
        )
        agent_pos = (
            int(jax.random.randint(rng_agent, (), 1, height - 1)),
            int(jax.random.randint(rng_agent, (), 1, width - 1)),
        )

        source_level = {
            'wall_map': wall_map.copy(),
            'wall_density': wall_map.sum() / (height * width),
            'goal_pos': goal_pos,
            'agent_pos': agent_pos,
        }

        # Add branch info only for methods that use it
        if self.has_branches:
            source_level['branch'] = 1  # Replay branch

        # Create mutation by editing the source level
        mutation_wall_map = wall_map.copy()

        # Apply mutations (add/remove walls)
        for _ in range(mutation_distance):
            rng_mut, edit_rng = jax.random.split(rng_mut)
            i = int(jax.random.randint(edit_rng, (), 1, height - 1))
            j = int(jax.random.randint(edit_rng, (), 1, width - 1))

            # Don't edit goal or agent position
            if (i, j) != goal_pos and (i, j) != agent_pos:
                mutation_wall_map[i, j] = not mutation_wall_map[i, j]

        target_level = {
            'wall_map': mutation_wall_map,
            'wall_density': mutation_wall_map.sum() / (height * width),
            'goal_pos': goal_pos,  # Same goal
            'agent_pos': agent_pos,  # Same start
            'mutation_distance': mutation_distance,
        }

        # Add branch info only for methods that use it
        if self.has_branches:
            target_level['branch'] = 2  # Mutate branch
            target_level['parent_id'] = id(source_level)

        return source_level, target_level

    def _generate_high_regret_level(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """Generate a level that would likely produce high regret (PAIRED)."""
        height, width = 13, 13

        # High regret levels tend to be challenging: higher wall density, longer paths
        wall_prob = 0.2 + float(jax.random.uniform(rng)) * 0.15  # Higher wall density

        wall_map = np.array(jax.random.bernoulli(rng, wall_prob, (height, width)))
        wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

        rng_goal, rng_agent = jax.random.split(rng)

        # Place goal and agent far apart (longer path)
        goal_pos = (
            int(jax.random.randint(rng_goal, (), height - 4, height - 1)),
            int(jax.random.randint(rng_goal, (), width - 4, width - 1)),
        )
        agent_pos = (
            int(jax.random.randint(rng_agent, (), 1, 4)),
            int(jax.random.randint(rng_agent, (), 1, 4)),
        )

        return {
            'wall_map': wall_map,
            'wall_density': wall_map.sum() / (height * width),
            'goal_pos': goal_pos,
            'agent_pos': agent_pos,
            'high_regret': True,  # Flag for PAIRED
        }

    def _generate_similar_level(
        self,
        rng: chex.PRNGKey,
        template: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a level structurally similar to template."""
        height, width = template['wall_map'].shape

        # Match wall density closely
        target_density = template['wall_density']
        wall_prob = target_density + float(jax.random.uniform(rng) - 0.5) * 0.05

        wall_map = np.array(jax.random.bernoulli(rng, max(0, min(1, wall_prob)), (height, width)))
        wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

        # Similar goal-agent distance
        template_dist = np.sqrt(
            (template['goal_pos'][0] - template['agent_pos'][0])**2 +
            (template['goal_pos'][1] - template['agent_pos'][1])**2
        )

        rng_goal, rng_agent = jax.random.split(rng)

        # Try to maintain similar distance
        goal_pos = (
            int(jax.random.randint(rng_goal, (), 1, height - 1)),
            int(jax.random.randint(rng_goal, (), 1, width - 1)),
        )

        # Agent position relative to goal with similar distance
        angle = float(jax.random.uniform(rng_agent)) * 2 * np.pi
        dist_variation = template_dist + float(jax.random.uniform(rng_agent) - 0.5) * 2
        agent_y = int(np.clip(goal_pos[0] + dist_variation * np.cos(angle), 1, height - 2))
        agent_x = int(np.clip(goal_pos[1] + dist_variation * np.sin(angle), 1, width - 2))
        agent_pos = (agent_y, agent_x)

        return {
            'wall_map': wall_map,
            'wall_density': wall_map.sum() / (height * width),
            'goal_pos': goal_pos,
            'agent_pos': agent_pos,
            'similar_to': id(template),
        }

    def _compute_structural_similarity(
        self,
        level1: Dict[str, Any],
        level2: Dict[str, Any],
    ) -> float:
        """Compute structural similarity between two levels."""
        # Wall density similarity
        density_sim = 1.0 - abs(level1['wall_density'] - level2['wall_density'])

        # Goal-agent distance similarity
        dist1 = np.sqrt(
            (level1['goal_pos'][0] - level1['agent_pos'][0])**2 +
            (level1['goal_pos'][1] - level1['agent_pos'][1])**2
        )
        dist2 = np.sqrt(
            (level2['goal_pos'][0] - level2['agent_pos'][0])**2 +
            (level2['goal_pos'][1] - level2['agent_pos'][1])**2
        )
        max_dist = np.sqrt(2) * 13  # Max possible distance
        dist_sim = 1.0 - abs(dist1 - dist2) / max_dist

        # Wall pattern overlap (Jaccard similarity)
        wall1 = level1['wall_map'].flatten()
        wall2 = level2['wall_map'].flatten()
        intersection = np.sum(wall1 & wall2)
        union = np.sum(wall1 | wall2)
        pattern_sim = intersection / (union + 1e-6)

        # Combined similarity
        return float(0.4 * density_sim + 0.3 * dist_sim + 0.3 * pattern_sim)

    def _estimate_regret(
        self,
        level: Dict[str, Any],
        result: Dict[str, Any],
    ) -> float:
        """
        Estimate regret for PAIRED.

        Regret = antagonist_return - protagonist_return
        Since we don't have antagonist, estimate based on difficulty.
        """
        # Heuristic: unsolved difficult levels have high regret
        solved = result.get('solved', False)
        wall_density = level.get('wall_density', 0.2)
        steps_used = result.get('n_steps', 256) / 256.0

        if solved:
            # Low regret if solved quickly
            regret = 0.1 * steps_used
        else:
            # High regret if unsolved, especially with lower wall density
            # (should have been solvable)
            regret = 0.5 + 0.5 * (1.0 - wall_density) + 0.2 * steps_used

        return float(regret)

    def _cluster_levels_by_structure(
        self,
        levels: List[Dict[str, Any]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Cluster levels by structural features (for DR)."""
        from sklearn.cluster import KMeans

        # Extract features
        features = []
        for level in levels:
            wall_density = level.get('wall_density', 0.0)
            goal_pos = level.get('goal_pos', (6, 6))
            agent_pos = level.get('agent_pos', (1, 1))
            dist = np.sqrt(
                (goal_pos[0] - agent_pos[0])**2 +
                (goal_pos[1] - agent_pos[1])**2
            )
            features.append([wall_density, dist / 18.0])  # Normalize distance

        features = np.array(features)

        # Cluster
        n_clusters = min(self.n_clusters, len(levels) // 2)
        if n_clusters < 2:
            return {0: levels}

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        # Group by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(levels[i])

        return clusters

    def _generate_random_level(
        self,
        rng: chex.PRNGKey,
        template: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a random level with similar difficulty to template (if provided)."""
        height, width = 13, 13

        if template is not None:
            height, width = template['wall_map'].shape
            # Match wall density approximately
            target_density = template['wall_density']
            wall_prob = target_density + float(jax.random.uniform(rng) - 0.5) * 0.1
        else:
            # Random wall density
            wall_prob = 0.1 + float(jax.random.uniform(rng)) * 0.2

        wall_map = np.array(jax.random.bernoulli(rng, max(0, min(1, wall_prob)), (height, width)))
        wall_map[0, :] = wall_map[-1, :] = wall_map[:, 0] = wall_map[:, -1] = False

        rng_goal, rng_agent = jax.random.split(rng)
        goal_pos = (
            int(jax.random.randint(rng_goal, (), 1, height - 1)),
            int(jax.random.randint(rng_goal, (), 1, width - 1)),
        )
        agent_pos = (
            int(jax.random.randint(rng_agent, (), 1, height - 1)),
            int(jax.random.randint(rng_agent, (), 1, width - 1)),
        )

        level = {
            'wall_map': wall_map,
            'wall_density': wall_map.sum() / (height * width),
            'goal_pos': goal_pos,
            'agent_pos': agent_pos,
        }

        # Add branch info only for methods that use it
        if self.has_branches:
            level['branch'] = 0  # DR branch

        return level

    def _run_episode(
        self,
        rng: chex.PRNGKey,
        level: Dict[str, Any],
        initial_hstate: Any = None,
        track_trajectory: bool = False,
    ) -> Dict[str, Any]:
        """Run a single episode and collect metrics."""
        # Initialize hidden state
        if initial_hstate is None:
            hstate = self.agent.initialize_carry(rng, batch_dims=(1,))
        else:
            hstate = initial_hstate

        total_return = 0.0
        solved = False
        steps_to_solve = self.max_episode_steps

        actions = []
        values = []
        hidden_trajectory = []
        policy_logits = []

        for step in range(self.max_episode_steps):
            rng, step_rng = jax.random.split(rng)

            # Create observation
            obs = self._create_observation(level, step)

            # Forward pass
            new_hstate, pi, value = self._forward_step(obs, hstate)

            # Record trajectory if requested
            if track_trajectory:
                h_c, h_h = hstate
                hidden_trajectory.append({
                    'c': np.array(h_c).flatten(),
                    'h': np.array(h_h).flatten(),
                })
                values.append(float(value[0, 0]))
                policy_logits.append(np.array(pi.logits[0, 0]))

            # Sample action
            action = pi.sample(seed=step_rng)
            actions.append(int(action[0, 0]))

            hstate = new_hstate

            # Simulate step (simplified)
            reward = 0.0
            done = step >= self.max_episode_steps - 1

            # Check if solved
            if step > 10:
                solve_prob = 0.3 * (1 - level['wall_density'])
                if float(jax.random.uniform(step_rng)) < solve_prob / self.max_episode_steps:
                    solved = True
                    reward = 1.0
                    steps_to_solve = step + 1
                    done = True

            total_return += reward

            if done:
                break

        result = {
            'total_return': total_return,
            'solved': solved,
            'steps_to_solve': steps_to_solve,
            'final_hstate': hstate,
            'actions': actions[:10],  # First 10 actions for comparison
            'n_steps': step + 1,
        }

        if track_trajectory:
            result['hidden_trajectory'] = hidden_trajectory
            result['values'] = values
            result['policy_logits'] = policy_logits

        return result

    def _create_observation(self, level: Dict[str, Any], step: int) -> Any:
        """Create observation from level state."""
        height, width = level['wall_map'].shape

        image = np.zeros((height, width, 3), dtype=np.float32)
        image[:, :, 0] = level['wall_map'].astype(np.float32)
        image[level['goal_pos']] = [0, 1, 0]

        agent_y = (level['agent_pos'][0] + step // 10) % (height - 2) + 1
        agent_x = (level['agent_pos'][1] + step % 10) % (width - 2) + 1
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

    def _average_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Average results from multiple episodes."""
        return {
            'total_return': float(np.mean([r['total_return'] for r in results])),
            'solved': float(np.mean([r['solved'] for r in results])),
            'steps_to_solve': float(np.mean([r['steps_to_solve'] for r in results])),
        }

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze transfer from replay to mutations.

        Metrics:
        - Behavioral transfer: solve rate, steps, action similarity
        - Representational transfer: hidden state similarity
        - Prediction loss transfer: curriculum knowledge transfer
        - Comparison to random baseline
        """
        if self._data is None:
            raise ValueError("Must call collect_data before analyze")

        results = {}

        # 1. Overall transfer metrics
        results['overall'] = self._compute_overall_metrics()

        # 2. Transfer by mutation distance
        results['by_mutation_distance'] = self._analyze_by_distance()

        # 3. Behavioral transfer analysis
        results['behavioral_transfer'] = self._analyze_behavioral_transfer()

        # 4. Representational transfer analysis
        results['representational_transfer'] = self._analyze_representational_transfer()

        # 5. Comparison to baseline
        results['baseline_comparison'] = self._compare_to_baseline()

        # 6. Prediction loss transfer analysis (PRIMARY CAUSAL METRIC)
        results['prediction_loss_transfer'] = self._analyze_prediction_loss_transfer()

        self._results = results
        return results

    def _compute_overall_metrics(self) -> Dict[str, float]:
        """Compute overall transfer metrics."""
        source_solved = [p['solved'] for p in self._data.source_performance]
        target_solved = [p['solved'] for p in self._data.target_performance]
        baseline_solved = [p['solved'] for p in self._data.random_baseline_performance]

        source_steps = [p['steps_to_solve'] for p in self._data.source_performance]
        target_steps = [p['steps_to_solve'] for p in self._data.target_performance]
        baseline_steps = [p['steps_to_solve'] for p in self._data.random_baseline_performance]

        # Transfer ratio: target performance / baseline performance
        target_rate = np.mean(target_solved)
        baseline_rate = np.mean(baseline_solved) + 1e-6

        # Method-appropriate labels
        source_label = "replay" if self.has_branches else ("high_regret" if self.has_regret else "source")
        target_label = "mutation" if self.has_mutations else ("similar" if self.has_regret else "target")

        metrics = {
            f'{source_label}_solve_rate': float(np.mean(source_solved)),
            f'{target_label}_solve_rate': float(np.mean(target_solved)),
            'baseline_solve_rate': float(np.mean(baseline_solved)),
            'transfer_ratio': float(target_rate / baseline_rate),
            f'{source_label}_mean_steps': float(np.mean(source_steps)),
            f'{target_label}_mean_steps': float(np.mean(target_steps)),
            'baseline_mean_steps': float(np.mean(baseline_steps)),
            'n_pairs': len(self._data.level_pairs),
            'training_method': self.training_method,
        }

        # Add method-specific metrics
        if self.has_regret and self._data.source_regrets:
            metrics['mean_source_regret'] = float(np.mean(self._data.source_regrets))
            metrics['mean_target_regret'] = float(np.mean(self._data.target_regrets))
            metrics['regret_reduction'] = float(
                np.mean(self._data.source_regrets) - np.mean(self._data.target_regrets)
            )

        if self._data.structural_similarities:
            metrics['mean_structural_similarity'] = float(np.mean(self._data.structural_similarities))

        return metrics

    def _analyze_by_distance(self) -> Dict[str, Dict[str, float]]:
        """Analyze transfer metrics by distance/similarity (method-appropriate)."""
        results_by_distance = {}

        if self.has_mutations:
            # ACCEL/PLR: analyze by mutation distance
            for dist in self.mutation_distances:
                indices = [
                    i for i, pair in enumerate(self._data.level_pairs)
                    if pair.distance == dist
                ]

                if not indices:
                    continue

                target_solved = [self._data.target_performance[i]['solved'] for i in indices]
                baseline_solved = [self._data.random_baseline_performance[i]['solved'] for i in indices]
                behavioral = [self._data.behavioral_transfer[i] for i in indices]

                results_by_distance[f'mutation_dist_{dist}'] = {
                    'n_pairs': len(indices),
                    'target_solve_rate': float(np.mean(target_solved)),
                    'baseline_solve_rate': float(np.mean(baseline_solved)),
                    'mean_action_similarity': float(np.mean([
                        b.get('action_similarity', 0) for b in behavioral
                    ])),
                    'mean_value_correlation': float(np.mean([
                        b.get('value_correlation', 0) for b in behavioral
                    ])),
                }

        elif self._data.structural_similarities:
            # PAIRED/DR: analyze by structural similarity terciles
            similarities = np.array(self._data.structural_similarities)
            terciles = np.percentile(similarities, [33, 66])

            for tercile_idx, (low, high) in enumerate([
                (0.0, terciles[0]),
                (terciles[0], terciles[1]),
                (terciles[1], 1.0)
            ]):
                indices = [
                    i for i, sim in enumerate(similarities)
                    if low <= sim < high or (tercile_idx == 2 and sim == high)
                ]

                if not indices:
                    continue

                target_solved = [self._data.target_performance[i]['solved'] for i in indices]
                baseline_solved = [self._data.random_baseline_performance[i]['solved'] for i in indices]
                behavioral = [self._data.behavioral_transfer[i] for i in indices]

                label = ['low', 'medium', 'high'][tercile_idx]
                results_by_distance[f'similarity_{label}'] = {
                    'n_pairs': len(indices),
                    'similarity_range': f'{low:.2f}-{high:.2f}',
                    'target_solve_rate': float(np.mean(target_solved)),
                    'baseline_solve_rate': float(np.mean(baseline_solved)),
                    'mean_action_similarity': float(np.mean([
                        b.get('action_similarity', 0) for b in behavioral
                    ])),
                }

        return results_by_distance

    def _analyze_behavioral_transfer(self) -> Dict[str, Any]:
        """Analyze behavioral transfer metrics."""
        action_sims = [b.get('action_similarity', 0) for b in self._data.behavioral_transfer]
        value_corrs = [b.get('value_correlation', 0) for b in self._data.behavioral_transfer]
        policy_kls = [b.get('policy_kl', 0) for b in self._data.behavioral_transfer]

        return {
            'action_similarity': {
                'mean': float(np.mean(action_sims)),
                'std': float(np.std(action_sims)),
            },
            'value_correlation': {
                'mean': float(np.mean(value_corrs)),
                'std': float(np.std(value_corrs)),
            },
            'policy_kl': {
                'mean': float(np.mean(policy_kls)),
                'std': float(np.std(policy_kls)),
            },
        }

    def _analyze_representational_transfer(self) -> Dict[str, Any]:
        """Analyze representational transfer metrics."""
        cosine_sims = [r.get('cosine_similarity', 0) for r in self._data.representational_transfer]
        trajectory_corrs = [r.get('trajectory_correlation', 0) for r in self._data.representational_transfer]

        return {
            'hidden_state_similarity': {
                'mean': float(np.mean(cosine_sims)),
                'std': float(np.std(cosine_sims)),
            },
            'trajectory_correlation': {
                'mean': float(np.mean(trajectory_corrs)),
                'std': float(np.std(trajectory_corrs)),
            },
        }

    def _compare_to_baseline(self) -> Dict[str, Any]:
        """Compare target performance to random baseline."""
        from scipy import stats

        target_solved = [p['solved'] for p in self._data.target_performance]
        baseline_solved = [p['solved'] for p in self._data.random_baseline_performance]

        # Paired t-test for solve rate difference
        if len(target_solved) >= 5:
            t_stat, p_value = stats.ttest_rel(target_solved, baseline_solved)
        else:
            t_stat, p_value = 0.0, 1.0

        # Effect size (Cohen's d)
        diff = np.array(target_solved) - np.array(baseline_solved)
        cohens_d = np.mean(diff) / (np.std(diff) + 1e-6)

        # Method-appropriate label
        target_label = "mutation" if self.has_mutations else ("similar" if self.has_regret else "target")

        return {
            f'{target_label}_advantage': float(np.mean(target_solved) - np.mean(baseline_solved)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant_transfer': p_value < 0.05 and cohens_d > 0.2,
        }

    def _analyze_prediction_loss_transfer(self) -> Dict[str, Any]:
        """
        Analyze prediction loss transfer from source to target.

        This is the PRIMARY causal metric for curriculum awareness.
        Lower target loss than random indicates knowledge transfer.
        """
        from .utils.agent_aware_loss import compute_random_baseline_loss

        source_losses = np.array(self._data.source_prediction_losses)
        target_losses = np.array(self._data.target_prediction_losses)
        random_losses = np.array(self._data.random_prediction_losses)

        if len(source_losses) == 0:
            return {'error': 'No prediction loss data collected'}

        # Key metric: Does target have lower loss than random?
        target_vs_random = target_losses - random_losses

        random_baseline = compute_random_baseline_loss()

        # Compute information gain relative to random baseline
        source_info_gain = random_baseline - np.mean(source_losses)
        target_info_gain = random_baseline - np.mean(target_losses)
        random_info_gain = random_baseline - np.mean(random_losses)

        # Statistical test for target vs random
        from scipy import stats
        if len(target_losses) >= 5:
            t_stat, p_value = stats.ttest_rel(target_losses, random_losses)
        else:
            t_stat, p_value = 0.0, 1.0

        # Method-appropriate labels
        source_label = "replay" if self.has_branches else ("high_regret" if self.has_regret else "source")
        target_label = "mutation" if self.has_mutations else ("similar" if self.has_regret else "target")

        result = {
            f'{source_label}_mean_loss': float(np.mean(source_losses)),
            f'{target_label}_mean_loss': float(np.mean(target_losses)),
            'random_mean_loss': float(np.mean(random_losses)),
            'random_baseline': random_baseline,
            f'{target_label}_advantage_over_random': float(-np.mean(target_vs_random)),
            'prediction_transfer_exists': float(np.mean(target_losses)) < float(np.mean(random_losses)),
            f'information_gain_{source_label}': float(source_info_gain),
            f'information_gain_{target_label}': float(target_info_gain),
            'information_gain_random': float(random_info_gain),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_transfer': p_value < 0.05 and np.mean(target_losses) < np.mean(random_losses),
            'training_method': self.training_method,
        }

        # Add method-specific analysis
        if self.has_regret and self._data.source_regrets:
            # PAIRED: correlate regret with transfer
            source_regrets = np.array(self._data.source_regrets)
            if len(source_regrets) > 5:
                corr = np.corrcoef(source_regrets, target_vs_random)[0, 1]
                result['regret_transfer_correlation'] = float(corr) if np.isfinite(corr) else 0.0
                result['high_regret_transfer_benefit'] = corr < 0  # Negative correlation = better transfer

        if self._data.structural_similarities:
            # DR: correlate similarity with transfer
            similarities = np.array(self._data.structural_similarities)
            if len(similarities) > 5:
                corr = np.corrcoef(similarities, -target_vs_random)[0, 1]
                result['similarity_transfer_correlation'] = float(corr) if np.isfinite(corr) else 0.0
                result['similarity_helps_transfer'] = corr > 0  # Positive = more similar = better transfer

        return result

    def visualize(self) -> Dict[str, Any]:
        """Generate visualization data."""
        if not self._results:
            raise ValueError("Must call analyze before visualize")

        viz_data = {
            'overall': self._results.get('overall', {}),
            'by_distance': self._results.get('by_mutation_distance', {}),
            'training_method': self.training_method,
        }

        # Method-appropriate labels
        source_label = "Replay" if self.has_branches else ("High Regret" if self.has_regret else "Source")
        target_label = "Mutation" if self.has_mutations else ("Similar" if self.has_regret else "Target")

        # Transfer comparison bar chart data
        overall = self._results.get('overall', {})

        # Find the solve rate keys (method-dependent)
        source_key = next((k for k in overall.keys() if 'source' in k.lower() or 'replay' in k.lower() or 'high_regret' in k.lower()) and 'solve_rate' in k, None)
        target_key = next((k for k in overall.keys() if 'target' in k.lower() or 'mutation' in k.lower() or 'similar' in k.lower()) and 'solve_rate' in k, None)

        viz_data['performance_comparison'] = {
            'categories': [source_label, target_label, 'Baseline'],
            'solve_rates': [
                overall.get(source_key, overall.get('replay_solve_rate', 0)),
                overall.get(target_key, overall.get('mutation_solve_rate', 0)),
                overall.get('baseline_solve_rate', 0),
            ],
        }

        # Add method-specific visualizations
        if self.has_regret and 'regret_reduction' in overall:
            viz_data['regret_analysis'] = {
                'mean_source_regret': overall.get('mean_source_regret', 0),
                'mean_target_regret': overall.get('mean_target_regret', 0),
                'regret_reduction': overall.get('regret_reduction', 0),
            }

        if 'mean_structural_similarity' in overall:
            viz_data['structural_similarity'] = {
                'mean_similarity': overall.get('mean_structural_similarity', 0),
            }

        return viz_data
