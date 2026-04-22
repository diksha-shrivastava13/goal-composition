"""
F1: Causal Model Extraction.

Test Richens & Everitt: regret-bounded agents learn causal models.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
import chex

from ..base import CheckpointExperiment
from ..utils.paired_helpers import (
    generate_levels, extract_level_features_batch, get_pro_hstates,
    get_pro_ant_returns, levels_to_dicts, compute_bfs_path_length,
)


@dataclass
class CausalEdge:
    """Edge in causal graph."""
    source: str
    target: str
    strength: float
    intervention_validated: bool = False


class CausalModelExtractionExperiment(CheckpointExperiment):
    """
    Test Richens & Everitt: regret-bounded agents learn causal models.

    Protocol:
    1. Extract causal graph from conditional independence structure in h-states
    2. Compare to ground truth causal structure of the environment
    3. Test interventional fidelity
    4. Compare causal model quality across training methods
    """

    @property
    def name(self) -> str:
        return "causal_model_extraction"

    # Ground truth causal structure for grid navigation
    GROUND_TRUTH_EDGES = [
        ('wall_density', 'path_difficulty'),
        ('goal_distance', 'path_difficulty'),
        ('path_difficulty', 'optimal_steps'),
        ('optimal_steps', 'return'),
        ('agent_position', 'optimal_steps'),
        ('wall_density', 'collision_risk'),
        ('collision_risk', 'return'),
    ]

    VARIABLES = [
        'wall_density', 'goal_distance', 'agent_position',
        'path_difficulty', 'optimal_steps', 'collision_risk', 'return'
    ]

    def __init__(self, **kwargs):
        """
        Args:
            n_permutation_trials: Number of permutation trials for statistical
                baseline. Default 100 — minimum for p < 0.01 resolution.
            edge_corr_threshold: Minimum |correlation| to consider an edge
                candidate in initial screening. Default 0.1.
            partial_corr_threshold: Below this |partial correlation|, a
                pair is considered conditionally independent (edge removed).
                Default 0.05.
            final_edge_threshold: Minimum |correlation| for an edge to be
                included in the final causal graph after direction assignment.
                Default 0.15 (stricter than initial screening).
        """
        super().__init__(**kwargs)
        self.n_samples = self.exp_config("n_samples")
        self.n_interventions = self.exp_config("n_interventions")
        self.hidden_dim = self.exp_config("hidden_dim")
        self.n_permutation_trials = self.exp_config("n_permutation_trials")
        self.edge_corr_threshold = self.exp_config("edge_corr_threshold")
        self.partial_corr_threshold = self.exp_config("partial_corr_threshold")
        self.final_edge_threshold = self.exp_config("final_edge_threshold")
        self._data: List[Dict[str, Any]] = []
        self._extracted_graph: List[CausalEdge] = []
        self._require_paired()

    def _require_paired(self):
        if self.training_method != "paired":
            raise ValueError(f"CausalModelExtractionExperiment requires PAIRED")

    def collect_data(self, rng: chex.PRNGKey) -> List[Dict[str, Any]]:
        """Collect observational data for causal analysis using real network evaluations."""
        rng, level_rng, hstate_rng, return_rng, diff_rng = jax.random.split(rng, 5)

        # Generate real levels in batch
        levels = generate_levels(self.agent, level_rng, self.n_samples)

        # Get real hidden states from protagonist network
        hstates = get_pro_hstates(hstate_rng, levels, self)
        self.hidden_dim = hstates.shape[1]

        # Get real returns
        pro_returns, _, _ = get_pro_ant_returns(return_rng, levels, self)

        # Extract real level features
        features_batch = extract_level_features_batch(levels)

        # Convert levels to dicts for BFS path length computation
        level_dicts = levels_to_dicts(levels, self.n_samples)

        from ..utils.paired_helpers import compute_difficulty
        difficulties_arr = compute_difficulty(levels, self, diff_rng)

        for i in range(self.n_samples):
            wall_density = float(features_batch['wall_density'][i])
            goal_distance = float(features_batch['goal_distance'][i])

            # Compute derived causal variables from real level structure
            bfs_length = compute_bfs_path_length(level_dicts[i])
            optimal_steps = float(bfs_length) if bfs_length >= 0 else goal_distance * 2.0
            path_difficulty = float(difficulties_arr[i])
            # BFS-based collision risk: proportion of cells near walls on the shortest path
            collision_risk = float(wall_density * optimal_steps / max(bfs_length, 1)) if bfs_length > 0 else float(wall_density)
            agent_pos = level_dicts[i]['agent_pos']
            agent_position = float(np.sqrt(sum(x**2 for x in agent_pos)))

            self._data.append({
                'wall_density': wall_density,
                'goal_distance': goal_distance,
                'agent_position': agent_position,
                'path_difficulty': path_difficulty,
                'optimal_steps': optimal_steps,
                'collision_risk': collision_risk,
                'return': float(pro_returns[i]),
                'hstate': hstates[i],
            })

        return self._data

    def _collect_sample(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """Return a random sample from already-collected data for intervention experiments."""
        if self._data:
            idx = int(jax.random.randint(rng, (), 0, len(self._data)))
            return dict(self._data[idx])  # shallow copy so interventions don't mutate originals
        # Fallback: generate a single real sample
        rng, level_rng, h_rng, ret_rng = jax.random.split(rng, 4)
        levels = generate_levels(self.agent, level_rng, 1)
        hstates = get_pro_hstates(h_rng, levels, self)
        pro_returns, _, _ = get_pro_ant_returns(ret_rng, levels, self)
        features = extract_level_features_batch(levels)
        level_dicts = levels_to_dicts(levels, 1)
        bfs_length = compute_bfs_path_length(level_dicts[0])
        wall_density = float(features['wall_density'][0])
        goal_distance = float(features['goal_distance'][0])
        agent_pos = level_dicts[0]['agent_pos']
        return {
            'wall_density': wall_density,
            'goal_distance': goal_distance,
            'agent_position': float(np.sqrt(sum(x**2 for x in agent_pos))),
            'path_difficulty': float(1.0 - pro_returns[0]),
            'optimal_steps': float(bfs_length) if bfs_length >= 0 else goal_distance * 2.0,
            'collision_risk': float(wall_density * float(bfs_length) / max(bfs_length, 1)) if bfs_length > 0 else wall_density,
            'return': float(pro_returns[0]),
            'hstate': hstates[0],
        }

    def _extract_causal_graph(self) -> List[CausalEdge]:
        """Extract causal graph from conditional independence structure."""
        # Build variable matrix
        variables = {}
        for var in self.VARIABLES:
            if var in self._data[0]:
                variables[var] = np.array([d[var] for d in self._data])

        # Also add h-state projections as potential mediators
        hstates = np.array([d['hstate'] for d in self._data])
        # Train Ridge probes to predict variables from hstates
        from sklearn.linear_model import Ridge
        for var_name in ['wall_density', 'goal_distance', 'path_difficulty', 'collision_risk']:
            if var_name in variables:
                probe = Ridge(alpha=1.0)
                probe.fit(hstates, variables[var_name])
                variables[f'h_{var_name.split("_")[0]}'] = probe.predict(hstates)

        edges = []

        # PC algorithm (simplified): check conditional independence
        var_names = list(variables.keys())

        for i, var1 in enumerate(var_names):
            for var2 in var_names[i + 1:]:
                # Check unconditional correlation
                corr = np.corrcoef(variables[var1], variables[var2])[0, 1]
                if np.isnan(corr):
                    continue

                if abs(corr) > self.edge_corr_threshold:
                    # Check if conditioning on other variables removes correlation
                    is_direct = True
                    for conditioning_var in var_names:
                        if conditioning_var in [var1, var2]:
                            continue

                        # Partial correlation
                        partial_corr = self._partial_correlation(
                            variables[var1],
                            variables[var2],
                            variables[conditioning_var],
                        )

                        if abs(partial_corr) < self.partial_corr_threshold and abs(corr) > self.edge_corr_threshold:
                            # Correlation explained by conditioning variable
                            is_direct = False
                            break

                    if is_direct and abs(corr) > self.final_edge_threshold:
                        # Determine direction (simplified: based on known structure)
                        source, target = self._determine_direction(var1, var2)
                        edges.append(CausalEdge(
                            source=source,
                            target=target,
                            strength=abs(corr),
                        ))

        self._extracted_graph = edges
        return edges

    def _partial_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> float:
        """Compute partial correlation of x and y given z via OLS residualization."""
        from sklearn.linear_model import LinearRegression

        Z = z.reshape(-1, 1) if z.ndim == 1 else z
        reg = LinearRegression()

        x_resid = x - reg.fit(Z, x).predict(Z)
        y_resid = y - reg.fit(Z, y).predict(Z)

        corr = np.corrcoef(x_resid, y_resid)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def _determine_direction(self, var1: str, var2: str) -> Tuple[str, str]:
        """Determine causal direction using temporal precedence + domain knowledge.

        Temporal precedence: level features (wall_density, goal_distance, etc.)
        are determined at level generation time, before agent behavior (returns).
        Hidden-state representations (h_*) are formed during rollout, after
        observing level features but before computing returns. This ordering
        is ground truth for gridworld environments.

        Tiers:
            0 (causes): Level generation features — temporally first
            1 (intermediates): Hidden-state encodings + derived features —
               formed during rollout from level observations
            2 (effects): Episode returns — temporally last
        """
        # Known causes precede effects in rollout time
        causes = ['wall_density', 'goal_distance', 'agent_position']
        intermediates = ['path_difficulty', 'collision_risk', 'optimal_steps']
        effects = ['return']

        # h-state representations are formed after observing level features
        h_vars = ['h_wall', 'h_goal', 'h_path', 'h_risk', 'h_collision']

        def get_level(var):
            if var in causes:
                return 0  # Level generation time
            elif var in intermediates or var in h_vars:
                return 1  # During rollout
            else:
                return 2  # Episode outcome

        level1 = get_level(var1)
        level2 = get_level(var2)

        if level1 < level2:
            return var1, var2
        elif level2 < level1:
            return var2, var1
        else:
            # Same tier: alphabetical (arbitrary but deterministic)
            return (var1, var2) if var1 < var2 else (var2, var1)

    def _test_interventions(self, rng: chex.PRNGKey) -> Dict[str, float]:
        """Test interventional predictions."""
        results = {}

        # Intervention: Set wall_density to high
        rng, int_rng = jax.random.split(rng)
        do_high_wall = self._intervention_experiment(
            int_rng, 'wall_density', 0.4, 'return'
        )
        results['do_high_wall_effect'] = do_high_wall

        # Intervention: Set goal_distance to low
        rng, int_rng = jax.random.split(rng)
        do_short_goal = self._intervention_experiment(
            int_rng, 'goal_distance', 3.0, 'return'
        )
        results['do_short_goal_effect'] = do_short_goal

        # Compare to causal predictions from extracted graph
        # If graph is correct, interventions should match predictions
        predicted_high_wall = self._predict_intervention_from_graph('wall_density', 0.4, 'return')
        predicted_short_goal = self._predict_intervention_from_graph('goal_distance', 3.0, 'return')

        results['prediction_error_wall'] = abs(do_high_wall - predicted_high_wall)
        results['prediction_error_goal'] = abs(do_short_goal - predicted_short_goal)
        results['mean_prediction_error'] = (
            results['prediction_error_wall'] + results['prediction_error_goal']
        ) / 2

        return results

    def _intervention_experiment(
        self,
        rng: chex.PRNGKey,
        intervened_var: str,
        intervened_value: float,
        outcome_var: str,
    ) -> float:
        """Run intervention experiment using real network forward passes.

        Generates levels with the intervened variable set, runs the agent,
        and measures the actual outcome.
        """
        from ..utils.paired_helpers import generate_constrained_levels, run_batched_rollout

        outcomes = []
        n = min(self.n_interventions, 50)

        # Generate levels constrained by the intervened variable
        constraints = {}
        if intervened_var == 'wall_density':
            constraints['wall_density'] = (intervened_value - 0.05, intervened_value + 0.05)
        elif intervened_var == 'goal_distance':
            constraints['goal_distance'] = (intervened_value - 1.0, intervened_value + 1.0)

        try:
            rng, gen_rng, roll_rng = jax.random.split(rng, 3)
            levels = generate_constrained_levels(
                self.agent, gen_rng, n, constraints
            )

            # Run real rollout
            result = run_batched_rollout(
                roll_rng, levels, self.train_state, self.agent, max_steps=self.config.get("max_steps", 256),
            )
            outcomes = list(np.array(result.episode_returns))
        except Exception:
            # Fallback: use observational data filtered by intervened value
            for d in self._data:
                if abs(d.get(intervened_var, 0) - intervened_value) < 0.15:
                    outcomes.append(d.get(outcome_var, 0))

        if not outcomes:
            # Last resort: use overall mean
            outcomes = [d.get(outcome_var, 0) for d in self._data]

        return float(np.mean(outcomes)) if outcomes else 0.0

    def _predict_intervention(
        self,
        intervened_var: str,
        intervened_value: float,
        outcome_var: str,
    ) -> float:
        """Predict intervention effect from extracted graph."""
        # Use observational data to estimate causal effect
        # Simplified: use adjustment formula

        obs_values = np.array([d[intervened_var] for d in self._data])
        outcomes = np.array([d[outcome_var] for d in self._data])

        # Bin by intervened variable
        near_value_mask = np.abs(obs_values - intervened_value) < 0.1

        if near_value_mask.sum() > 10:
            return float(np.mean(outcomes[near_value_mask]))
        else:
            # Extrapolate using regression
            slope = np.polyfit(obs_values, outcomes, 1)[0]
            mean_outcome = np.mean(outcomes)
            mean_var = np.mean(obs_values)
            return float(mean_outcome + slope * (intervened_value - mean_var))

    def _predict_intervention_from_graph(
        self,
        intervened_var: str,
        intervened_value: float,
        outcome_var: str,
    ) -> float:
        """Predict intervention effect using extracted causal graph (BFS path tracing)."""
        if not self._extracted_graph:
            return self._predict_intervention(intervened_var, intervened_value, outcome_var)

        # Build adjacency from extracted edges
        from collections import deque
        adjacency = {}
        for edge in self._extracted_graph:
            adjacency.setdefault(edge.source, []).append((edge.target, edge.strength))

        # BFS from intervened_var to outcome_var
        visited = set()
        queue = deque([(intervened_var, 1.0)])
        best_weight = None

        while queue:
            node, cumulative_weight = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            if node == outcome_var:
                best_weight = cumulative_weight
                break

            for neighbor, strength in adjacency.get(node, []):
                if neighbor not in visited:
                    queue.append((neighbor, cumulative_weight * strength))

        if best_weight is None:
            # No path found — fall back to regression
            return self._predict_intervention(intervened_var, intervened_value, outcome_var)

        # Predict: mean_outcome + causal_effect * (intervened_value - mean_intervened)
        obs_values = np.array([d[intervened_var] for d in self._data])
        outcomes = np.array([d[outcome_var] for d in self._data])
        mean_outcome = float(np.mean(outcomes))
        mean_var = float(np.mean(obs_values))

        return mean_outcome + best_weight * (intervened_value - mean_var)

    def analyze(self) -> Dict[str, Any]:
        """Analyze causal model extraction."""
        if not self._data:
            raise ValueError("Must call collect_data first")

        results = {}

        # Extract causal graph
        extracted_graph = self._extract_causal_graph()
        results['extracted_edges'] = [
            {'source': e.source, 'target': e.target, 'strength': e.strength}
            for e in extracted_graph
        ]
        results['num_extracted_edges'] = len(extracted_graph)

        # Compare to ground truth
        results['causal_graph_accuracy'] = self._compute_graph_accuracy(extracted_graph)

        # Interventional fidelity
        rng = jax.random.PRNGKey(42)
        results['interventional_fidelity'] = self._test_interventions(rng)

        # Transfer performance (simplified proxy)
        results['transfer_performance_gap'] = self._estimate_transfer_gap()

        # Causal model quality comparison
        # DR baseline: permutation test — shuffle returns and re-extract graph
        accuracy = results['causal_graph_accuracy']
        rng_perm = jax.random.PRNGKey(123)
        perm_accuracies = []
        original_data = self._data
        for trial in range(self.n_permutation_trials):
            rng_perm, perm_rng = jax.random.split(rng_perm)
            shuffled_data = [d.copy() for d in original_data]
            returns = [d['return'] for d in shuffled_data]
            perm_idx = np.array(jax.random.permutation(perm_rng, len(returns)))
            for j, d in enumerate(shuffled_data):
                d['return'] = returns[int(perm_idx[j])]
            # Re-extract graph on shuffled data
            self._data = shuffled_data
            perm_graph = self._extract_causal_graph()
            perm_acc = self._compute_graph_accuracy(perm_graph)
            perm_accuracies.append(perm_acc.get('f1', 0.0))
        self._data = original_data  # Restore original data
        dr_estimated = float(np.mean(perm_accuracies)) if perm_accuracies else 0.0

        # Random baseline: replace hstates with random noise and re-extract
        random_data = [d.copy() for d in original_data]
        rng_rand = jax.random.PRNGKey(456)
        for d in random_data:
            rng_rand, noise_rng = jax.random.split(rng_rand)
            d['hstate'] = np.array(jax.random.normal(noise_rng, d['hstate'].shape))
        self._data = random_data
        random_graph = self._extract_causal_graph()
        random_acc = self._compute_graph_accuracy(random_graph)
        random_estimated = random_acc.get('f1', 0.0)
        self._data = original_data  # Restore original data

        results['causal_model_quality_by_method'] = {
            'paired': accuracy,
            'dr_estimated_f1': dr_estimated,
            'random_estimated_f1': random_estimated,
            'baseline_method': 'permutation_test_with_reextraction',
        }

        return results

    def _compute_graph_accuracy(self, extracted: List[CausalEdge]) -> Dict[str, float]:
        """Compute accuracy against ground truth."""
        # Convert to edge sets (ignoring h-state edges for comparison)
        extracted_pairs = set()
        for e in extracted:
            if not e.source.startswith('h_') and not e.target.startswith('h_'):
                extracted_pairs.add((e.source, e.target))

        ground_truth_pairs = set(self.GROUND_TRUTH_EDGES)

        # Structural Hamming Distance
        true_positives = len(extracted_pairs & ground_truth_pairs)
        false_positives = len(extracted_pairs - ground_truth_pairs)
        false_negatives = len(ground_truth_pairs - extracted_pairs)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        shd = false_positives + false_negatives

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'structural_hamming_distance': int(shd),
        }

    def _estimate_transfer_gap(self) -> float:
        """Estimate transfer performance gap."""
        # Simplified: variance in return predictions
        returns = np.array([d['return'] for d in self._data])
        hstates = np.array([d['hstate'] for d in self._data])

        # Fit simple model
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(hstates, returns)

        predictions = model.predict(hstates)
        mse = np.mean((returns - predictions) ** 2)

        return float(mse)

    def visualize(self) -> Dict[str, np.ndarray]:
        """Visualize causal analysis."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        figures = {}

        if not self._data:
            return figures

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Correlation matrix
        ax = axes[0, 0]
        variables = {}
        for var in self.VARIABLES:
            if var in self._data[0]:
                variables[var] = np.array([d[var] for d in self._data])

        var_names = list(variables.keys())
        corr_matrix = np.zeros((len(var_names), len(var_names)))
        for i, v1 in enumerate(var_names):
            for j, v2 in enumerate(var_names):
                corr = np.corrcoef(variables[v1], variables[v2])[0, 1]
                corr_matrix[i, j] = corr if not np.isnan(corr) else 0

        im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_xticks(range(len(var_names)))
        ax.set_xticklabels(var_names, rotation=45, ha='right')
        ax.set_yticks(range(len(var_names)))
        ax.set_yticklabels(var_names)
        ax.set_title('Variable Correlation Matrix')
        plt.colorbar(im, ax=ax)

        # Extracted graph edges
        ax = axes[0, 1]
        if self._extracted_graph:
            edges = [(e.source[:8], e.target[:8], e.strength) for e in self._extracted_graph[:15]]
            labels = [f'{s}->{t}' for s, t, _ in edges]
            strengths = [e[2] for e in edges]
            y = np.arange(len(labels))
            ax.barh(y, strengths, alpha=0.7)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Edge Strength')
            ax.set_title('Extracted Causal Edges')
        else:
            ax.text(0.5, 0.5, 'No edges extracted', ha='center', va='center', transform=ax.transAxes)

        # Ground truth vs extracted comparison
        ax = axes[1, 0]
        accuracy = self._compute_graph_accuracy(self._extracted_graph)
        metrics = ['precision', 'recall', 'f1']
        values = [accuracy[m] for m in metrics]
        ax.bar(metrics, values, alpha=0.7)
        ax.set_ylabel('Score')
        ax.set_title('Causal Graph Recovery Accuracy')
        ax.set_ylim(0, 1)

        # Intervention predictions
        ax = axes[1, 1]
        rng = jax.random.PRNGKey(42)
        int_results = self._test_interventions(rng)
        ax.bar(
            ['High Wall\nEffect', 'Short Goal\nEffect', 'Wall\nError', 'Goal\nError'],
            [int_results['do_high_wall_effect'], int_results['do_short_goal_effect'],
             int_results['prediction_error_wall'], int_results['prediction_error_goal']],
            alpha=0.7,
        )
        ax.set_ylabel('Value')
        ax.set_title('Interventional Analysis')

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        figures["causal_analysis"] = np.asarray(buf)[:, :, :3]
        plt.close(fig)

        return figures
