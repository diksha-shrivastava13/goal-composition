"""
Base experiment class for interpretability experiments.

All experiments follow the same pattern:
1. Load checkpoint
2. Collect data (activations, predictions, etc.)
3. Run analysis
4. Generate metrics and visualizations

Supports multiple training methods:
- accel: ACCEL with DR→Replay→Mutate branches
- plr: Prioritized Level Replay
- robust_plr: Robust PLR
- paired: PAIRED adversarial curriculum (no branches, regret-driven)
- dr: Domain Randomization (no curriculum structure)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Set
import os
import json

import jax
import jax.numpy as jnp
import numpy as np
import chex


# Training methods and their characteristics
TRAINING_METHODS = {
    "accel": {
        "has_branches": True,
        "has_replay_buffer": True,
        "has_mutations": True,
        "has_adversary": False,
        "has_regret": False,
        "branch_count": 3,  # DR=0, Replay=1, Mutate=2
    },
    "plr": {
        "has_branches": True,
        "has_replay_buffer": True,
        "has_mutations": False,
        "has_adversary": False,
        "has_regret": False,
        "branch_count": 2,  # DR=0, Replay=1
    },
    "robust_plr": {
        "has_branches": True,
        "has_replay_buffer": True,
        "has_mutations": False,
        "has_adversary": False,
        "has_regret": False,
        "branch_count": 2,  # DR=0, Replay=1
    },
    "paired": {
        "has_branches": False,
        "has_replay_buffer": False,
        "has_mutations": False,
        "has_adversary": True,
        "has_regret": True,
        "branch_count": 0,
    },
    "dr": {
        "has_branches": False,
        "has_replay_buffer": False,
        "has_mutations": False,
        "has_adversary": False,
        "has_regret": False,
        "branch_count": 0,
    },
}


def get_method_properties(training_method: str) -> Dict[str, Any]:
    """Get properties for a training method."""
    if training_method not in TRAINING_METHODS:
        raise ValueError(
            f"Unknown training method: {training_method}. "
            f"Available: {list(TRAINING_METHODS.keys())}"
        )
    return TRAINING_METHODS[training_method]


class BaseExperiment(ABC):
    """
    Abstract base class for interpretability experiments.

    Subclasses must implement:
    - collect_data: Gather necessary data from agent
    - analyze: Run the analysis
    - visualize: Create visualizations
    """

    def __init__(
        self,
        agent,
        train_state,
        config: dict,
        output_dir: Optional[str] = None,
        training_method: str = "accel",
    ):
        """
        Initialize experiment.

        Args:
            agent: Trained agent instance
            train_state: Loaded train state with params
            config: Experiment configuration
            output_dir: Directory to save results
            training_method: Training method used (accel, plr, robust_plr, paired, dr)
        """
        self.agent = agent
        self.train_state = train_state
        self.config = config
        self.output_dir = output_dir or "."
        self.training_method = training_method

        # Get method-specific properties
        self.method_properties = get_method_properties(training_method)

        self.data = {}
        self.results = {}
        self.figures = {}

    @property
    def has_branches(self) -> bool:
        """Whether the training method uses curriculum branches."""
        return self.method_properties["has_branches"]

    @property
    def has_replay_buffer(self) -> bool:
        """Whether the training method uses a replay buffer."""
        return self.method_properties["has_replay_buffer"]

    @property
    def has_mutations(self) -> bool:
        """Whether the training method uses level mutations."""
        return self.method_properties["has_mutations"]

    @property
    def has_adversary(self) -> bool:
        """Whether the training method uses an adversarial level generator."""
        return self.method_properties["has_adversary"]

    @property
    def has_regret(self) -> bool:
        """Whether the training method uses regret-based curriculum."""
        return self.method_properties["has_regret"]

    @property
    def branch_count(self) -> int:
        """Number of curriculum branches (0 if no branches)."""
        return self.method_properties["branch_count"]

    @property
    @abstractmethod
    def name(self) -> str:
        """Experiment name for logging and file naming."""
        pass

    @abstractmethod
    def collect_data(self, rng: chex.PRNGKey) -> Dict[str, Any]:
        """
        Collect necessary data for the experiment.

        This typically involves running the agent on levels and
        collecting activations, predictions, etc.

        Args:
            rng: Random key for reproducibility

        Returns:
            Dictionary of collected data
        """
        pass

    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """
        Run analysis on collected data.

        Returns:
            Dictionary of analysis results
        """
        pass

    @abstractmethod
    def visualize(self) -> Dict[str, np.ndarray]:
        """
        Create visualizations of results.

        Returns:
            Dictionary mapping figure names to numpy arrays (RGB images)
        """
        pass

    def run(self, rng: chex.PRNGKey, save: bool = True) -> Dict[str, Any]:
        """
        Run the full experiment pipeline.

        Args:
            rng: Random key
            save: Whether to save results to disk

        Returns:
            Dictionary with all results
        """
        print(f"Running experiment: {self.name}")

        # Collect data
        print("  Collecting data...")
        self.data = self.collect_data(rng)

        # Analyze
        print("  Analyzing...")
        self.results = self.analyze()

        # Visualize
        print("  Creating visualizations...")
        self.figures = self.visualize()

        # Save if requested
        if save:
            self.save()

        return {
            "name": self.name,
            "data": self.data,
            "results": self.results,
            "figures": self.figures,
        }

    def save(self):
        """Save results and figures to disk."""
        exp_dir = os.path.join(self.output_dir, self.name)
        os.makedirs(exp_dir, exist_ok=True)

        # Save results as JSON
        results_path = os.path.join(exp_dir, "results.json")
        serializable_results = self._make_serializable(self.results)
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        # Save figures
        import matplotlib.pyplot as plt
        for fig_name, fig_array in self.figures.items():
            fig_path = os.path.join(exp_dir, f"{fig_name}.png")
            plt.imsave(fig_path, fig_array)

        print(f"  Results saved to: {exp_dir}")

    def _make_serializable(self, obj):
        """Convert numpy/jax arrays to lists for JSON serialization."""
        if isinstance(obj, (np.ndarray, jnp.ndarray)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        else:
            return obj


class CheckpointExperiment(BaseExperiment):
    """
    Experiment that only needs checkpoint (no training-time hooks).

    Most experiments fall into this category - they can be run
    post-hoc on any saved checkpoint.
    """

    @property
    def requires_training_hooks(self) -> bool:
        return False


class TrainingTimeExperiment(BaseExperiment):
    """
    Experiment that requires data collected during training.

    These experiments need special hooks in the training loop
    to collect data (e.g., symbolic regression, behavioral coupling).
    """

    @property
    def requires_training_hooks(self) -> bool:
        return True

    @abstractmethod
    def training_hook(
        self,
        train_state,
        metrics: dict,
        step: int,
    ) -> dict:
        """
        Hook called during training to collect data.

        Args:
            train_state: Current train state
            metrics: Current training metrics
            step: Training step number

        Returns:
            Data to store for later analysis
        """
        pass
