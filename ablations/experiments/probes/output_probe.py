"""
Output probes for policy and value analysis.

Contains probes that analyze agent outputs (policy, value) rather than
hidden states, testing whether curriculum information leaks into behavior.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import jax
import jax.numpy as jnp
import chex


class PolicyEntropyAnalyzer:
    """
    Analyzes policy entropy patterns across curriculum branches.

    Tests hypothesis: Replay levels have lower entropy (more confident)
    than DR or Mutate levels (less practiced).
    """

    def __init__(self):
        self.branch_entropies = {0: [], 1: [], 2: []}  # DR, Replay, Mutate
        self.branch_names = {0: "DR", 1: "Replay", 2: "Mutate"}

    def add_sample(
        self,
        policy_logits: np.ndarray,
        branch: int,
    ):
        """
        Add policy sample for a branch.

        Args:
            policy_logits: Logits from policy head, shape (n_actions,) or (batch, n_actions)
            branch: Branch index (0=DR, 1=Replay, 2=Mutate)
        """
        logits = np.asarray(policy_logits)
        if logits.ndim == 1:
            logits = logits[None, :]

        # Compute entropy for each sample in batch
        probs = _softmax(logits)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)

        self.branch_entropies[branch].extend(entropy.tolist())

    def analyze(self) -> Dict[str, any]:
        """
        Analyze entropy patterns across branches.

        Returns:
            Dict with per-branch statistics and comparisons
        """
        results = {}

        for branch_id, entropies in self.branch_entropies.items():
            if len(entropies) > 0:
                name = self.branch_names[branch_id]
                results[f"{name}_mean_entropy"] = float(np.mean(entropies))
                results[f"{name}_std_entropy"] = float(np.std(entropies))
                results[f"{name}_n_samples"] = len(entropies)

        # Compare branches
        if len(self.branch_entropies[1]) > 0 and len(self.branch_entropies[0]) > 0:
            # Replay vs DR
            replay_mean = np.mean(self.branch_entropies[1])
            dr_mean = np.mean(self.branch_entropies[0])
            results["replay_vs_dr_diff"] = float(replay_mean - dr_mean)
            results["replay_lower_entropy"] = replay_mean < dr_mean

        if len(self.branch_entropies[2]) > 0 and len(self.branch_entropies[1]) > 0:
            # Mutate vs Replay
            mutate_mean = np.mean(self.branch_entropies[2])
            replay_mean = np.mean(self.branch_entropies[1])
            results["mutate_vs_replay_diff"] = float(mutate_mean - replay_mean)

        return results


class BranchClassifier:
    """
    Classifies curriculum branch from policy/value outputs.

    Tests: Does agent behavior implicitly encode curriculum awareness?
    If branch can be predicted from outputs, the agent's behavior
    systematically differs by curriculum context.
    """

    def __init__(self, cv_folds: int = 5):
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, multi_class="multinomial")
        self.fitted = False

    def fit(
        self,
        policy_logits: np.ndarray,
        values: np.ndarray,
        branches: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train classifier to predict branch from outputs.

        Args:
            policy_logits: Policy logits, shape (n_samples, n_actions)
            values: Value estimates, shape (n_samples,)
            branches: Branch labels, shape (n_samples,)

        Returns:
            Dict with cross-validation metrics
        """
        policy_logits = np.asarray(policy_logits)
        values = np.asarray(values).reshape(-1, 1)
        branches = np.asarray(branches)

        # Compute policy features
        probs = _softmax(policy_logits)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1, keepdims=True)
        max_prob = np.max(probs, axis=-1, keepdims=True)

        # Concatenate features
        X = np.concatenate([policy_logits, values, entropy, max_prob], axis=-1)
        X_scaled = self.scaler.fit_transform(X)

        # Cross-validation
        scores = cross_val_score(
            self.model, X_scaled, branches,
            cv=self.cv_folds, scoring="accuracy"
        )

        # Fit on full data
        self.model.fit(X_scaled, branches)
        self.fitted = True

        # Random baseline
        unique_classes = np.unique(branches)
        random_baseline = 1.0 / len(unique_classes)

        return {
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
            "random_baseline": random_baseline,
            "above_random": float(np.mean(scores)) > random_baseline + 0.05,
            "scores": scores.tolist(),
        }

    def predict(
        self,
        policy_logits: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """Predict branch from outputs."""
        if not self.fitted:
            raise RuntimeError("Classifier must be fitted first")

        policy_logits = np.asarray(policy_logits)
        values = np.asarray(values).reshape(-1, 1)

        probs = _softmax(policy_logits)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1, keepdims=True)
        max_prob = np.max(probs, axis=-1, keepdims=True)

        X = np.concatenate([policy_logits, values, entropy, max_prob], axis=-1)
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)


class OutputProbe:
    """
    General probe for curriculum properties from agent outputs.

    Similar to property probes but operates on policy/value outputs
    rather than hidden states.
    """

    def __init__(
        self,
        task: str = "regression",
        alpha: float = 1.0,
        cv_folds: int = 5,
    ):
        self.task = task
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        self.model = None
        self.fitted = False

    def fit(
        self,
        policy_logits: np.ndarray,
        values: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, float]:
        """
        Fit probe to predict target from outputs.

        Args:
            policy_logits: Policy logits, shape (n_samples, n_actions)
            values: Value estimates, shape (n_samples,)
            targets: Target to predict

        Returns:
            Dict with metrics
        """
        policy_logits = np.asarray(policy_logits)
        values = np.asarray(values).reshape(-1, 1)
        targets = np.asarray(targets)

        # Build feature vector
        probs = _softmax(policy_logits)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1, keepdims=True)
        max_prob = np.max(probs, axis=-1, keepdims=True)
        action_probs_sorted = np.sort(probs, axis=-1)[:, ::-1]  # Top-k probs

        X = np.concatenate([
            policy_logits,
            values,
            entropy,
            max_prob,
            action_probs_sorted[:, :3],  # Top 3 action probs
        ], axis=-1)

        X_scaled = self.scaler.fit_transform(X)

        if self.task == "regression":
            self.model = Ridge(alpha=self.alpha)
            scoring = "r2"
        else:
            self.model = LogisticRegression(C=1/self.alpha, max_iter=1000)
            scoring = "accuracy"

        scores = cross_val_score(
            self.model, X_scaled, targets,
            cv=self.cv_folds, scoring=scoring
        )

        self.model.fit(X_scaled, targets)
        self.fitted = True

        return {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "scores": scores.tolist(),
        }


class ValueCalibrationAnalyzer:
    """
    Analyzes value function calibration by branch type.

    Tests: Is value calibrated uniformly across curriculum branches,
    or does it differ (e.g., better on Replay, worse on DR)?
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.branch_data = {0: {"values": [], "returns": []},
                           1: {"values": [], "returns": []},
                           2: {"values": [], "returns": []}}
        self.branch_names = {0: "DR", 1: "Replay", 2: "Mutate"}

    def add_sample(
        self,
        value: float,
        actual_return: float,
        branch: int,
    ):
        """Add a (value, return) pair for a branch."""
        self.branch_data[branch]["values"].append(value)
        self.branch_data[branch]["returns"].append(actual_return)

    def compute_ece(self, values: np.ndarray, returns: np.ndarray) -> float:
        """Compute Expected Calibration Error."""
        if len(values) == 0:
            return 0.0

        values = np.asarray(values)
        returns = np.asarray(returns)

        bin_edges = np.linspace(values.min() - 1e-5, values.max() + 1e-5, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            mask = (values >= bin_edges[i]) & (values < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_conf = values[mask].mean()
                bin_acc = returns[mask].mean()
                bin_size = mask.sum()
                ece += bin_size * np.abs(bin_conf - bin_acc)

        return float(ece / len(values))

    def analyze(self) -> Dict[str, float]:
        """Compute calibration metrics per branch."""
        results = {}

        for branch_id, data in self.branch_data.items():
            name = self.branch_names[branch_id]
            values = np.array(data["values"])
            returns = np.array(data["returns"])

            if len(values) > 10:
                results[f"{name}_ece"] = self.compute_ece(values, returns)
                results[f"{name}_correlation"] = float(np.corrcoef(values, returns)[0, 1])
                results[f"{name}_mae"] = float(np.mean(np.abs(values - returns)))
                results[f"{name}_n_samples"] = len(values)

        return results


class ActorCriticDivergenceTracker:
    """
    Tracks divergence between actor and critic representations over training.

    If actor and critic develop different representations, this suggests
    specialized goal encoding in each head.
    """

    def __init__(self):
        self.history = {
            "steps": [],
            "policy_entropy_mean": [],
            "value_variance": [],
            "policy_value_correlation": [],
        }

    def add_checkpoint(
        self,
        step: int,
        policy_logits: np.ndarray,
        values: np.ndarray,
    ):
        """
        Record metrics at a training checkpoint.

        Args:
            step: Training step
            policy_logits: Policy logits, shape (n_samples, n_actions)
            values: Value estimates, shape (n_samples,)
        """
        policy_logits = np.asarray(policy_logits)
        values = np.asarray(values)

        probs = _softmax(policy_logits)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)

        self.history["steps"].append(step)
        self.history["policy_entropy_mean"].append(float(np.mean(entropy)))
        self.history["value_variance"].append(float(np.var(values)))

        # Correlation between max action prob and value
        max_probs = np.max(probs, axis=-1)
        if np.std(max_probs) > 0 and np.std(values) > 0:
            corr = np.corrcoef(max_probs, values)[0, 1]
        else:
            corr = 0.0
        self.history["policy_value_correlation"].append(float(corr))

    def get_divergence_metrics(self) -> Dict[str, any]:
        """Compute divergence metrics from history."""
        if len(self.history["steps"]) < 2:
            return {"insufficient_data": True}

        steps = np.array(self.history["steps"])
        entropy = np.array(self.history["policy_entropy_mean"])
        variance = np.array(self.history["value_variance"])
        corr = np.array(self.history["policy_value_correlation"])

        return {
            "entropy_trend": float(np.polyfit(steps, entropy, 1)[0]),
            "variance_trend": float(np.polyfit(steps, variance, 1)[0]),
            "correlation_trend": float(np.polyfit(steps, corr, 1)[0]),
            "final_correlation": float(corr[-1]),
            "history": self.history,
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax along last axis."""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
