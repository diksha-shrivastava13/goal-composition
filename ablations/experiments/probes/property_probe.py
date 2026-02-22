"""
Property probes for level/curriculum feature prediction.

Contains linear, MLP, and distributed probes for analyzing what information
is encoded in agent hidden states.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, f1_score
import jax.numpy as jnp
import chex


class LinearPropertyProbe:
    """
    Linear probe for predicting level properties from hidden states.

    Uses Ridge regression for continuous targets and Logistic regression
    for classification tasks.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        task: str = "regression",
        cv_folds: int = 5,
    ):
        """
        Args:
            alpha: Regularization strength
            task: "regression" or "classification"
            cv_folds: Number of cross-validation folds
        """
        self.alpha = alpha
        self.task = task
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        self.model = None
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Fit probe and return cross-validation metrics.

        Args:
            X: Hidden states, shape (n_samples, hidden_dim)
            y: Targets, shape (n_samples,) or (n_samples, n_targets)

        Returns:
            Dict with cv metrics (mean_score, std_score, scores)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        if self.task == "regression":
            self.model = Ridge(alpha=self.alpha)
            scoring = "r2"
        else:
            self.model = LogisticRegression(C=1/self.alpha, max_iter=1000)
            scoring = "accuracy"

        # Cross-validation
        scores = cross_val_score(
            self.model, X_scaled, y,
            cv=self.cv_folds, scoring=scoring
        )

        # Fit on full data
        self.model.fit(X_scaled, y)
        self.fitted = True

        return {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "scores": scores.tolist(),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for new hidden states."""
        if not self.fitted:
            raise RuntimeError("Probe must be fitted before prediction")
        X_scaled = self.scaler.transform(np.asarray(X))
        return self.model.predict(X_scaled)

    def get_weights(self) -> np.ndarray:
        """Get learned weights for interpretability."""
        if not self.fitted:
            raise RuntimeError("Probe must be fitted first")
        return self.model.coef_


class MLPPropertyProbe:
    """
    Non-linear MLP probe for detecting non-linear encodings in hidden states.

    Compares with LinearPropertyProbe to assess if information is linearly
    encoded or requires non-linear decoding.
    """

    def __init__(
        self,
        hidden_layers: Tuple[int, ...] = (128, 64),
        alpha: float = 1e-4,
        task: str = "regression",
        cv_folds: int = 5,
        max_iter: int = 500,
    ):
        """
        Args:
            hidden_layers: Tuple of hidden layer sizes
            alpha: L2 regularization
            task: "regression" or "classification"
            cv_folds: Number of cross-validation folds
            max_iter: Maximum training iterations
        """
        self.hidden_layers = hidden_layers
        self.alpha = alpha
        self.task = task
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.model = None
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Fit probe and return cross-validation metrics."""
        X = np.asarray(X)
        y = np.asarray(y)

        X_scaled = self.scaler.fit_transform(X)

        if self.task == "regression":
            self.model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                alpha=self.alpha,
                max_iter=self.max_iter,
                early_stopping=True,
                validation_fraction=0.1,
            )
            scoring = "r2"
        else:
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layers,
                alpha=self.alpha,
                max_iter=self.max_iter,
                early_stopping=True,
                validation_fraction=0.1,
            )
            scoring = "accuracy"

        scores = cross_val_score(
            self.model, X_scaled, y,
            cv=self.cv_folds, scoring=scoring
        )

        self.model.fit(X_scaled, y)
        self.fitted = True

        return {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "scores": scores.tolist(),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for new hidden states."""
        if not self.fitted:
            raise RuntimeError("Probe must be fitted before prediction")
        X_scaled = self.scaler.transform(np.asarray(X))
        return self.model.predict(X_scaled)


class DistributedProbe:
    """
    Probe different components of hidden state separately.

    Tests whether information is distributed across LSTM components (c vs h)
    or concentrated in specific regions.
    """

    def __init__(
        self,
        probe_type: str = "linear",
        alpha: float = 1.0,
        task: str = "regression",
        cv_folds: int = 5,
        lstm_hidden_dim: int = 256,
    ):
        """
        Args:
            probe_type: "linear" or "mlp"
            alpha: Regularization strength
            task: "regression" or "classification"
            cv_folds: Number of cross-validation folds
            lstm_hidden_dim: Hidden dimension of LSTM (for splitting c/h)
        """
        self.probe_type = probe_type
        self.alpha = alpha
        self.task = task
        self.cv_folds = cv_folds
        self.lstm_hidden_dim = lstm_hidden_dim

        # Create probes for each component
        self.probes = {}

    def _create_probe(self):
        """Create a new probe instance."""
        if self.probe_type == "linear":
            return LinearPropertyProbe(
                alpha=self.alpha,
                task=self.task,
                cv_folds=self.cv_folds,
            )
        else:
            return MLPPropertyProbe(
                alpha=self.alpha,
                task=self.task,
                cv_folds=self.cv_folds,
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Fit probes on different components and return metrics.

        Args:
            X: Flattened hidden state (c, h), shape (n_samples, 2*lstm_hidden_dim)
            y: Targets

        Returns:
            Dict mapping component name to metrics
        """
        X = np.asarray(X)
        y = np.asarray(y)

        results = {}

        # Full hidden state
        self.probes["full"] = self._create_probe()
        results["full"] = self.probes["full"].fit(X, y)

        # Cell state (c)
        c = X[:, :self.lstm_hidden_dim]
        self.probes["cell"] = self._create_probe()
        results["cell"] = self.probes["cell"].fit(c, y)

        # Hidden state (h)
        h = X[:, self.lstm_hidden_dim:]
        self.probes["hidden"] = self._create_probe()
        results["hidden"] = self.probes["hidden"].fit(h, y)

        # First half of each
        c_first = c[:, :self.lstm_hidden_dim//2]
        h_first = h[:, :self.lstm_hidden_dim//2]

        self.probes["cell_first_half"] = self._create_probe()
        results["cell_first_half"] = self.probes["cell_first_half"].fit(c_first, y)

        self.probes["hidden_first_half"] = self._create_probe()
        results["hidden_first_half"] = self.probes["hidden_first_half"].fit(h_first, y)

        return results

    def get_component_importance(self) -> Dict[str, float]:
        """
        Get relative importance of each component for prediction.

        Returns:
            Dict mapping component name to importance score (R² or accuracy)
        """
        return {
            name: probe.model.score(
                probe.scaler.transform(np.zeros((1, probe.scaler.n_features_in_))),
                np.zeros(1)
            ) if hasattr(probe, 'fitted') and probe.fitted else 0.0
            for name, probe in self.probes.items()
        }


def train_probe(
    hidden_states: np.ndarray,
    targets: np.ndarray,
    probe_type: str = "linear",
    task: str = "regression",
    **kwargs,
) -> Tuple[Union[LinearPropertyProbe, MLPPropertyProbe], Dict[str, float]]:
    """
    Convenience function to train a probe.

    Args:
        hidden_states: Hidden states, shape (n_samples, hidden_dim)
        targets: Targets
        probe_type: "linear" or "mlp"
        task: "regression" or "classification"
        **kwargs: Additional arguments for probe

    Returns:
        (trained_probe, metrics)
    """
    if probe_type == "linear":
        probe = LinearPropertyProbe(task=task, **kwargs)
    elif probe_type == "mlp":
        probe = MLPPropertyProbe(task=task, **kwargs)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")

    metrics = probe.fit(hidden_states, targets)
    return probe, metrics


def evaluate_probe(
    probe: Union[LinearPropertyProbe, MLPPropertyProbe],
    hidden_states: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate a trained probe on new data.

    Args:
        probe: Trained probe
        hidden_states: Test hidden states
        targets: Test targets

    Returns:
        Dict with evaluation metrics
    """
    predictions = probe.predict(hidden_states)
    targets = np.asarray(targets)

    if probe.task == "regression":
        r2 = r2_score(targets, predictions)
        mae = np.mean(np.abs(targets - predictions))
        return {"r2": float(r2), "mae": float(mae)}
    else:
        acc = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average="weighted")
        return {"accuracy": float(acc), "f1": float(f1)}


def compute_probe_comparison(
    hidden_states: np.ndarray,
    targets: np.ndarray,
    task: str = "regression",
) -> Dict[str, Dict[str, float]]:
    """
    Compare linear vs non-linear probes.

    Useful for determining if information is linearly encoded.

    Args:
        hidden_states: Hidden states
        targets: Targets
        task: "regression" or "classification"

    Returns:
        Dict with results for each probe type and comparison
    """
    linear_probe, linear_metrics = train_probe(
        hidden_states, targets, probe_type="linear", task=task
    )
    mlp_probe, mlp_metrics = train_probe(
        hidden_states, targets, probe_type="mlp", task=task
    )

    # Gap indicates non-linear encoding
    gap = mlp_metrics["mean_score"] - linear_metrics["mean_score"]

    return {
        "linear": linear_metrics,
        "mlp": mlp_metrics,
        "nonlinearity_gap": float(gap),
        "is_nonlinear": gap > 0.05,  # Threshold for meaningful gap
    }
