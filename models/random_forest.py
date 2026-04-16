"""
Random Forest Baseline for NIDS classification.

Provides a strong traditional machine learning baseline to compare
against the deep learning models.
"""

import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Optional
import numpy as np

class RandomForestModel:
    """
    Wrapper for Scikit-learn RandomForestClassifier to match project interface.
    """
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1 # Use all available cores
        )

    def train(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the training data."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def save(self, path: Path):
        """Save the model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path):
        """Load the model from disk."""
        self.model = joblib.load(path)
