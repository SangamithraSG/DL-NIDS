"""
Weighted Ensemble Model for NIDS classification.

Combines the predictions of multiple models (BiLSTM, Hybrid, RF)
to improve overall detection accuracy and robustness.
"""

import numpy as np
from typing import List, Dict, Any

class EnsembleModel:
    """
    Combines probabilities from multiple models using weighted averaging.
    """
    def __init__(self, weights: List[float] = None):
        """
        Parameters
        ----------
        weights : List[float]
            Weights for each model. If None, equal weights are used.
        """
        self.weights = weights

    def predict(self, model_probs: List[np.ndarray]) -> np.ndarray:
        """
        Combine probabilities and return final class predictions.
        
        Parameters
        ----------
        model_probs : List[np.ndarray]
            List of probability matrices from each model, each of shape [N, C].
        """
        if self.weights is None:
            self.weights = [1.0 / len(model_probs)] * len(model_probs)
        
        assert len(self.weights) == len(model_probs), "Weights must match number of models."
        
        # Weighted average of probabilities
        ensemble_probs = np.zeros_like(model_probs[0])
        for w, p in zip(self.weights, model_probs):
            ensemble_probs += w * p
            
        # Get final class predictions
        y_pred = np.argmax(ensemble_probs, axis=1)
        
        return y_pred, ensemble_probs
