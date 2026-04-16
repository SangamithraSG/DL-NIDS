"""
Class imbalance handling for NSL-KDD.

Provides SMOTE oversampling and class-weight computation to address the
strong imbalance between Normal vs. U2R / R2L traffic in NSL-KDD.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

from utils.config import RANDOM_SEED, SMOTE_STRATEGY
from utils.logger import get_logger

logger = get_logger(__name__)


class SMOTEBalancer:
    """
    SMOTE (Synthetic Minority Over-sampling Technique) balancer.

    Wraps imbalanced-learn's SMOTE with sensible defaults for NSL-KDD.
    SMOTE is only applied to the training split — never test/val.

    Parameters
    ----------
    strategy : str | dict
        SMOTE sampling strategy (passed directly to SMOTE).
        'auto' = resample all minority classes to match the majority class.
    random_state : int
        Random seed for reproducibility.
    k_neighbors : int
        Number of nearest neighbours used by SMOTE.
    """

    def __init__(
        self,
        strategy: str = SMOTE_STRATEGY,
        random_state: int = RANDOM_SEED,
        k_neighbors: int = 5,
    ) -> None:
        self.strategy = strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self._smote: Optional[SMOTE] = None

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit SMOTE and resample the training data.

        Parameters
        ----------
        X : np.ndarray, shape [n_samples, n_features]
            Feature matrix.
        y : np.ndarray, shape [n_samples]
            Integer label vector.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (X_resampled, y_resampled) with synthetic minority samples added.
        """
        unique, counts = np.unique(y, return_counts=True)
        logger.info("Class distribution BEFORE SMOTE:")
        for cls, cnt in zip(unique, counts):
            logger.info(f"  class {cls}: {cnt:,}")

        self._smote = SMOTE(
            sampling_strategy=self.strategy,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors,
        )
        X_res, y_res = self._smote.fit_resample(X, y)

        unique_r, counts_r = np.unique(y_res, return_counts=True)
        logger.info("Class distribution AFTER SMOTE:")
        for cls, cnt in zip(unique_r, counts_r):
            logger.info(f"  class {cls}: {cnt:,}")

        return X_res.astype(np.float32), y_res.astype(np.int64)


def compute_class_weights(
    y: np.ndarray,
    classes: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    """
    Compute balanced class weights for use in PyTorch loss functions.

    Uses scikit-learn's 'balanced' scheme:
        weight[i] = n_samples / (n_classes * count[i])

    Parameters
    ----------
    y : np.ndarray
        Integer label vector (training labels, pre- or post-SMOTE).
    classes : np.ndarray, optional
        Explicit class array. Defaults to sorted unique values in y.

    Returns
    -------
    Dict[int, float]
        Mapping {class_index: weight}.
    """
    if classes is None:
        classes = np.unique(y)

    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Class weights computed: {weight_dict}")
    return weight_dict


def weights_to_tensor(
    weight_dict: Dict[int, float],
    num_classes: int,
    device: str = "cpu",
):
    """
    Convert a class-weight dict to a PyTorch FloatTensor.

    Parameters
    ----------
    weight_dict : Dict[int, float]
        Output of compute_class_weights().
    num_classes : int
        Total number of classes (tensor length).
    device : str
        Target device string ('cpu' or 'cuda').

    Returns
    -------
    torch.Tensor
        1-D FloatTensor of length num_classes.
    """
    import torch
    w = [weight_dict.get(i, 1.0) for i in range(num_classes)]
    return torch.tensor(w, dtype=torch.float32, device=device)
