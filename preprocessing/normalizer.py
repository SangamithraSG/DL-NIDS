"""
Feature normalisation for NSL-KDD numeric columns.

Fits a MinMaxScaler on training data and applies it to both splits.
The fitted scaler is persisted to disk for consistent inference.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

from utils.config import (
    CATEGORICAL_FEATURES,
    LABEL_COL,
    SCALER_PATH,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Columns that are never used as input features
_NON_FEATURE_COLS = [
    LABEL_COL,
    "label_binary",
    "label_category",
    "label_multiclass",
]


class FeatureNormalizer:
    """
    MinMaxScaler wrapper that operates on all numeric feature columns.

    The scaler is fitted on training data only, then applied to test data
    to prevent data leakage.

    Attributes
    ----------
    scaler : MinMaxScaler
        Underlying sklearn scaler.
    numeric_cols : List[str]
        Column names that were scaled (set during fit).
    """

    def __init__(self) -> None:
        """Initialise with an unfitted MinMaxScaler."""
        self.scaler: Optional[MinMaxScaler] = None
        self.numeric_cols: List[str] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FeatureNormalizer":
        """
        Fit the scaler on training data.

        Only numeric columns (excluding label and categorical columns that
        have already been encoded) are scaled.

        Parameters
        ----------
        df : pd.DataFrame
            Encoded training DataFrame (categoricals already one-hot-expanded).

        Returns
        -------
        FeatureNormalizer
            self, for method chaining.
        """
        exclude = set(_NON_FEATURE_COLS + CATEGORICAL_FEATURES)
        self.numeric_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(df[self.numeric_cols])
        logger.info(f"Normalizer fitted on {len(self.numeric_cols)} numeric columns")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted scaler to a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to scale (must contain the numeric columns seen during fit).

        Returns
        -------
        pd.DataFrame
            DataFrame with numeric feature columns scaled to [0, 1].

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted yet.
        """
        if self.scaler is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        df = df.copy()
        # Only scale columns that exist in current df (test may lack some)
        cols_present = [c for c in self.numeric_cols if c in df.columns]
        df[cols_present] = self.scaler.transform(df[cols_present])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on and transform the given DataFrame (convenience method).

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.

        Returns
        -------
        pd.DataFrame
            Scaled DataFrame.
        """
        return self.fit(df).transform(df)

    def get_feature_matrix(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract the scaled feature matrix (X) from a transformed DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Already-transformed DataFrame.

        Returns
        -------
        Tuple[np.ndarray, List[str]]
            (X array of shape [n_samples, n_features], list of feature names)
        """
        exclude = set(_NON_FEATURE_COLS)
        feature_cols = [c for c in df.columns if c not in exclude]
        X = df[feature_cols].values.astype(np.float32)
        return X, feature_cols

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = SCALER_PATH) -> None:
        """
        Persist the fitted scaler to disk.

        Parameters
        ----------
        path : Path
            Destination file path (default from config).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "numeric_cols": self.numeric_cols}, path)
        logger.info(f"Normalizer saved → {path}")

    def load(self, path: Path = SCALER_PATH) -> "FeatureNormalizer":
        """
        Load a previously fitted scaler from disk.

        Parameters
        ----------
        path : Path
            Source file path (default from config).

        Returns
        -------
        FeatureNormalizer
            self with scaler populated.

        Raises
        ------
        FileNotFoundError
            If the scaler file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")
        payload = joblib.load(path)
        self.scaler = payload["scaler"]
        self.numeric_cols = payload["numeric_cols"]
        logger.info(f"Normalizer loaded ← {path}")
        return self
