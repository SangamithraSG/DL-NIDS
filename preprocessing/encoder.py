"""
Feature encoding for NSL-KDD categorical columns.

Fits a OneHotEncoder on training data and transforms both train/test.
Encoder state is serialised to disk with joblib for reuse at inference.
"""

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

from utils.config import CATEGORICAL_FEATURES, ENCODER_PATH, LABEL_COL
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEncoder:
    """
    One-hot encoder wrapper for NSL-KDD categorical features.

    Attributes
    ----------
    categorical_features : List[str]
        Column names to be encoded.
    encoder : OneHotEncoder
        Underlying sklearn encoder (fitted after calling fit_transform).
    feature_names_out : List[str]
        Output feature names post one-hot expansion.
    """

    def __init__(self, categorical_features: List[str] = CATEGORICAL_FEATURES) -> None:
        """
        Initialise the FeatureEncoder.

        Parameters
        ----------
        categorical_features : List[str]
            Names of categorical columns to one-hot encode.
        """
        self.categorical_features: List[str] = categorical_features
        self.encoder: Optional[OneHotEncoder] = None
        self.feature_names_out: List[str] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        """
        Fit the encoder on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame containing categorical columns.

        Returns
        -------
        FeatureEncoder
            self (for method chaining).
        """
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.encoder.fit(df[self.categorical_features])
        self.feature_names_out = list(
            self.encoder.get_feature_names_out(self.categorical_features)
        )
        logger.info(
            f"Encoder fitted on {len(self.categorical_features)} categorical features → "
            f"{len(self.feature_names_out)} OHE columns"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted encoder to a DataFrame.

        Drops the original categorical columns and appends the one-hot
        encoded columns to produce a fully numeric DataFrame (excluding
        label columns).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with categorical columns replaced by
            one-hot encoded columns.

        Raises
        ------
        RuntimeError
            If the encoder has not been fitted yet.
        """
        if self.encoder is None:
            raise RuntimeError("Encoder has not been fitted. Call fit() first.")

        ohe_array = self.encoder.transform(df[self.categorical_features])
        ohe_df = pd.DataFrame(ohe_array, columns=self.feature_names_out, index=df.index)

        # Drop original categoricals; keep all other columns
        remaining = df.drop(columns=self.categorical_features)
        result = pd.concat([remaining, ohe_df], axis=1)
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on and transform the provided DataFrame (convenience method).

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.

        Returns
        -------
        pd.DataFrame
            Encoded DataFrame.
        """
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = ENCODER_PATH) -> None:
        """
        Serialise the fitted encoder to disk using joblib.

        Parameters
        ----------
        path : Path
            Destination file path (default from config).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"encoder": self.encoder, "feature_names_out": self.feature_names_out},
            path,
        )
        logger.info(f"Encoder saved → {path}")

    def load(self, path: Path = ENCODER_PATH) -> "FeatureEncoder":
        """
        Load a previously fitted encoder from disk.

        Parameters
        ----------
        path : Path
            Source file path (default from config).

        Returns
        -------
        FeatureEncoder
            self with encoder populated.

        Raises
        ------
        FileNotFoundError
            If the encoder file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Encoder file not found: {path}")
        payload = joblib.load(path)
        self.encoder = payload["encoder"]
        self.feature_names_out = payload["feature_names_out"]
        logger.info(f"Encoder loaded ← {path}")
        return self

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_all_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Return the full ordered list of feature names after encoding.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame (used to infer non-categorical column names).

        Returns
        -------
        List[str]
            Feature names in the same order as columns produced by transform().
        """
        non_cat = [
            c for c in df.columns
            if c not in self.categorical_features
            and c not in (LABEL_COL, "label_binary", "label_category", "label_multiclass")
        ]
        return non_cat + self.feature_names_out
