"""
NSL-KDD Dataset Loader.
Handles raw record ingestion, column assignment, and categorical mapping.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

from utils.config import (
    COLUMN_NAMES,
    LABEL_COL,
    DIFFICULTY_COL,
    CATEGORICAL_FEATURES,
    ATTACK_CATEGORY_MAP,
    CATEGORY_TO_INT,
    TRAIN_FILE,
    TEST_FILE,
)
from utils.logger import get_logger

logger = get_logger(__name__)

def load_raw(path: Path) -> pd.DataFrame:
    """Read a raw NSL-KDD text file into a structured DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset artifact missing: {path}")

    logger.info(f"Ingesting raw stream: {path.name}")
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    return df

def _apply_label_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich dataframe with binary and multi-class target vectors."""
    raw_label = df[LABEL_COL].str.strip().str.rstrip(".")

    df["label_binary"] = (raw_label != "normal").astype(int)
    df["label_category"] = raw_label.map(lambda x: ATTACK_CATEGORY_MAP.get(x.lower(), "DoS"))
    df["label_multiclass"] = df["label_category"].map(CATEGORY_TO_INT)
    return df

def load_dataset(
    train_path: Path = TRAIN_FILE,
    test_path: Path = TEST_FILE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and synchronize train/test splits with identical schemas."""
    train_df = load_raw(train_path)
    test_df  = load_raw(test_path)

    train_df = _apply_label_schema(train_df)
    test_df  = _apply_label_schema(test_df)

    # Clean non-feature metadata
    train_df.drop(columns=[DIFFICULTY_COL], inplace=True, errors="ignore")
    test_df.drop(columns=[DIFFICULTY_COL], inplace=True, errors="ignore")

    return train_df, test_df
