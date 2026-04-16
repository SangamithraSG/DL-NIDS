"""
Preprocessing pipeline orchestration for DL-NIDS.
Full sequence: Ingestion → Encoding → Normalization → Balancing → Splitting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from preprocessing.loader import load_dataset
from preprocessing.encoder import FeatureEncoder
from preprocessing.normalizer import FeatureNormalizer
from preprocessing.balancer import SMOTEBalancer, compute_class_weights, weights_to_tensor
from utils.config import (
    RANDOM_SEED, TEST_SIZE, VAL_SIZE,
    SEQ_LEN, LABEL_COL, CATEGORICAL_FEATURES,
    PROCESSED_TRAIN, PROCESSED_TEST,
    ENCODER_PATH, SCALER_PATH,
    TrainingConfig
)
from utils.logger import get_logger

logger = get_logger(__name__)
# Initialize default config for parameter defaults
_cfg = TrainingConfig()

class TimeSeriesDataset(torch.utils.data.Dataset):
    """Memory-efficient sliding window dataset for spatio-temporal modeling."""
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len
        self.n_samples = len(X) - seq_len + 1
        
        if self.n_samples <= 0:
            raise ValueError(f"Insufficient samples ({len(X)}) for sequence length {seq_len}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_val = self.y[idx + self.seq_len - 1]
        return x_seq, y_val

def _to_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seq_len: Optional[int] = None,
) -> DataLoader:
    """Consolidate features and targets into a production-ready DataLoader."""
    if seq_len is not None:
        ds = TimeSeriesDataset(X, y, seq_len)
    else:
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers, 
        pin_memory=False
    )

def run_pipeline(
    multiclass: bool = True,
    apply_smote: bool = True,
    batch_size: int = _cfg.batch_size,
    seq_len: int = SEQ_LEN,
    num_workers: int = _cfg.num_workers,
) -> Dict:
    """
    Execute the definitive data transformation pipeline.
    
    Verified attributes in TrainingConfig: batch_size, num_workers.
    """
    # 1. Load Data
    train_df, test_df = load_dataset()
    target_col = "label_multiclass" if multiclass else "label_binary"
    num_classes = 5 if multiclass else 2

    # 2. Sequential Encoding
    encoder = FeatureEncoder()
    train_enc = encoder.fit_transform(train_df)
    test_enc  = encoder.transform(test_df)
    encoder.save(ENCODER_PATH)

    # 3. Dynamic Normalization
    normalizer = FeatureNormalizer()
    train_norm = normalizer.fit_transform(train_enc)
    test_norm  = normalizer.transform(test_enc)
    normalizer.save(SCALER_PATH)

    # 4. Feature Extraction
    meta_cols = {LABEL_COL, "label_binary", "label_category", "label_multiclass"}
    f_cols = [c for c in train_norm.columns if c not in meta_cols]

    X_full = train_norm[f_cols].values.astype(np.float32)
    y_full = train_norm[target_col].values.astype(np.int64)
    X_test = test_norm[f_cols].values.astype(np.float32)
    y_test = test_norm[target_col].values.astype(np.int64)

    # 5. Stratified Splitting
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_SEED,
        stratify=y_full,
    )

    # 6. Minority Balancing (SMOTE)
    if apply_smote:
        X_train, y_train = SMOTEBalancer().fit_resample(X_train, y_train)

    # 7. Class Weight Calibration
    cw_tensor = weights_to_tensor(compute_class_weights(y_train), num_classes)

    # 8. Loader Synthesis
    loaders = {
        "loader_train": _to_dataloader(X_train, y_train, batch_size, True,  num_workers),
        "loader_val":   _to_dataloader(X_val,   y_val,   batch_size, False, num_workers),
        "loader_test":  _to_dataloader(X_test,  y_test,  batch_size, False, num_workers),
        "seq_loader_train": _to_dataloader(X_train, y_train, batch_size, True,  num_workers, seq_len),
        "seq_loader_val":   _to_dataloader(X_val,   y_val,   batch_size, False, num_workers, seq_len),
        "seq_loader_test":  _to_dataloader(X_test,  y_test,  batch_size, False, num_workers, seq_len),
    }

    return {
        **loaders,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "class_weight_tensor": cw_tensor,
        "feature_names": f_cols,
        "num_classes": num_classes,
        "input_dim": X_train.shape[1],
        "seq_len": seq_len,
    }
