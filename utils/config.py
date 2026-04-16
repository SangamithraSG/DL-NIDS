"""
Central configuration module for DL-NIDS.
This is the FINAL, EXHAUSTIVE, SINGLE SOURCE OF TRUTH for all paths, constants, and hyperparameters.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ─── CORE PATH RESOLUTION ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_DIR         = PROJECT_ROOT
DATA_DIR         = PROJECT_ROOT / "data"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
MODELS_DIR       = SAVED_MODELS_DIR # Alias for legacy consistency
LOGS_DIR         = PROJECT_ROOT / "logs"
ARTIFACTS_DIR    = PROJECT_ROOT / "artifacts"
REPORTS_DIR      = PROJECT_ROOT / "reports"
DASHBOARD_DIR    = PROJECT_ROOT / "dashboard"

# Ensure all critical folders exist
for _dir in [DATA_DIR, SAVED_MODELS_DIR, LOGS_DIR, ARTIFACTS_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── DATASET ARTIFACTS ────────────────────────────────────────────────────────
TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE  = DATA_DIR / "KDDTest+.txt"

SCALER_PATH      = ARTIFACTS_DIR / "scaler.joblib"
ENCODER_PATH     = ARTIFACTS_DIR / "encoder.joblib"
PROCESSED_TRAIN  = ARTIFACTS_DIR / "train_processed.npz"
PROCESSED_TEST   = ARTIFACTS_DIR / "test_processed.npz"

# ─── MODEL CHECKPOINTS ────────────────────────────────────────────────────────
CHECKPOINT_AUTOENCODER = SAVED_MODELS_DIR / "autoencoder_best.pt"
CHECKPOINT_LSTM        = SAVED_MODELS_DIR / "lstm_best.pt"
CHECKPOINT_CNN         = SAVED_MODELS_DIR / "cnn_best.pt"
CHECKPOINT_HYBRID      = SAVED_MODELS_DIR / "hybrid_best.pt"
CHECKPOINT_RF          = SAVED_MODELS_DIR / "random_forest.joblib"

# ─── NSL-KDD SCHEMA & MAPPING ────────────────────────────────────────────────
LABEL_COL      = "label"
DIFFICULTY_COL = "difficulty_level"

COLUMN_NAMES: List[str] = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level",
]

CATEGORICAL_FEATURES: List[str] = ["protocol_type", "service", "flag"]

ATTACK_CATEGORY_MAP: Dict[str, str] = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
    "mailbomb": "DoS", "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L",
    "snmpguess": "R2L", "worm": "R2L", "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}

CATEGORY_TO_INT: Dict[str, int] = {
    "Normal": 0, "DoS": 1, "Probe": 2, "R2L": 3, "U2R": 4
}
INT_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_INT.items()}

CLASS_NAMES_BINARY     = ["Normal", "Attack"]
CLASS_NAMES_MULTICLASS = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# ─── HYPERPARAMETERS ──────────────────────────────────────────────────────────
SEQ_LEN          = 10
RANDOM_SEED      = 42
TEST_SIZE        = 0.15
VAL_SIZE         = 0.15
SMOTE_STRATEGY   = "auto"

@dataclass
class TrainingConfig:
    """Consolidated training parameters for all models."""
    batch_size:      int   = 512
    learning_rate:   float = 1e-3
    weight_decay:    float = 1e-4
    num_epochs:      int   = 100
    patience:        int   = 10
    grad_clip:       float = 1.0
    mixed_precision: bool  = True
    num_workers:     int   = 0     # Safe default for Windows
    t_max:           int   = 50    # CosineAnnealingLR T_max
    dropout:         float = 0.4
    hidden_dim:      int   = 128
    num_layers:      int   = 2

@dataclass
class AutoencoderConfig:
    anomaly_percentile: float = 95.0
    encoder_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    dropout: float = 0.3

# ─── DASHBOARD SETTINGS ────────────────────────────────────────────────────────
DASHBOARD_TITLE = "DL-NIDS | Intelligent Intrusion Detection"
DASHBOARD_ICON  = "🛡️"

# ─── LOGGING ──────────────────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FILE   = LOGS_DIR / "dl_nids.log"
