"""
Reproducibility seed setter for DL-NIDS.
Call set_seed() once at program start to ensure deterministic behaviour
across Python, NumPy, and PyTorch (CPU and CUDA).
"""

import os
import random
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Parameters
    ----------
    seed : int
        The global random seed value (default: 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.warning("PyTorch not found — skipping torch seed setup.")

    logger.info(f"Global seed set to {seed}")
