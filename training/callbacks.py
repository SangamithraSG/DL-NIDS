"""
Training callbacks for DL-NIDS.

Includes EarlyStopping and ModelCheckpointing to ensure efficient training
and preservation of the best performing model states.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """
    def __init__(self, patience: int = 10, verbose: bool = False, delta: float = 0, path: str = 'checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement. 
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = Path(path)
        self.trace_func = trace_func

    def __call__(self, val_loss: float, model: torch.nn.Module):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class ModelCheckpoint:
    """
    Periodically saves model checkpoints.
    """
    def __init__(self, path: Path, monitor: str = 'val_f1', mode: str = 'max'):
        self.path = Path(path)
        self.monitor = monitor
        self.mode = mode
        self.best_val = -np.inf if mode == 'max' else np.inf
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def step(self, current_val: float, model: torch.nn.Module):
        improved = False
        if self.mode == 'max':
            if current_val > self.best_val:
                improved = True
        else:
            if current_val < self.best_val:
                improved = True
        
        if improved:
            self.best_val = current_val
            torch.save(model.state_dict(), self.path)
            # logger.info(f"Model saved to {self.path} (Best {self.monitor}: {current_val:.4f})")
            return True
        return False
