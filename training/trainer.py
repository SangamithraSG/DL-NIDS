"""
Unified training engine for DL-NIDS models.
Handles the training loop, validation, and testing for all PyTorch models.
"""

import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from utils.config import TrainingConfig, LOGS_DIR
from utils.logger import get_logger
from training.callbacks import EarlyStopping, ModelCheckpoint

logger = get_logger(__name__)

class Trainer:
    """Unified training and evaluation coordinator with V2 Config support."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        config: TrainingConfig = TrainingConfig(),
        model_name: str = "model",
        is_autoencoder: bool = False
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.model_name = model_name
        self.is_autoencoder = is_autoencoder

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.t_max
        )

        self.scaler = GradScaler(enabled=config.mixed_precision)
        self.history: List[Dict[str, float]] = []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Execute a single epoch of training."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.config.mixed_precision):
                if self.is_autoencoder:
                    loss = self.model.get_reconstruction_loss(data)
                else:
                    target = target.to(self.device)
                    output = self.model(data)
                    if hasattr(self.model, 'criterion'):
                        loss = self.model.criterion(output, target)
                    else:
                        loss = torch.nn.functional.cross_entropy(output, target)

            self.scaler.scale(loss).backward()
            
            if not self.is_autoencoder:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> float:
        """Verify model performance on unseen validation samples."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(self.device)
                if self.is_autoencoder:
                    loss = self.model.get_reconstruction_loss(data)
                else:
                    target = target.to(self.device)
                    output = self.model(data)
                    if hasattr(self.model, 'criterion'):
                        loss = self.model.criterion(output, target)
                    else:
                        loss = torch.nn.functional.cross_entropy(output, target)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, checkpoint_path: Path):
        """Standardized training loop with Early Stopping and history persistence."""
        epochs = self.config.num_epochs
        early_stopping = EarlyStopping(patience=self.config.patience, path=checkpoint_path, verbose=True)
        
        logger.info(f"Initiating training cycle for engine: {self.model_name}")
        
        for epoch in range(1, epochs + 1):
            start = time.time()
            t_loss = self.train_epoch(train_loader)
            v_loss = self.validate(val_loader)
            self.scheduler.step()
            
            duration = time.time() - start
            self.history.append({'epoch': epoch, 'train_loss': t_loss, 'val_loss': v_loss, 'duration': duration})
            
            logger.info(f"Epoch {epoch}/{epochs} | Loss: {t_loss:.4f} | Val: {v_loss:.4f} | {duration:.1f}s")
            
            early_stopping(v_loss, self.model)
            if early_stopping.early_stop:
                logger.info("Convergence detected. Early stopping triggered.")
                break
                
        # Persist Analysis Artifacts
        self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        pd.DataFrame(self.history).to_csv(LOGS_DIR / f"{self.model_name}_history.csv", index=False)

    def calibrate_autoencoder_threshold(self, val_loader: DataLoader):
        """Estimate the anomaly detection threshold using the 95th percentile of normal reconstruction error."""
        if not self.is_autoencoder: return
        self.model.eval()
        losses = []
        with torch.no_grad():
            for data, _ in val_loader:
                losses.extend(self.model.get_reconstruction_loss(data.to(self.device), reduction='none').cpu().numpy())
        self.model.threshold = np.percentile(losses, 95.0)
        logger.info(f"Anomaly threshold calibrated: {self.model.threshold:.6f}")

    def get_predictions(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate high-fidelity predictions and probabilities for evaluation."""
        self.model.eval()
        y_true, y_pred, y_probs = [], [], []
        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(self.device)
                if self.is_autoencoder:
                    preds = self.model.predict(data)
                    probs = self.model.get_reconstruction_loss(data, reduction='none')
                else:
                    output = self.model(data)
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(output, dim=1)
                y_pred.extend(preds.cpu().numpy())
                y_probs.extend(probs.cpu().numpy())
                y_true.extend(target.numpy())
        return np.array(y_true), np.array(y_pred), np.array(y_probs)
