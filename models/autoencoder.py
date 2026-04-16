"""
Autoencoder for Anomaly Detection in DL-NIDS.

This model is trained exclusively on 'Normal' traffic to learn the 
distribution of benign network activity. At inference, high reconstruction 
error indicates an anomaly (attack).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from utils.config import AutoencoderConfig

class Autoencoder(nn.Module):
    """
    PyTorch Autoencoder model for anomaly detection.
    
    Architecture:
    - Encoder: FC(input_dim -> 64 -> 32 -> 16)
    - Decoder: FC(16 -> 32 -> 64 -> input_dim)
    """

    def __init__(self, input_dim: int, config: AutoencoderConfig = AutoencoderConfig()):
        """
        Initialise the Autoencoder.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        config : AutoencoderConfig
            Configuration containing layer dimensions and dropout rates.
        """
        super(Autoencoder, self).__init__()
        
        dims = config.encoder_dims # [64, 32, 16]
        
        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dims[0]),
            nn.BatchNorm1d(dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(dims[0], dims[1]),
            nn.BatchNorm1d(dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(dims[1], dims[2]),
            nn.BatchNorm1d(dims[2]),
            nn.ReLU(),
        )
        
        # ── Decoder ───────────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(dims[2], dims[1]),
            nn.BatchNorm1d(dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(dims[1], dims[0]),
            nn.BatchNorm1d(dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(dims[0], input_dim),
            nn.Sigmoid() # Scale output to [0, 1] range as features are normalized
        )
        
        self.threshold = 0.0 # To be set after training based on validation data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Reconstructed feature tensor.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_reconstruction_loss(self, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        Compute the MSE reconstruction loss for the input.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.
        reduction : str
            Type of reduction ('mean', 'none').

        Returns
        -------
        torch.Tensor
            Reconstruction loss.
        """
        reconstructed = self.forward(x)
        if reduction == 'none':
            # Compute MSE per sample: (input - reconstructed)^2 summed over features
            return torch.mean((x - reconstructed) ** 2, dim=1)
        else:
            return nn.functional.mse_loss(reconstructed, x, reduction=reduction)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict if samples are anomalous based on the learned threshold.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Binary predictions (0: Normal, 1: Anomaly).
        """
        losses = self.get_reconstruction_loss(x, reduction='none')
        return (losses > self.threshold).long()
