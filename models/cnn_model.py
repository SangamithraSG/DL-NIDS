"""
1D Convolutional Neural Network for NIDS classification.

Extracts local spatial features from sequences of network packets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from utils.config import TrainingConfig

class CNNClassifier(nn.Module):
    """
    1D CNN Classifier for sequence data.
    """
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int, 
        config: TrainingConfig = TrainingConfig()
    ):
        super(CNNClassifier, self).__init__()
        
        # input: [batch, input_dim, seq_len] -> PyTorch Conv1D expects channel first
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # After two convs (padding=1) and one pool (kernel=2) on seq_len=10:
        # conv1: 10 -> 10
        # conv2: 10 -> 10
        # pool: 10 -> 5
        self.fc_input_dim = 128 * 5
        
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expects x: [batch, seq_len, input_dim]
        We transpose it to [batch, input_dim, seq_len] for Conv1D.
        """
        x = x.transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.dropout(x)
        logits = self.fc(x)
        
        return logits
