"""
Hybrid CNN-BiLSTM with Attention for NIDS classification.

Combines the spatial feature extraction of CNNs with the temporal
modeling of LSTMs for a robust intrusion detection system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from utils.config import TrainingConfig
from models.lstm_model import Attention

class CNNBiLSTMHybrid(nn.Module):
    """
    Hybrid model: CNN -> BiLSTM -> Attention -> FC
    """
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int, 
        config: TrainingConfig = TrainingConfig()
    ):
        super(CNNBiLSTMHybrid, self).__init__()
        
        self.hidden_dim = config.hidden_dim
        
        # ── CNN Section ─────────────────────────────────────────────────────
        # input: [batch, input_dim, seq_len]
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        # ── BiLSTM Section ──────────────────────────────────────────────────
        # input: [batch, 128, seq_len] -> we transpose back to [batch, seq_len, 128]
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # ── Attention Section ───────────────────────────────────────────────
        self.attention = Attention(self.hidden_dim)
        
        # ── FC Head ──────────────────────────────────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
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
        """
        # CNN Part
        x = x.transpose(1, 2) # [batch, input_dim, seq_len]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # LSTM Part
        x = x.transpose(1, 2) # [batch, seq_len, 128]
        lstm_out, _ = self.lstm(x)
        
        # Attention
        context, self.attn_weights = self.attention(lstm_out)
        
        # Head
        logits = self.fc(context)
        
        return logits
