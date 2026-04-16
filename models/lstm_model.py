"""
Bidirectional LSTM with Attention for sequence-based NIDS classification.

This model processes a sequence of 10 network packets to capture temporal
dependencies and uses an attention mechanism to identify critical time steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from utils.config import TrainingConfig

class Attention(nn.Module):
    """
    Self-attention mechanism to weight the importance of different time steps.
    """
    def __init__(self, hidden_dim: int):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1) # *2 for bidirectional

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        lstm_output : torch.Tensor
            [batch_size, seq_len, hidden_dim * 2]
        
        Returns
        -------
        context : torch.Tensor
            [batch_size, hidden_dim * 2]
        weights : torch.Tensor
            [batch_size, seq_len]
        """
        # Calculate scores: [batch_size, seq_len, 1]
        attn_weights = self.attn(lstm_output)
        
        # Softmax over time steps: [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum: [batch_size, hidden_dim * 2]
        context = torch.sum(attn_weights * lstm_output, dim=1)
        
        return context, attn_weights.squeeze(-1)

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM Classifier with Attention.
    """
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int, 
        config: TrainingConfig = TrainingConfig()
    ):
        super(BiLSTMClassifier, self).__init__()
        
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        
        # ── BiLSTM ───────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        # ── Attention ────────────────────────────────────────────────────────
        self.attention = Attention(self.hidden_dim)
        
        # ── Fully Connected Head ─────────────────────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, num_classes)
        )
        
        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor [batch_size, seq_len, input_dim]
        
        Returns
        -------
        logits : torch.Tensor [batch_size, num_classes]
        """
        # LSTM output: [batch_size, seq_len, hidden_dim * 2]
        lstm_out, _ = self.lstm(x)
        
        # Attention: [batch_size, hidden_dim * 2]
        context, self.attn_weights = self.attention(lstm_out)
        
        # Head
        logits = self.fc(context)
        
        return logits

    def get_attention_weights(self) -> torch.Tensor:
        """Helper to retrieve weights for visualization."""
        return self.attn_weights
