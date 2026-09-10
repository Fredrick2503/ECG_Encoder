"""
RR-Interval Atrial Fibrillation Detector for MIT-BIH AF Database.
Processes continuous RR intervals and delta-RR sequences with statistical feature heads and BiGRU/BiLSTM.
"""

import torch
import torch.nn as nn


class RRAFDetector(nn.Module):
    """
    RR-Interval based AF Detector:
    - Input: [Batch, Seq_len, 2] (RR interval in seconds and delta-RR)
    - 1D Temporal Convolution layer
    - 2-layer Bidirectional GRU / LSTM sequence encoder
    - Statistical summary feature integration (Mean, Std, RMSSD, pNN50 approximation)
    - Binary AF classification head
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Classifier integrating sequence representation and stat features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Seq_len, 2]
        rr = x[:, :, 0]  # [B, Seq_len]
        drr = x[:, :, 1]  # [B, Seq_len]

        # Calculate differentiable statistical summary features
        mean_rr = torch.mean(rr, dim=1, keepdim=True)
        std_rr = torch.std(rr, dim=1, keepdim=True)
        rmssd = torch.sqrt(torch.mean(drr ** 2, dim=1, keepdim=True) + 1e-8)
        pnn50_approx = torch.mean(torch.sigmoid((torch.abs(drr) - 0.05) * 50.0), dim=1, keepdim=True)
        stats = torch.cat([mean_rr, std_rr, rmssd, pnn50_approx], dim=1)  # [B, 4]

        # Sequence modeling
        proj = self.input_proj(x)  # [B, Seq_len, hidden_dim]
        gru_out, _ = self.gru(proj)  # [B, Seq_len, hidden_dim * 2]
        seq_rep = torch.mean(gru_out, dim=1)  # [B, hidden_dim * 2]

        combined = torch.cat([seq_rep, stats], dim=1)  # [B, hidden_dim * 2 + 4]
        logits = self.classifier(combined)  # [B, num_classes]
        return logits
