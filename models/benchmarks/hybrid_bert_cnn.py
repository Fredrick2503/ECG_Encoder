"""
Hybrid BERT-CNN Architecture for ECG Representation & Beat Classification.
Integrates bidirectional transformer self-attention with multi-scale 1D CNN heads and cross-layer residual gating.
"""

import math
import torch
import torch.nn as nn
from models.benchmarks.ecg_former import PositionalEncoding


class MultiScaleConvBlock(nn.Module):
    """Multi-scale 1D convolution block capturing short, medium, and wide ECG morphology waves."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_channels = out_channels // 4
        self.b1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels),
            nn.GELU()
        )
        self.b2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(branch_channels),
            nn.GELU()
        )
        self.b3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(branch_channels),
            nn.GELU()
        )
        self.b4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class HybridBERTCNN(nn.Module):
    """
    Hybrid BERT-CNN:
    - Multi-scale 1D CNN morphology extractor
    - Bidirectional Transformer Encoder with [CLS] pooling
    - Cross-gating module between CNN feature representations and Transformer representations
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        cnn_hidden: int = 64,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # CNN Branch
        self.cnn_block1 = MultiScaleConvBlock(in_channels, cnn_hidden)
        self.pool1 = nn.MaxPool1d(2)
        self.cnn_block2 = MultiScaleConvBlock(cnn_hidden, d_model)
        self.pool2 = nn.MaxPool1d(2)

        # BERT-style Transformer Branch
        self.pos_encoder = PositionalEncoding(d_model, max_len=500, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Gated Fusion
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        c1 = self.pool1(self.cnn_block1(x))
        c2 = self.pool2(self.cnn_block2(c1))  # [B, d_model, L_down]

        cnn_pool = torch.mean(c2, dim=2)  # [B, d_model]

        tokens = c2.transpose(1, 2)  # [B, L_down, d_model]
        tokens = self.pos_encoder(tokens)
        trans_out = self.transformer(tokens)  # [B, L_down, d_model]
        trans_pool = torch.mean(trans_out, dim=1)  # [B, d_model]

        # Gated Fusion
        concat = torch.cat([cnn_pool, trans_pool], dim=1)  # [B, d_model * 2]
        g = self.gate(concat)  # [B, d_model]
        fused = g * trans_pool + (1.0 - g) * cnn_pool  # [B, d_model]

        logits = self.classifier(fused)
        return logits
