"""
CNN-Transformer Hybrid Architecture for ECG Beat / Rhythm Classification.
Combines 1D residual convolutional token embedding with Multi-Head Self-Attention Transformer encoder.
"""

import math
import torch
import torch.nn as nn
from models.benchmarks.ecg_former import PositionalEncoding


class CNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer:
    - Multi-layer 1D ConvStem for local morphological feature representation
    - Transformer Encoder for global contextual sequence relationships
    - Global average pooling + MLP classifier
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        stem_channels: tuple = (32, 64, 128),
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 1D Convolutional Stem
        conv_layers = []
        curr_in = in_channels
        for c in stem_channels:
            conv_layers.extend([
                nn.Conv1d(curr_in, c, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(c),
                nn.SiLU(inplace=True),
                nn.Dropout(p=dropout * 0.5)
            ])
            curr_in = c

        self.conv_stem = nn.Sequential(*conv_layers)
        self.proj = nn.Linear(stem_channels[-1], d_model) if stem_channels[-1] != d_model else nn.Identity()

        # Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model, max_len=500, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        stem_out = self.conv_stem(x)  # [B, stem_channels[-1], L_down]
        tokens = stem_out.transpose(1, 2)  # [B, L_down, stem_channels[-1]]
        tokens = self.proj(tokens)  # [B, L_down, d_model]

        seq = self.pos_encoder(tokens)
        encoded = self.transformer_encoder(seq)  # [B, L_down, d_model]
        pooled = torch.mean(self.norm(encoded), dim=1)  # [B, d_model]
        logits = self.classifier(pooled)
        return logits
