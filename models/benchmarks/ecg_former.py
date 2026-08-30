"""
ECGFormer Architecture for ECG Sequence & Beat Classification.
Uses 1D Patch Tokenization, Learnable [CLS] Token, Positional Encoding, and Transformer Encoder Layers.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ECGFormer(nn.Module):
    """
    ECGFormer: Pure Transformer Encoder architecture for ECG signals.
    - 1D Patch Projection / Tokenizer
    - Multi-Head Self-Attention layers
    - [CLS] Token & LayerNorm classification
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        patch_size: int = 14,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # 1D Patch Tokenizer
        self.patch_embed = nn.Conv1d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
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

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        B = x.shape[0]
        tokens = self.patch_embed(x).transpose(1, 2)  # [B, Num_patches, d_model]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        seq = torch.cat([cls_tokens, tokens], dim=1)  # [B, Num_patches + 1, d_model]

        seq = self.pos_encoder(seq)
        encoded = self.transformer_encoder(seq)  # [B, Seq_len, d_model]
        cls_rep = self.norm(encoded[:, 0])  # [B, d_model]
        logits = self.classifier(cls_rep)
        return logits
