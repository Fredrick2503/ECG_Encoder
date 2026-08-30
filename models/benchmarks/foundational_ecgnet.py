"""
FoundationalECGNet Architecture for Multi-Lead Multi-Dataset Representation (PTB-XL + CinC).
Integrates multi-scale 1D residual temporal attention blocks with cross-lead tokenization and multi-label cardiac diagnosis.
"""

import math
import torch
import torch.nn as nn
from models.benchmarks.ecg_former import PositionalEncoding


class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(4, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(4, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1)
        return x * w


class SEResNetBlock1D(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.se = SqueezeExcitation1D(out_c)

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + res)


class FoundationalECGNet(nn.Module):
    """
    FoundationalECGNet:
    - 12-lead multi-channel foundation encoder
    - Deep 1D SE-ResNet hierarchical temporal representations
    - Bidirectional multi-head cross-lead attention transformer
    - Multi-label classification head for PTB-XL (5 superclasses or fine-grained) + CinC diagnostic targets
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        base_channels: int = 64,
        d_model: int = 128,
        nhead: int = 8,
        num_trans_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Hierarchical SE-ResNet Backbone
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = SEResNetBlock1D(base_channels, base_channels, stride=1)
        self.layer2 = SEResNetBlock1D(base_channels, base_channels * 2, stride=2)
        self.layer3 = SEResNetBlock1D(base_channels * 2, d_model, stride=2)

        # Transformer Sequence Aggregator
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.norm = nn.LayerNorm(d_model)

        # Multi-label diagnostic classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 12, L]
        stem_out = self.stem(x)
        l1 = self.layer1(stem_out)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)  # [B, d_model, L_down]

        tokens = l3.transpose(1, 2)  # [B, L_down, d_model]
        tokens = self.pos_encoder(tokens)
        encoded = self.transformer(tokens)  # [B, L_down, d_model]
        rep = torch.mean(self.norm(encoded), dim=1)  # [B, d_model]
        return rep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rep = self.extract_features(x)
        logits = self.classifier(rep)
        return logits
