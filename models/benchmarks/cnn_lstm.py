"""
CNN-LSTM Architecture for ECG Beat / Rhythm Classification.
Combines 1D Convolutional feature representation with Bidirectional LSTM sequence modeling.
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    CNN-LSTM baseline model:
    - 3-stage 1D CNN blocks for spatial/morphological feature extraction
    - 2-layer Bidirectional LSTM for temporal dependency modeling
    - Dense MLP classification head
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        cnn_channels: tuple = (32, 64, 128),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 1D CNN Feature Extractor
        layers = []
        curr_in = in_channels
        for c in cnn_channels:
            layers.extend([
                nn.Conv1d(curr_in, c, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(p=dropout * 0.5)
            ])
            curr_in = c
        self.cnn = nn.Sequential(*layers)

        # BiLSTM Sequence Modeler
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        feats = self.cnn(x)  # [B, C_out, L_down]
        # Transpose for LSTM: [B, L_down, C_out]
        feats = feats.transpose(1, 2)
        lstm_out, _ = self.lstm(feats)  # [B, L_down, hidden * 2]
        # Global average pool over sequence
        pooled = torch.mean(lstm_out, dim=1)  # [B, hidden * 2]
        logits = self.classifier(pooled)
        return logits
