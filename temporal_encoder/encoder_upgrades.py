import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class ECGTransformer(nn.Module):
    """
    Transformer-based Encoder for temporal representation learning of ECG signals.
    """
    def __init__(
        self,
        input_size: int = 12,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        num_classes: int = 5
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1D CNN Input Projection (reduces sequence length and maps leads to d_model)
        self.projector = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=d_model, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_leads, signal_length)
        Returns:
            torch.Tensor: Latent representation of shape (batch_size, d_model)
        """
        # x is (batch, leads, length) -> projector expects (batch, leads, length)
        proj = self.projector(x)  # (batch, d_model, length/2)
        proj = proj.transpose(1, 2)  # (batch, length/2, d_model)
        
        # Add Positional Encoding
        encoded = self.pos_encoder(proj)
        
        # Transformer pass
        out = self.transformer(encoded)  # (batch, length/2, d_model)
        
        # Global Average Pooling along temporal dimension
        z = out.mean(dim=1)  # (batch, d_model)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.get_representation(x)
        return self.fc(z)


class ECGMultiScaleCNN(nn.Module):
    """
    Multi-Scale CNN + BiLSTM Neural Network for temporal ECG representation learning.
    """
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 5
    ):
        super().__init__()
        
        # Parallel Conv1D branches
        self.conv_small = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv_medium = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv_large = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=51, padding=25),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Downsampler to save computation and control temporal resolution
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # BiLSTM processing concatenated features (32 * 3 = 96 channels)
        self.lstm = nn.LSTM(
            input_size=96,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_leads, signal_length)
        Returns:
            torch.Tensor: Latent representation of shape (batch_size, hidden_size * 2)
        """
        # Apply parallel branches
        feat_small = self.conv_small(x)
        feat_medium = self.conv_medium(x)
        feat_large = self.conv_large(x)
        
        # Concatenate branches along the channel dimension
        feat = torch.cat([feat_small, feat_medium, feat_large], dim=1)  # (batch, 96, length)
        
        # MaxPool to downsample length
        feat_pooled = self.pool(feat)  # (batch, 96, length/2)
        
        # Transpose for LSTM (batch, length/2, 96)
        lstm_in = feat_pooled.transpose(1, 2)
        
        output, (hidden, cell) = self.lstm(lstm_in)
        
        # Concatenate final forward and backward states
        z = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.get_representation(x)
        return self.fc(z)


class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # Prevent division by zero or negative size
        reduction = max(1, min(reduction, channels))
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.fc(x).view(b, c, 1)
        return x * y


class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None, use_se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=stride, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=15, stride=1, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.se = SqueezeExcitation1D(out_channels) if use_se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if getattr(self, 'se', None) is not None:
            out = self.se(out)
        out += identity
        out = self.relu(out)
        return out



class ECGResNet1D(nn.Module):
    """
    Standard ResNet-18 1D architecture for ECG signal representation learning.
    Now supports Squeeze-and-Excitation (SE) channel attention.
    """
    def __init__(
        self,
        input_size: int = 12,
        num_classes: int = 5,
        layers: list = [2, 2, 2, 2],
        base_filters: int = 64,
        dropout: float = 0.2,
        use_se: bool = False
    ):
        super().__init__()
        self.use_se = use_se
        self.in_channels = base_filters
        self.conv = nn.Conv1d(input_size, base_filters, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(base_filters, layers[0], stride=1)
        self.layer2 = self._make_layer(base_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_filters * 8, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(base_filters * 8, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        layers = []
        layers.append(ResBlock1D(self.in_channels, out_channels, stride, downsample, use_se=self.use_se))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock1D(self.in_channels, out_channels, use_se=self.use_se))
        return nn.Sequential(*layers)

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_leads, signal_length)
        Returns:
            torch.Tensor: Latent representation of shape (batch_size, base_filters * 8)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.get_representation(x)
        return self.fc(z)


class ECGBiGRU(nn.Module):
    def __init__(self, input_size: int = 12, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        x_transposed = x.transpose(1, 2)
        output, hidden = self.gru(x_transposed)
        z = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return z
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.get_representation(x)
        return self.fc(z)


class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size: int = 12, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        x_transposed = x.transpose(1, 2)
        output, _ = self.lstm(x_transposed)
        attn_weights = torch.softmax(self.attn(output), dim=1)
        z = torch.sum(output * attn_weights, dim=1)
        return z
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.get_representation(x)
        return self.fc(z)


class ECGCNNBiLSTMTransformer(nn.Module):
    def __init__(self, input_size: int = 12, hidden_size: int = 128, d_model: int = 128, nhead: int = 8, num_layers: int = 2, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.proj = nn.Linear(hidden_size * 2, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = feat.transpose(1, 2)
        lstm_out, _ = self.lstm(feat)
        trans_in = self.proj(lstm_out)
        encoded = self.pos_encoder(trans_in)
        out = self.transformer(encoded)
        z = out.mean(dim=1)
        return z
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.get_representation(x)
        return self.fc(z)


