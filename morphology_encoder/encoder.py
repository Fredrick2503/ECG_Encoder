import torch
import torch.nn as nn

class ResBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ECGMorphologyEncoder(nn.Module):
    """
    2D ResNet Architecture to process GAF, Spectrogram, or Scalogram representations of 12-lead ECG.
    Extracts a 512-dimensional morphology representation (Z_morphology).
    """
    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        layers: list = [2, 2, 2, 2],
        base_filters: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_channels = base_filters
        
        self.conv = nn.Conv2d(input_channels, base_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(base_filters, layers[0], stride=1)
        self.layer2 = self._make_layer(base_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_filters * 8, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head
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
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(ResBlock2D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock2D(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts Z_morphology.
        
        Args:
            x: Input tensor of shape (batch_size, 12, H, W)
            
        Returns:
            torch.Tensor: Latent representation of shape (batch_size, 512)
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
