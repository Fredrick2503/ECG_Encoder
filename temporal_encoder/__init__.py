"""
Temporal Encoder Package
========================
Exports temporal representation architectures for 12-lead ECG signals.
"""

from temporal_encoder.encoder import (
    ECGBiLSTM,
    ECGReconstructionDecoder,
    ECGResNet1D,
    ECGTransformer,
    ECGMultiScaleCNN,
    SqueezeExcitation1D,
    ResBlock1D,
)
from temporal_encoder.encoder_upgrades import (
    ECGResNet1D as ECGResNet1DUpgraded,
)

__all__ = [
    "ECGBiLSTM",
    "ECGReconstructionDecoder",
    "ECGResNet1D",
    "ECGTransformer",
    "ECGMultiScaleCNN",
    "SqueezeExcitation1D",
    "ResBlock1D",
    "ECGResNet1DUpgraded",
]