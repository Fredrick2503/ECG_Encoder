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
from temporal_encoder.predictor import TemporalPredictor
from temporal_encoder.evaluator import TemporalEvaluator
from temporal_encoder.explainer import TemporalSaliencyExplainer

__all__ = [
    "ECGBiLSTM",
    "ECGReconstructionDecoder",
    "ECGResNet1D",
    "ECGTransformer",
    "ECGMultiScaleCNN",
    "SqueezeExcitation1D",
    "ResBlock1D",
    "TemporalPredictor",
    "TemporalEvaluator",
    "TemporalSaliencyExplainer",
]
