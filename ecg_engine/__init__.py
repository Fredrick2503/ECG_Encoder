"""
ECG Foundation Representation System Engine Package
===================================================
Unified, frozen, multi-modal foundation encoder and diagnostic classification engine for 12-lead ECG.
"""

from ecg_engine.interfaces import (
    EngineConfig,
    FusedRepresentationResult,
    DiagnosticPredictionResult,
    DEFAULT_CLASS_NAMES,
    BaseEncoder,
    BaseClassifier,
    BasePreprocessor,
    BaseBiomarkerService,
)
from ecg_engine.engine import ECGEncoderEngine
from ecg_engine.preprocessor import SignalPreprocessor
from ecg_engine.biomarker_service import BiomarkerService
from ecg_engine.wrappers import (
    TemporalEncoderWrapper,
    MorphologyEncoderWrapper,
    BiomarkerEncoderWrapper,
)
from ecg_engine.fusion import FusionEngine
from ecg_engine.classifier import DiagnosticClassifier

__all__ = [
    "ECGEncoderEngine",
    "EngineConfig",
    "FusedRepresentationResult",
    "DiagnosticPredictionResult",
    "DEFAULT_CLASS_NAMES",
    "BaseEncoder",
    "BaseClassifier",
    "BasePreprocessor",
    "BaseBiomarkerService",
    "SignalPreprocessor",
    "BiomarkerService",
    "TemporalEncoderWrapper",
    "MorphologyEncoderWrapper",
    "BiomarkerEncoderWrapper",
    "FusionEngine",
    "DiagnosticClassifier",
]
