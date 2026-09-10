"""
ECG Foundation Representation System - Engine Interfaces & Protocols
====================================================================
Defines abstract contracts and data transfer objects following SOLID principles.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn

# Standard PTB-XL Diagnostic Superclasses
DEFAULT_CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]


@dataclass
class EngineConfig:
    """Configuration data class for ECGEncoderEngine dependency injection."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    temporal_model_path: Optional[str] = "models/C5_full_dataset.pt"
    morphology_model_path: Optional[str] = "models/morphology_encoder_v1.pt"
    biomarker_model_path: Optional[str] = "biomarkers/attention_mlp_cwt.pt"
    classifier_model_path: Optional[str] = "models/classification_mlp.pt"
    thresholds_path: Optional[str] = "models/classification_mlp_thresholds.npy"
    imputer_path: Optional[str] = "biomarkers/imputer_cwt.pkl"
    scaler_path: Optional[str] = "biomarkers/scaler_cwt.pkl"
    class_names: List[str] = field(default_factory=lambda: list(DEFAULT_CLASS_NAMES))
    sampling_rate: int = 100
    target_length: int = 1000
    num_leads: int = 12


@dataclass
class FusedRepresentationResult:
    """Encapsulates the multimodal latent representations extracted by the engine."""
    z_fused: np.ndarray        # (batch_size, 1056)
    z_temporal: np.ndarray     # (batch_size, 512)
    z_morphology: np.ndarray   # (batch_size, 512)
    z_biomarker: np.ndarray    # (batch_size, 32)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "z_fused": self.z_fused,
            "z_temporal": self.z_temporal,
            "z_morphology": self.z_morphology,
            "z_biomarker": self.z_biomarker,
        }


@dataclass
class DiagnosticPredictionResult:
    """Encapsulates multi-label diagnostic classification outcomes."""
    probabilities: np.ndarray        # (batch_size, 5) float [0, 1]
    predictions: np.ndarray          # (batch_size, 5) binary {0, 1}
    class_names: List[str]
    thresholds: np.ndarray           # (5,) calibrated decision boundaries
    detected_conditions: List[List[str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probabilities": self.probabilities,
            "predictions": self.predictions,
            "class_names": self.class_names,
            "thresholds": self.thresholds,
            "detected_conditions": self.detected_conditions,
        }


class BaseEncoder(ABC, nn.Module):
    """Abstract Base Class for representation learning encoders."""
    
    @abstractmethod
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent embedding vector from input tensor."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the dimension of the produced representation."""
        pass


class BaseClassifier(ABC, nn.Module):
    """Abstract Base Class for diagnostic classifiers."""
    
    @abstractmethod
    def predict_proba(self, z: torch.Tensor) -> torch.Tensor:
        """Return class probabilities [0, 1] from representation."""
        pass

    @abstractmethod
    def predict(self, z: torch.Tensor, thresholds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return binary multi-label predictions."""
        pass


class BasePreprocessor(ABC):
    """Abstract Base Class for signal preprocessors."""
    
    @abstractmethod
    def preprocess(self, signal: np.ndarray) -> np.ndarray:
        """Normalize, filter, and structure raw ECG signal."""
        pass


class BaseBiomarkerService(ABC):
    """Abstract Base Class for clinical biomarker feature extraction & preparation."""
    
    @abstractmethod
    def extract_or_impute(
        self,
        signal: Optional[np.ndarray] = None,
        raw_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Produce standardized, missingness-masked biomarker feature vector."""
        pass
