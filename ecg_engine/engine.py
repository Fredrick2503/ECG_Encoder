"""
ECG Foundation Representation System - Master Engine
====================================================
Unified, object-oriented facade providing complete representation learning and diagnostic classification.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch

from ecg_engine.interfaces import (
    EngineConfig,
    FusedRepresentationResult,
    DiagnosticPredictionResult,
    DEFAULT_CLASS_NAMES,
)
from ecg_engine.preprocessor import SignalPreprocessor
from ecg_engine.biomarker_service import BiomarkerService
from ecg_engine.wrappers import (
    TemporalEncoderWrapper,
    MorphologyEncoderWrapper,
    BiomarkerEncoderWrapper,
)
from ecg_engine.fusion import FusionEngine
from ecg_engine.classifier import DiagnosticClassifier


class ECGEncoderEngine:
    """
    Unified High-Level Engine for the ECG Foundation Representation System.
    
    Adheres to SOLID principles:
    - Single Responsibility: Orchestrates preprocessing, multimodal encoders, fusion, and classification.
    - Open/Closed: Extensible to new encoders via BaseEncoder interface.
    - Dependency Inversion: Configurable with custom models, thresholds, and preprocessors.
    
    Example Usage:
    >>> engine = ECGEncoderEngine()
    >>> ecg_signal = np.random.randn(12, 1000) # 12-lead ECG, 1000 samples
    >>> rep = engine.encode(ecg_signal)
    >>> print("Fused shape:", rep.z_fused.shape) # (1, 1056)
    >>> pred = engine.predict(ecg_signal)
    >>> print("Detected conditions:", pred.detected_conditions)
    """
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        
        # Initialize pipeline components via dependency injection
        self.preprocessor = SignalPreprocessor(
            target_length=self.config.target_length,
            num_leads=self.config.num_leads
        )
        self.biomarker_service = BiomarkerService(
            imputer_path=self.config.imputer_path,
            scaler_path=self.config.scaler_path
        )
        self.temporal_encoder = TemporalEncoderWrapper(
            model_path=self.config.temporal_model_path,
            device=str(self.device)
        )
        self.morphology_encoder = MorphologyEncoderWrapper(
            model_path=self.config.morphology_model_path,
            device=str(self.device)
        )
        self.biomarker_encoder = BiomarkerEncoderWrapper(
            model_path=self.config.biomarker_model_path,
            device=str(self.device)
        )
        self.fusion_engine = FusionEngine()
        self.classifier = DiagnosticClassifier(
            model_path=self.config.classifier_model_path,
            thresholds_path=self.config.thresholds_path,
            class_names=self.config.class_names,
            device=str(self.device)
        )

    def encode(
        self,
        ecg_signal: np.ndarray | torch.Tensor,
        biomarker_features: Optional[np.ndarray] = None
    ) -> FusedRepresentationResult:
        """
        Extracts temporal (512), morphology (512), biomarker (32), and fused (1056) embeddings.
        
        Args:
            ecg_signal: 12-lead ECG signal array or tensor.
            biomarker_features: Optional raw or imputed clinical biomarker features.
            
        Returns:
            FusedRepresentationResult containing extracted representations.
        """
        x_norm = self.preprocessor.preprocess(ecg_signal).to(self.device)
        batch_size = x_norm.shape[0]

        # 1. Temporal Representation (512-D)
        z_t = self.temporal_encoder.get_representation(x_norm)

        # 2. Morphology Representation (512-D)
        z_m = self.morphology_encoder.get_representation(x_norm)

        # 3. Biomarker Representation (32-D)
        x_bio = self.biomarker_service.extract_or_impute(
            signal=ecg_signal,
            raw_features=biomarker_features,
            batch_size=batch_size
        ).to(self.device)
        z_b = self.biomarker_encoder.get_representation(x_bio)

        # 4. Multimodal Fusion (1056-D)
        z_fused = self.fusion_engine.fuse(z_t, z_m, z_b)

        return FusedRepresentationResult(
            z_fused=z_fused.detach().cpu().numpy(),
            z_temporal=z_t.detach().cpu().numpy(),
            z_morphology=z_m.detach().cpu().numpy(),
            z_biomarker=z_b.detach().cpu().numpy(),
            metadata={"batch_size": batch_size, "device": str(self.device)}
        )

    def predict(
        self,
        ecg_signal: Optional[np.ndarray | torch.Tensor] = None,
        z_fused: Optional[np.ndarray | torch.Tensor] = None,
        biomarker_features: Optional[np.ndarray] = None,
        custom_thresholds: Optional[np.ndarray] = None
    ) -> DiagnosticPredictionResult:
        """
        Performs multi-label diagnostic classification on ECG signal or pre-extracted z_fused.
        
        Args:
            ecg_signal: 12-lead ECG signal (if z_fused is not provided).
            z_fused: Pre-extracted 1056-D fused representation (optional).
            biomarker_features: Optional biomarker features.
            custom_thresholds: Optional override for decision thresholds.
            
        Returns:
            DiagnosticPredictionResult containing probabilities, binary predictions, and condition tags.
        """
        if z_fused is None:
            if ecg_signal is None:
                raise ValueError("Either ecg_signal or z_fused must be provided.")
            rep_res = self.encode(ecg_signal, biomarker_features=biomarker_features)
            z_fused_tensor = torch.from_numpy(rep_res.z_fused).float().to(self.device)
        else:
            if isinstance(z_fused, np.ndarray):
                z_fused_tensor = torch.from_numpy(z_fused).float().to(self.device)
            else:
                z_fused_tensor = z_fused.to(self.device)

        probs_tensor = self.classifier.predict_proba(z_fused_tensor)
        preds_tensor = self.classifier.predict(z_fused_tensor, thresholds=custom_thresholds)

        probs = probs_tensor.detach().cpu().numpy()
        preds = preds_tensor.detach().cpu().numpy()
        th = custom_thresholds if custom_thresholds is not None else self.classifier.thresholds

        # Map binary predictions to diagnostic labels
        detected_conditions = []
        for row in preds:
            detected = [
                self.classifier.class_names[idx]
                for idx, val in enumerate(row)
                if val == 1
            ]
            if not detected:
                detected = ["NORM (Presumed Normal)"]
            detected_conditions.append(detected)

        return DiagnosticPredictionResult(
            probabilities=probs,
            predictions=preds,
            class_names=self.classifier.class_names,
            thresholds=th,
            detected_conditions=detected_conditions
        )

    def process(
        self,
        ecg_signal: np.ndarray | torch.Tensor,
        biomarker_features: Optional[np.ndarray] = None
    ) -> Tuple[FusedRepresentationResult, DiagnosticPredictionResult]:
        """
        Convenience method executing both encoding and classification in a single call.
        """
        rep = self.encode(ecg_signal, biomarker_features=biomarker_features)
        pred = self.predict(z_fused=rep.z_fused)
        return rep, pred
