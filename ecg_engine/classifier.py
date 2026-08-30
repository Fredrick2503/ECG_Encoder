"""
ECG Foundation Representation System - Fusion & Classification
==============================================================
Implements deterministic multimodal latent fusion and calibrated multi-label classification.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from ecg_engine.interfaces import BaseClassifier, DEFAULT_CLASS_NAMES
from classification.classifier import MLPClassifier


class FusionEngine:
    """
    Combines representations across temporal (512), morphology (512), and biomarker (32) modalities
    into the unified 1056-dimensional space: z_fused = [z_t, z_m, z_b].
    """
    def __init__(self):
        self.output_dim = 1056

    def fuse(self, z_t: torch.Tensor, z_m: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Concatenates latent vectors along feature dimension.
        """
        return torch.cat([z_t, z_m, z_b], dim=1)


class DiagnosticClassifier(BaseClassifier):
    """
    Calibrated Multi-Layer Perceptron (MLP) Classifier.
    Predicts diagnostic superclasses (NORM, MI, STTC, CD, HYP) from z_fused in R^1056.
    """
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = "models/classification_mlp.pt",
        thresholds_path: Optional[Union[str, Path]] = "models/classification_mlp_thresholds.npy",
        class_names: Optional[List[str]] = None,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        self.class_names = class_names or list(DEFAULT_CLASS_NAMES)
        self.model = MLPClassifier(input_dim=1056, hidden_dim=256, num_classes=len(self.class_names)).to(device)
        
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device, weights_only=False)
            except TypeError:
                state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
            
        # Load optimal decision thresholds
        if thresholds_path and os.path.exists(thresholds_path):
            self.thresholds = np.load(thresholds_path)
        else:
            self.thresholds = np.full(len(self.class_names), 0.5, dtype=np.float32)

    def predict_proba(self, z_fused: torch.Tensor) -> torch.Tensor:
        """
        Returns class probabilities in [0, 1].
        """
        z_fused = z_fused.to(self.device)
        with torch.no_grad():
            logits = self.model(z_fused)
            probs = torch.sigmoid(logits)
            return probs

    def predict(self, z_fused: torch.Tensor, thresholds: Optional[Union[np.ndarray, torch.Tensor]] = None) -> torch.Tensor:
        """
        Returns binary multi-label predictions based on calibrated thresholds.
        """
        probs = self.predict_proba(z_fused)
        
        if thresholds is None:
            th = torch.tensor(self.thresholds, dtype=torch.float32, device=self.device)
        elif isinstance(thresholds, np.ndarray):
            th = torch.from_numpy(thresholds).float().to(self.device)
        else:
            th = thresholds.to(self.device)
            
        preds = (probs >= th).long()
        return preds

    def forward(self, z_fused: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(z_fused)
