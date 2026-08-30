"""
ECG Foundation Representation System - Encoder Wrappers
=======================================================
Implements concrete BaseEncoder wrappers for the 3 frozen modalities.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn

from ecg_engine.interfaces import BaseEncoder
from temporal_encoder.encoder_upgrades import ECGResNet1D
from morphology_encoder.encoder import ECGMorphologyEncoder
from morphology_encoder.conversion import ecg_to_spectrogram
from biomarkers.models import AttentionMLPAutoencoder


class TemporalEncoderWrapper(BaseEncoder):
    """
    Wrapper for 1D Temporal ResNet-SE Foundation Encoder.
    Outputs: z_temporal in R^512.
    """
    def __init__(self, model_path: Optional[Union[str, Path]] = "models/C5_full_dataset.pt", device: str = "cpu"):
        super().__init__()
        self._output_dim = 512
        self.device = device
        self.model = ECGResNet1D(num_classes=5, use_se=True).to(device)
        
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device, weights_only=False)
            except TypeError:
                state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Preprocessed tensor of shape (batch, 12, length)
        Returns:
            torch.Tensor: (batch, 512)
        """
        x = x.to(self.device)
        with torch.no_grad():
            return self.model.get_representation(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_representation(x)


class MorphologyEncoderWrapper(BaseEncoder):
    """
    Wrapper for 2D Morphology ResNet Encoder.
    Converts 12-lead signal to 2D time-frequency spectrograms and extracts morphology embeddings.
    Outputs: z_morphology in R^512.
    """
    def __init__(self, model_path: Optional[Union[str, Path]] = "models/morphology_encoder_v1.pt", device: str = "cpu"):
        super().__init__()
        self._output_dim = 512
        self.device = device
        self.model = ECGMorphologyEncoder(input_channels=12, num_classes=5).to(device)
        
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device, weights_only=False)
            except TypeError:
                state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Preprocessed tensor of shape (batch, 12, length)
        Returns:
            torch.Tensor: (batch, 512)
        """
        x = x.to(self.device)
        with torch.no_grad():
            spec = ecg_to_spectrogram(x)
            return self.model.get_representation(spec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_representation(x)


class BiomarkerEncoderWrapper(BaseEncoder):
    """
    Wrapper for Attention MLP Biomarker Autoencoder.
    Outputs: z_biomarker in R^32.
    """
    def __init__(self, model_path: Optional[Union[str, Path]] = "biomarkers/attention_mlp_best.pt", device: str = "cpu"):
        super().__init__()
        self._output_dim = 32
        self.device = device
        self.model = AttentionMLPAutoencoder(input_dim=50, latent_dim=32).to(device)
        
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device, weights_only=False)
            except TypeError:
                state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def get_representation(self, x_joint: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_joint: Joint biomarker tensor of shape (batch, 50)
        Returns:
            torch.Tensor: (batch, 32)
        """
        x_joint = x_joint.to(self.device)
        with torch.no_grad():
            return self.model.encode(x_joint)

    def forward(self, x_joint: torch.Tensor) -> torch.Tensor:
        return self.get_representation(x_joint)
