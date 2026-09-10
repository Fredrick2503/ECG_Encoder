"""
ECG Foundation Representation System - Preprocessor Implementation
=================================================================
Handles lead normalization, baseline removal, and signal dimension alignment.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from ecg_engine.interfaces import BasePreprocessor


class SignalPreprocessor(BasePreprocessor):
    """
    Concrete preprocessor for 12-lead ECG time-series signals.
    Standardizes shape to (12, target_length) and applies robust z-score normalization.
    """
    def __init__(self, target_length: int = 1000, num_leads: int = 12):
        self.target_length = target_length
        self.num_leads = num_leads

    def preprocess(self, signal: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Converts arbitrary 12-lead ECG signal into standardized PyTorch tensor of shape (batch, 12, target_length).
        
        Args:
            signal: numpy array or torch Tensor with shape (12, L), (L, 12), (batch, 12, L), or (batch, L, 12).
            
        Returns:
            torch.Tensor: Float32 tensor of shape (batch_size, 12, target_length).
        """
        if isinstance(signal, np.ndarray):
            x = torch.from_numpy(signal).float()
        else:
            x = signal.clone().float()

        # Handle dimensions
        if x.ndim == 2:
            # Single sample: check if shape is (12, L) or (L, 12)
            if x.shape[0] == self.num_leads:
                x = x.unsqueeze(0)  # (1, 12, L)
            elif x.shape[1] == self.num_leads:
                x = x.transpose(0, 1).unsqueeze(0)  # (1, 12, L)
            else:
                # Default transpose assuming (L, leads)
                x = x.transpose(0, 1).unsqueeze(0)
        elif x.ndim == 3:
            # Batch sample: check if shape is (batch, 12, L) or (batch, L, 12)
            if x.shape[1] != self.num_leads and x.shape[2] == self.num_leads:
                x = x.transpose(1, 2)  # (batch, 12, L)
        else:
            raise ValueError(f"Expected 2D or 3D tensor/array, got shape {x.shape}")

        # Resample / interpolate along time dimension if length does not match target_length
        if x.shape[2] != self.target_length:
            x = F.interpolate(x, size=self.target_length, mode='linear', align_corners=False)

        # Per-lead zero-mean, unit-variance normalization (Z-score)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-8
        x_norm = (x - mean) / std

        return x_norm
