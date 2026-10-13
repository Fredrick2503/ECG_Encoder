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
    Supports native 2-second (200 samples @ 100Hz) and 10-second (1000 samples @ 100Hz) modes.
    """
    def __init__(
        self,
        target_length: Optional[int] = 1000,
        num_leads: int = 12,
        auto_adapt_duration: bool = True
    ):
        self.target_length = target_length
        self.num_leads = num_leads
        self.auto_adapt_duration = auto_adapt_duration

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

        # Resample / interpolate along time dimension if needed
        if self.target_length is not None:
            if self.auto_adapt_duration and x.shape[2] <= 300:
                # Native ultra-short mode (2.0s @ 100Hz = 200 samples)
                if x.shape[2] != 200:
                    x = F.interpolate(x, size=200, mode='linear', align_corners=False)
            elif self.auto_adapt_duration and x.shape[2] in (200, 1000):
                pass
            elif x.shape[2] != self.target_length:
                x = F.interpolate(x, size=self.target_length, mode='linear', align_corners=False)

        # Per-lead normalization: for short windows (<= 300 samples), use isoelectric
        # median baseline subtraction to prevent QRS/T-wave area from depressing the
        # baseline and creating false ST elevation (STEMI / STTC false alarms).
        if x.shape[2] <= 300:
            baseline = torch.median(x, dim=2, keepdim=True).values
        else:
            baseline = x.mean(dim=2, keepdim=True)

        std = x.std(dim=2, keepdim=True) + 1e-8
        x_norm = (x - baseline) / std

        return x_norm
