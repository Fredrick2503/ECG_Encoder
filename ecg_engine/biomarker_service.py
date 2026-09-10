"""
ECG Foundation Representation System - Biomarker Service
========================================================
Extracts or formats clinical biomarkers into joint missingness-masked feature vectors (50-D).
"""

from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
from ecg_engine.interfaces import BaseBiomarkerService


FEATURES = [
    "RR_Mean", "QRS_Duration", "PR_Interval", "QT_Interval", "QTc_Bazett",
    "ST_Duration", "P_wave_Duration", "R_Amplitude", "P_Amplitude", "T_Amplitude",
    "ST_Deviation", "Q_Amplitude", "R_S_Ratio", "QRS_Energy", "SDNN",
    "RMSSD", "pNN50", "pNN20", "SDRR_RMSSD_Ratio", "HRV_Triangular_Index",
    "LF_Power", "HF_Power", "LF_HF_Ratio", "Total_Power", "Sample_Entropy"
]


class BiomarkerService(BaseBiomarkerService):
    """
    Manages clinical biomarker preprocessing, scaling, and missingness binary masking.
    Output is a 50-dimensional joint representation: [Scaled_Features (25), Missingness_Mask (25)].
    """
    def __init__(
        self,
        imputer_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None
    ):
        self.imputer = None
        self.scaler = None

        if imputer_path is None:
            if os.path.exists("biomarkers/imputer_cwt.pkl"):
                imputer_path = "biomarkers/imputer_cwt.pkl"
            else:
                imputer_path = "biomarkers/imputer.pkl"

        if scaler_path is None:
            if os.path.exists("biomarkers/scaler_cwt.pkl"):
                scaler_path = "biomarkers/scaler_cwt.pkl"
            else:
                scaler_path = "biomarkers/scaler.pkl"
        
        if imputer_path and os.path.exists(imputer_path):
            with open(imputer_path, "rb") as f:
                self.imputer = pickle.load(f)
                
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

    @property
    def num_features(self) -> int:
        if self.scaler is not None and hasattr(self.scaler, "n_features_in_"):
            return self.scaler.n_features_in_
        if self.imputer is not None and hasattr(self.imputer, "n_features_in_"):
            return self.imputer.n_features_in_
        return 24

    def extract_or_impute(
        self,
        signal: Optional[np.ndarray | torch.Tensor] = None,
        raw_features: Optional[np.ndarray] = None,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Formats or imputes raw clinical features into standard joint input for AttentionMLPAutoencoder.
        
        Args:
            signal: Optional raw signal.
            raw_features: Optional pre-extracted feature array (24, 25, 48, or 50 dimensions).
            batch_size: Number of samples in batch.
            
        Returns:
            torch.Tensor: Shape (batch_size, 2 * num_features).
        """
        target_n = self.num_features
        if raw_features is not None:
            feats = np.asarray(raw_features, dtype=np.float32)
            if feats.ndim == 1:
                feats = feats.reshape(1, -1)
            
            if feats.shape[1] in (48, 50):
                if feats.shape[1] == target_n * 2:
                    return torch.from_numpy(feats).float()
                elif feats.shape[1] > target_n * 2:
                    # Truncate
                    feats_sub = np.concatenate([
                        feats[:, :target_n],
                        feats[:, feats.shape[1]//2 : feats.shape[1]//2 + target_n]
                    ], axis=1)
                    return torch.from_numpy(feats_sub).float()
                else:
                    # Pad
                    pad_val = np.zeros((feats.shape[0], target_n * 2 - feats.shape[1]), dtype=np.float32)
                    feats_padded = np.concatenate([feats, pad_val], axis=1)
                    return torch.from_numpy(feats_padded).float()

            if feats.shape[1] in (24, 25):
                if feats.shape[1] > target_n:
                    feats = feats[:, :target_n]
                elif feats.shape[1] < target_n:
                    pad_val = np.zeros((feats.shape[0], target_n - feats.shape[1]), dtype=np.float32)
                    feats = np.concatenate([feats, pad_val], axis=1)

                mask = (~np.isnan(feats)).astype(np.float32)
                if self.imputer is not None:
                    feats_imp = self.imputer.transform(feats)
                else:
                    feats_imp = np.nan_to_num(feats, nan=0.0)
                    
                if self.scaler is not None:
                    feats_scaled = self.scaler.transform(feats_imp)
                else:
                    feats_scaled = feats_imp
                    
                joint = np.concatenate([feats_scaled, mask], axis=1) # 2 * target_n
                return torch.from_numpy(joint.astype(np.float32)).float()

        # Fallback: create default zero/neutral feature vectors with missingness mask = 0
        feats_scaled = np.zeros((batch_size, target_n), dtype=np.float32)
        mask = np.zeros((batch_size, target_n), dtype=np.float32)
        joint = np.concatenate([feats_scaled, mask], axis=1)
        return torch.from_numpy(joint).float()
