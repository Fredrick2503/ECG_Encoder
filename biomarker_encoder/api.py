import torch
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Union, List
from representation_generation.biomarker_extractor import ECGFeatureExtractor
from biomarker_encoder.models import AttentionMLPAutoencoder, BetaVAE, FTTransformerAutoencoder

class ECGBiomarkerEncoderAPI:
    def __init__(
        self,
        model_type: str,
        checkpoint_path: str,
        scaler_path: str,
        imputer_path: str,
        feature_cols_path: str,
        input_dim: int = 50,
        latent_dim: int = 32,
        device: str = None,
        model_kwargs: dict = None
    ):
        self.model_type = model_type.lower()
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load scaler, imputer, and feature columns
        self.scaler = joblib.load(scaler_path)
        self.imputer = joblib.load(imputer_path)
        self.feature_cols = joblib.load(feature_cols_path)
        
        # Instantiate model
        kwargs = model_kwargs if model_kwargs else {}
        if self.model_type == "attention_mlp":
            self.model = AttentionMLPAutoencoder(input_dim=input_dim, latent_dim=latent_dim, **kwargs)
        elif self.model_type == "beta_vae":
            self.model = BetaVAE(input_dim=input_dim, latent_dim=latent_dim, **kwargs)
        elif self.model_type == "ft_transformer":
            self.model = FTTransformerAutoencoder(input_dim=input_dim, latent_dim=latent_dim, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Load weights
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Extractor for raw signal inputs (fs=500 by default)
        self.extractor = ECGFeatureExtractor(fs=500, leads="II")

    def extract_embeddings_from_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Extract latent embeddings from pre-extracted features DataFrame."""
        # Align features with expected column order
        X = features_df[self.feature_cols].copy()
        
        # Outlier handling (clamping to the fitted scaler's range limits or generic clipping)
        # Clip features to be safe
        for col in X.columns:
            q_low = X[col].quantile(0.01)
            q_high = X[col].quantile(0.99)
            X[col] = np.clip(X[col], q_low, q_high)
            
        # Impute
        X_filled = X.copy()
        for col in X_filled.columns:
            if X_filled[col].isna().all():
                X_filled[col] = 0.0
        X_imputed = self.imputer.transform(X_filled)
        
        # Scale
        X_scaled = self.scaler.transform(X_imputed)
        
        # Forward pass through model to get latent representation
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            if hasattr(self.model, "loss_function"):  # VAE
                mu, logvar = self.model.encode(X_tensor)
                embeddings = mu.cpu().numpy()
            else:
                if self.model_type == "ft_transformer":
                    embeddings = self.model.encode(X_tensor).cpu().numpy()
                else:
                    # Attention MLP forward has encode
                    embeddings = self.model.encode(X_tensor).cpu().numpy()
                    
        return embeddings

    def extract_embeddings(self, raw_signal: np.ndarray, fs: int = 500) -> np.ndarray:
        """Extract latent embeddings directly from raw ECG signal."""
        # Update extractor fs if different
        self.extractor.fs = fs
        
        # Extract features
        features = self.extractor.extract(raw_signal, leads="II")
        features_dict = features.to_dict()
        
        # Convert to DataFrame
        df = pd.DataFrame([features_dict])
        
        return self.extract_embeddings_from_features(df)
