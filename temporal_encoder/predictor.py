import torch
import numpy as np
from typing import Union

class TemporalPredictor:
    """
    Inference helper that wraps a trained encoder model to extract 
    classification probabilities, binary predictions, and latent representations.
    """
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def predict_proba(self, dataloader: torch.utils.data.DataLoader) -> np.ndarray:
        """Computes sigmoid classification probabilities for the dataset."""
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                signals = batch[0] if isinstance(batch, (list, tuple)) else batch
                signals = signals.to(self.device)
                
                logits = self.model(signals)
                probs = torch.sigmoid(logits)
                probabilities.append(probs.cpu().numpy())
                
        return np.concatenate(probabilities, axis=0)

    def predict(self, dataloader: torch.utils.data.DataLoader, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary outputs based on thresholding probabilities."""
        probs = self.predict_proba(dataloader)
        return (probs >= threshold).astype(np.float32)

    def get_embeddings(self, dataloader: torch.utils.data.DataLoader) -> np.ndarray:
        """Extracts representation embeddings (z) from the encoder model."""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                signals = batch[0] if isinstance(batch, (list, tuple)) else batch
                signals = signals.to(self.device)
                
                # Check if model has get_representation method
                if hasattr(self.model, "get_representation"):
                    z = self.model.get_representation(signals)
                else:
                    # Fallback
                    z = self.model(signals)
                    
                embeddings.append(z.cpu().numpy())
                
        return np.concatenate(embeddings, axis=0)
