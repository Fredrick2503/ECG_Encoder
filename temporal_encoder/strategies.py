from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
import numpy as np

class BaseSSLStrategy(ABC):
    """Abstract base class for all Self-Supervised Learning (SSL) pretraining strategies."""
    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        decoder: Optional[nn.Module],
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the pretraining loss for the strategy.
        
        Args:
            model: The shared encoder model (ECGBiLSTM).
            decoder: Optional decoder model for reconstruction tasks.
            x: Input batch tensor of shape (batch_size, num_leads, signal_length).
            
        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        pass


class ReconstructionLearningStrategy(BaseSSLStrategy):
    """
    Reconstruction pretraining strategy.
    Encodes the complete signal and reconstructs it using the decoder.
    """
    def __init__(self):
        self.criterion = nn.MSELoss()

    def compute_loss(
        self,
        model: nn.Module,
        decoder: Optional[nn.Module],
        x: torch.Tensor
    ) -> torch.Tensor:
        if decoder is None:
            raise ValueError("Reconstruction strategy requires a decoder.")
            
        z = model.get_representation(x)
        x_hat = decoder(z)
        return self.criterion(x, x_hat)


class MaskedAutoencoderStrategy(BaseSSLStrategy):
    """
    Masked Autoencoder (MAE) pretraining strategy.
    Randomly masks portions of the time-steps and reconstructs the masked regions.
    """
    def __init__(self, mask_ratio: float = 0.3):
        """
        Args:
            mask_ratio: Fraction of the signal length to mask (default 30%).
        """
        self.mask_ratio = mask_ratio

    def compute_loss(
        self,
        model: nn.Module,
        decoder: Optional[nn.Module],
        x: torch.Tensor
    ) -> torch.Tensor:
        if decoder is None:
            raise ValueError("MAE strategy requires a decoder.")
            
        batch_size, num_leads, length = x.shape
        
        # 1. Create a binary mask over time-steps: shape (batch_size, 1, length)
        # 1 represents visible, 0 represents masked
        mask = torch.ones((batch_size, 1, length), device=x.device, dtype=x.dtype)
        
        # Generate random mask positions for each batch element
        num_masked_samples = int(self.mask_ratio * length)
        for i in range(batch_size):
            masked_indices = torch.randperm(length)[:num_masked_samples]
            mask[i, 0, masked_indices] = 0.0
            
        # 2. Mask the signal (set masked indices to 0)
        x_masked = x * mask
        
        # 3. Encode visible signal
        z = model.get_representation(x_masked)
        
        # 4. Decode full signal
        x_hat = decoder(z)
        
        # 5. Compute MSE loss only on masked regions
        masked_loss_mask = 1.0 - mask
        loss = torch.sum(((x - x_hat) * masked_loss_mask) ** 2) / (torch.sum(masked_loss_mask) * num_leads + 1e-8)
        return loss


class ContrastiveLearningStrategy(BaseSSLStrategy):
    """
    Contrastive Learning pretraining strategy using InfoNCE loss.
    Creates two augmented views of each signal and maximizes their similarity.
    """
    def __init__(self, temperature: float = 0.1, projection_dim: int = 64, latent_dim: int = 256):
        """
        Args:
            temperature: Scaling factor for InfoNCE similarity.
            projection_dim: Output dimension of projection head.
            latent_dim: Input latent representation dimension (hidden_size * 2).
        """
        self.temperature = temperature
        # Projection head to map representations to a lower-dimensional sphere
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, projection_dim)
        )

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Applies random augmentation: Gaussian noise + random scale."""
        # 1. Add small random Gaussian noise
        noise = torch.randn_like(x) * 0.05
        # 2. Apply random scaling factor
        scale = (torch.rand(x.shape[0], 1, 1, device=x.device) * 0.4 + 0.8) # Range: [0.8, 1.2]
        return (x + noise) * scale

    def compute_loss(
        self,
        model: nn.Module,
        decoder: Optional[nn.Module],
        x: torch.Tensor
    ) -> torch.Tensor:
        # Move projection head parameters to the correct device dynamically
        self.projection_head = self.projection_head.to(x.device)
        
        # 1. Create two augmented views
        x1 = self._augment(x)
        x2 = self._augment(x)
        
        # 2. Get representations
        z1 = model.get_representation(x1)
        z2 = model.get_representation(x2)
        
        # 3. Project to contrastive head
        p1 = nn.functional.normalize(self.projection_head(z1), dim=1)
        p2 = nn.functional.normalize(self.projection_head(z2), dim=1)
        
        # 4. Compute SimCLR / InfoNCE loss
        batch_size = x.shape[0]
        out = torch.cat([p1, p2], dim=0)
        
        # Compute cosine similarity matrix (2B, 2B)
        sim = torch.exp(torch.matmul(out, out.T) / self.temperature)
        
        # Mask out self-similarity elements on the diagonal
        mask = torch.eye(2 * batch_size, device=x.device).bool()
        sim = sim.masked_fill(mask, 0.0)
        
        # Extract similarities for positive pairs: (p1_i, p2_i)
        pos1 = torch.exp(torch.sum(p1 * p2, dim=-1) / self.temperature)
        pos = torch.cat([pos1, pos1], dim=0)
        
        # Loss formula: -log( pos_i / sum(sim_i) )
        loss = -torch.log(pos / (torch.sum(sim, dim=1) + 1e-8))
        return torch.mean(loss)
