import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)

class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on Cui et al. (CVPR 2019).
    Re-weights the BCE loss per-class based on the effective number of samples.
    """
    def __init__(self, samples_per_class, num_classes=5, beta=0.9999):
        super().__init__()
        if not isinstance(samples_per_class, torch.Tensor):
            samples_per_class = torch.tensor(samples_per_class, dtype=torch.float32)
            
        effective_num = (1.0 - torch.pow(beta, samples_per_class)) / (1.0 - beta)
        weights = 1.0 / (effective_num + 1e-8)
        # Normalize weights so they sum to the number of classes
        weights = weights / weights.sum() * num_classes
        self.register_buffer('weights', weights)

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weighted_bce = bce * self.weights.unsqueeze(0)
        return weighted_bce.mean()

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification.
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        xs_p = torch.sigmoid(logits)
        xs_n = 1.0 - xs_p

        # Asymmetric Clipping for negatives
        if self.clip is not None and self.clip > 0:
            xs_n = (xs_n + self.clip).clamp(max=1.0)

        # Basic BCE terms
        loss_pos = targets * torch.log(xs_p.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_n.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_pos > 0:
            loss_pos *= (1.0 - xs_p) ** self.gamma_pos
        if self.gamma_neg > 0:
            loss_neg *= (1.0 - xs_n) ** self.gamma_neg

        loss = loss_pos + loss_neg
        return -loss.mean()
