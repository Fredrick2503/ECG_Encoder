import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Focal Loss for binary / multi-label classification.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Classification.
    Ref: https://arxiv.org/abs/2009.14119
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xs_p = torch.sigmoid(x)
        xs_n = 1.0 - xs_p

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_n = (xs_n + self.clip).clamp(max=1.0)

        # Positive and negative loss
        loss_pos = y * torch.log(xs_p.clamp(min=self.eps))
        loss_neg = (1 - y) * torch.log(xs_n.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_pos > 0:
            loss_pos *= (1 - xs_p) ** self.gamma_pos
        if self.gamma_neg > 0:
            loss_neg *= (1 - xs_n) ** self.gamma_neg

        loss = loss_pos + loss_neg
        
        if self.reduction == 'mean':
            return -loss.mean()
        elif self.reduction == 'sum':
            return -loss.sum()
        else:
            return -loss
