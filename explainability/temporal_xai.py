import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, LayerGradCam, Occlusion

class TemporalExplainer:
    """
    Explainer for Temporal Encoders (e.g., ECGResNet1D, ECGBiLSTM, ECGTransformer).
    Uses Captum to compute attributions via Integrated Gradients, Grad-CAM, and Occlusion.
    """
    def __init__(self, model: nn.Module, device: str = None):
        self.device = device if device else next(model.parameters()).device
        self.model = model.to(self.device)
        self.model.eval()

        self.ig = IntegratedGradients(self.model)
        self.occlusion = Occlusion(self.model)
        self.grad_cam = None

    def set_gradcam_layer(self, target_layer: nn.Module):
        """Sets the target layer for Grad-CAM."""
        self.grad_cam = LayerGradCam(self.model, target_layer)

    def explain_ig(self, x: torch.Tensor, target_class: int, baseline: torch.Tensor = None, n_steps: int = 50) -> torch.Tensor:
        """
        Computes Integrated Gradients for the input.
        Args:
            x: Input tensor of shape (batch, leads, time)
            target_class: Target class index.
            baseline: Baseline tensor. If None, uses a zero tensor.
            n_steps: Number of approximation steps for IG.
        Returns:
            Attribution mask of the same shape as x.
        """
        x = x.to(self.device)
        if baseline is None:
            baseline = torch.zeros_like(x).to(self.device)
        else:
            baseline = baseline.to(self.device)

        x.requires_grad_()
        attributions, delta = self.ig.attribute(
            x,
            baselines=baseline,
            target=target_class,
            n_steps=n_steps,
            return_convergence_delta=True
        )
        return attributions

    def explain_gradcam(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Computes Grad-CAM for the input. Requires `set_gradcam_layer` to be called first.
        """
        if self.grad_cam is None:
            raise ValueError("Grad-CAM target layer not set. Call `set_gradcam_layer` first.")
            
        x = x.to(self.device)
        x.requires_grad_()
        attributions = self.grad_cam.attribute(x, target=target_class)
        # Interpolate attributions to match input size
        attributions = torch.nn.functional.interpolate(
            attributions, 
            size=x.shape[2], 
            mode='linear', 
            align_corners=False
        )
        return attributions

    def explain_occlusion(self, x: torch.Tensor, target_class: int, sliding_window_shapes: tuple = (1, 50)) -> torch.Tensor:
        """
        Computes Occlusion sensitivity.
        Args:
            sliding_window_shapes: Tuple of (leads, time_window) for the window to mask. 
                                   Example: (1, 50) masks 50 time steps on 1 lead at a time.
        """
        x = x.to(self.device)
        attributions = self.occlusion.attribute(
            x,
            target=target_class,
            sliding_window_shapes=sliding_window_shapes, 
            strides=(1, sliding_window_shapes[-1] // 2) if len(sliding_window_shapes) == 2 else None,
            baselines=0
        )
        return attributions
