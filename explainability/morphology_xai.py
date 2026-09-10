import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, LayerGradCam, Occlusion
from morphology_encoder.conversion import ecg_to_gaf, ecg_to_spectrogram

class EndToEndMorphologyWrapper(nn.Module):
    """
    Wraps the 1D-to-2D conversion and the Morphology Encoder/Classifier.
    This enables Integrated Gradients to compute attributions directly on the 1D waveform.
    """
    def __init__(self, encoder: nn.Module, conversion_type: str = 'gasf', target_length: int = 64, n_fft: int = 64, hop_length: int = 32):
        super().__init__()
        self.encoder = encoder
        self.conversion_type = conversion_type.lower()
        self.target_length = target_length
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: 1D ECG waveform (batch, leads, signal_length)
        """
        if self.conversion_type in ['gasf', 'gadf']:
            # 2D Conversion
            x_2d = ecg_to_gaf(x, type=self.conversion_type, target_length=self.target_length)
        elif self.conversion_type == 'spectrogram':
            x_2d = ecg_to_spectrogram(x, n_fft=self.n_fft, hop_length=self.hop_length)
        else:
            raise ValueError(f"Unknown conversion type: {self.conversion_type}")
            
        return self.encoder(x_2d)

class MorphologyExplainer:
    """
    Explainer for Morphology Encoders.
    Supports Integrated Gradients on the 1D waveform, and 2D Grad-CAM on the feature maps.
    """
    def __init__(self, model: nn.Module, conversion_type: str = 'gasf', device: str = None, **conversion_kwargs):
        self.device = device if device else next(model.parameters()).device
        self.model = model.to(self.device)
        self.model.eval()
        
        # End-to-end wrapper for 1D attributions
        self.wrapper = EndToEndMorphologyWrapper(self.model, conversion_type=conversion_type, **conversion_kwargs).to(self.device)
        self.wrapper.eval()
        
        self.ig = IntegratedGradients(self.wrapper)
        self.occlusion = Occlusion(self.wrapper)
        self.grad_cam = None

    def set_gradcam_layer(self, target_layer: nn.Module):
        """Sets the target layer for 2D Grad-CAM (using the original model, not wrapper)."""
        self.grad_cam = LayerGradCam(self.model, target_layer)

    def explain_ig_1d(self, x_1d: torch.Tensor, target_class: int, baseline: torch.Tensor = None, n_steps: int = 50) -> torch.Tensor:
        """
        Computes Integrated Gradients on the 1D input waveform.
        """
        x_1d = x_1d.to(self.device)
        if baseline is None:
            baseline = torch.zeros_like(x_1d).to(self.device)
        else:
            baseline = baseline.to(self.device)

        x_1d.requires_grad_()
        attributions, delta = self.ig.attribute(
            x_1d,
            baselines=baseline,
            target=target_class,
            n_steps=n_steps,
            return_convergence_delta=True
        )
        return attributions
        
    def explain_occlusion_1d(self, x_1d: torch.Tensor, target_class: int, sliding_window_shapes: tuple = (1, 50)) -> torch.Tensor:
        """
        Computes Occlusion sensitivity on the 1D input waveform.
        """
        x_1d = x_1d.to(self.device)
        attributions = self.occlusion.attribute(
            x_1d,
            target=target_class,
            sliding_window_shapes=sliding_window_shapes,
            strides=(1, sliding_window_shapes[-1] // 2) if len(sliding_window_shapes) == 2 else None,
            baselines=0
        )
        return attributions

    def explain_gradcam_2d(self, x_2d: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Computes Grad-CAM for the 2D input (GAF/Spectrogram).
        Requires `set_gradcam_layer` to be called first.
        """
        if self.grad_cam is None:
            raise ValueError("Grad-CAM target layer not set. Call `set_gradcam_layer` first.")
            
        x_2d = x_2d.to(self.device)
        x_2d.requires_grad_()
        attributions = self.grad_cam.attribute(x_2d, target=target_class)
        # Interpolate attributions to match input size (H, W)
        attributions = torch.nn.functional.interpolate(
            attributions, 
            size=x_2d.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        return attributions

    def explain_lead_specific_gradcam_2d(self, x_2d: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Computes lead-specific Guided Grad-CAM for the 2D input (GAF/Spectrogram).
        Combines the spatial/spectral attribution of Grad-CAM with lead-specific gradients
        to preserve differences in attribution between leads.
        """
        if self.grad_cam is None:
            raise ValueError("Grad-CAM target layer not set. Call `set_gradcam_layer` first.")
            
        x_2d = x_2d.to(self.device).clone().detach()
        x_2d.requires_grad = True
        
        # Forward pass
        output = self.model(x_2d)
        score = output[0, target_class]
        
        # Backward pass w.r.t input to capture lead-specific fine-grained details
        self.model.zero_grad()
        score.backward(retain_graph=True)
        input_grad = x_2d.grad.data.clone()
        
        # Compute standard Grad-CAM
        gradcam_attr = self.grad_cam.attribute(x_2d, target=target_class)
        gradcam_attr = torch.nn.functional.interpolate(
            gradcam_attr, 
            size=x_2d.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Guided Grad-CAM: Element-wise multiply Grad-CAM activation mask with positive/absolute input gradients
        # This resolves lead-specific contributions and preserves inter-lead variance
        lead_attr = gradcam_attr * torch.abs(input_grad)
        return lead_attr

