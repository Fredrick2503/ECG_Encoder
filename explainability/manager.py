import torch
import torch.nn as nn
from explainability.temporal_xai import TemporalExplainer
from explainability.morphology_xai import MorphologyExplainer

class XAIManager:
    """
    High-level API for XAI across different ECG Encoder types.
    Dispatches to the correct explainer (Temporal or Morphology).
    """
    def __init__(self, model: nn.Module, encoder_type: str, device: str = None, **kwargs):
        """
        Args:
            model: The trained encoder/classifier model.
            encoder_type: 'temporal' or 'morphology'.
            device: Device to run the model on.
            **kwargs: Additional arguments for specific explainers (e.g. conversion_type for morphology).
        """
        self.encoder_type = encoder_type.lower()
        
        if self.encoder_type == 'temporal':
            self.explainer = TemporalExplainer(model, device=device)
        elif self.encoder_type == 'morphology':
            self.explainer = MorphologyExplainer(model, device=device, **kwargs)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Use 'temporal' or 'morphology'.")

    def set_gradcam_layer(self, target_layer: nn.Module):
        self.explainer.set_gradcam_layer(target_layer)

    def explain(self, x: torch.Tensor, target_class: int, method: str = 'ig', **kwargs) -> torch.Tensor:
        """
        Generates explanation for the input.
        
        Args:
            x: Input tensor (batch, leads, time)
            target_class: Target class for explanation.
            method: 'ig' (Integrated Gradients), 'gradcam', or 'occlusion'.
            **kwargs: Additional parameters for the specific method.
            
        Returns:
            Attribution mask tensor.
        """
        method = method.lower()
        if self.encoder_type == 'temporal':
            if method == 'ig':
                return self.explainer.explain_ig(x, target_class, **kwargs)
            elif method == 'gradcam':
                return self.explainer.explain_gradcam(x, target_class)
            elif method == 'occlusion':
                return self.explainer.explain_occlusion(x, target_class, **kwargs)
            else:
                raise ValueError(f"Method {method} not supported for Temporal models.")
                
        elif self.encoder_type == 'morphology':
            if method == 'ig':
                return self.explainer.explain_ig_1d(x, target_class, **kwargs)
            elif method == 'gradcam':
                # GradCAM expects the 2D input (x here should be 2D, or caller handles conversion)
                # For a seamless 1D-to-2D GradCAM experience, the caller should pass 2D or we do it here.
                # Assuming caller passes the already converted 2D tensor for gradcam_2d.
                lead_specific = kwargs.get('lead_specific', False)
                if lead_specific:
                    return self.explainer.explain_lead_specific_gradcam_2d(x, target_class)
                return self.explainer.explain_gradcam_2d(x, target_class)
            elif method == 'occlusion':
                return self.explainer.explain_occlusion_1d(x, target_class, **kwargs)
            else:
                raise ValueError(f"Method {method} not supported for Morphology models.")
                
        return None
