import torch
import numpy as np

class TemporalSaliencyExplainer:
    """
    Computes gradient-based saliency maps to identify which regions and leads 
    of the ECG signal the model attributes to specific diagnostic outcomes.
    """
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def explain(self, signal: np.ndarray, class_idx: int) -> np.ndarray:
        """
        Computes absolute gradient saliency map for the given ECG signal and class.
        
        Args:
            signal: Single ECG signal array of shape (num_leads, signal_length).
            class_idx: Index of the target prediction class.
            
        Returns:
            np.ndarray: Saliency map of shape (num_leads, signal_length).
        """
        self.model.eval()
        
        # Add batch dimension and convert to torch tensor with gradients enabled
        signal_tensor = torch.tensor(signal, dtype=torch.float32, device=self.device).unsqueeze(0)
        signal_tensor.requires_grad = True
        
        # Forward pass
        logits = self.model(signal_tensor)
        
        # Select target class logit
        target_logit = logits[0, class_idx]
        
        # Backward pass
        self.model.zero_grad()
        target_logit.backward()
        
        # Extract gradients
        gradients = signal_tensor.grad.detach().cpu().squeeze(0).numpy()
        
        # Saliency is the absolute value of gradients
        return np.abs(gradients)
