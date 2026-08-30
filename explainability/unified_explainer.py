import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from captum.attr import IntegratedGradients

from morphology_encoder.conversion import ecg_to_spectrogram

class EndToEndUnifiedModel(nn.Module):
    """
    End-to-end PyTorch wrapper connecting raw 1D waveforms and raw biomarkers
    to the final unified classifier predictions.
    """
    def __init__(self, temp_model, morph_model, bio_model, classifier_model):
        super().__init__()
        self.temp_model = temp_model
        self.morph_model = morph_model
        self.bio_model = bio_model
        self.classifier_model = classifier_model
        
        # Keep sub-models in eval mode
        self.temp_model.eval()
        self.morph_model.eval()
        self.bio_model.eval()
        self.classifier_model.eval()

    def forward(self, signal_1d, biomarkers):
        # signal_1d: (batch, 12, 1000)
        # biomarkers: (batch, 50)
        
        # 1. Extract Temporal representations (512-D)
        zt = self.temp_model.get_representation(signal_1d)
        
        # 2. Convert and extract Morphology representations (512-D)
        spec = ecg_to_spectrogram(signal_1d)
        zm = self.morph_model.get_representation(spec)
        
        # 3. Extract Biomarker representations (32-D)
        zb = self.bio_model.encode(biomarkers)
        
        # 4. Concatenate to joint space Z_fused (1056-D)
        z_fused = torch.cat([zt, zm, zb], dim=1)
        
        # 5. Classifier logits
        logits = self.classifier_model(z_fused)
        return logits

class UnifiedExplainer:
    """
    Unified Explainable AI pipeline computing joint attributions across 
    waveform time-series, tabular biomarkers, and fusion representation slices.
    """
    def __init__(self, temp_model, morph_model, bio_model, classifier_model, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_model = temp_model.to(self.device)
        self.morph_model = morph_model.to(self.device)
        self.bio_model = bio_model.to(self.device)
        self.classifier_model = classifier_model.to(self.device)
        
        # Wrapper for end-to-end attributions
        self.wrapper = EndToEndUnifiedModel(
            self.temp_model, self.morph_model, self.bio_model, self.classifier_model
        ).to(self.device)
        self.wrapper.eval()
        
        self.ig_joint = IntegratedGradients(self.wrapper)
        self.ig_classifier = IntegratedGradients(self.classifier_model)

    def explain_instance(self, signal_1d: torch.Tensor, biomarkers: torch.Tensor, target_class: int, n_steps: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes joint Integrated Gradients w.r.t 1D waveforms and tabular biomarkers.
        Returns:
            Tuple[np.ndarray, np.ndarray]: (signal_attributions, biomarker_attributions)
        """
        sig_tensor = signal_1d.to(self.device).clone().detach()
        bio_tensor = biomarkers.to(self.device).clone().detach()
        
        sig_tensor.requires_grad = True
        bio_tensor.requires_grad = True
        
        baseline_sig = torch.zeros_like(sig_tensor).to(self.device)
        baseline_bio = torch.zeros_like(bio_tensor).to(self.device)
        
        attributions = self.ig_joint.attribute(
            inputs=(sig_tensor, bio_tensor),
            baselines=(baseline_sig, baseline_bio),
            target=target_class,
            n_steps=n_steps
        )
        
        sig_attr = attributions[0].detach().cpu().numpy()
        bio_attr = attributions[1].detach().cpu().numpy()
        
        return sig_attr, bio_attr

    def explain_modality_contributions(self, signal_1d: torch.Tensor, biomarkers: torch.Tensor, target_class: int) -> Dict[str, float]:
        """
        Computes the relative attribution/importance percentage of each encoder modality branch.
        """
        self.wrapper.eval()
        with torch.no_grad():
            sig_tensor = signal_1d.to(self.device)
            bio_tensor = biomarkers.to(self.device)
            
            zt = self.temp_model.get_representation(sig_tensor)
            spec = ecg_to_spectrogram(sig_tensor)
            zm = self.morph_model.get_representation(spec)
            zb = self.bio_model.encode(bio_tensor)
            
            z_fused = torch.cat([zt, zm, zb], dim=1)
            
        z_fused.requires_grad = True
        baseline_z = torch.zeros_like(z_fused).to(self.device)
        
        # Compute Integrated Gradients at the classifier level w.r.t joint space
        attr_z = self.ig_classifier.attribute(
            z_fused,
            baselines=baseline_z,
            target=target_class
        ).detach().cpu().numpy()[0]
        
        abs_attr = np.abs(attr_z)
        
        temp_sum = np.sum(abs_attr[0:512])
        morph_sum = np.sum(abs_attr[512:1024])
        bio_sum = np.sum(abs_attr[1024:1056])
        
        total = temp_sum + morph_sum + bio_sum + 1e-8
        
        return {
            "temporal": float(temp_sum / total),
            "morphology": float(morph_sum / total),
            "biomarker": float(bio_sum / total)
        }
