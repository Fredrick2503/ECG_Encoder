import unittest
import torch
from explainability.unified_explainer import EndToEndUnifiedModel, UnifiedExplainer
from temporal_encoder.encoder import ECGResNet1D
from morphology_encoder.encoder import ECGMorphologyEncoder
from biomarkers.models import AttentionMLPAutoencoder
from classification.classifier import MLPClassifier

class TestUnifiedExplainer(unittest.TestCase):
    def test_pipeline_shapes(self):
        # Instantiate dummy models
        temp_model = ECGResNet1D(num_classes=5)
        morph_model = ECGMorphologyEncoder(input_channels=12, num_classes=5)
        bio_model = AttentionMLPAutoencoder(input_dim=48, latent_dim=32)
        classifier_model = MLPClassifier(input_dim=1056, hidden_dim=256, num_classes=5)
        
        # Test wrapper forward
        wrapper = EndToEndUnifiedModel(temp_model, morph_model, bio_model, classifier_model)
        
        dummy_sig = torch.randn(2, 12, 1000)
        dummy_bio = torch.randn(2, 48)
        
        logits = wrapper(dummy_sig, dummy_bio)
        self.assertEqual(logits.shape, (2, 5))
        
        # Test explainer instance
        explainer = UnifiedExplainer(temp_model, morph_model, bio_model, classifier_model, device="cpu")
        sig_attr, bio_attr = explainer.explain_instance(dummy_sig[0:1], dummy_bio[0:1], target_class=0, n_steps=2)
        
        self.assertEqual(sig_attr.shape, (1, 12, 1000))
        self.assertEqual(bio_attr.shape, (1, 48))
        
        # Test modality contributions
        contributions = explainer.explain_modality_contributions(dummy_sig[0:1], dummy_bio[0:1], target_class=0)
        self.assertIn("temporal", contributions)
        self.assertIn("morphology", contributions)
        self.assertIn("biomarker", contributions)
        self.assertAlmostEqual(sum(contributions.values()), 1.0, places=4)

if __name__ == "__main__":
    unittest.main()
