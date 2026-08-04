import unittest
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from temporal_encoder.encoder import ECGBiLSTM, ECGReconstructionDecoder
from temporal_encoder.strategies import (
    ReconstructionLearningStrategy,
    MaskedAutoencoderStrategy,
    ContrastiveLearningStrategy
)
from temporal_encoder.trainer import TemporalTrainer
from temporal_encoder.predictor import TemporalPredictor
from temporal_encoder.evaluator import TemporalEvaluator
from temporal_encoder.explainer import TemporalSaliencyExplainer

class TestTemporalEncoder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.num_leads = 12
        self.length = 100
        self.hidden_size = 64  # small size for tests
        self.num_classes = 5
        
        # Create toy data
        self.signals = np.random.randn(self.batch_size, self.num_leads, self.length).astype(np.float32)
        self.labels = np.random.randint(0, 2, (self.batch_size, self.num_classes)).astype(np.float32)
        
        self.dataset = TensorDataset(torch.tensor(self.signals), torch.tensor(self.labels))
        self.dataloader = DataLoader(self.dataset, batch_size=2)
        
        # Initialize model
        self.model = ECGBiLSTM(
            input_size=self.num_leads,
            hidden_size=self.hidden_size,
            num_layers=1,
            num_classes=self.num_classes
        )
        self.decoder = ECGReconstructionDecoder(
            latent_dim=self.hidden_size * 2,
            num_leads=self.num_leads,
            signal_length=self.length
        )

    def test_encoder_shapes(self):
        x = torch.tensor(self.signals)
        logits = self.model(x)
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        
        z = self.model.get_representation(x)
        self.assertEqual(z.shape, (self.batch_size, self.hidden_size * 2))

    def test_reconstruction_decoder(self):
        z = torch.randn(self.batch_size, self.hidden_size * 2)
        reconstruction = self.decoder(z)
        self.assertEqual(reconstruction.shape, (self.batch_size, self.num_leads, self.length))

    def test_reconstruction_strategy(self):
        x = torch.tensor(self.signals)
        strategy = ReconstructionLearningStrategy()
        loss = strategy.compute_loss(self.model, self.decoder, x)
        self.assertGreater(loss.item(), 0.0)

    def test_mae_strategy(self):
        x = torch.tensor(self.signals)
        strategy = MaskedAutoencoderStrategy(mask_ratio=0.4)
        loss = strategy.compute_loss(self.model, self.decoder, x)
        self.assertGreater(loss.item(), 0.0)

    def test_contrastive_strategy(self):
        x = torch.tensor(self.signals)
        strategy = ContrastiveLearningStrategy(temperature=0.1, projection_dim=16, latent_dim=self.hidden_size * 2)
        loss = strategy.compute_loss(self.model, None, x)
        self.assertGreater(loss.item(), 0.0)

    def test_trainer_fit_pretraining(self):
        trainer = TemporalTrainer(self.model, lr=1e-3)
        strategy = ReconstructionLearningStrategy()
        history = trainer.fit(
            train_loader=self.dataloader,
            epochs=2,
            is_pretraining=True,
            strategy=strategy,
            decoder=self.decoder
        )
        self.assertEqual(len(history["train_loss"]), 2)
        self.assertLess(history["train_loss"][-1], history["train_loss"][0] + 1.0)

    def test_trainer_fit_supervised(self):
        trainer = TemporalTrainer(self.model, lr=1e-3)
        history = trainer.fit(
            train_loader=self.dataloader,
            val_loader=self.dataloader,
            epochs=2,
            is_pretraining=False
        )
        self.assertEqual(len(history["train_loss"]), 2)
        self.assertEqual(len(history["val_loss"]), 2)

    def test_predictor_and_evaluator(self):
        predictor = TemporalPredictor(self.model)
        probs = predictor.predict_proba(self.dataloader)
        self.assertEqual(probs.shape, (self.batch_size, self.num_classes))
        
        preds = predictor.predict(self.dataloader, threshold=0.5)
        self.assertEqual(preds.shape, (self.batch_size, self.num_classes))
        
        embeddings = predictor.get_embeddings(self.dataloader)
        self.assertEqual(embeddings.shape, (self.batch_size, self.hidden_size * 2))
        
        metrics = TemporalEvaluator.evaluate(self.labels, probs)
        self.assertIn("subset_accuracy", metrics)
        self.assertIn("hamming_loss", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("macro_auc", metrics)

    def test_explainer(self):
        explainer = TemporalSaliencyExplainer(self.model)
        single_signal = self.signals[0]
        saliency = explainer.explain(single_signal, class_idx=2)
        self.assertEqual(saliency.shape, (self.num_leads, self.length))
        self.assertTrue((saliency >= 0.0).all())

if __name__ == "__main__":
    unittest.main()
