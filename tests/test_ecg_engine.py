"""
Unit and Integration Tests for ECGEncoderEngine
===============================================
Verifies OOP contracts, shape consistency, and multi-label prediction capabilities.
"""

import unittest
import numpy as np
import torch
from ecg_engine import (
    ECGEncoderEngine,
    EngineConfig,
    SignalPreprocessor,
    FusionEngine,
    FusedRepresentationResult,
    DiagnosticPredictionResult,
    DEFAULT_CLASS_NAMES,
)


class TestECGEncoderEngine(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.config = EngineConfig(device="cpu")
        cls.engine = ECGEncoderEngine(config=cls.config)

    def test_preprocessor_shapes(self):
        preprocessor = SignalPreprocessor(target_length=1000, num_leads=12)
        
        # Test (12, 1000)
        s1 = np.random.randn(12, 1000)
        out1 = preprocessor.preprocess(s1)
        self.assertEqual(out1.shape, (1, 12, 1000))
        
        # Test (1000, 12)
        s2 = np.random.randn(1000, 12)
        out2 = preprocessor.preprocess(s2)
        self.assertEqual(out2.shape, (1, 12, 1000))
        
        # Test Batch (4, 12, 500) -> Resamples to 1000
        s3 = np.random.randn(4, 12, 500)
        out3 = preprocessor.preprocess(s3)
        self.assertEqual(out3.shape, (4, 12, 1000))

    def test_fusion_engine_dimensions(self):
        fusion = FusionEngine()
        zt = torch.randn(2, 512)
        zm = torch.randn(2, 512)
        zb = torch.randn(2, 32)
        zf = fusion.fuse(zt, zm, zb)
        self.assertEqual(zf.shape, (2, 1056))

    def test_end_to_end_encoding(self):
        batch_signals = np.random.randn(2, 12, 1000)
        rep = self.engine.encode(batch_signals)
        
        self.assertIsInstance(rep, FusedRepresentationResult)
        self.assertEqual(rep.z_temporal.shape, (2, 512))
        self.assertEqual(rep.z_morphology.shape, (2, 512))
        self.assertEqual(rep.z_biomarker.shape, (2, 32))
        self.assertEqual(rep.z_fused.shape, (2, 1056))

    def test_end_to_end_prediction(self):
        batch_signals = np.random.randn(2, 12, 1000)
        pred = self.engine.predict(ecg_signal=batch_signals)
        
        self.assertIsInstance(pred, DiagnosticPredictionResult)
        self.assertEqual(pred.probabilities.shape, (2, 5))
        self.assertEqual(pred.predictions.shape, (2, 5))
        self.assertEqual(len(pred.detected_conditions), 2)
        self.assertEqual(pred.class_names, DEFAULT_CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()
