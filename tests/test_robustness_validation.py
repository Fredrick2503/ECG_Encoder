import unittest
import numpy as np
from classification.run_robustness_validation import (
    apply_baseline_wander,
    apply_powerline_interference,
    apply_high_frequency_noise,
    apply_single_lead_mask,
    apply_multiple_leads_mask
)

class TestRobustnessNoiseModels(unittest.TestCase):
    def setUp(self):
        # Mock 12-lead ECG signal of length 1000
        self.signal = np.random.randn(12, 1000).astype(np.float32)
        
    def test_noise_shapes(self):
        # Baseline wander shape check
        x_wander = apply_baseline_wander(self.signal)
        self.assertEqual(x_wander.shape, (12, 1000))
        
        # Powerline interference shape check
        x_powerline = apply_powerline_interference(self.signal)
        self.assertEqual(x_powerline.shape, (12, 1000))
        
        # High frequency noise shape check
        x_hf = apply_high_frequency_noise(self.signal)
        self.assertEqual(x_hf.shape, (12, 1000))
        
        # Lead masking shape check
        x_single_masked = apply_single_lead_mask(self.signal, lead_idx=1)
        self.assertEqual(x_single_masked.shape, (12, 1000))
        self.assertTrue(np.all(x_single_masked[1, :] == 0.0))
        
        # Chest leads masking check
        x_chest_masked = apply_multiple_leads_mask(self.signal, lead_indices=range(6, 12))
        self.assertEqual(x_chest_masked.shape, (12, 1000))
        for idx in range(6, 12):
            self.assertTrue(np.all(x_chest_masked[idx, :] == 0.0))

if __name__ == "__main__":
    unittest.main()
