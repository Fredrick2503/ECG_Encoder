import unittest
import numpy as np
from data_management.ecg_record import ECGRecord
from preprocessing.validation import SignalValidator
from preprocessing.filters import ButterworthFilter, NotchFilter, FIRFilter, WaveletDenoise
from preprocessing.normalization import ZScoreNormalizer, MinMaxNormalizer, RobustNormalizer
from preprocessing.segmentation import FixedWindowSegmenter, SlidingWindowSegmenter, PanTompkinsSegmenter
from preprocessing.outlier_detection import DBSCANOutlierDetector
from preprocessing.balancing import ECGDatasetBalancer
from preprocessing.manager import PreprocessingManager

class TestSignalValidator(unittest.TestCase):
    def setUp(self):
        self.validator = SignalValidator(min_length=100, min_amplitude=0.01)

    def test_valid_signal(self):
        # Sine wave with some random noise
        t = np.linspace(0, 1, 200)
        lead = np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.1, 200)
        signal = np.vstack([lead, lead * 0.5])
        is_valid, errors = self.validator.validate(signal)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_short_signal(self):
        signal = np.zeros((2, 50))
        is_valid, errors = self.validator.validate(signal)
        self.assertFalse(is_valid)
        self.assertTrue(any("length" in err for err in errors))

    def test_flatline(self):
        signal = np.zeros((2, 200))
        is_valid, errors = self.validator.validate(signal)
        self.assertFalse(is_valid)
        self.assertTrue(any("flatline" in err or "insufficient amplitude" in err for err in errors))

    def test_nans(self):
        signal = np.ones((2, 200))
        signal[0, 50] = np.nan
        # Exceed max nan ratio (0.1 by default)
        signal[1, :50] = np.nan
        is_valid, errors = self.validator.validate(signal)
        self.assertFalse(is_valid)
        self.assertTrue(any("NaN/Inf ratio" in err for err in errors))

    def test_clean_nans(self):
        signal = np.ones((1, 200))
        signal[0, 10] = np.nan
        cleaned = self.validator.clean_signal(signal)
        self.assertFalse(np.isnan(cleaned).any())
        self.assertEqual(cleaned[0, 10], 1.0)


class TestFilters(unittest.TestCase):
    def setUp(self):
        self.fs = 500
        t = np.linspace(0, 2, self.fs * 2)
        # Sine wave combining 1 Hz (baseline wander), 10 Hz (ECG-like), and 60 Hz (powerline)
        self.lead = np.sin(2 * np.pi * 1 * t) + np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)
        self.signal = np.vstack([self.lead, self.lead])

    def test_butterworth_bandpass(self):
        filt = ButterworthFilter(lowcut=2.0, highcut=45.0)
        out = filt.process(self.signal, self.fs)
        self.assertEqual(out.shape, self.signal.shape)
        # Power at 1 Hz and 60 Hz should be heavily attenuated
        self.assertTrue(np.std(out) < np.std(self.signal))

    def test_notch(self):
        filt = NotchFilter(notch_freq=60.0)
        out = filt.process(self.signal, self.fs)
        self.assertEqual(out.shape, self.signal.shape)

    def test_fir(self):
        filt = FIRFilter(lowcut=2.0, highcut=45.0)
        out = filt.process(self.signal, self.fs)
        self.assertEqual(out.shape, self.signal.shape)

    def test_wavelet_denoise(self):
        filt = WaveletDenoise(wavelet="db4", level=3)
        out = filt.process(self.signal, self.fs)
        self.assertEqual(out.shape, self.signal.shape)


class TestNormalization(unittest.TestCase):
    def setUp(self):
        self.signal = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]])

    def test_z_score(self):
        norm = ZScoreNormalizer()
        out = norm.process(self.signal, 500)
        np.testing.assert_almost_equal(np.mean(out), 0.0, decimal=5)
        np.testing.assert_almost_equal(np.std(out), 1.0, decimal=5)

    def test_min_max(self):
        norm = MinMaxNormalizer(feature_range=(0.0, 1.0))
        out = norm.process(self.signal, 500)
        self.assertEqual(np.min(out), 0.0)
        self.assertEqual(np.max(out), 1.0)

    def test_robust(self):
        norm = RobustNormalizer()
        out = norm.process(self.signal, 500)
        self.assertEqual(np.median(out), 0.0)


class TestSegmentation(unittest.TestCase):
    def setUp(self):
        self.signal = np.random.normal(0, 1, (2, 2500))

    def test_fixed_window(self):
        seg = FixedWindowSegmenter(window_size=1000)
        out = seg.process(self.signal, 500)
        # Expected: 2 segments of length 1000
        self.assertEqual(out.shape, (2, 2, 1000))

    def test_sliding_window(self):
        seg = SlidingWindowSegmenter(window_size=1000, overlap=500)
        out = seg.process(self.signal, 500)
        # Expected: start at 0, 500, 1000, 1500 -> 4 segments
        self.assertEqual(out.shape, (4, 2, 1000))

    def test_pan_tompkins(self):
        # Create a synthetic ECG record with simulated R-peaks
        fs = 500
        length = fs * 5  # 5 seconds
        t = np.linspace(0, 5, length)
        lead = np.zeros(length)
        
        # Add high amplitude pulses representing QRS complexes at 1s, 2s, 3s, 4s
        for peak_sec in [1.0, 2.0, 3.0, 4.0]:
            idx = int(peak_sec * fs)
            lead[idx-10:idx+10] = np.sin(np.linspace(-np.pi, np.pi, 20)) * 2.0
            
        signal = np.vstack([lead, lead])
        
        seg = PanTompkinsSegmenter(pre_r_samples=100, post_r_samples=150)
        out = seg.process(signal, fs)
        # Expected: 4 heartbeats segmented
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(out.shape[1], 2)
        self.assertEqual(out.shape[2], 250)  # pre_r + post_r = 100 + 150


class TestOutlierAndBalancing(unittest.TestCase):
    def test_dbscan_outliers(self):
        # 10 similar records, 1 extreme outlier record
        records = []
        for i in range(10):
            records.append(np.random.normal(0, 0.1, (2, 500)))
        records.append(np.random.normal(10.0, 5.0, (2, 500)))  # Extreme outlier
        
        signals = np.array(records)
        detector = DBSCANOutlierDetector(eps=2.0, min_samples=3)
        inliers, outliers = detector.detect_outliers(signals)
        
        self.assertIn(10, outliers)
        self.assertNotIn(10, inliers)

    def test_dataset_balancer(self):
        # Balanced resampler test
        # Generate 15 samples of Class 0, 15 samples of Class 1 (to satisfy SMOTE class minimums)
        signals = np.random.normal(0, 1, (30, 2, 100))
        labels = np.array([0] * 15 + [1] * 15)
        
        balancer = ECGDatasetBalancer(random_state=42)
        signals_res, labels_res = balancer.balance_dataset(signals, labels)
        
        self.assertEqual(signals_res.shape[1:], (2, 100))
        self.assertEqual(len(signals_res), len(labels_res))


class TestPreprocessingManager(unittest.TestCase):
    def test_profiles(self):
        # Signal of length 1500
        signal = np.random.normal(0, 0.5, (12, 1500))
        # Ensure lead 1 has standard R-peak shape for Pan-Tompkins
        signal[1, 500] = 5.0
        signal[1, 1000] = 5.0
        
        record = ECGRecord(
            record_id="101",
            signal=signal,
            sampling_rate=500,
            leads=[f"L{i}" for i in range(12)]
        )
        
        manager = PreprocessingManager()
        
        # Test Temporal Profile
        rec_temp = manager.preprocess_record(record, "temporal")
        # Segmentation: 1500 // 1000 = 1 segment
        self.assertEqual(rec_temp.signal.shape, (1, 12, 1000))
        
        # Test Morphology Profile
        rec_morph = manager.preprocess_record(record, "morphology")
        self.assertEqual(rec_morph.signal.ndim, 3)
        self.assertEqual(rec_morph.signal.shape[1], 12)
        self.assertEqual(rec_morph.signal.shape[2], 400) # pre_r + post_r = 150 + 250
        
        # Test Biomarker Profile
        rec_bio = manager.preprocess_record(record, "biomarker")
        # Biomarker doesn't segment by default, so stays 2D
        self.assertEqual(rec_bio.signal.shape, (12, 1500))


if __name__ == "__main__":
    unittest.main()
