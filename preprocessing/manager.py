from typing import Dict, Any, Optional
from data_management.ecg_record import ECGRecord
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.validation import SignalValidator
from preprocessing.filters import ButterworthFilter, NotchFilter, WaveletDenoise, FIRFilter
from preprocessing.normalization import ZScoreNormalizer, MinMaxNormalizer, RobustNormalizer
from preprocessing.segmentation import FixedWindowSegmenter, SlidingWindowSegmenter, PanTompkinsSegmenter

class PreprocessingManager:
    """
    Facade class that manages preprocessing profiles (Temporal, Morphology, Biomarker)
    and executes the preprocessing pipeline on ECGRecord objects.
    """
    def __init__(self, validation_params: Optional[Dict[str, Any]] = None):
        """
        Args:
            validation_params: Parameters to configure the default SignalValidator.
        """
        self.validation_params = validation_params or {}
        self.validator = SignalValidator(**self.validation_params)

    def get_pipeline(self, profile_name: str, custom_config: Optional[Dict[str, Any]] = None) -> PreprocessingPipeline:
        """
        Factory method to construct a PreprocessingPipeline for a given profile.
        
        Args:
            profile_name: Name of the profile ('temporal', 'morphology', 'biomarker').
            custom_config: Optional overrides for step parameters.
            
        Returns:
            PreprocessingPipeline: The constructed pipeline.
        """
        cfg = custom_config or {}
        steps = []
        name = profile_name.lower()
        
        if name == "temporal":
            # 1. Butterworth Bandpass (0.5 - 45 Hz)
            lowcut = cfg.get("lowcut", 0.5)
            highcut = cfg.get("highcut", 45.0)
            order = cfg.get("order", 4)
            steps.append(ButterworthFilter(lowcut=lowcut, highcut=highcut, order=order))
            
            # 2. Notch filter (60 Hz)
            notch_freq = cfg.get("notch_freq", 60.0)
            steps.append(NotchFilter(notch_freq=notch_freq))
            
            # 3. Fixed Window Segmentation (e.g. 1000 samples)
            window_size = cfg.get("window_size", 1000)
            steps.append(FixedWindowSegmenter(window_size=window_size))
            
            # 4. Z-score normalization
            steps.append(ZScoreNormalizer())
            
        elif name == "morphology":
            # 1. Wavelet Denoising (db4, 4 levels)
            wavelet = cfg.get("wavelet", "db4")
            level = cfg.get("level", 4)
            steps.append(WaveletDenoise(wavelet=wavelet, level=level))
            
            # 2. Pan-Tompkins beat-based segmentation
            pre_r = cfg.get("pre_r_samples", 150)
            post_r = cfg.get("post_r_samples", 250)
            target_lead = cfg.get("target_lead_idx", 1)
            steps.append(PanTompkinsSegmenter(pre_r_samples=pre_r, post_r_samples=post_r, target_lead_idx=target_lead))
            
            # 3. Min-Max normalization (0 to 1)
            feature_range = cfg.get("feature_range", (0.0, 1.0))
            steps.append(MinMaxNormalizer(feature_range=feature_range))
            
        elif name == "biomarker":
            # 1. Highpass Butterworth (0.5 Hz) to remove baseline wander
            lowcut = cfg.get("lowcut", 0.5)
            order = cfg.get("order", 4)
            steps.append(ButterworthFilter(lowcut=lowcut, highcut=None, order=order))
            
            # 2. Notch filter (60 Hz)
            notch_freq = cfg.get("notch_freq", 60.0)
            steps.append(NotchFilter(notch_freq=notch_freq))
            
            # 3. Robust scaling
            steps.append(RobustNormalizer())
            
            # Optional sliding window
            if "window_size" in cfg:
                window_size = cfg["window_size"]
                overlap = cfg.get("overlap", window_size // 2)
                steps.append(SlidingWindowSegmenter(window_size=window_size, overlap=overlap))
                
        else:
            raise ValueError(f"Unknown preprocessing profile name: '{profile_name}'. Available: 'temporal', 'morphology', 'biomarker'.")
            
        return PreprocessingPipeline(steps=steps, validator=self.validator)

    def preprocess_record(
        self,
        record: ECGRecord,
        profile_name: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> ECGRecord:
        """
        Preprocesses a single ECGRecord using the specified profile.
        
        Args:
            record: The raw input ECGRecord.
            profile_name: Preprocessing profile name ('temporal', 'morphology', 'biomarker').
            custom_config: Optional parameters overrides.
            
        Returns:
            ECGRecord: A new ECGRecord containing the preprocessed signal.
        """
        pipeline = self.get_pipeline(profile_name, custom_config)
        preprocessed_signal = pipeline.process(record.signal, record.sampling_rate)
        
        return ECGRecord(
            record_id=record.record_id,
            signal=preprocessed_signal,
            sampling_rate=record.sampling_rate,
            leads=record.leads,
            labels=record.labels,
            metadata=record.metadata,
            patient_id=record.patient_id,
            age=record.age,
            sex=record.sex
        )
