import numpy as np
from preprocessing.filters import PreprocessingStep

class ZScoreNormalizer(PreprocessingStep):
    """
    Applies Z-score standardisation (mean=0, std=1) to each lead independently.
    """
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def process(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        normalized = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            lead = signal[i]
            mean = np.mean(lead)
            std = np.std(lead)
            normalized[i] = (lead - mean) / (std if std > 0 else self.epsilon)
        return normalized


class MinMaxNormalizer(PreprocessingStep):
    """
    Scales each lead independently to a specified range [feature_range[0], feature_range[1]].
    """
    def __init__(self, feature_range: tuple = (0.0, 1.0), epsilon: float = 1e-8):
        self.feature_range = feature_range
        self.epsilon = epsilon

    def process(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        normalized = np.zeros_like(signal)
        min_val, max_val = self.feature_range
        for i in range(signal.shape[0]):
            lead = signal[i]
            lead_min = np.min(lead)
            lead_max = np.max(lead)
            
            denom = lead_max - lead_min
            # Normalize to 0-1
            lead_norm = (lead - lead_min) / (denom if denom > 0 else self.epsilon)
            
            # Scale to range
            normalized[i] = lead_norm * (max_val - min_val) + min_val
        return normalized


class RobustNormalizer(PreprocessingStep):
    """
    Scales each lead independently using the median and interquartile range (IQR).
    This normalizer is robust to outliers and extreme artifacts.
    """
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def process(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        normalized = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            lead = signal[i]
            median = np.median(lead)
            q75, q25 = np.percentile(lead, [75, 25])
            iqr = q75 - q25
            normalized[i] = (lead - median) / (iqr + self.epsilon)
        return normalized
