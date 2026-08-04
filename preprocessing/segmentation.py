import numpy as np
import scipy.signal
from typing import List, Tuple
from preprocessing.filters import PreprocessingStep

class FixedWindowSegmenter(PreprocessingStep):
    """
    Segments the ECG signal into contiguous, non-overlapping windows of a fixed length.
    If the signal is shorter than the window size, it will be zero-padded.
    """
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size

    def process(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        num_leads, length = signal.shape
        
        if length < self.window_size:
            # Pad with zeros
            padded = np.zeros((num_leads, self.window_size), dtype=signal.dtype)
            padded[:, :length] = signal
            return padded[np.newaxis, :, :]  # Shape: (1, num_leads, window_size)
            
        num_segments = length // self.window_size
        segments = []
        for i in range(num_segments):
            start = i * self.window_size
            end = start + self.window_size
            segments.append(signal[:, start:end])
            
        return np.array(segments)  # Shape: (num_segments, num_leads, window_size)


class SlidingWindowSegmenter(PreprocessingStep):
    """
    Segments the ECG signal into overlapping windows using a sliding window.
    """
    def __init__(self, window_size: int = 1000, overlap: int = 500):
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        if self.stride <= 0:
            raise ValueError("Overlap must be strictly less than window_size.")

    def process(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        num_leads, length = signal.shape
        
        if length < self.window_size:
            padded = np.zeros((num_leads, self.window_size), dtype=signal.dtype)
            padded[:, :length] = signal
            return padded[np.newaxis, :, :]
            
        segments = []
        start = 0
        while start + self.window_size <= length:
            end = start + self.window_size
            segments.append(signal[:, start:end])
            start += self.stride
            
        return np.array(segments)  # Shape: (num_segments, num_leads, window_size)


class PanTompkinsSegmenter(PreprocessingStep):
    """
    QRS complex detection based on the Pan-Tompkins algorithm to segment individual heartbeats.
    Each segmented beat is centered around the detected R-peak.
    """
    def __init__(self, pre_r_samples: int = 150, post_r_samples: int = 250, target_lead_idx: int = 1):
        """
        Args:
            pre_r_samples: Number of samples to include before the R-peak (e.g. PR interval).
            post_r_samples: Number of samples to include after the R-peak (e.g. QT interval).
            target_lead_idx: Lead index to run peak detection on (typically Lead II, index 1).
        """
        self.pre_r_samples = pre_r_samples
        self.post_r_samples = post_r_samples
        self.target_lead_idx = target_lead_idx

    def _detect_r_peaks(self, lead_signal: np.ndarray, fs: int) -> np.ndarray:
        # 1. Bandpass filter (5 - 15 Hz)
        nyquist = 0.5 * fs
        low = 5.0 / nyquist
        high = 15.0 / nyquist
        b, a = scipy.signal.butter(3, [low, high], btype='bandpass')
        filtered = scipy.signal.filtfilt(b, a, lead_signal)
        
        # 2. Derivative filter: y(n) = 1/8 [2x(n) + x(n-1) - x(n-3) - 2x(n-4)]
        # We can implement this using a standard convolution
        h_der = np.array([2.0, 1.0, 0.0, -1.0, -2.0]) / 8.0
        derivative = scipy.signal.filtfilt(h_der, [1.0], filtered)
        
        # 3. Squaring
        squared = derivative ** 2
        
        # 4. Moving window integration (150ms window)
        window_len = int(0.15 * fs)
        moving_avg = np.convolve(squared, np.ones(window_len) / window_len, mode='same')
        
        # 5. Peak detection using scipy find_peaks
        # Minimum peak distance of 300ms (to prevent T-peak triggering)
        min_dist = int(0.3 * fs)
        peaks, _ = scipy.signal.find_peaks(
            moving_avg, 
            distance=min_dist,
            prominence=np.max(moving_avg) * 0.15
        )
        
        # Refine peak locations on the original raw signal within a small search window
        refined_peaks = []
        search_win = int(0.05 * fs)  # 50ms search window
        for peak in peaks:
            start = max(0, peak - search_win)
            end = min(len(lead_signal), peak + search_win)
            refined_peak = start + np.argmax(np.abs(lead_signal[start:end]))
            refined_peaks.append(refined_peak)
            
        return np.array(refined_peaks)

    def process(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        num_leads, length = signal.shape
        
        # Fallback if target lead is out of index bounds
        lead_idx = min(self.target_lead_idx, num_leads - 1)
        lead_signal = signal[lead_idx]
        
        r_peaks = self._detect_r_peaks(lead_signal, sampling_rate)
        
        beat_len = self.pre_r_samples + self.post_r_samples
        beats = []
        
        for peak in r_peaks:
            start = peak - self.pre_r_samples
            end = peak + self.post_r_samples
            
            # Boundary checks: zero-pad if beat extends beyond signal boundaries
            if start < 0 or end > length:
                beat = np.zeros((num_leads, beat_len), dtype=signal.dtype)
                sig_start = max(0, start)
                sig_end = min(length, end)
                
                beat_start = max(0, -start)
                beat_end = beat_start + (sig_end - sig_start)
                
                beat[:, beat_start:beat_end] = signal[:, sig_start:sig_end]
                beats.append(beat)
            else:
                beats.append(signal[:, start:end])
                
        if len(beats) == 0:
            # Fallback if no peaks detected: return single zero-padded window from center of signal
            fallback_start = max(0, (length - beat_len) // 2)
            fallback_end = min(length, fallback_start + beat_len)
            fallback_beat = np.zeros((num_leads, beat_len), dtype=signal.dtype)
            fallback_beat[:, :fallback_end - fallback_start] = signal[:, fallback_start:fallback_end]
            return fallback_beat[np.newaxis, :, :]
            
        return np.array(beats)  # Shape: (num_beats, num_leads, beat_len)
