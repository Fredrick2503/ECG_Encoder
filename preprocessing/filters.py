from abc import ABC, abstractmethod
from typing import Optional
import warnings
import numpy as np
import scipy.signal
import pywt

class PreprocessingStep(ABC):
    """Abstract base class for a single preprocessing step."""
    @abstractmethod
    def process(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Processes the input signal.
        
        Args:
            signal: numpy array of shape (num_leads, signal_length)
            sampling_rate: frequency in Hz
            
        Returns:
            np.ndarray: processed signal of shape (num_leads, signal_length)
        """
        pass


class ButterworthFilter(PreprocessingStep):
    """
    Butterworth filter (lowpass, highpass, or bandpass) using zero-phase filtering (filtfilt).
    """
    def __init__(
        self,
        lowcut: Optional[float] = 0.5,
        highcut: Optional[float] = 45.0,
        order: int = 4
    ):
        """
        Args:
            lowcut: Low-frequency cutoff in Hz (None for lowpass filter).
            highcut: High-frequency cutoff in Hz (None for highpass filter).
            order: Order of the filter.
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def process(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        nyquist = 0.5 * sampling_rate
        
        if self.lowcut is not None and self.highcut is not None:
            # Bandpass
            low = self.lowcut / nyquist
            high = self.highcut / nyquist
            b, a = scipy.signal.butter(self.order, [low, high], btype='bandpass')
        elif self.lowcut is not None:
            # Highpass
            low = self.lowcut / nyquist
            b, a = scipy.signal.butter(self.order, low, btype='highpass')
        elif self.highcut is not None:
            # Lowpass
            high = self.highcut / nyquist
            b, a = scipy.signal.butter(self.order, high, btype='lowpass')
        else:
            return signal
            
        filtered = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            filtered[i] = scipy.signal.filtfilt(b, a, signal[i])
            
        return filtered


class NotchFilter(PreprocessingStep):
    """
    IIR Notch filter to remove powerline interference (typically 50 Hz or 60 Hz).
    """
    def __init__(self, notch_freq: float = 60.0, Q: float = 30.0):
        """
        Args:
            notch_freq: Frequency to reject in Hz (default 60 Hz).
            Q: Quality factor (defines bandwidth of notch filter).
        """
        self.notch_freq = notch_freq
        self.Q = Q

    def process(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        nyquist = 0.5 * sampling_rate
        w0 = self.notch_freq / nyquist

        if not (0.0 < w0 < 1.0):
            warnings.warn(
                f"Notch frequency {self.notch_freq} Hz is not valid for sampling rate {sampling_rate} Hz. "
                f"Skipping NotchFilter. Valid normalized frequency must satisfy 0 < w0 < 1."
            )
            return signal

        b, a = scipy.signal.iirnotch(w0, self.Q)
        filtered = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            filtered[i] = scipy.signal.filtfilt(b, a, signal[i])
            
        return filtered


class FIRFilter(PreprocessingStep):
    """
    FIR bandpass/lowpass/highpass filter using the window method.
    """
    def __init__(
        self,
        lowcut: Optional[float] = 0.5,
        highcut: Optional[float] = 45.0,
        numtaps: int = 101
    ):
        """
        Args:
            lowcut: Low-frequency cutoff in Hz.
            highcut: High-frequency cutoff in Hz.
            numtaps: Number of filter coefficients (must be odd).
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.numtaps = numtaps if numtaps % 2 == 1 else numtaps + 1

    def process(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        nyquist = 0.5 * sampling_rate
        
        if self.lowcut is not None and self.highcut is not None:
            # Bandpass
            taps = scipy.signal.firwin(self.numtaps, [self.lowcut, self.highcut], pass_zero=False, fs=sampling_rate)
        elif self.lowcut is not None:
            # Highpass
            taps = scipy.signal.firwin(self.numtaps, self.lowcut, pass_zero=False, fs=sampling_rate)
        elif self.highcut is not None:
            # Lowpass
            taps = scipy.signal.firwin(self.numtaps, self.highcut, pass_zero=True, fs=sampling_rate)
        else:
            return signal
            
        filtered = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            # Zero-phase FIR filter using filtfilt
            filtered[i] = scipy.signal.filtfilt(taps, [1.0], signal[i])
            
        return filtered


class WaveletDenoise(PreprocessingStep):
    """
    Denoising using discrete wavelet transform thresholding.
    Supports standard wavelets (e.g., db4, db8) and soft thresholding.
    """
    def __init__(self, wavelet: str = "db4", level: int = 4):
        """
        Args:
            wavelet: Wavelet name (e.g. 'db4', 'db8').
            level: Decomposition level.
        """
        self.wavelet = wavelet
        self.level = level

    def process(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        denoised = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            lead_signal = signal[i]
            
            # Wavelet decomposition
            coeffs = pywt.wavedec(lead_signal, self.wavelet, level=self.level)
            
            # Don't threshold approximation coefficients (coeffs[0]), only detail coefficients
            # Calculate Universal Threshold (Donaho & Johnstone) using MAD of the highest level detail
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(lead_signal)))
            
            # Apply soft thresholding to details
            new_coeffs = [coeffs[0]]
            for detail in coeffs[1:]:
                new_coeffs.append(pywt.threshold(detail, value=threshold, mode='soft'))
                
            # Wavelet reconstruction
            denoised[i] = pywt.waverec(new_coeffs, self.wavelet)[:len(lead_signal)]
            
        return denoised
