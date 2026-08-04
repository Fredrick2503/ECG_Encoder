from typing import Tuple, List
import numpy as np

class SignalValidator:
    """
    Validates ECG signals for quality issues like flatlines, NaNs, 
    extreme amplitudes, and insufficient signal duration.
    """
    def __init__(
        self,
        min_length: int = 500,
        max_nan_ratio: float = 0.1,
        min_amplitude: float = 0.01,
        max_amplitude: float = 15.0,
        flatline_threshold: float = 1e-5
    ):
        """
        Args:
            min_length: Minimum number of samples required.
            max_nan_ratio: Maximum fraction of NaN values allowed before rejecting the signal.
            min_amplitude: Minimum standard deviation of amplitude to prevent flatline/low signal.
            max_amplitude: Maximum absolute peak amplitude allowed (to filter out extreme noise).
            flatline_threshold: Standard deviation threshold below which a lead is considered flat.
        """
        self.min_length = min_length
        self.max_nan_ratio = max_nan_ratio
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.flatline_threshold = flatline_threshold

    def validate(self, signal: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validates the given ECG signal.
        
        Args:
            signal: Numpy array of shape (num_leads, signal_length)
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_error_messages)
        """
        errors = []
        
        if not isinstance(signal, np.ndarray):
            return False, ["Signal must be a numpy ndarray."]
            
        if signal.ndim != 2:
            return False, [f"Signal must be 2D of shape (num_leads, length), got shape {signal.shape}."]
            
        num_leads, length = signal.shape
        
        if length < self.min_length:
            errors.append(f"Signal length {length} is shorter than minimum required length {self.min_length}.")
            
        # NaN / Inf Check
        total_elements = signal.size
        nan_count = np.isnan(signal).sum() + np.isinf(signal).sum()
        nan_ratio = nan_count / total_elements if total_elements > 0 else 1.0
        
        if nan_ratio > self.max_nan_ratio:
            errors.append(f"NaN/Inf ratio {nan_ratio:.3f} exceeds maximum allowed ratio {self.max_nan_ratio}.")
            
        # Lead-specific checks
        for lead_idx in range(num_leads):
            lead_signal = signal[lead_idx]
            
            # Clean NaNs locally for amplitude checks (only if within bounds)
            valid_mask = np.isfinite(lead_signal)
            if not np.any(valid_mask):
                errors.append(f"Lead {lead_idx} contains no finite values.")
                continue
                
            lead_valid = lead_signal[valid_mask]
            
            # Standard deviation check for flatline
            lead_std = np.std(lead_valid)
            if lead_std < self.flatline_threshold:
                errors.append(f"Lead {lead_idx} is a flatline (std = {lead_std:.6f} < {self.flatline_threshold}).")
                
            # Amplitude bounds check
            lead_max = np.max(np.abs(lead_valid))
            if lead_max > self.max_amplitude:
                errors.append(f"Lead {lead_idx} exceeds maximum allowed amplitude ({lead_max:.2f} > {self.max_amplitude}).")
                
            if lead_std < self.min_amplitude:
                errors.append(f"Lead {lead_idx} has insufficient amplitude (std = {lead_std:.4f} < {self.min_amplitude}).")
                
        return len(errors) == 0, errors

    def clean_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Cleans minor issues (like single NaNs by linear interpolation).
        
        Args:
            signal: Numpy array of shape (num_leads, signal_length)
            
        Returns:
            np.ndarray: Interpolated/cleaned signal
        """
        cleaned = signal.copy()
        for lead_idx in range(cleaned.shape[0]):
            lead_signal = cleaned[lead_idx]
            nans = np.isnan(lead_signal)
            if not np.any(nans):
                continue
            x = lambda z: z.nonzero()[0]
            lead_signal[nans] = np.interp(x(nans), x(~nans), lead_signal[~nans])
        return cleaned
