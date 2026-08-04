from typing import List, Optional
import numpy as np
from preprocessing.filters import PreprocessingStep
from preprocessing.validation import SignalValidator

class PreprocessingPipeline:
    """
    Chains and executes multiple preprocessing steps sequentially.
    Automatically handles 3D segmented signals by applying non-segmenter steps
    to each segment independently.
    """
    def __init__(
        self,
        steps: List[PreprocessingStep],
        validator: Optional[SignalValidator] = None
    ):
        """
        Args:
            steps: List of preprocessing steps to execute.
            validator: Optional SignalValidator to reject poor quality inputs.
        """
        self.steps = steps
        self.validator = validator

    def process(self, signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Processes the input signal through all registered steps.
        
        Args:
            signal: Raw numpy signal array of shape (num_leads, length)
            sampling_rate: Sampling frequency in Hz
            
        Returns:
            np.ndarray: Preprocessed signal (2D or 3D if segmented)
        """
        # 1. Validation
        if self.validator is not None:
            is_valid, errors = self.validator.validate(signal)
            if not is_valid:
                raise ValueError(f"Signal validation failed: {'; '.join(errors)}")
            # Apply interpolation to minor NaN occurrences
            signal = self.validator.clean_signal(signal)

        # 2. Sequential execution
        current_signal = signal.copy()
        for step in self.steps:
            # Check if the step is a segmenter class
            is_segmenter = "Segmenter" in step.__class__.__name__
            
            if current_signal.ndim == 3 and not is_segmenter:
                # Signal is already segmented into shape (num_segments, num_leads, window_size).
                # Apply 2D processing step to each segment individually.
                processed_segments = []
                for segment in current_signal:
                    processed_segments.append(step.process(segment, sampling_rate))
                current_signal = np.array(processed_segments)
            else:
                current_signal = step.process(current_signal, sampling_rate)

        return current_signal
