from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class ECGRecord:
    """
    Domain model representing a single ECG recording.
    
    Attributes:
        record_id (str): Unique identifier for the record.
        signal (np.ndarray): The ECG signal array. Standard shape is (num_leads, signal_length)
                             or (signal_length, num_leads).
        sampling_rate (int): Sampling frequency in Hz.
        leads (List[str]): Names of the leads in order (e.g., standard 12 leads).
        labels (np.ndarray): Encoded label array (e.g. multi-hot binary vector).
        metadata (Dict[str, Any]): Additional clinical or dataset-specific metadata.
        patient_id (Optional[int]): ID of the patient.
        age (Optional[float]): Patient age.
        sex (Optional[str]): Patient sex ('M', 'F', or None).
    """
    record_id: str
    signal: np.ndarray
    sampling_rate: int
    leads: List[str]
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    metadata: Dict[str, Any] = field(default_factory=dict)
    patient_id: Optional[int] = None
    age: Optional[float] = None
    sex: Optional[str] = None

    def __post_init__(self):
        # Validate signal type and shape
        if not isinstance(self.signal, np.ndarray):
            raise TypeError(f"Signal must be a numpy.ndarray, got {type(self.signal)}")
        
        # Ensure signal is float32
        if self.signal.dtype != np.float32:
            self.signal = self.signal.astype(np.float32)

    @property
    def num_leads(self) -> int:
        """Returns the number of leads in the ECG record."""
        return len(self.leads)

    @property
    def signal_length(self) -> int:
        """Returns the length of the ECG signal (number of samples per lead)."""
        # If signal is (num_leads, length)
        if self.signal.ndim == 2:
            if self.signal.shape[0] == self.num_leads:
                return self.signal.shape[1]
            else:
                return self.signal.shape[0]
        return len(self.signal)

    def get_lead_signal(self, lead_name: str) -> np.ndarray:
        """Retrieves the signal for a specific lead by name."""
        if lead_name not in self.leads:
            raise ValueError(f"Lead '{lead_name}' not found in record. Available: {self.leads}")
        idx = self.leads.index(lead_name)
        
        # Determine orientation of signal
        if self.signal.ndim == 2:
            if self.signal.shape[0] == self.num_leads:
                return self.signal[idx, :]
            else:
                return self.signal[:, idx]
        raise ValueError("Signal must be 2-dimensional to extract a specific lead.")
