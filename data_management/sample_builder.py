from typing import List, Callable, Optional, Tuple, Union, Any
import torch
from torch.utils.data import Dataset
import numpy as np

from data_management.loader import BaseLoader
from data_management.ecg_record import ECGRecord

class ECGDataset(Dataset):
    """
    A PyTorch Dataset wrapping ECG loader and record building.
    """
    
    def __init__(
        self,
        record_ids: List[Union[int, str]],
        loader: BaseLoader,
        preprocessor: Optional[Any] = None,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ):
        """
        Args:
            record_ids (List[Union[int, str]]): List of identifiers to fetch records for.
            loader (BaseLoader): Loader to load ECGRecord objects.
            preprocessor: Preprocessing pipeline to clean signals (optional).
            transform: Optional transform callable to apply augmentation/tensor formatting.
        """
        self.record_ids = list(record_ids)
        self.loader = loader
        self.preprocessor = preprocessor
        self.transform = transform

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads the ECG record, preprocesses the signal, and returns tensors.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (signal_tensor, label_tensor)
        """
        record_id = self.record_ids[idx]
        record = self.loader.load_record(record_id)
        
        signal = record.signal
        
        # Apply preprocessing if available
        if self.preprocessor is not None:
            # We assume preprocessor has a process() method taking a numpy array
            if hasattr(self.preprocessor, "process"):
                signal = self.preprocessor.process(signal)
            elif callable(self.preprocessor):
                signal = self.preprocessor(signal)
                
        # Apply transform/augmentations
        if self.transform is not None:
            signal = self.transform(signal)
            
        # Convert signal to torch tensor
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        
        # Convert label to torch tensor
        label_tensor = torch.tensor(record.labels, dtype=torch.float32)
        
        return signal_tensor, label_tensor
