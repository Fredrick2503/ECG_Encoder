from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

class BaseLabelEncoder(ABC):
    """Abstract Base Class for all label encoders."""
    
    @abstractmethod
    def encode(self, labels: Union[List[str], str]) -> np.ndarray:
        """Encodes labels into a numerical format."""
        pass

    @abstractmethod
    def decode(self, encoded: np.ndarray) -> List[str]:
        """Decodes a numerical representation back to original labels."""
        pass


class PTBXLLabelEncoder(BaseLabelEncoder):
    """Multi-hot binary label encoder for PTB-XL diagnostic classes."""
    
    def __init__(self, classes: List[str]):
        self.classes = list(classes)
        self.num_classes = len(self.classes)

    def encode(self, diagnostic_classes: List[str]) -> np.ndarray:
        """
        Encodes a list of diagnostic classes into a multi-hot binary vector.
        
        Args:
            diagnostic_classes (List[str]): List of labels.
            
        Returns:
            np.ndarray: Multi-hot binary vector of shape (num_classes,).
        """
        vector = np.zeros(self.num_classes, dtype=np.float32)
        for cls in diagnostic_classes:
            if cls in self.classes:
                idx = self.classes.index(cls)
                vector[idx] = 1.0
        return vector

    def decode(self, encoded: np.ndarray, threshold: float = 0.5) -> List[str]:
        """
        Decodes a multi-hot binary vector back into a list of diagnostic classes.
        
        Args:
            encoded (np.ndarray): Array of probabilities or binary values.
            threshold (float): Threshold to consider a class present.
            
        Returns:
            List[str]: Decoded diagnostic class names.
        """
        decoded = []
        for i, val in enumerate(encoded):
            if val >= threshold:
                decoded.append(self.classes[i])
        return decoded
