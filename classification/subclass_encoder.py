import numpy as np
from typing import List, Union

PTBXL_SUBCLASSES = [
    "NORM", "IMI", "AMI", "STTC", "LVH", "LAFB/LPFB", "ISC_", "ISCA", 
    "IRBBB", "_AVB", "IVCD", "CLBBB", "NST_", "ISCI", "LAO/LAE", "CRBBB", 
    "LMI", "RVH", "WPW", "ILBBB", "RAO/RAE", "SEHYP", "PMI"
]

class PTBXLSubclassLabelEncoder:
    """
    Multi-hot binary label encoder for PTB-XL diagnostic subclasses (23 classes).
    """
    def __init__(self, classes: List[str] = PTBXL_SUBCLASSES):
        self.classes = list(classes)
        self.num_classes = len(self.classes)

    def encode(self, subclasses: List[str]) -> np.ndarray:
        """
        Encodes a list of diagnostic subclasses into a multi-hot binary vector of shape (23,).
        """
        vector = np.zeros(self.num_classes, dtype=np.float32)
        for sub in subclasses:
            if sub in self.classes:
                idx = self.classes.index(sub)
                vector[idx] = 1.0
        return vector

    def decode(self, encoded: np.ndarray, threshold: float = 0.5) -> List[str]:
        """
        Decodes a multi-hot binary vector back into subclass list names.
        """
        decoded = []
        for i, val in enumerate(encoded):
            if val >= threshold:
                decoded.append(self.classes[i])
        return decoded
