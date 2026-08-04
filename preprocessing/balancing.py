import numpy as np
from typing import Tuple
from imblearn.combine import SMOTEENN

class ECGDatasetBalancer:
    """
    Balances ECG dataset distributions using SMOTE-ENN (SMOTE + Edited Nearest Neighbors).
    Reshapes 3D multi-lead signals (num_samples, num_leads, length) into a 2D feature matrix
    for compatibility with imbalanced-learn, and supports multi-hot label targets
    using a label power-set transformation.
    """
    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: Seed for reproducibility of synthetic sample generation.
        """
        self.random_state = random_state

    def balance_dataset(self, signals: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balances the ECG dataset using SMOTE-ENN.
        
        Args:
            signals: Numpy array of shape (num_samples, num_leads, length)
            labels: Numpy array of shape (num_samples, num_classes) or (num_samples,)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Resampled (signals, labels)
        """
        num_samples, num_leads, length = signals.shape
        
        # Flatten time-series channels to 2D matrix
        signals_2d = signals.reshape(num_samples, -1)
        
        is_multilabel = labels.ndim == 2 and labels.shape[1] > 1
        
        if is_multilabel:
            # Map each unique multi-hot combination to a unique class ID (Power-set method)
            unique_classes, class_ids = np.unique(labels, axis=0, return_inverse=True)
            y = class_ids
        else:
            y = labels.squeeze()
            unique_classes = None

        # Check class counts: SMOTE requires at least 6 samples per class by default (for 5-NN)
        # If class frequencies are too low, skip balancing to prevent errors
        unique_y, counts = np.unique(y, return_counts=True)
        if len(unique_y) < 2 or np.min(counts) < 6:
            print("Warning: Class frequencies are too low to apply SMOTE-ENN. Skipping dataset balancing.")
            return signals, labels
            
        smote_enn = SMOTEENN(random_state=self.random_state)
        signals_resampled, y_resampled = smote_enn.fit_resample(signals_2d, y)
        
        # Reshape 2D signal matrix back to 3D time series tensor
        signals_resampled_3d = signals_resampled.reshape(-1, num_leads, length)
        
        if is_multilabel and unique_classes is not None:
            labels_resampled = unique_classes[y_resampled]
        else:
            labels_resampled = y_resampled
            
        return signals_resampled_3d, labels_resampled
