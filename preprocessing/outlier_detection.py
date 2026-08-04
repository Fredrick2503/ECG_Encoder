import numpy as np
from typing import Tuple
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class DBSCANOutlierDetector:
    """
    Detects outliers in a batch of ECG records using DBSCAN clustering.
    Extracts statistical features from each record and flags anomalies.
    """
    def __init__(self, eps: float = 3.0, min_samples: int = 5):
        """
        Args:
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        """
        self.eps = eps
        self.min_samples = min_samples

    def _extract_features(self, signals: np.ndarray) -> np.ndarray:
        """
        Extracts statistical descriptors for each lead of each record.
        
        Args:
            signals: Numpy array of shape (num_records, num_leads, length)
            
        Returns:
            np.ndarray: Feature matrix of shape (num_records, num_leads * num_features)
        """
        N, C, L = signals.shape
        features = []
        
        for i in range(N):
            rec_features = []
            for j in range(C):
                lead = signals[i, j]
                mean = np.mean(lead)
                std = np.std(lead)
                min_val = np.min(lead)
                max_val = np.max(lead)
                energy = np.sum(lead ** 2) / L
                
                rec_features.extend([mean, std, min_val, max_val, energy])
            features.append(rec_features)
            
        return np.array(features)

    def detect_outliers(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identifies inliers and outliers in the batch of ECG signals.
        
        Args:
            signals: Numpy array of shape (num_records, num_leads, length)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (clean_indices, outlier_indices)
        """
        if len(signals) < self.min_samples:
            # Insufficient samples to construct meaningful density-based clusters
            return np.arange(len(signals)), np.array([], dtype=int)
            
        features = self._extract_features(signals)
        
        # Standardize features before applying distance-based clustering
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(scaled_features)
        
        # DBSCAN labels outliers as -1
        inlier_indices = np.where(labels != -1)[0]
        outlier_indices = np.where(labels == -1)[0]
        
        return inlier_indices, outlier_indices
