"""
=========================================================
Biomarker Encoder Evaluation Metrics

Computes advanced evaluation metrics and optimizes class-wise
decision thresholds for multilabel ECG classification.

Author : ECG Intelligence System
=========================================================
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)

class BiomarkerMetrics:
    def __init__(self, thresholds: Union[float, np.ndarray, List[float]] = 0.5):
        self.thresholds = thresholds

    def threshold_predictions(self, probabilities: np.ndarray) -> np.ndarray:
        """Binarize predicted probabilities using stored decision thresholds."""
        probs = np.asarray(probabilities, dtype=np.float32)
        if isinstance(self.thresholds, (float, int)):
            return (probs >= self.thresholds).astype(int)
        
        # Apply class-wise thresholds
        thresh_arr = np.asarray(self.thresholds, dtype=np.float32)
        preds = np.zeros_like(probs, dtype=int)
        for c in range(probs.shape[1]):
            t = thresh_arr[c] if c < len(thresh_arr) else 0.5
            preds[:, c] = (probs[:, c] >= t).astype(int)
        return preds

    def optimize_thresholds(self, y_true: np.ndarray, probabilities: np.ndarray, method: str = "f1") -> np.ndarray:
        """
        Optimizes thresholds per class on validation data.
        Supported methods: 'f1' (maximizes class-specific F1), 'youden' (maximizes sensitivity + specificity - 1).
        """
        y_true_arr = np.asarray(y_true, dtype=int)
        probs_arr = np.asarray(probabilities, dtype=np.float32)
        num_classes = y_true_arr.shape[1]
        
        opt_thresholds = np.zeros(num_classes)
        
        for c in range(num_classes):
            best_score = -np.inf
            best_thresh = 0.5
            
            # Sweep threshold candidate values
            for thresh in np.arange(0.01, 1.0, 0.01):
                preds = (probs_arr[:, c] >= thresh).astype(int)
                
                if method.lower() == "youden":
                    tp = np.sum((y_true_arr[:, c] == 1) & (preds == 1))
                    fn = np.sum((y_true_arr[:, c] == 1) & (preds == 0))
                    tn = np.sum((y_true_arr[:, c] == 0) & (preds == 0))
                    fp = np.sum((y_true_arr[:, c] == 0) & (preds == 1))
                    
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    score = sensitivity + specificity - 1
                else:  # Default to F1 optimization
                    score = f1_score(y_true_arr[:, c], preds, zero_division=0)
                
                if score > best_score:
                    best_score = score
                    best_thresh = thresh
            
            opt_thresholds[c] = best_thresh
            
        self.thresholds = opt_thresholds
        return opt_thresholds

    def evaluate(self, y_true: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray = None) -> pd.DataFrame:
        """Computes all research-grade classification metrics."""
        results = {}
        y_true_arr = np.asarray(y_true, dtype=int)
        preds_arr = np.asarray(predictions, dtype=int)

        # 1. Exact Match Accuracy (Subset Accuracy)
        results["Subset Accuracy"] = accuracy_score(y_true_arr, preds_arr)

        # 2. Balanced Accuracy (Macro Average of Class-wise balanced accuracy)
        bal_accs = []
        for c in range(y_true_arr.shape[1]):
            bal_accs.append(balanced_accuracy_score(y_true_arr[:, c], preds_arr[:, c]))
        results["Balanced Accuracy"] = np.mean(bal_accs)

        # 3. Precision (Micro, Macro, Weighted)
        results["Precision Micro"] = precision_score(y_true_arr, preds_arr, average="micro", zero_division=0)
        results["Precision Macro"] = precision_score(y_true_arr, preds_arr, average="macro", zero_division=0)
        results["Precision Weighted"] = precision_score(y_true_arr, preds_arr, average="weighted", zero_division=0)

        # 4. Recall (Micro, Macro, Weighted)
        results["Recall Micro"] = recall_score(y_true_arr, preds_arr, average="micro", zero_division=0)
        results["Recall Macro"] = recall_score(y_true_arr, preds_arr, average="macro", zero_division=0)
        results["Recall Weighted"] = recall_score(y_true_arr, preds_arr, average="weighted", zero_division=0)

        # 5. F1-score (Micro, Macro, Weighted)
        results["F1 Micro"] = f1_score(y_true_arr, preds_arr, average="micro", zero_division=0)
        results["F1 Macro"] = f1_score(y_true_arr, preds_arr, average="macro", zero_division=0)
        results["F1 Weighted"] = f1_score(y_true_arr, preds_arr, average="weighted", zero_division=0)

        # 6. Hamming Loss & Jaccard Index
        results["Hamming Loss"] = hamming_loss(y_true_arr, preds_arr)
        results["Jaccard Micro"] = jaccard_score(y_true_arr, preds_arr, average="micro", zero_division=0)
        results["Jaccard Macro"] = jaccard_score(y_true_arr, preds_arr, average="macro", zero_division=0)

        # 7. Matthews Correlation Coefficient (MCC)
        try:
            results["Matthews Correlation"] = matthews_corrcoef(y_true_arr.ravel(), preds_arr.ravel())
        except Exception:
            results["Matthews Correlation"] = np.nan

        # 8. Probability Metrics (ROC-AUC, PR-AUC)
        if probabilities is not None:
            probs_arr = np.asarray(probabilities, dtype=np.float32)
            try:
                results["ROC AUC Micro"] = roc_auc_score(y_true_arr, probs_arr, average="micro")
            except Exception:
                results["ROC AUC Micro"] = np.nan
            try:
                results["ROC AUC Macro"] = roc_auc_score(y_true_arr, probs_arr, average="macro")
            except Exception:
                results["ROC AUC Macro"] = np.nan
            try:
                results["PR AUC Micro"] = average_precision_score(y_true_arr, probs_arr, average="micro")
            except Exception:
                results["PR AUC Micro"] = np.nan
            try:
                results["PR AUC Macro"] = average_precision_score(y_true_arr, probs_arr, average="macro")
            except Exception:
                results["PR AUC Macro"] = np.nan

        return pd.DataFrame(results, index=[0])

    def print_results(self, results: pd.DataFrame):
        print("=" * 70)
        print("BIOMARKER ENCODER PERFORMANCE METRICS")
        print("=" * 70)
        for metric, value in results.iloc[0].items():
            if pd.isna(value):
                print(f"{metric:<30}: N/A")
            else:
                print(f"{metric:<30}: {value:.4f}")
        print("=" * 70)