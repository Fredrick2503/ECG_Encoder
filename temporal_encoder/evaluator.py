import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict, Any

class TemporalEvaluator:
    """
    Computes performance metrics for multi-label ECG classification models.
    Supports multi-hot target labels.
    """
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculates performance metrics.
        
        Args:
            y_true: Ground truth binary targets shape (N, num_classes).
            y_pred_proba: Predicted probability values shape (N, num_classes).
            threshold: Probability threshold for binary prediction.
            
        Returns:
            Dict[str, float]: Calculated metrics.
        """
        y_pred = (y_pred_proba >= threshold).astype(np.float32)
        
        # 1. Exact match accuracy (Subset Accuracy)
        subset_acc = accuracy_score(y_true, y_pred)
        
        # 2. Hamming Loss (per-element error rate)
        hamming_loss = np.mean(y_true != y_pred)
        
        # 3. Macro F1 score
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        # 4. Macro ROC-AUC score
        try:
            macro_auc = roc_auc_score(y_true, y_pred_proba, average="macro")
        except ValueError:
            # Fallback if validation batch doesn't contain positive labels for some classes
            macro_auc = 0.5
            
        return {
            "subset_accuracy": float(subset_acc),
            "hamming_loss": float(hamming_loss),
            "macro_f1": float(macro_f1),
            "macro_auc": float(macro_auc)
        }
