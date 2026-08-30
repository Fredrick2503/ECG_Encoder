import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def compute_binary_ece(labels, probs, num_bins=10):
    """
    Computes Expected Calibration Error (ECE) for a binary classification task.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n_samples = len(labels)
    
    for m in range(num_bins):
        bin_lower = bin_boundaries[m]
        bin_upper = bin_boundaries[m + 1]
        
        # Get samples in this bin
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += prop_in_bin * np.abs(accuracy_in_bin - avg_confidence_in_bin)
            
    return float(ece)

def calculate_metrics(labels, probs, thresholds=None):
    """
    Computes ROC-AUC, F1 scores, Subset Accuracy, per-class F1, and ECE.
    """
    num_classes = labels.shape[1]
    
    # 1. Thresholds
    if thresholds is None:
        thresholds = np.full(num_classes, 0.5)
        
    preds = (probs >= thresholds).astype(int)
    
    # 2. ROC-AUC
    try:
        macro_auc = roc_auc_score(labels, probs, average='macro')
        micro_auc = roc_auc_score(labels, probs, average='micro')
    except Exception:
        macro_auc = np.nan
        micro_auc = np.nan
        
    # 3. F1
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    
    # 4. Subset Accuracy (Exact Match Ratio)
    subset_acc = accuracy_score(labels, preds)
    
    # 5. Per-class F1 and ECE
    per_class_f1 = []
    per_class_ece = []
    per_class_auc = []
    
    for c in range(num_classes):
        f1_c = f1_score(labels[:, c], preds[:, c], zero_division=0)
        ece_c = compute_binary_ece(labels[:, c], probs[:, c])
        try:
            auc_c = roc_auc_score(labels[:, c], probs[:, c])
        except Exception:
            auc_c = np.nan
            
        per_class_f1.append(float(f1_c))
        per_class_ece.append(float(ece_c))
        per_class_auc.append(float(auc_c))
        
    macro_ece = float(np.mean(per_class_ece))
    
    return {
        "macro_auc": float(macro_auc),
        "micro_auc": float(micro_auc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "subset_acc": float(subset_acc),
        "per_class_f1": per_class_f1,
        "per_class_auc": per_class_auc,
        "per_class_ece": per_class_ece,
        "macro_ece": macro_ece
    }
