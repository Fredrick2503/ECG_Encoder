import os
import sys
import subprocess
import argparse
import time

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

# Add workspace to path
sys.path.append("c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder")

from data_management.dataset_factory import DatasetFactory
from data_management.label_encoder import PTBXLLabelEncoder, BinaryLabelEncoder
from temporal_encoder.predictor import TemporalPredictor
from temporal_encoder.evaluator import TemporalEvaluator

device = "cuda" if torch.cuda.is_available() else "cpu"
tracking_uri = "file:///c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/mlruns"
mlflow.set_tracking_uri(tracking_uri)

def run_training(script: str, balance_mode: str, epochs: int):
    cmd = [
        ".venv\\Scripts\\python.exe",
        script,
        "--balance_mode", balance_mode,
        "--epochs", str(epochs),
        "--num_workers", "4"
    ]
    print(f"\n>>> Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def get_best_model_run_id(experiment_name: str, balance_mode: str):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found!")
        
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'!")
        
    # Filter by balance_mode parameter
    col_bal_mode = "params.balance_mode"
    if col_bal_mode not in runs.columns:
        raise ValueError(f"No balance_mode parameter found in logged runs of {experiment_name}.")
        
    filtered = runs[runs[col_bal_mode] == balance_mode]
    finished = filtered[filtered["status"] == "FINISHED"]
    if finished.empty:
        raise ValueError(f"No finished runs found for balance_mode={balance_mode} in experiment '{experiment_name}'!")
        
    # Get the most recent run
    most_recent = finished.sort_values(by="start_time", ascending=False).iloc[0]
    return most_recent["run_id"]

def evaluate_experiment_ensemble(balance_mode: str):
    print(f"\n==================================================")
    print(f"EVALUATING ENSEMBLE FOR: {balance_mode.upper()}")
    print(f"==================================================")
    
    # 1. Load dataset statistics
    _, val_ds, test_ds, loader = DatasetFactory.create_datasets(
        dataset_type="ptbxl", download=False, resolution="lr", balance_mode=balance_mode
    )
    
    classes = loader.label_encoder.classes
    num_classes = len(classes)
    
    # Extract labels directly from metadata
    def extract_labels_from_metadata(ds):
        labels_list = []
        for rec_id in ds.record_ids:
            row = loader.metadata_df.loc[rec_id]
            diag_classes = loader.parser.get_diagnostic_classes(row.get("scp_codes", {}))
            encoded = loader.label_encoder.encode(diag_classes)
            labels_list.append(encoded)
        return np.array(labels_list)
        
    val_labels = extract_labels_from_metadata(val_ds)
    test_labels = extract_labels_from_metadata(test_ds)
    
    # Get model run IDs
    resnet_run_id = get_best_model_run_id("ECG_ResNet_Final", balance_mode)
    transformer_run_id = get_best_model_run_id("ECG_Transformer_Final", balance_mode)
    
    # Load Models
    print(f"Loading ResNet (Run: {resnet_run_id}) on {device}...")
    resnet_model = mlflow.pytorch.load_model(f"runs:/{resnet_run_id}/model").to(device)
    resnet_model.eval()
    
    print(f"Loading Transformer (Run: {transformer_run_id}) on {device}...")
    transformer_model = mlflow.pytorch.load_model(f"runs:/{transformer_run_id}/model").to(device)
    transformer_model.eval()
    
    # Loaders (num_workers=4 for fast parallel loading)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=(device=="cuda"))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=(device=="cuda"))
    
    # Run Inference
    resnet_predictor = TemporalPredictor(resnet_model, device=device)
    resnet_val_probs = resnet_predictor.predict_proba(val_loader)
    resnet_test_probs = resnet_predictor.predict_proba(test_loader)
    
    transformer_predictor = TemporalPredictor(transformer_model, device=device)
    transformer_val_probs = transformer_predictor.predict_proba(val_loader)
    transformer_test_probs = transformer_predictor.predict_proba(test_loader)
    
    # Search for optimal ensemble weight (w * ResNet + (1 - w) * Transformer)
    best_weight = 0.5
    best_val_subset_acc = -1.0
    for w in np.linspace(0.0, 1.0, 11):
        ensemble_val_probs = w * resnet_val_probs + (1 - w) * transformer_val_probs
        preds = (ensemble_val_probs >= 0.5).astype(int)
        subset_acc = np.mean(np.all(preds == val_labels, axis=1))
        if subset_acc > best_val_subset_acc:
            best_val_subset_acc = subset_acc
            best_weight = w
            
    print(f"Optimal weight: ResNet={best_weight:.2f}, Transformer={1.0-best_weight:.2f}")
    
    # Ensemble probabilities
    ensemble_val_probs = best_weight * resnet_val_probs + (1 - best_weight) * transformer_val_probs
    ensemble_test_probs = best_weight * resnet_test_probs + (1 - best_weight) * transformer_test_probs
    
    # Class-by-class threshold optimization
    best_thresholds = np.ones(num_classes) * 0.5
    for c in range(num_classes):
        best_f1 = -1.0
        best_t = 0.5
        for t in np.linspace(0.01, 0.99, 100):
            preds = (ensemble_val_probs[:, c] >= t).astype(int)
            targets = val_labels[:, c]
            tp = np.sum((preds == 1) & (targets == 1))
            fp = np.sum((preds == 1) & (targets == 0))
            fn = np.sum((preds == 0) & (targets == 1))
            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[c] = best_t
        
    # Evaluate Ensemble with Optimized Thresholds on Test Set
    ensemble_preds_opt = np.zeros_like(ensemble_test_probs)
    for c in range(num_classes):
        ensemble_preds_opt[:, c] = (ensemble_test_probs[:, c] >= best_thresholds[c]).astype(int)
        
    opt_subset_acc = np.mean(np.all(ensemble_preds_opt == test_labels, axis=1))
    opt_hamming = np.mean(ensemble_preds_opt != test_labels)
    opt_macro_f1 = f1_score(test_labels, ensemble_preds_opt, average="macro", zero_division=0)
    
    # ROC-AUC calculation
    try:
        opt_macro_auc = roc_auc_score(test_labels, ensemble_test_probs, average="macro")
    except ValueError:
        opt_macro_auc = 0.5
        
    # Compile per-class metrics dictionary
    per_class_results = []
    for c in range(num_classes):
        cls_name = classes[c]
        thresh = best_thresholds[c]
        y_true_c = test_labels[:, c]
        y_pred_c = ensemble_preds_opt[:, c]
        y_prob_c = ensemble_test_probs[:, c]
        
        prec = precision_score(y_true_c, y_pred_c, zero_division=0)
        rec = recall_score(y_true_c, y_pred_c, zero_division=0)
        f1 = f1_score(y_true_c, y_pred_c, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true_c, y_prob_c)
        except ValueError:
            auc = 0.5
            
        support = int(np.sum(y_true_c))
        
        per_class_results.append({
            "class": cls_name,
            "threshold": thresh,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "support": support
        })
        
    # Overall summary metrics
    summary = {
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "norm_val_count": int(np.sum(val_labels[:, 0])),
        "norm_test_count": int(np.sum(test_labels[:, 0])),
        "resnet_weight": best_weight,
        "transformer_weight": 1.0 - best_weight,
        "subset_accuracy": opt_subset_acc,
        "macro_f1": opt_macro_f1,
        "macro_auc": opt_macro_auc,
        "hamming_loss": opt_hamming,
        "per_class": per_class_results
    }
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Orchestrate 4 data-balancing experiments")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs per run")
    args = parser.parse_args()
    
    modes = ["average", "max", "min", "binary"]
    results = {}
    
    t_start = time.time()
    
    for mode in modes:
        print(f"\n\n======================================================================")
        print(f"STARTING EXPERIMENT: {mode.upper()} ({args.epochs} epochs)")
        print(f"======================================================================")
        
        # 1. Train models
        run_training("train_final_resnet.py", mode, args.epochs)
        run_training("train_final_transformer.py", mode, args.epochs)
        
        # 2. Evaluate
        results[mode] = evaluate_experiment_ensemble(mode)
        
    # Write report
    report_path = "c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/outputs/reports/experiments_comparison_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# ECG Foundation Representation - Balancing Experiments Report\n\n")
        f.write(f"This report compares the performance of 4 distinct data balancing / filtering experiments on the PTB-XL dataset.\n")
        f.write(f"Evaluation completed on: `{time.strftime('%Y-%m-%d %H:%M:%S')}`. Total duration: `{time.time()-t_start:.1f}s`.\n\n")
        
        f.write("## 1. Overview Comparison Table\n\n")
        f.write("| Experiment / Mode | Val Size | Test Size | ResNet W | Trans. W | Subset Acc | Macro F1 | Macro AUC | Hamming Loss |\n")
        f.write("|-------------------|----------|-----------|----------|----------|------------|----------|-----------|--------------|\n")
        for mode in modes:
            r = results[mode]
            f.write(f"| `{mode}` | {r['val_size']} | {r['test_size']} | {r['resnet_weight']:.2f} | {r['transformer_weight']:.2f} | {r['subset_accuracy']:.4f} | {r['macro_f1']:.4f} | {r['macro_auc']:.4f} | {r['hamming_loss']:.4f} |\n")
            
        f.write("\n---\n\n## 2. Detailed Per-Class Breakdown per Experiment\n\n")
        
        for mode in modes:
            r = results[mode]
            f.write(f"### Experiment: `{mode}`\n\n")
            f.write("| Class | Threshold | Precision | Recall | F1-Score | ROC-AUC | Support |\n")
            f.write("|-------|-----------|-----------|--------|----------|---------|---------|\n")
            for pc in r["per_class"]:
                f.write(f"| {pc['class']} | {pc['threshold']:.4f} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['auc']:.4f} | {pc['support']} |\n")
            f.write("\n")
            
    print(f"\n======================================================================")
    print(f"ALL EXPERIMENTS COMPLETE! Report generated at outputs/reports/experiments_comparison_report.md")
    print(f"======================================================================")

if __name__ == "__main__":
    main()
