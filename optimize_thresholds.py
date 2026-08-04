import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_management.dataset_factory import DatasetFactory
from temporal_encoder.predictor import TemporalPredictor
from temporal_encoder.evaluator import TemporalEvaluator

device = "cuda" if torch.cuda.is_available() else "cpu"

def optimize():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # 1. Fetch best run model
    print("Fetching the best model from ECG_TemporalEncoder_Optimized experiment...")
    experiment = mlflow.get_experiment_by_name("ECG_TemporalEncoder_Optimized")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    finished_runs = runs[runs["status"] == "FINISHED"]
    
    if finished_runs.empty:
        print("No completed runs found in ECG_TemporalEncoder_Optimized!")
        return
        
    best_run = finished_runs.sort_values(by="metrics.test_macro_auc", ascending=False).iloc[0]
    run_id = best_run["run_id"]
    print(f"Loading model from Run ID: {run_id} (Test Macro AUC: {best_run['metrics.test_macro_auc']:.4f})")
    
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri).to(device)
    model.eval()
    
    # 2. Load dataset
    print("Loading PTB-XL dataset...")
    _, val_ds, test_ds, _ = DatasetFactory.create_datasets(
        dataset_type="ptbxl", download=False, resolution="lr"
    )
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # Collect ground truths
    val_labels = []
    for _, lbls in val_loader:
        val_labels.append(lbls.numpy())
    val_labels = np.concatenate(val_labels, axis=0)
    
    test_labels = []
    for _, lbls in test_loader:
        test_labels.append(lbls.numpy())
    test_labels = np.concatenate(test_labels, axis=0)
    
    # Get predictions
    predictor = TemporalPredictor(model, device=device)
    print("Predicting probabilities on validation set...")
    val_probs = predictor.predict_proba(val_loader)
    print("Predicting probabilities on test set...")
    test_probs = predictor.predict_proba(test_loader)
    
    # 3. Optimize thresholds class-by-class
    print("Optimizing thresholds class-by-class on validation set...")
    num_classes = val_labels.shape[1]
    best_thresholds = np.ones(num_classes) * 0.5
    
    for c in range(num_classes):
        best_f1 = -1.0
        best_t = 0.5
        for t in np.linspace(0.01, 0.99, 100):
            preds = (val_probs[:, c] >= t).astype(int)
            targets = val_labels[:, c]
            # Calculate F1 for class c
            tp = np.sum((preds == 1) & (targets == 1))
            fp = np.sum((preds == 1) & (targets == 0))
            fn = np.sum((preds == 0) & (targets == 1))
            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[c] = best_t
        print(f"  Class {c} (Threshold: {best_t:.4f}, Validation F1: {best_f1:.4f})")
        
    # Evaluate with default threshold (0.5)
    default_preds = (test_probs >= 0.5).astype(int)
    default_subset_acc = np.mean(np.all(default_preds == test_labels, axis=1))
    print(f"\nDefault threshold (0.5) Test Subset Accuracy: {default_subset_acc:.4f}")
    
    # Evaluate with optimized thresholds
    opt_preds = np.zeros_like(test_probs)
    for c in range(num_classes):
        opt_preds[:, c] = (test_probs[:, c] >= best_thresholds[c]).astype(int)
        
    opt_subset_acc = np.mean(np.all(opt_preds == test_labels, axis=1))
    print(f"Optimized thresholds Test Subset Accuracy: {opt_subset_acc:.4f}")

if __name__ == "__main__":
    optimize()
