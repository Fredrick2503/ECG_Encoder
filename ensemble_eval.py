import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_management.dataset_factory import DatasetFactory
from temporal_encoder.predictor import TemporalPredictor
from temporal_encoder.evaluator import TemporalEvaluator
from sklearn.metrics import f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_best_model_from_experiment(experiment_name: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found!")
        
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    finished_runs = runs[runs["status"] == "FINISHED"]
    if finished_runs.empty:
        raise ValueError(f"No finished runs found in experiment '{experiment_name}'!")
        
    # Sort by test_macro_auc if exists, otherwise test_subset_accuracy
    metric_cols = finished_runs.columns
    sort_col = None
    for col in ["metrics.test_macro_auc", "metrics.test_subset_accuracy", "metrics.val_loss"]:
        if col in metric_cols:
            sort_col = col
            break
            
    if sort_col:
        ascending = (sort_col == "metrics.val_loss")
        best_run = finished_runs.sort_values(by=sort_col, ascending=ascending).iloc[0]
    else:
        best_run = finished_runs.iloc[0]
        
    run_id = best_run["run_id"]
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading best model from {experiment_name} (Run ID: {run_id})...")
    model = mlflow.pytorch.load_model(model_uri).to(device)
    model.eval()
    return model

def main():
    tracking_uri = "file:///c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/mlruns"
    
    # 1. Load the models
    resnet_model = load_best_model_from_experiment("ECG_ResNet_Final", tracking_uri)
    transformer_model = load_best_model_from_experiment("ECG_Transformer_Final", tracking_uri)
    
    # 2. Load dataset
    print("\nLoading PTB-XL dataset...")
    _, val_ds, test_ds, _ = DatasetFactory.create_datasets(
        dataset_type="ptbxl", download=False, resolution="lr"
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    # Collect ground truths
    val_labels = []
    for _, lbls in val_loader:
        val_labels.append(lbls.numpy())
    val_labels = np.concatenate(val_labels, axis=0)
    
    test_labels = []
    for _, lbls in test_loader:
        test_labels.append(lbls.numpy())
    test_labels = np.concatenate(test_labels, axis=0)
    
    # 3. Get predictions (probabilities)
    print("\nRunning inference with ResNet...")
    resnet_predictor = TemporalPredictor(resnet_model, device=device)
    resnet_val_probs = resnet_predictor.predict_proba(val_loader)
    resnet_test_probs = resnet_predictor.predict_proba(test_loader)
    
    print("Running inference with Transformer...")
    transformer_predictor = TemporalPredictor(transformer_model, device=device)
    transformer_val_probs = transformer_predictor.predict_proba(val_loader)
    transformer_test_probs = transformer_predictor.predict_proba(test_loader)
    
    # 4. Search for best ensemble weight (w * ResNet + (1 - w) * Transformer)
    print("\nSearching for best ensemble weight on validation set...")
    best_weight = 0.5
    best_val_subset_acc = -1.0
    
    # Grid search for weight
    for w in np.linspace(0.0, 1.0, 11):
        ensemble_val_probs = w * resnet_val_probs + (1 - w) * transformer_val_probs
        # Use simple 0.5 threshold to evaluate weight
        preds = (ensemble_val_probs >= 0.5).astype(int)
        subset_acc = np.mean(np.all(preds == val_labels, axis=1))
        print(f"  Weight ResNet={w:.1f}, Transformer={1-w:.1f} => Val Subset Accuracy: {subset_acc:.4f}")
        if subset_acc > best_val_subset_acc:
            best_val_subset_acc = subset_acc
            best_weight = w
            
    print(f"\nOptimal Ensemble Weights: ResNet = {best_weight:.2f}, Transformer = {1.0 - best_weight:.2f}")
    
    # Apply optimal weight to compute combined probabilities
    ensemble_val_probs = best_weight * resnet_val_probs + (1 - best_weight) * transformer_val_probs
    ensemble_test_probs = best_weight * resnet_test_probs + (1 - best_weight) * transformer_test_probs
    
    # 5. Optimize thresholds class-by-class on validation set
    num_classes = val_labels.shape[1]
    best_thresholds = np.ones(num_classes) * 0.5
    
    print("\nOptimizing thresholds class-by-class on validation set...")
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
        print(f"  Class {c} (Threshold: {best_t:.4f}, Validation F1: {best_f1:.4f})")
        
    # 6. Evaluate Baseline and Final Optimized Models on Test Set
    print("\n==================================================")
    print("FINAL TEST METRICS EVALUATION")
    print("==================================================")
    
    # Baseline ResNet (0.5 threshold)
    resnet_test_preds = (resnet_test_probs >= 0.5).astype(int)
    resnet_metrics = TemporalEvaluator.evaluate(test_labels, resnet_test_probs, threshold=0.5)
    print(f"Baseline ResNet (Threshold=0.5):")
    print(f"  Subset Accuracy: {resnet_metrics['subset_accuracy']:.4f}")
    print(f"  Macro ROC-AUC:   {resnet_metrics['macro_auc']:.4f}")
    print(f"  Macro F1-Score:  {resnet_metrics['macro_f1']:.4f}")
    print(f"  Hamming Loss:    {resnet_metrics['hamming_loss']:.4f}")
    
    # Baseline Transformer (0.5 threshold)
    transformer_test_preds = (transformer_test_probs >= 0.5).astype(int)
    transformer_metrics = TemporalEvaluator.evaluate(test_labels, transformer_test_probs, threshold=0.5)
    print(f"\nBaseline Transformer (Threshold=0.5):")
    print(f"  Subset Accuracy: {transformer_metrics['subset_accuracy']:.4f}")
    print(f"  Macro ROC-AUC:   {transformer_metrics['macro_auc']:.4f}")
    print(f"  Macro F1-Score:  {transformer_metrics['macro_f1']:.4f}")
    print(f"  Hamming Loss:    {transformer_metrics['hamming_loss']:.4f}")
    
    # Ensemble with default 0.5 threshold
    ensemble_preds_default = (ensemble_test_probs >= 0.5).astype(int)
    ensemble_default_metrics = TemporalEvaluator.evaluate(test_labels, ensemble_test_probs, threshold=0.5)
    print(f"\nEnsemble (Threshold=0.5):")
    print(f"  Subset Accuracy: {ensemble_default_metrics['subset_accuracy']:.4f}")
    print(f"  Macro ROC-AUC:   {ensemble_default_metrics['macro_auc']:.4f}")
    print(f"  Macro F1-Score:  {ensemble_default_metrics['macro_f1']:.4f}")
    print(f"  Hamming Loss:    {ensemble_default_metrics['hamming_loss']:.4f}")
    
    # Ensemble with Optimized Thresholds
    ensemble_preds_opt = np.zeros_like(ensemble_test_probs)
    for c in range(num_classes):
        ensemble_preds_opt[:, c] = (ensemble_test_probs[:, c] >= best_thresholds[c]).astype(int)
        
    opt_subset_acc = np.mean(np.all(ensemble_preds_opt == test_labels, axis=1))
    opt_hamming = np.mean(ensemble_preds_opt != test_labels)
    opt_macro_f1 = f1_score(test_labels, ensemble_preds_opt, average="macro", zero_division=0)
    opt_macro_auc = TemporalEvaluator.evaluate(test_labels, ensemble_test_probs)["macro_auc"]
    
    print(f"\nEnsemble with OPTIMIZED THRESHOLDS:")
    print(f"  Subset Accuracy: {opt_subset_acc:.4f}")
    print(f"  Macro ROC-AUC:   {opt_macro_auc:.4f}")
    print(f"  Macro F1-Score:  {opt_macro_f1:.4f}")
    print(f"  Hamming Loss:    {opt_hamming:.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()
