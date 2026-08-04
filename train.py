import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from data_management.dataset_factory import DatasetFactory
from temporal_encoder.encoder import ECGBiLSTM, ECGReconstructionDecoder
from temporal_encoder.strategies import (
    ReconstructionLearningStrategy,
    MaskedAutoencoderStrategy,
    ContrastiveLearningStrategy
)
from temporal_encoder.trainer import TemporalTrainer
from temporal_encoder.predictor import TemporalPredictor
from temporal_encoder.evaluator import TemporalEvaluator

def run_experiment(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    test_labels: np.ndarray,
    pretrain_epochs: int = 5,
    finetune_epochs: int = 5,
    device: str = "cpu"
):
    results = {}
    
    # ----------------------------------------------------
    # 1. Pure Supervised Baseline
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("Running Experiment 1: Pure Supervised Baseline")
    print("="*50)
    model_baseline = ECGBiLSTM(input_size=12, hidden_size=128, num_layers=2, num_classes=5)
    trainer_baseline = TemporalTrainer(model_baseline, lr=1e-3, device=device)
    
    trainer_baseline.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=finetune_epochs,
        is_pretraining=False
    )
    
    predictor_baseline = TemporalPredictor(model_baseline, device=device)
    probs_baseline = predictor_baseline.predict_proba(test_loader)
    metrics_baseline = TemporalEvaluator.evaluate(test_labels, probs_baseline)
    results["Supervised Baseline"] = metrics_baseline
    
    # ----------------------------------------------------
    # 2. Reconstruction Pretraining + Fine-tuning
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("Running Experiment 2: Reconstruction SSL Pretraining + Fine-Tuning")
    print("="*50)
    model_recon = ECGBiLSTM(input_size=12, hidden_size=128, num_layers=2, num_classes=5)
    decoder_recon = ECGReconstructionDecoder(latent_dim=256, num_leads=12, signal_length=1000)
    trainer_recon = TemporalTrainer(model_recon, lr=1e-3, device=device)
    recon_strategy = ReconstructionLearningStrategy()
    
    print("Pretraining encoder via Reconstruction...")
    trainer_recon.fit(
        train_loader=train_loader,
        epochs=pretrain_epochs,
        is_pretraining=True,
        strategy=recon_strategy,
        decoder=decoder_recon
    )
    
    print("Fine-tuning pretrained Reconstruction model...")
    trainer_recon.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=finetune_epochs,
        is_pretraining=False
    )
    
    predictor_recon = TemporalPredictor(model_recon, device=device)
    probs_recon = predictor_recon.predict_proba(test_loader)
    metrics_recon = TemporalEvaluator.evaluate(test_labels, probs_recon)
    results["Reconstruction SSL"] = metrics_recon

    # ----------------------------------------------------
    # 3. MAE Pretraining + Fine-tuning
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("Running Experiment 3: Masked Autoencoder (MAE) SSL Pretraining + Fine-Tuning")
    print("="*50)
    model_mae = ECGBiLSTM(input_size=12, hidden_size=128, num_layers=2, num_classes=5)
    decoder_mae = ECGReconstructionDecoder(latent_dim=256, num_leads=12, signal_length=1000)
    trainer_mae = TemporalTrainer(model_mae, lr=1e-3, device=device)
    mae_strategy = MaskedAutoencoderStrategy(mask_ratio=0.3)
    
    print("Pretraining encoder via MAE...")
    trainer_mae.fit(
        train_loader=train_loader,
        epochs=pretrain_epochs,
        is_pretraining=True,
        strategy=mae_strategy,
        decoder=decoder_mae
    )
    
    print("Fine-tuning pretrained MAE model...")
    trainer_mae.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=finetune_epochs,
        is_pretraining=False
    )
    
    predictor_mae = TemporalPredictor(model_mae, device=device)
    probs_mae = predictor_mae.predict_proba(test_loader)
    metrics_mae = TemporalEvaluator.evaluate(test_labels, probs_mae)
    results["Masked Autoencoder (MAE)"] = metrics_mae

    # ----------------------------------------------------
    # 4. Contrastive Pretraining + Fine-tuning
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("Running Experiment 4: Contrastive Learning (SimCLR) SSL Pretraining + Fine-Tuning")
    print("="*50)
    model_contrastive = ECGBiLSTM(input_size=12, hidden_size=128, num_layers=2, num_classes=5)
    trainer_contrastive = TemporalTrainer(model_contrastive, lr=1e-3, device=device)
    contrastive_strategy = ContrastiveLearningStrategy(temperature=0.1, projection_dim=64, latent_dim=256)
    
    print("Pretraining encoder via Contrastive Learning...")
    trainer_contrastive.fit(
        train_loader=train_loader,
        epochs=pretrain_epochs,
        is_pretraining=True,
        strategy=contrastive_strategy
    )
    
    print("Fine-tuning pretrained Contrastive model...")
    trainer_contrastive.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=finetune_epochs,
        is_pretraining=False
    )
    
    predictor_contrastive = TemporalPredictor(model_contrastive, device=device)
    probs_contrastive = predictor_contrastive.predict_proba(test_loader)
    metrics_contrastive = TemporalEvaluator.evaluate(test_labels, probs_contrastive)
    results["Contrastive SSL"] = metrics_contrastive

    # ----------------------------------------------------
    # Print Comparison Table
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("EXPERIMENT BENCHMARK SUMMARY")
    print("="*50)
    headers = ["Model / Strategy", "Subset Accuracy", "Hamming Loss", "Macro F1", "Macro AUC"]
    print(f"| {' | '.join(headers)} |")
    print(f"| {' | '.join(['---' for _ in headers])} |")
    for name, metrics in results.items():
        print(f"| {name} "
              f"| {metrics['subset_accuracy']:.4f} "
              f"| {metrics['hamming_loss']:.4f} "
              f"| {metrics['macro_f1']:.4f} "
              f"| {metrics['macro_auc']:.4f} |")
    print("="*50)
    
    return results

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Data (Low resolution for speed/resource efficiency)
    print("Loading PTB-XL dataset (100Hz resolution)...")
    train_loader, val_loader, test_loader, loader = DatasetFactory.create_dataloaders(
        dataset_type="ptbxl",
        download=True,
        resolution="lr",
        batch_size=16
    )
    
    # 2. Extract ground truth labels for evaluation
    test_labels = []
    for _, lbls in test_loader:
        test_labels.append(lbls.numpy())
    test_labels = np.concatenate(test_labels, axis=0)
    
    print(f"Data loading complete. Test set labels shape: {test_labels.shape}")
    
    # Check if we are running in lightweight / toy mode (e.g. only 1 record)
    # If so, we decrease epochs or run smaller iterations to complete quickly
    is_lightweight = len(test_loader.dataset) <= 1
    pretrain_eps = 2 if is_lightweight else 5
    finetune_eps = 2 if is_lightweight else 10
    
    if is_lightweight:
        print("Lightweight dataset detected. Running with minimal epochs (2 pretrain, 2 finetune) for verification.")
    else:
        print("Full dataset detected. Running benchmark (5 pretrain, 10 finetune).")
        
    run_experiment(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_labels=test_labels,
        pretrain_epochs=pretrain_eps,
        finetune_epochs=finetune_eps,
        device=device
    )
