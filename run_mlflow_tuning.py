import os
import argparse
import mlflow
import mlflow.pytorch
import torch
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def train_and_log_trial(
    lr: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    pretrain_strategy: str,
    pretrain_epochs: int,
    finetune_epochs: int,
    resolution: str,
    batch_size: int,
    run_name: str,
    num_records: int = -1
):
    """Runs a single trial: pretraining (if selected), fine-tuning, and evaluation."""
    
    # 1. Load Data
    print(f"\nLoading PTB-XL dataset ({resolution} resolution, batch_size={batch_size})...")
    train_ds, val_ds, test_ds, loader = DatasetFactory.create_datasets(
        dataset_type="ptbxl",
        download=False,  # Already verified/downloaded
        resolution=resolution
    )
    
    if num_records > 0:
        print(f"Limiting dataset size to {num_records} records...")
        train_ds.record_ids = train_ds.record_ids[:num_records]
        val_ds.record_ids = val_ds.record_ids[:max(5, int(num_records * 0.15))]
        test_ds.record_ids = test_ds.record_ids[:max(5, int(num_records * 0.15))]
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Get ground truth labels from test set
    test_labels = []
    for _, lbls in test_loader:
        test_labels.append(lbls.numpy())
    test_labels = np.concatenate(test_labels, axis=0)

    # 2. Start Nested MLflow Run
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        # Log trial hyperparameters
        mlflow.log_params({
            "learning_rate": lr,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "pretrain_strategy": pretrain_strategy if pretrain_strategy else "Supervised (None)",
            "pretrain_epochs": pretrain_epochs,
            "finetune_epochs": finetune_epochs,
            "resolution": resolution,
            "batch_size": batch_size
        })

        # 3. Instantiate model
        model = ECGBiLSTM(
            input_size=12,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=5
        ).to(device)
        
        # Override dropout inside LSTM if needed
        model.lstm.dropout = dropout if num_layers > 1 else 0.0

        # 4. Handle SSL Pretraining (if selected)
        if pretrain_strategy and pretrain_epochs > 0:
            print(f"--- Pretraining Phase: {pretrain_strategy} ({pretrain_epochs} epochs) ---")
            trainer_pretrain = TemporalTrainer(model, lr=lr, device=device)
            decoder = ECGReconstructionDecoder(latent_dim=hidden_size * 2, num_leads=12, signal_length=1000 if resolution == "lr" else 5000)
            
            if pretrain_strategy.lower() == "reconstruction":
                strategy = ReconstructionLearningStrategy()
            elif pretrain_strategy.lower() == "mae":
                strategy = MaskedAutoencoderStrategy(mask_ratio=0.3)
            elif pretrain_strategy.lower() == "contrastive":
                strategy = ContrastiveLearningStrategy(temperature=0.1, projection_dim=64, latent_dim=hidden_size * 2)
            else:
                raise ValueError(f"Unknown SSL pretraining strategy: {pretrain_strategy}")

            trainer_pretrain.fit(
                train_loader=train_loader,
                epochs=pretrain_epochs,
                is_pretraining=True,
                strategy=strategy,
                decoder=decoder
            )

        # 5. Downstream Fine-Tuning / Supervised Phase
        print(f"--- Fine-Tuning / Supervised Training Phase ({finetune_epochs} epochs) ---")
        trainer = TemporalTrainer(model, lr=lr, device=device)
        
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=finetune_epochs,
            is_pretraining=False
        )

        # 6. Evaluation
        print("Evaluating on test set...")
        predictor = TemporalPredictor(model, device=device)
        probs = predictor.predict_proba(test_loader)
        metrics = TemporalEvaluator.evaluate(test_labels, probs)
        
        print(f"Results for trial {run_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            if not np.isnan(v):
                mlflow.log_metric(k, v)

        # Log PyTorch Model artifact
        mlflow.pytorch.log_model(model, artifact_path="model", serialization_format="pickle")

        return metrics

def run_tuning_sweep(args):
    # Set tracking database and experiment
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Clean up any active runs
    try:
        mlflow.end_run()
    except Exception:
        pass

    # Define hyperparameter grid
    # If in dry-run mode, we do a minimal run
    if args.dry_run:
        lrs = [1e-3]
        hiddens = [128]
        pretrains = [None]
        finetune_eps = 1
        pretrain_eps = 0
        print("Running in DRY RUN mode (1 trial, 1 epoch)...")
    else:
        lrs = [1e-3, 5e-4, 1e-4]
        hiddens = [128, 256]
        pretrains = [None, "mae", "contrastive"]
        finetune_eps = args.finetune_epochs
        pretrain_eps = args.pretrain_epochs
        print("Starting complete hyperparameter sweep...")

    # Start Parent Run
    with mlflow.start_run(run_name="temporal_encoder_tuning_parent") as parent_run:
        mlflow.log_params({
            "sweep_type": "grid_search",
            "resolution": args.resolution,
            "batch_size": args.batch_size
        })

        best_acc = -1
        best_params = {}
        trial_num = 1

        for lr in lrs:
            for hidden in hiddens:
                for pretrain in pretrains:
                    run_name = f"trial_{trial_num}_lr_{lr}_hidden_{hidden}_ssl_{pretrain if pretrain else 'none'}"
                    print(f"\n==================================================")
                    print(f"Starting Trial {trial_num}: {run_name}")
                    print(f"==================================================")

                    try:
                        metrics = train_and_log_trial(
                            lr=lr,
                            hidden_size=hidden,
                            num_layers=2,
                            dropout=0.3,
                            pretrain_strategy=pretrain,
                            pretrain_epochs=pretrain_eps,
                            finetune_epochs=finetune_eps,
                            resolution=args.resolution,
                            batch_size=args.batch_size,
                            run_name=run_name,
                            num_records=args.num_records
                        )

                        acc = metrics["subset_accuracy"]
                        if acc > best_acc:
                            best_acc = acc
                            best_params = {
                                "lr": lr,
                                "hidden_size": hidden,
                                "pretrain_strategy": pretrain
                            }
                    except Exception as e:
                        print(f"Trial failed with exception: {e}")
                        continue

                    trial_num += 1

        # Log best overall details to parent run
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_subset_accuracy", best_acc)
        
        print("\n==================================================")
        print("Hyperparameter tuning sweep completed!")
        print(f"Best Subset Accuracy achieved: {best_acc:.4f}")
        print(f"Best Configuration: {best_params}")
        print(f"Run 'mlflow ui' on '{args.tracking_uri}' to view the results dashboard.")
        print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Temporal Encoder MLflow Hyperparameter Sweeper")
    parser.add_argument("--tracking_uri", type=str, default="sqlite:///mlflow.db", help="MLflow tracking URI")
    parser.add_argument("--experiment_name", type=str, default="ECG_TemporalEncoder_Tuning", help="MLflow experiment name")
    parser.add_argument("--resolution", type=str, default="lr", help="ECG resolution (lr=100Hz or hr=500Hz)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--pretrain_epochs", type=int, default=5, help="Number of pretraining epochs")
    parser.add_argument("--finetune_epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--num_records", type=int, default=-1, help="Number of records to load (default -1 loads all)")
    parser.add_argument("--dry_run", action="store_true", help="Run a fast 1-epoch sanity check")

    args = parser.parse_args()
    run_tuning_sweep(args)
