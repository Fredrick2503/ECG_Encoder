import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import argparse
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader

from data_management.dataset_factory import DatasetFactory
from temporal_encoder.encoder_upgrades import ECGTransformer
from temporal_encoder.predictor import TemporalPredictor
from temporal_encoder.evaluator import TemporalEvaluator
from utils.losses import FocalLoss, AsymmetricLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def train_final_transformer(args):
    # Set tracking database and experiment
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Clean up active runs
    try:
        mlflow.end_run()
    except Exception:
        pass

    # 1. Load Data
    print(f"\nLoading the FULL PTB-XL dataset ({args.resolution} resolution)...")
    train_ds, val_ds, test_ds, loader = DatasetFactory.create_datasets(
        dataset_type="ptbxl",
        download=False,
        resolution=args.resolution
    )

    print(f"Loaded records: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # Reduce batch size and set num_workers to 0 to keep system load extremely light
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    test_labels = []
    for _, lbls in test_loader:
        test_labels.append(lbls.numpy())
    test_labels = np.concatenate(test_labels, axis=0)

    # 2. Instantiate Model with Trial 3 Optimal Parameters
    model = ECGTransformer(
        input_size=12,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_classes=5
    ).to(device)

    # Start MLflow run
    run_name = f"final_transformer_full_dataset_bs{args.batch_size}_epochs{args.epochs}"
    print(f"\n==================================================")
    print(f"Starting Final Run: {run_name}")
    print(f"Targeting parameters from Trial 3: d_model={args.d_model}, nhead={args.nhead}, layers={args.num_layers}, lr={args.lr}")
    print(f"==================================================")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "resolution": args.resolution,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "loss_type": args.loss_type
        })

        if args.loss_type == "focal":
            criterion = FocalLoss()
        elif args.loss_type == "asl":
            criterion = AsymmetricLoss()
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=3, verbose=True
        )

        best_val_loss = float("inf")
        best_model_state = None
        early_stopping_counter = 0
        early_stopping_patience = 8

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            for signals, labels in train_loader:
                signals, labels = signals.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(signals)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * signals.size(0)
            epoch_train_loss = train_loss / len(train_loader.dataset)

            # Validate Epoch
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for signals, labels in val_loader:
                    signals, labels = signals.to(device), labels.to(device)
                    logits = model(signals)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * signals.size(0)
                    
                    probs = torch.sigmoid(logits)
                    val_preds.append(probs.cpu().numpy())
                    val_targets.append(labels.cpu().numpy())
                    
            epoch_val_loss = val_loss / len(val_loader.dataset)
            scheduler.step(epoch_val_loss)

            val_preds = np.concatenate(val_preds, axis=0)
            val_targets = np.concatenate(val_targets, axis=0)
            val_metrics = TemporalEvaluator.evaluate(val_targets, val_preds)

            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            for k, v in val_metrics.items():
                if not np.isnan(v):
                    mlflow.log_metric(f"val_{k}", v, step=epoch)
            
            print(f"Epoch {epoch}/{args.epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f} - Val Subset Acc: {val_metrics['subset_accuracy']:.4f}")

            # Checkpoint & Early Stopping Check
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                early_stopping_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print("  => Validation loss improved. Saving checkpoint.")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}. Restoring best checkpoint.")
                    break

        # Restore best model weights
        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

        # Final Evaluation on Test Set
        print("\nEvaluating best model on test set...")
        predictor = TemporalPredictor(model, device=device)
        test_probs = predictor.predict_proba(test_loader)
        metrics = TemporalEvaluator.evaluate(test_labels, test_probs)

        print("\n==================================================")
        print("FINAL TEST METRICS:")
        print("==================================================")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            if not np.isnan(v):
                mlflow.log_metric(f"test_{k}", v)
        print("==================================================")

        # Log model
        mlflow.pytorch.log_model(model, artifact_path="model", serialization_format="pickle")
        print("Model successfully logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final ECG Transformer Training on Full Dataset")
    parser.add_argument("--tracking_uri", type=str, default="file:///c:/Users/fredr/OneDrive/Desktop/Projects/ECG_Encoder/mlruns", help="MLflow tracking URI")
    parser.add_argument("--experiment_name", type=str, default="ECG_Transformer_Final", help="MLflow experiment name")
    parser.add_argument("--resolution", type=str, default="lr", help="ECG resolution (lr or hr)")
    parser.add_argument("--batch_size", type=int, default=16, help="Reduced batch size to keep GPU usage light")
    parser.add_argument("--num_workers", type=int, default=0, help="0 workers avoids CPU multiprocessing overload")
    
    # Trial 3 Parameters
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=1e-05)
    parser.add_argument("--epochs", type=int, default=40, help="Higher training budget with early stopping")
    parser.add_argument("--loss_type", type=str, default="bce", choices=["bce", "focal", "asl"], help="Loss function type")

    args = parser.parse_args()
    train_final_transformer(args)
