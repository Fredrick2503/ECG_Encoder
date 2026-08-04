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

def train_and_evaluate_optimized(args):
    # Set tracking database and experiment
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Clean up active runs
    try:
        mlflow.end_run()
    except Exception:
        pass

    # 1. Load Data
    print(f"\nLoading PTB-XL dataset ({args.resolution} resolution)...")
    train_ds, val_ds, test_ds, loader = DatasetFactory.create_datasets(
        dataset_type="ptbxl",
        download=False,
        resolution=args.resolution
    )

    if args.num_records > 0:
        print(f"Dry run / Debug mode: limiting dataset to {args.num_records} records...")
        train_ds.record_ids = train_ds.record_ids[:args.num_records]
        val_ds.record_ids = val_ds.record_ids[:max(5, int(args.num_records * 0.15))]
        test_ds.record_ids = test_ds.record_ids[:max(5, int(args.num_records * 0.15))]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    test_labels = []
    for _, lbls in test_loader:
        test_labels.append(lbls.numpy())
    test_labels = np.concatenate(test_labels, axis=0)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")

    # Start MLflow run
    with mlflow.start_run(run_name=args.run_name) as run:
        # Log parameters
        mlflow.log_params({
            "learning_rate": args.lr,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout_lstm": args.dropout_lstm,
            "dropout_fc": args.dropout_fc,
            "pretrain_strategy": args.pretrain_strategy if args.pretrain_strategy else "None",
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "resolution": args.resolution,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay
        })

        # 2. Instantiate Model
        model = ECGBiLSTM(
            input_size=12,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_classes=5
        ).to(device)

        # Set custom regularization
        model.lstm.dropout = args.dropout_lstm if args.num_layers > 1 else 0.0
        # Classification head dropout is located in model.fc[2]
        if len(model.fc) > 2 and isinstance(model.fc[2], torch.nn.Dropout):
            model.fc[2].p = args.dropout_fc

        # 3. Handle SSL Pretraining (if selected)
        if args.pretrain_strategy and args.pretrain_epochs > 0:
            print(f"\n--- Starting SSL Pretraining Phase: {args.pretrain_strategy} ({args.pretrain_epochs} epochs) ---")
            trainer_pretrain = TemporalTrainer(model, lr=args.lr, device=device)
            decoder = ECGReconstructionDecoder(
                latent_dim=args.hidden_size * 2, 
                num_leads=12, 
                signal_length=1000 if args.resolution == "lr" else 5000
            )

            if args.pretrain_strategy.lower() == "reconstruction":
                strategy = ReconstructionLearningStrategy()
            elif args.pretrain_strategy.lower() == "mae":
                strategy = MaskedAutoencoderStrategy(mask_ratio=0.3)
            elif args.pretrain_strategy.lower() == "contrastive":
                strategy = ContrastiveLearningStrategy(temperature=0.1, projection_dim=64, latent_dim=args.hidden_size * 2)
            else:
                raise ValueError(f"Unknown SSL pretraining strategy: {args.pretrain_strategy}")

            trainer_pretrain.fit(
                train_loader=train_loader,
                epochs=args.pretrain_epochs,
                is_pretraining=True,
                strategy=strategy,
                decoder=decoder
            )

        # 4. Supervised Training Loop with LR Scheduler & Early Stopping
        print(f"\n--- Starting Supervised Training / Fine-Tuning Phase ({args.finetune_epochs} epochs) ---")
        
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3, verbose=True
        )

        best_val_loss = float("inf")
        early_stopping_counter = 0
        early_stopping_patience = 7
        best_model_state = None

        for epoch in range(1, args.finetune_epochs + 1):
            # Train Epoch
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
            epoch_train_loss = train_loss / len(train_ds)

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
                    
            epoch_val_loss = val_loss / len(val_ds)
            
            # Step the scheduler
            scheduler.step(epoch_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Compute Epoch Metrics
            val_preds = np.concatenate(val_preds, axis=0)
            val_targets = np.concatenate(val_targets, axis=0)
            metrics = TemporalEvaluator.evaluate(val_targets, val_preds)

            print(f"Epoch {epoch}/{args.finetune_epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f} - Val Acc: {metrics['subset_accuracy']:.4f} - LR: {current_lr:.6f}")

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)
            for k, v in metrics.items():
                if not np.isnan(v):
                    mlflow.log_metric(f"val_{k}", v, step=epoch)

            # Early Stopping and Checkpointing Check
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

        # 5. Final Evaluation on Test Set
        print("\nEvaluating best model on test set...")
        predictor = TemporalPredictor(model, device=device)
        test_probs = predictor.predict_proba(test_loader)
        test_metrics = TemporalEvaluator.evaluate(test_labels, test_probs)

        print("\n==================================================")
        print("FINAL TEST METRICS:")
        print("==================================================")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
            if not np.isnan(v):
                mlflow.log_metric(f"test_{k}", v)
        print("==================================================")

        # Log PyTorch Model
        mlflow.pytorch.log_model(model, artifact_path="model", serialization_format="pickle")
        print("Model successfully logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Temporal Encoder Optimized Trainer")
    parser.add_argument("--tracking_uri", type=str, default="sqlite:///mlflow.db", help="MLflow tracking URI")
    parser.add_argument("--experiment_name", type=str, default="ECG_TemporalEncoder_Optimized", help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default="ecg_blstm_optimized_run", help="MLflow run name")
    parser.add_argument("--resolution", type=str, default="lr", help="ECG resolution (lr=100Hz or hr=500Hz)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--hidden_size", type=int, default=256, help="LSTM hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM layers")
    parser.add_argument("--dropout_lstm", type=float, default=0.4, help="LSTM dropout rate")
    parser.add_argument("--dropout_fc", type=float, default=0.5, help="Classification head dropout rate")
    parser.add_argument("--pretrain_strategy", type=str, default="mae", help="SSL pretraining strategy")
    parser.add_argument("--pretrain_epochs", type=int, default=5, help="Number of pretraining epochs")
    parser.add_argument("--finetune_epochs", type=int, default=15, help="Number of fine-tuning epochs")
    parser.add_argument("--num_records", type=int, default=-1, help="Limit number of records loaded (default -1 loads all)")

    args = parser.parse_args()
    train_and_evaluate_optimized(args)
