import optuna
import torch
import logging
from biomarker_encoder.models import AttentionMLPAutoencoder, BetaVAE, FTTransformerAutoencoder
from biomarker_encoder.trainer import BiomarkerTrainer

logger = logging.getLogger("BiomarkerTuning")

def run_optuna_study(model_type: str, input_dim: int, train_loader, val_loader, n_trials: int = 10, epochs: int = 15):
    """
    Tune hyperparameters for the specified model_type using Optuna.
    model_type can be: 'attention_mlp', 'beta_vae', or 'ft_transformer'
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        # Common parameters
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        latent_dim = trial.suggest_int("latent_dim", 16, 64, step=16)
        
        # Architecture parameters
        if model_type == "attention_mlp":
            hidden_units = trial.suggest_int("hidden_units", 64, 256, step=64)
            dropout = trial.suggest_float("dropout", 0.1, 0.4)
            num_heads = trial.suggest_categorical("num_heads", [2, 4])
            
            # MultiheadAttention requires hidden_units to be divisible by num_heads
            # Adjust hidden_units if not divisible
            if hidden_units % num_heads != 0:
                hidden_units = (hidden_units // num_heads) * num_heads
                if hidden_units == 0:
                    hidden_units = num_heads
            
            model = AttentionMLPAutoencoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                dropout=dropout,
                num_heads=num_heads,
                hidden_units=hidden_units
            )
            
        elif model_type == "beta_vae":
            hidden_units = trial.suggest_int("hidden_units", 64, 256, step=64)
            beta = trial.suggest_float("beta", 0.1, 5.0)
            model = BetaVAE(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_units=hidden_units,
                beta=beta
            )
            
        elif model_type == "ft_transformer":
            d_model = trial.suggest_categorical("d_model", [32, 64])
            nhead = trial.suggest_categorical("nhead", [2, 4])
            num_layers = trial.suggest_int("num_layers", 2, 3)
            ffn_dim = trial.suggest_int("ffn_dim", 64, 128, step=32)
            dropout = trial.suggest_float("dropout", 0.1, 0.3)
            
            model = FTTransformerAutoencoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                ffn_dim=ffn_dim,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        # Instantiate trainer
        trainer = BiomarkerTrainer(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            patience=5,
            checkpoint_path=f"temp_best_{model_type}.pt",
            mixed_precision=True
        )
        
        # Fit model on training split and return validation loss
        try:
            _, _, val_history = trainer.fit(train_loader, val_loader, epochs=epochs)
            val_loss = min(val_history)
            return val_loss
        except Exception as e:
            logger.error(f"Error during Optuna trial: {e}")
            return float("inf")

    logger.info(f"Starting Optuna search for {model_type}...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Best trial value for {model_type}: {study.best_value}")
    logger.info(f"Best params for {model_type}: {study.best_params}")
    return study.best_params
