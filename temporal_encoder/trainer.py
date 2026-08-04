import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from temporal_encoder.strategies import BaseSSLStrategy

class TemporalTrainer:
    """
    Handles model training and validation loops for both supervised classification 
    and self-supervised pretraining tasks.
    """
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: str = "cpu"
    ):
        """
        Args:
            model: The neural network encoder model.
            lr: Learning rate.
            device: Training device ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.supervised_criterion = nn.BCEWithLogitsLoss()  # Multi-label binary cross-entropy

    def train_supervised_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Runs a single epoch of supervised training."""
        self.model.train()
        total_loss = 0.0
        
        for signals, labels in dataloader:
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(signals)
            loss = self.supervised_criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * signals.size(0)
            
        return total_loss / len(dataloader.dataset)

    def validate_supervised(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluates loss on a validation set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for signals, labels in dataloader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(signals)
                loss = self.supervised_criterion(logits, labels)
                total_loss += loss.item() * signals.size(0)
                
        return total_loss / len(dataloader.dataset)

    def train_pretrain_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        strategy: BaseSSLStrategy,
        decoder: Optional[nn.Module] = None
    ) -> float:
        """Runs a single epoch of self-supervised pretraining."""
        self.model.train()
        if decoder is not None:
            decoder.train()
            decoder = decoder.to(self.device)
            
        total_loss = 0.0
        
        # Track both encoder and decoder parameters
        params = list(self.model.parameters())
        if decoder is not None:
            params += list(decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.optimizer.param_groups[0]['lr'])
        
        for batch in dataloader:
            # Dataloader can return (signal, label) or just signals.
            # Handle both gracefully.
            if isinstance(batch, (list, tuple)):
                signals = batch[0]
            else:
                signals = batch
                
            signals = signals.to(self.device)
            
            optimizer.zero_grad()
            loss = strategy.compute_loss(self.model, decoder, signals)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * signals.size(0)
            
        return total_loss / len(dataloader.dataset)

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 5,
        is_pretraining: bool = False,
        strategy: Optional[BaseSSLStrategy] = None,
        decoder: Optional[nn.Module] = None
    ) -> Dict[str, list]:
        """
        Executes the complete training loop.
        
        Returns:
            Dict[str, list]: History log of losses.
        """
        history = {"train_loss": [], "val_loss": []}
        
        # Check if MLflow tracking is active
        try:
            import mlflow
            mlflow_active = mlflow.active_run() is not None
        except ImportError:
            mlflow_active = False
            
        for epoch in range(1, epochs + 1):
            if is_pretraining:
                if strategy is None:
                    raise ValueError("Pretraining requires an SSL strategy.")
                loss = self.train_pretrain_epoch(train_loader, strategy, decoder)
                print(f"Epoch {epoch}/{epochs} [Pretraining] - Loss: {loss:.4f}")
                history["train_loss"].append(loss)
                if mlflow_active:
                    mlflow.log_metric("pretrain_loss", loss, step=epoch)
            else:
                train_loss = self.train_supervised_epoch(train_loader)
                val_loss = self.validate_supervised(val_loader) if val_loader is not None else 0.0
                print(f"Epoch {epoch}/{epochs} [Supervised] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                if mlflow_active:
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                
        return history

