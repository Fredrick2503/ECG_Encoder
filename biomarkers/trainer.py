import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import logging

logger = logging.getLogger("BiomarkerTrainer")

class BiomarkerTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
        checkpoint_path: str = "best_model.pt",
        mixed_precision: bool = True
    ):
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.mixed_precision = mixed_precision and (self.device.type == "cuda")
        
        self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            self.optimizer.zero_grad()
            
            N = batch_x.size(1) // 2
            orig_x = batch_x[:, :N]
            
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                if hasattr(self.model, "loss_function"):  # VAE
                    reconstructed, latent, mu, logvar, class_logits = self.model(batch_x)
                    vae_loss, recon_loss, kld_loss = self.model.loss_function(reconstructed, orig_x, mu, logvar)
                    class_loss = nn.functional.binary_cross_entropy_with_logits(class_logits, batch_y)
                    loss = vae_loss + class_loss
                else:  # Standard Autoencoders
                    reconstructed, latent, class_logits = self.model(batch_x)
                    recon_loss = nn.functional.mse_loss(reconstructed, orig_x)
                    class_loss = nn.functional.binary_cross_entropy_with_logits(class_logits, batch_y)
                    loss = recon_loss + class_loss
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * batch_x.size(0)
            
        return total_loss / len(dataloader.dataset)

    def val_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                N = batch_x.size(1) // 2
                orig_x = batch_x[:, :N]
                
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    if hasattr(self.model, "loss_function"):  # VAE
                        reconstructed, latent, mu, logvar, class_logits = self.model(batch_x)
                        vae_loss, recon_loss, kld_loss = self.model.loss_function(reconstructed, orig_x, mu, logvar)
                        class_loss = nn.functional.binary_cross_entropy_with_logits(class_logits, batch_y)
                        loss = vae_loss + class_loss
                    else:  # Standard Autoencoders
                        reconstructed, latent, class_logits = self.model(batch_x)
                        recon_loss = nn.functional.mse_loss(reconstructed, orig_x)
                        class_loss = nn.functional.binary_cross_entropy_with_logits(class_logits, batch_y)
                        loss = recon_loss + class_loss
                
                total_loss += loss.item() * batch_x.size(0)
                
        return total_loss / len(dataloader.dataset)

    def fit(self, train_loader, val_loader, epochs: int = 100):
        best_loss = float("inf")
        best_model_wts = copy.deepcopy(self.model.state_dict())
        patience_counter = 0
        
        train_history = []
        val_history = []
        
        logger.info(f"Training started on device: {self.device}")
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.val_epoch(val_loader)
            
            train_history.append(train_loss)
            val_history.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                os.makedirs(os.path.dirname(os.path.abspath(self.checkpoint_path)), exist_ok=True)
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                    
        self.model.load_state_dict(best_model_wts)
        return self.model, train_history, val_history
