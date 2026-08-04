import torch
import torch.nn as nn

class ECGBiLSTM(nn.Module):
    """
    Bidirectional LSTM neural network for learning temporal representations of ECG signals.
    Supports downstream multi-label supervised classification and self-supervised pretraining.
    """
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5
    ):
        """
        Args:
            input_size: Number of input channels/leads (default 12).
            hidden_size: Hidden dimension size of LSTM.
            num_layers: Number of LSTM layers.
            num_classes: Number of target prediction classes.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the latent representation embedding (z) from the ECG signal.
        
        Args:
            x: Input tensor of shape (batch_size, num_leads, signal_length)
            
        Returns:
            torch.Tensor: Latent representation of shape (batch_size, hidden_size * 2)
        """
        # Transpose from (batch, leads, length) to (batch, length, leads) for LSTM recurrent steps
        x_transposed = x.transpose(1, 2)
        
        # LSTM forward pass
        output, (hidden, cell) = self.lstm(x_transposed)
        
        # hidden shape: (num_layers * num_directions, batch_size, hidden_size)
        # Concatenate the final forward direction hidden state (index -2) and backward direction (index -1)
        z = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            x: Input tensor of shape (batch_size, num_leads, signal_length)
            
        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes)
        """
        z = self.get_representation(x)
        return self.fc(z)


class ECGReconstructionDecoder(nn.Module):
    """
    Lightweight decoder used during Reconstruction Learning and Masked Autoencoder tasks
    to reconstruct the original 12-lead ECG signal from the latent embedding (z).
    """
    def __init__(self, latent_dim: int, num_leads: int = 12, signal_length: int = 1000):
        """
        Args:
            latent_dim: Dimension of latent embedding input (usually hidden_size * 2).
            num_leads: Number of output ECG leads.
            signal_length: Output time series length.
        """
        super().__init__()
        self.num_leads = num_leads
        self.signal_length = signal_length
        
        # MLP architecture to project latent dimensions back to full signal space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_leads * signal_length)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent vector back into a reconstructed signal.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            torch.Tensor: Reconstructed signal of shape (batch_size, num_leads, signal_length)
        """
        flat_reconstruction = self.decoder(z)
        return flat_reconstruction.view(-1, self.num_leads, self.signal_length)
