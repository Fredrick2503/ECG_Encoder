import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.relu(out + residual)


class AttentionMLPAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, dropout: float = 0.3, num_heads: int = 4, hidden_units: int = 128, num_classes: int = 5):
        super().__init__()
        # Encoder
        self.bn = nn.BatchNorm1d(input_dim)
        self.enc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.enc2 = nn.Linear(256, hidden_units)
        
        self.res_block = ResidualBlock(hidden_units, dropout)
        
        # Self-Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_units, num_heads=num_heads, dropout=dropout)
        
        self.enc3 = nn.Linear(hidden_units, 64)
        self.enc4 = nn.Linear(64, latent_dim)
        
        # Decoder (reconstructs only original features: input_dim // 2)
        self.dec1 = nn.Linear(latent_dim, 32)
        self.dec2 = nn.Linear(32, 64)
        self.dec3 = nn.Linear(64, hidden_units)
        self.dec4 = nn.Linear(hidden_units, 256)
        self.dec_out = nn.Linear(256, input_dim // 2)
        
        # Classification Head
        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        x = self.bn(x)
        x = self.relu(self.enc1(x))
        x = self.dropout(x)
        x = self.enc2(x)
        
        x = self.res_block(x)
        
        # Self-attention requires [SeqLen, Batch, EmbedDim]
        x_seq = x.unsqueeze(0)
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        x = attn_out.squeeze(0)
        
        x = self.relu(self.enc3(x))
        latent = self.enc4(x)
        return latent

    def decode(self, latent):
        x = self.relu(self.dec1(latent))
        x = self.relu(self.dec2(x))
        x = self.relu(self.dec3(x))
        x = self.relu(self.dec4(x))
        out = self.dec_out(x)
        return out

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        class_logits = self.classifier(latent)
        return reconstructed, latent, class_logits


class BetaVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_units: int = 128, beta: float = 1.0, num_classes: int = 5):
        super().__init__()
        self.beta = beta
        
        # Encoder
        self.bn = nn.BatchNorm1d(input_dim)
        self.enc1 = nn.Linear(input_dim, 256)
        self.enc2 = nn.Linear(256, hidden_units)
        self.enc3 = nn.Linear(hidden_units, 64)
        self.enc4 = nn.Linear(64, 32)
        
        # Mean and log variance
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder (reconstructs only original features: input_dim // 2)
        self.dec1 = nn.Linear(latent_dim, 32)
        self.dec2 = nn.Linear(32, 64)
        self.dec3 = nn.Linear(64, hidden_units)
        self.dec4 = nn.Linear(hidden_units, 256)
        self.dec_out = nn.Linear(256, input_dim // 2)
        
        # Classification Head
        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        x = self.bn(x)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        out = self.dec_out(x)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        class_logits = self.classifier(mu)  # Use mu for classification
        return reconstructed, z, mu, logvar, class_logits

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + self.beta * kld_loss
        return total_loss, recon_loss, kld_loss


class FeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_model))
        self.bias = nn.Parameter(torch.randn(num_features, d_model))
        self.num_features = num_features

    def forward(self, x):
        x_unsqueezed = x.unsqueeze(-1)  # [Batch, NumFeatures, 1]
        tokens = x_unsqueezed * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return tokens


class FTTransformerAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, d_model: int = 64, nhead: int = 4, num_layers: int = 2, ffn_dim: int = 128, dropout: float = 0.2, num_classes: int = 5):
        super().__init__()
        self.tokenizer = FeatureTokenizer(input_dim, d_model)
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.enc_fc1 = nn.Linear(d_model, 64)
        self.enc_fc2 = nn.Linear(64, latent_dim)
        self.relu = nn.ReLU()
        
        # Decoder (reconstructs only original features: input_dim // 2)
        self.dec1 = nn.Linear(latent_dim, 64)
        self.dec2 = nn.Linear(64, 128)
        self.dec3 = nn.Linear(128, 256)
        self.dec_out = nn.Linear(256, input_dim // 2)
        
        # Classification Head
        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        tokens = self.tokenizer(x)  # [Batch, NumFeatures, d_model]
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat([cls_tokens, tokens], dim=1)
        trans_out = self.transformer(x_seq)
        cls_representation = trans_out[:, 0, :]
        out = self.relu(self.enc_fc1(cls_representation))
        latent = self.enc_fc2(out)
        return latent

    def decode(self, latent):
        x = self.relu(self.dec1(latent))
        x = self.relu(self.dec2(x))
        x = self.relu(self.dec3(x))
        out = self.dec_out(x)
        return out

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        class_logits = self.classifier(latent)
        return reconstructed, latent, class_logits
