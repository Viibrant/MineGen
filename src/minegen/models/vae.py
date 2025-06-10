"""VAE model implementation from notebooks/vae.ipynb"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from einops.layers.torch import Rearrange
from typing import Tuple, Optional


class VAE(LightningModule):
    """Variational Autoencoder for 3D schematic generation with category prediction."""

    def __init__(
        self,
        latent_dim: int = 64,
        embedding_size: int = 512,
        num_blocks: int = 512,
        num_categories: int = 20,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.latent_dim = latent_dim
        self.embedding_size = embedding_size
        self.num_blocks = num_blocks
        self.num_categories = num_categories
        self.learning_rate = learning_rate
        self.beta = beta

        # Block embedding
        self.block_embedding = nn.Embedding(
            num_blocks, embedding_size, padding_idx=0, scale_grad_by_freq=True
        )

        # Encoder
        self.encoder = nn.Sequential(
            Rearrange("b d h w -> b (d h w)"),
            nn.Linear(16 * 16 * 16, hidden_dim * 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Category classifier
        self.category_classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_categories),
        )

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, 16 * 16 * 16 * num_blocks),
            Rearrange("b (d h w c) -> b c d h w", d=16, h=16, w=16, c=num_blocks),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space."""
        # Embed blocks
        x_embedded = self.block_embedding(x.long())
        x_embedded = x_embedded.mean(dim=-1)  # Average embeddings
        
        # Encode
        h = self.encoder(x_embedded)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        h = self.decoder_input(z)
        h = F.relu(h)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        category_logits = self.category_classifier(z)
        return recon_x, category_logits, mu, logvar

    def loss_function(
        self, 
        recon_x: torch.Tensor, 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor,
        category_logits: torch.Tensor,
        categories: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss."""
        # Reconstruction loss
        recon_loss = F.cross_entropy(recon_x, x.long(), reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Category classification loss
        category_loss = F.binary_cross_entropy_with_logits(category_logits, categories)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss + category_loss
        
        return total_loss, recon_loss, kl_loss, category_loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, categories = batch
        recon_x, category_logits, mu, logvar = self(x)
        
        total_loss, recon_loss, kl_loss, category_loss = self.loss_function(
            recon_x, x, mu, logvar, category_logits, categories
        )
        
        self.log('train/total_loss', total_loss, prog_bar=True)
        self.log('train/recon_loss', recon_loss)
        self.log('train/kl_loss', kl_loss)
        self.log('train/category_loss', category_loss)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, categories = batch
        recon_x, category_logits, mu, logvar = self(x)
        
        total_loss, recon_loss, kl_loss, category_loss = self.loss_function(
            recon_x, x, mu, logvar, category_logits, categories
        )
        
        self.log('val/total_loss', total_loss, prog_bar=True)
        self.log('val/recon_loss', recon_loss)
        self.log('val/kl_loss', kl_loss)
        self.log('val/category_loss', category_loss)
        
        # Log accuracy
        pred_categories = torch.sigmoid(category_logits) > 0.5
        accuracy = (pred_categories == categories.bool()).float().mean()
        self.log('val/accuracy', accuracy)
        
        return total_loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def generate(self, num_samples: int = 1, category: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate new samples."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            samples = self.decode(z)
            return torch.argmax(samples, dim=1)