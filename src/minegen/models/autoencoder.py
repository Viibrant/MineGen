"""AutoEncoder models from notebooks/train_embeddings.ipynb"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from typing import Tuple


class AutoEncoder(LightningModule):
    """Basic AutoEncoder for schematic reconstruction and classification."""

    def __init__(
        self,
        num_categories: int = 19,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_categories = num_categories

        # Encoder
        self.encoder = nn.Sequential(
            self._conv_layer(1, 64, 3, 2, 1),
            self._conv_layer(64, 128, 3, 2, 1),  # 128, 8, 8, 8
            nn.Flatten(),
            nn.Linear(65536, 128),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(128),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, num_categories),
            nn.Softmax(dim=1)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 65536),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(65536),
            nn.Unflatten(1, (128, 8, 8, 8)),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(32, 1, kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(1),
        )

    def _conv_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.MaxPool3d(2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        encoded = self.encoder(x)
        y_hat = self.classifier(encoded)
        decoded = self.decoder(encoded)
        return y_hat, decoded

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, target = batch
        x = x.unsqueeze(1)  # Add channel dimension
        
        y_hat, decoded = self(x)
        
        clf_loss = F.cross_entropy(y_hat, target.squeeze(1))
        rec_loss = F.mse_loss(decoded, x)
        
        total_loss = clf_loss + rec_loss
        
        self.log('train/clf_loss', clf_loss)
        self.log('train/rec_loss', rec_loss)
        self.log('train/total_loss', total_loss, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, target = batch
        x = x.unsqueeze(1)
        
        y_hat, decoded = self(x)
        
        clf_loss = F.cross_entropy(y_hat, target.squeeze(1))
        rec_loss = F.mse_loss(decoded, x)
        total_loss = clf_loss + rec_loss
        
        # Calculate accuracy
        pred = torch.argmax(y_hat, dim=1)
        target_idx = torch.argmax(target.squeeze(1), dim=1)
        accuracy = (pred == target_idx).float().mean()
        
        self.log('val/clf_loss', clf_loss)
        self.log('val/rec_loss', rec_loss)
        self.log('val/total_loss', total_loss, prog_bar=True)
        self.log('val/accuracy', accuracy)
        
        return total_loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)


class VoxelAutoencoder(LightningModule):
    """Voxel-based autoencoder for 3D data."""

    def __init__(
        self,
        embedding_dim: int = 128,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, embedding_dim, kernel_size=3, stride=2, padding=1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embedding_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def training_step(self, batch, batch_idx):
        """Training step."""
        x = batch.unsqueeze(1).float() / 255.0
        x_hat, z = self(x)
        loss = F.mse_loss(x_hat, x)
        
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x = batch.unsqueeze(1).float() / 255.0
        x_hat, z = self(x)
        loss = F.mse_loss(x_hat, x)
        
        self.log('val/loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class VoxelEmbedding(nn.Module):
    """Voxel embedding layer for block ID embeddings."""

    def __init__(self, num_embeddings: int = 256, embedding_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Reshape input to be 1D tensor
        original_shape = x.shape
        x = x.view(-1)
        
        # Embed the input
        x = self.embedding(x)
        
        # Reshape output to original spatial dimensions + embedding dim
        x = x.view(*original_shape, -1)
        return x