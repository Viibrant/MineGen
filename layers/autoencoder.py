import torch
import torch.nn as nn
import pytorch_lightning as pl
from layers import conv_block, deconv_block
from typing import Optional


class VAE(pl.LightningModule):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.conv1 = nn.Conv3d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 32, 3, stride=2, padding=0)
        self.linear1 = nn.Linear(3 * 3 * 3 * 32, 128)
        self.linear_mu = nn.Linear(128, latent_dim)
        self.linear_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 3 * 32),
            nn.ReLU(True),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.ConvTranspose3d(8, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_lin(z)
        x = x.view(x.size(0), 32, 3, 3, 3)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, mu, logvar = self(x)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss
