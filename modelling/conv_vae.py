"""
Convolutional Autoencoder for voxel data, with a classification head. 
Uses a 3D convolutional encoder, followed by a 1D linear layer, followed by a 3D convolutional decoder.
"""

from torch import nn
import lightning.pytorch as pl


class AutoEncoder(nn.Module):
    def _conv_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.MaxPool3d(2),
        )

    def _linear_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(inplace=True),
        )

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            self._conv_layer(1, 64, 3, 2, 1),
            self._conv_layer(64, 128, 3, 2, 1),  # 128, 8, 8, 8
            nn.Flatten(),
            nn.Linear(65536, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
        )

        self.classifier = nn.Sequential(nn.Linear(128, 19), nn.Softmax(dim=1))

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
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        y_hat = self.classifier(encoded)
        decoded = self.decoder(encoded)
        return y_hat, decoded

class VAE(pl.LightningModule):
    def __init__(self, latent_dim, embedding_size, num_blocks=512, num_categories=20, hidden_dim=128):
        super().__init__()
        self.save_hyperparameters()
        self.embedding_size = embedding_size
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.embedding = nn.Embedding(num_blocks, embedding_size)
        self.encoder = nn.Sequential(
            nn.Conv3d(embedding_size, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.GELU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 1 * 1 * 1, self.hidden_dim),
            nn.GELU(),
        )
        self.linear_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(self.hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            Rearrange("b (d c h w) -> b c d h w", c=128, d=1, h=1, w=1),
            ResidualBlock(128, 32, 1, stride=1),
            ResidualBlock(32, 16, 3, stride=3),
            nn.ConvTranspose3d(16, 8, 3, stride=3, padding=1),
            nn.BatchNorm3d(8),
            nn.GELU(),
            nn.ConvTranspose3d(
                8, embedding_size, 3, stride=2, padding=0, output_padding=1
            ),
            nn.BatchNorm3d(embedding_size)
        )

        # self.category_predictor = nn.Sequential(
        #     nn.Linear(latent_dim, num_categories),
        #     nn.BatchNorm1d(num_categories),
        #     nn.GELU(),
        #     nn.Linear(num_categories, num_categories * 2),
        #     nn.BatchNorm1d(num_categories * 2),
        #     nn.GELU(),
        #     nn.Linear(num_categories * 2, num_categories),
        # )
        
        self.category_predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.BatchNorm1d(latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, num_categories),
            nn.BatchNorm1d(num_categories),
            nn.GELU(),
            nn.Linear(num_categories, num_categories),
            nn.BatchNorm1d(num_categories),
            nn.GELU(),
        )

    def encode(self, x):
        x = self.embedding(x)
        x = x.permute(
            0, 4, 1, 2, 3
        )  # Move the embedding dimension to the channel dimension
        x = self.encoder(x)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder(z)
        # x = x.permute(0, 2, 3, 4, 1)  # Move the embedding dimension back to the end
        return x

    def forward(self, x):
        x_mu, logvar = self.encode(x)
        z = self.reparameterize(x_mu, logvar)
        recon_x = self.decode(z)
        category_logits = self.category_predictor(z)
        return recon_x, category_logits, x_mu, logvar


    def training_step(self, batch, batch_idx):
        x, y = batch
        recon_x, category_logits, x_mu, logvar = self(x)
        # recon_loss = F.cross_entropy(recon_x.view(-1, recon_x.shape[-1]), x.view(-1), reduction='sum')
        recon_loss = F.cross_entropy(recon_x, x)
        category_loss = F.cross_entropy(category_logits, y)
        kld_loss = -0.5 * torch.sum(1 + logvar - x_mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss + category_loss
        self.log("train/train_loss", loss)
        self.log("train/recon_loss", recon_loss)
        self.log("train/kld_loss", kld_loss)
        self.log("train/category_loss", category_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        recon_x, category_logits, x_mu, logvar = self(x)
        # recon_loss = F.cross_entropy(recon_x.view(-1, recon_x.shape[-1]), x.view(-1), reduction='sum')
        recon_loss = F.cross_entropy(recon_x, x)
        category_loss = F.cross_entropy(category_logits, y)
        kld_loss = -0.5 * torch.sum(1 + logvar - x_mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss + category_loss
        self.log("val/val_loss", loss)
        self.log("val/recon_loss", recon_loss)
        self.log("val/kld_loss", kld_loss)
        self.log("val/category_loss", category_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer