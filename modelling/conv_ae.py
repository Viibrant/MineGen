"""
Convolutional Autoencoder for voxel data, with a classification head. 
Uses a 3D convolutional encoder, followed by a 1D linear layer, followed by a 3D convolutional decoder.
"""

from torch import nn


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
