from torch import nn
from typing import Type


class ResidualBlock(nn.Module):
    """3D Residual block with configurable activation and normalization."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        activation: Type[nn.Module] = nn.GELU,
        norm: Type[nn.Module] = nn.BatchNorm3d,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            norm(out_channels),
            activation(),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size, bias=False),
            norm(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            ),
            norm(out_channels),
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)