import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Optional

from .attention import TransformerLayer
from .blocks import RearrangeModule

try:
    from positional_encodings.torch_encodings import PositionalEncoding3D, Summer
    HAS_POSITIONAL_ENCODINGS = True
except ImportError:
    HAS_POSITIONAL_ENCODINGS = False


class NestedTransformer(nn.Module):
    """Nested transformer for hierarchical processing of 3D data."""

    def __init__(
        self, 
        patch_size: int, 
        embed_dim: int, 
        num_heads: int, 
        num_layers: int, 
        in_channels: Optional[int] = None
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels or embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.pool = nn.MaxPool3d(2)
        
        # Use positional encoding if available, otherwise identity
        if HAS_POSITIONAL_ENCODINGS:
            self.positional = Summer(PositionalEncoding3D(embed_dim))
        else:
            self.positional = nn.Identity()
            
        self.transformer_layers = nn.Sequential(
            *[TransformerLayer(self.in_channels, num_heads) for _ in range(num_layers)]
        )
        self.conv = nn.LazyConv3d(embed_dim, 1)

        self.model_patch = nn.Sequential(
            self.pool,
            RearrangeModule(self.patch_size),
            Rearrange("b c p1 p2 p3 s1 s2 s3 -> b (p1 s1) (p2 s2) (p3 s3) c"),
            self.positional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assumes B, C, D, H, W
        B, C, D, H, W = x.shape

        # Rearrange to (B, T, N, C)
        x = rearrange(
            x,
            "b c (p1 s1) (p2 s2) (p3 s3) -> b (p1 p2 p3) (s1 s2 s3) c",
            s1=self.patch_size,
            s2=self.patch_size,
            s3=self.patch_size,
            b=B,
        )

        x = self.transformer_layers(x)
        x = x.reshape(B, C, D, H, W)
        x = self.conv(x)
        x = self.pool(x)

        return x