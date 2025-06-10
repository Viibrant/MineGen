"""Experimental models from notebooks/temp.py"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange
from einops.layers.torch import Rearrange
from lightning.pytorch import LightningModule
from typing import List, Optional

from ..layers import NestedTransformer, ResidualBlock, RearrangeModule

try:
    from positional_encodings.torch_encodings import PositionalEncoding3D, Summer
    HAS_POSITIONAL_ENCODINGS = True
except ImportError:
    HAS_POSITIONAL_ENCODINGS = False


class ExperimentalModel(LightningModule):
    """Experimental hierarchical model from temp.py notebook."""

    def __init__(
        self,
        patch_size: Optional[List[int]] = None,
        embed_dim: Optional[List[int]] = None,
        num_layers: int = 2,
        num_heads: int = 8,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if not embed_dim:
            embed_dim = [128, 256, 512]
        if not patch_size:
            patch_size = [4, 4, 4, 4]

        self.learning_rate = learning_rate
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Create the hierarchical transformer layers
        self.hierarchical_transformers = nn.ModuleList([
            NestedTransformer(
                p, e, num_heads, num_layers,
                in_channels=embed_dim[i-1] if i > 0 else embed_dim[0]
            )
            for i, (p, e) in enumerate(zip(patch_size, embed_dim))
        ])

        # Create the transpose convolutional layers
        self.conv_t_final = nn.LazyConvTranspose3d(embed_dim[-1], kernel_size=1)

        self.conv_transpose = nn.ModuleList([
            nn.LazyConvTranspose3d(e, kernel_size=2, stride=2)
            for e in reversed(embed_dim)
        ])
        self.conv_transpose.append(nn.LazyConvTranspose3d(embed_dim[0], kernel_size=2, stride=2))

        # Add positional encoding for the main model
        if HAS_POSITIONAL_ENCODINGS:
            self.positional = Summer(PositionalEncoding3D(embed_dim[0]))
        else:
            self.positional = nn.Identity()

        self.norm = nn.LayerNorm(embed_dim[-1])
        self.conv = nn.LazyConv3d(embed_dim[-1] * 2, 1)

        self.projection = nn.Sequential(
            RearrangeModule(self.patch_size[0]),
            Rearrange("b c p1 p2 p3 s1 s2 s3 -> b p1 p2 p3 (s1 s2 s3 c)"),
            nn.LazyLinear(embed_dim[0]),
            self.positional,
            Rearrange("b d h w c -> b c d h w")
        )

        self.res_blocks = nn.ModuleList([
            ResidualBlock(e, e, kernel_size=1, stride=1) for e in reversed(embed_dim)
        ])

        self.res_raw = nn.Sequential(
            nn.LazyConv3d(embed_dim[0] // 2, kernel_size=1),
            Rearrange("b c d h w -> b d h w c"),
            nn.LayerNorm(embed_dim[0] // 2),
            nn.GELU(),
            Rearrange("b d h w c -> b c d h w"),
        )

        self.res_final = nn.Sequential(
            ResidualBlock(embed_dim[0], embed_dim[0] // 2, stride=1, kernel_size=2),
            nn.ConvTranspose3d(embed_dim[0] // 2, embed_dim[0] // 2, kernel_size=2, stride=2),
            nn.ConvTranspose3d(embed_dim[0] // 2, embed_dim[0] // 2, kernel_size=2, stride=2)
        )

        self.embed_layer = nn.Sequential(
            nn.Embedding(512, embed_dim[0] // 2, scale_grad_by_freq=True),
            Rearrange("b d h w c -> b c d h w")
        )

        self.conv_final = nn.LazyConv3d(512, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.embed_layer(x.int())
        raw_projection = Variable(self.res_raw(x))

        encoder_outputs = []

        # Patch Projection
        x = self.projection(x)
        patch_proj = x
        encoder_outputs.append(x)

        for nest in self.hierarchical_transformers:
            x = nest(x)
            encoder_outputs.append(x)

        x = rearrange(x, "b c d h w -> b d h w c")
        x = self.norm(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        x = self.conv(x)

        # Upscale the final hidden state
        x = self.conv_t_final(x)

        # Iterate over the layers in reverse order, and upscale
        for i, layer in enumerate(self.conv_transpose[1:]):
            previous_state = Variable(encoder_outputs[-i-1])

            # Pass through the residual block
            previous_state = self.res_blocks[i](previous_state)

            # Add previous hidden state
            x += previous_state

            # Pass through another residual block
            x = self.res_blocks[i](x)

            # Transpose convolute for upsampling
            x = layer(x)

        # Add the patch projection
        x += patch_proj

        # Pass through the final residual block
        x = self.res_final(x)

        # Add raw projection
        x += raw_projection

        # Final convolution
        x = self.conv_final(x)

        return x

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, _ = batch
        x = x.abs()
        x_hat = self(x)
        
        loss = F.cross_entropy(x_hat, x.long())
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, _ = batch
        x = x.abs()
        x_hat = self(x)
        
        loss = F.cross_entropy(x_hat, x.long())
        self.log('val/loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)