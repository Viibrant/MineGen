"""Hierarchical transformer model from notebooks integrated into main codebase."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from ..layers import NestedTransformer, ResidualBlock


class HierarchicalTransformerModel(nn.Module):
    """
    Hierarchical transformer model for schematic generation.
    Based on the lolModel from patch.ipynb notebook.
    """

    def __init__(
        self,
        in_size: int = 16,
        c_embed: int = 16,
        num_cat: int = 20,
        num_heads: int = 8,
        num_layers: int = 16,
        patch_size: int = 8,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.c_embed = c_embed
        self.num_cat = num_cat

        num_hierarchies = int(math.log2(in_size))

        # Initialize the hierarchical transformers for downscaling
        self.downscale_hierarchies = nn.ModuleList([
            NestedTransformer(
                self.patch_size,
                self.c_embed * 2**i,
                num_heads,
                num_layers,
                in_channels=int(self.c_embed * 2**(i-1)) if i != 0 else None
            )
            for i in range(num_hierarchies)
        ])

        self.conv = nn.LazyConv3d(512, 1)

        # Initialize the upsampling layers for each hierarchy
        self.upscale_hierarchies = nn.ModuleList([
            nn.LazyConvTranspose3d(
                int(self.c_embed * 2**(i-1)) if i != 1 else self.c_embed,
                2,
                stride=2
            )
            for i in range(num_hierarchies-1, -1, -1)
        ])

        self.conv_upscale = nn.LazyConv3d(self.c_embed, 1)

        self.embedding = nn.Sequential(
            nn.Embedding(768, self.c_embed, scale_grad_by_freq=True, padding_idx=0),
            Rearrange("b d h w c -> b c d h w")
        )

        self.classifier = nn.Sequential(
            nn.Conv3d(512, 256, 3, stride=2),
            nn.ReLU(),
            nn.Conv3d(256, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv3d(128, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_cat),
        )

        self.res_blocks_pre = nn.ModuleList([
            ResidualBlock(self.c_embed * 2**i, self.c_embed * 2**i, 3)
            for i in range(num_hierarchies-1, -1, -1)
        ])
        self.res_blocks_post = nn.ModuleList([
            ResidualBlock(self.c_embed * 2**i, self.c_embed * 2**i, 3)
            for i in range(num_hierarchies-1, -1, -1)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction logits and category predictions."""
        seq = self.embedding(x.abs().long())

        # Downscale through the hierarchies and store intermediate representations
        downscale_outputs = []
        for downscale in self.downscale_hierarchies:
            seq = downscale(seq)
            downscale_outputs.append(seq)

        # Upscale through the hierarchies and add in skip connections from downscale path
        for upscale, down_out, res_pre, res_post in zip(
            self.upscale_hierarchies,
            reversed(downscale_outputs),
            self.res_blocks_pre,
            self.res_blocks_post
        ):
            seq = res_pre(down_out) + seq   # Apply ResBlock before addition
            seq = res_post(seq)             # Apply ResBlock after addition
            seq = upscale(seq)

        seq = self.conv_upscale(seq)
        seq = self.conv(seq)

        cat = self.classifier(seq)

        return seq, cat