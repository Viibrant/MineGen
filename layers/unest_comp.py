from positional_encodings.torch_encodings import (
    PositionalEncoding3D,
    Summer,
    PositionalEncoding2D,
)
from lightning import LightningModule
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from .block import RearrangeModule
from .transformer import NestedTransformer
from .residual import ResidualBlock


class LitModel(LightningModule):
    def __init__(self, patch_size=None, embed_dim=None, num_layers=2, num_heads=8):
        super().__init__()

        patch_size = patch_size or [4, 4, 4, 4]
        embed_dim = embed_dim or [128, 256, 512]

        self.hierarchical_transformers = nn.ModuleList(
            [
                NestedTransformer(
                    p,
                    e,
                    num_heads,
                    num_layers,
                    in_channels=embed_dim[i - 1] if i > 0 else embed_dim[0],
                )
                for i, (p, e) in enumerate(zip(patch_size, embed_dim))
            ]
        )

        self.conv_t_final = nn.LazyConvTranspose3d(embed_dim[-1], kernel_size=1)
        self.conv_transpose = nn.ModuleList(
            [
                nn.LazyConvTranspose3d(e, kernel_size=2, stride=2)
                for e in reversed(embed_dim)
            ]
            + [nn.LazyConvTranspose3d(embed_dim[0], kernel_size=2, stride=2)]
        )

        self.positional = Summer(PositionalEncoding3D(embed_dim[0]))
        self.norm = nn.LayerNorm(embed_dim[-1])
        self.conv = nn.LazyConv3d(embed_dim[-1] * 2, 1)

        self.projection = nn.Sequential(
            RearrangeModule(patch_size[0]),
            Rearrange("b c p1 p2 p3 s1 s2 s3 -> b p1 p2 p3 (s1 s2 s3 c)"),
            nn.LazyLinear(embed_dim[0]),
            self.positional,
            Rearrange("b d h w c -> b c d h w"),
        )

        self.res_raw = nn.Sequential(
            nn.LazyConv3d(embed_dim[0] // 2, kernel_size=1),
            Rearrange("b c d h w -> b d h w c"),
            nn.LayerNorm(embed_dim[0] // 2),
            nn.GELU(),
            Rearrange("b d h w c -> b c d h w"),
        )

        self.res_final = nn.Sequential(
            ResidualBlock(embed_dim[0], embed_dim[0] // 2, stride=1, kernel_size=2),
            nn.ConvTranspose3d(
                embed_dim[0] // 2, embed_dim[0] // 2, kernel_size=2, stride=2
            ),
            nn.ConvTranspose3d(
                embed_dim[0] // 2, embed_dim[0] // 2, kernel_size=2, stride=2
            ),
        )

        self.embed_layer = nn.Sequential(
            nn.Embedding(512, embed_dim[0] // 2, scale_grad_by_freq=True),
            Rearrange("b d h w c -> b c d h w"),
        )

        self.conv_final = nn.LazyConv3d(512, kernel_size=1)

        self.classifier = nn.Sequential(
            nn.LazyConv3d(32, kernel_size=2, stride=2),
            nn.LazyConv3d(19, kernel_size=2, stride=2)
            # nn.AvgPool3d(1),
            # Rearrange("b c d h w -> b (d h w) c"),
            # nn.Linear(embed_dim[0], 19)
        )

    # The rest of the class remains the same

    def forward(self, x):
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
            x += Variable(encoder_outputs[-i - 1])
            x = layer(x)

        # Add the patch projection
        x += patch_proj

        # Pass through the final residual block
        x = self.res_final(x)

        # Add raw projection
        x += raw_projection

        # x = self.conv_final(x)

        x = self.classifier(x)
        print(x.shape)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.abs()

        y_hat = self(x)

        # Use binary_cross_entropy_with_logits to handle one-hot encoded style labels
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
