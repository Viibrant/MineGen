from positional_encodings.torch_encodings import (
    PositionalEncoding3D,
    Summer,
    PositionalEncoding2D,
)
from torch import nn
from .block import RearrangeModule
from einops import rearrange
from einops.layers.torch import Rearrange


class Attention(nn.Module):
    """
    This is much like `.vision_transformer.Attention` but uses *localised* self attention by accepting an input with
     an extra "image block" dim
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x is shape: B (batch_size), T (image blocks), N (seq length per image block), C (embed dim)
        """
        b, t, n, c = x.shape
        # result of next line is (qkv, B, num (H)eads, T, N, (C')hannels per head)
        qkv = (
            self.qkv(x)
            .reshape(b, t, n, 3, self.num_heads, c // self.num_heads)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 3, 4, 1).reshape(b, t, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x  # (B, T, N, C)


class TransformerLayer(nn.Module):
    """
    This is much like `.vision_transformer.Block` but:
        - Called TransformerLayer here to allow for "block" as defined in the paper ("non-overlapping image blocks")
        - Uses modified Attention layer that handles the "block" dimension
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.drop_path(self.attn(y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class NestedTransformer(nn.Module):
    """
    lol
    """
    def __init__(self, patch_size, embed_dim, num_heads, num_layers, in_channels=None):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels or embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.pool = nn.MaxPool3d(2)
        # TODO: FIGURE THIS OUT
        # self.positional = Summer(PositionalEncoding3D(embed_dim))
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

    def forward(self, x):
        x = self.model_patch(x)  # (B, D, H, W, C)
        B, D, H, W, C = x.shape

        # Rearrange to (B, T, N, C)
        x = rearrange(
            x,
            "b (p1 s1) (p2 s2) (p3 s3) c -> b (p1 p2 p3) (s1 s2 s3) c",
            s1=self.patch_size,
            s2=self.patch_size,
            s3=self.patch_size,
            b=x.shape[0],
        )

        x = self.transformer_layers(x)
        x = x.reshape(B, C, D, H, W)
        x = self.conv(x)

        return x
