# %%
%pip install positional-encodings[pytorch]

# %%
import pathlib
import os

os.chdir(pathlib.Path().absolute() / "..")
os.getcwd()

# %%
from data import build_dataset
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from einops.layers.torch import Rearrange
from einops import rearrange
from tqdm import tqdm
import torch
import math
import numpy as np
from positional_encodings.torch_encodings import PositionalEncoding3D, Summer, PositionalEncoding2D

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
raw = build_dataset(transform=None)
loader = DataLoader(raw, batch_size=1, shuffle=True)

# %%
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
        qkv = self.qkv(x).reshape(b, t, n, 3, self.num_heads, c // self.num_heads).permute(3, 0, 4, 1, 2, 5)
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
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
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
            nn.Dropout(drop)
        )

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.drop_path(self.attn(y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# %%
class RearrangeModule(nn.Module):
    """# Blockify/partition/whatever
    
    Input shape: (B, C, D, H, W) 
    Output shape: (B, C, pS, pW, pD, sW, sH, sD)"""
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, x):
        _, _, D, _, _ = x.shape
        p_l = D // self.patch_size
        
        # Split into blocks
        x = rearrange(x, f"b c (p1 s1) (p2 s2) (p3 s3) -> b c p1 p2 p3 s1 s2 s3", p1=p_l, p2=p_l, p3=p_l)
        
        # Aggregate blocks into partitions
        # x = rearrange(x, "b c p1 p2 p3 s1 s2 s3 -> b (p1 p2 p3) (s1 s2 s3) c")

        return x

# %%
class NestedTransformer(nn.Module):
    def __init__(self, patch_size, embed_dim, num_heads, num_layers, in_channels=None):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels or embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.pool = nn.MaxPool3d(2)
        #TODO: FIGURE THIS OUT
        # self.positional = Summer(PositionalEncoding3D(embed_dim))
        self.positional = nn.Identity()
        self.transformer_layers = nn.Sequential(*[TransformerLayer(self.in_channels, num_heads) for _ in range(num_layers)])
        self.conv = nn.LazyConv3d(embed_dim, 1)

        self.model_patch = nn.Sequential(
            self.pool,
            RearrangeModule(self.patch_size),
            Rearrange("b c p1 p2 p3 s1 s2 s3 -> b (p1 s1) (p2 s2) (p3 s3) c"),
            self.positional
        )

    def forward(self, x):
        x = self.model_patch(x) # (B, D, H, W, C)
        B, D, H, W, C = x.shape
        
        # Rearrange to (B, T, N, C)
        x = rearrange(x, "b (p1 s1) (p2 s2) (p3 s3) c -> b (p1 p2 p3) (s1 s2 s3) c", s1=self.patch_size, s2=self.patch_size, s3=self.patch_size, b=x.shape[0])
        
        x = self.transformer_layers(x)
        x = x.reshape(B, C, D, H, W)
        x = self.conv(x)

        return x


# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation=nn.GELU, norm=nn.BatchNorm3d):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, bias=False),
            norm(out_channels),
            activation(),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size, bias=False),
            norm(out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            norm(out_channels)
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


# %%
class Model(nn.Module):
    def __init__(self, patch_size=None, embed_dim=None, num_layers=2, num_heads=8):
        super().__init__()

        if not embed_dim:
            embed_dim = [128, 256, 512]
        if not patch_size:
            patch_size = [4, 4, 4, 4]
        

        # Create the hierarchical transformer layers
        # embed_dim = [embed_dim[0]] + embed_dim
        self.hierarchical_transformers = nn.ModuleList([
            NestedTransformer(p, e, num_heads, num_layers, in_channels=embed_dim[i-1] if i > 0 else embed_dim[0]) 
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
        self.positional = Summer(PositionalEncoding3D(embed_dim[0]))

        self.norm = nn.LayerNorm(embed_dim[-1])
        self.conv = nn.LazyConv3d(embed_dim[-1] * 2, 1)

        self.patch_size = patch_size
        self.embed_dim = embed_dim

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
            # Rearrange("b c d h w -> b d h w c"),
            nn.LazyConv3d(embed_dim[0] // 2, kernel_size=1),
            Rearrange("b c d h w -> b d h w c"),
            nn.LayerNorm(embed_dim[0] // 2),
            nn.GELU(),
            Rearrange("b d h w c -> b c d h w"),
        )

        self.res_final = nn.Sequential(
            ResidualBlock(embed_dim[0], embed_dim[0] // 2, stride=1, kernel_size=2),
            nn.ConvTranspose3d(embed_dim[0] // 2, embed_dim[0] // 2, kernel_size=2, stride=2),
            # nn.LazyLinear(embed_dim[0] // 2),
            nn.ConvTranspose3d(embed_dim[0] // 2, embed_dim[0] // 2, kernel_size=2, stride=2)
        )
        
        # self.collapse = nn.Sequential(
        #     nn.Softmax(dim=1)
        # )

        self.embed_layer = nn.Sequential(
            nn.Embedding(512, embed_dim[0] // 2, scale_grad_by_freq=True) ,
            Rearrange("b d h w c -> b c d h w")
        )

        self.conv_final = nn.LazyConv3d(512, kernel_size=1)


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
        
        del nest

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

            # Pass result through another residual block
            x = self.res_blocks[i](x)

            # Transpose convolute for upsampling
            x = layer(x)
            
            
        del encoder_outputs
        del previous_state
        
        # Add the patch projection
        x += patch_proj
        del patch_proj

        # Pass through the final residual block
        x = self.res_final(x)
        
        # Add raw projection
        x += raw_projection
        del raw_projection

        # Probability map
        # x = self.collapse(x)
        # x = torch.argmax(x, dim=1)

        x = self.conv_final(x)


        return x


BATCH_SIZE = 1
loader = DataLoader(raw, batch_size=BATCH_SIZE, shuffle=True)
SHAPE = (BATCH_SIZE, 1, 128, 128, 128)
patch_size = 4

torch.cuda.empty_cache()
model = Model(patch_size=[4, 4, 4], embed_dim=[128, 256, 512]).to(device)

for x, y in loader:    
    x = x.abs().to(device)
    x_h = model(x)

    print(x_h.shape)
    print(x_h[0][:10, :10, :10])
    break

# %%
torch.cuda.empty_cache()

# %%
criterion = nn.CrossEntropyLoss()
dataset = build_dataset(None)
# optimiser = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001)
optimiser = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
N_SAMPLES = (len(dataset) // 10) // 15

training_data = torch.utils.data.Subset(dataset, range(0, N_SAMPLES))
training_data = DataLoader(training_data, batch_size=1, shuffle=True, num_workers=4, pin_memory=False)

torch.backends.cudnn.benchmark = True


# %%
from torch.utils.tensorboard import SummaryWriter

# %%
num_epochs = 5
writer = SummaryWriter()
torch.cuda.empty_cache()

model.zero_grad(set_to_none=True)
for epoch in range(num_epochs):
    train_loss = 0
    for schem_data, target in (pbar := tqdm(training_data)):
        optimiser.zero_grad()
        
        # Move to GPU
        schem_data = schem_data.abs().to(device)
        # target = target.to(device)
        
        # Forward pass
        y_hat = model.forward(schem_data)
        y_hat = Variable(y_hat, requires_grad=True)
        # torch variable

        loss = criterion(y_hat, schem_data.long())
        # loss.requires_grad = True
        
        writer.add_scalar('Loss/train', loss.item(), epoch)
        loss.backward()
        optimiser.step()
        train_loss += loss.item()
        
        pbar.set_description(f"Epoch {epoch+1}, Training Loss: {train_loss:.6f}")

writer.flush()


