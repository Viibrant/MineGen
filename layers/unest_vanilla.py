from positional_encodings.torch_encodings import PositionalEncoding3D, Summer, PositionalEncoding2D
from torch import nn
from torch.autograd import Variable
from einops import rearrange, Rearrange
from .block import RearrangeModule
from .transformer import NestedTransformer
from .residual import ResidualBlock

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
            
            # Pass through the residual block and add
            x = x + self.res_blocks[i](previous_state)


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

        gc.collect()

        return x


BATCH_SIZE = 1
loader = DataLoader(raw, batch_size=BATCH_SIZE, shuffle=True)
SHAPE = (BATCH_SIZE, 1, 128, 128, 128)
patch_size = 4

torch.cuda.empty_cache()
model = Model(patch_size=[4, 4, 4], embed_dim=[64, 128, 256]).to(device)

for x, y in loader:    
#     x = x.abs().to(device)
#     x_h = model(x)

#     print(x_h.shape)
#     print(x_h[0][:10, :10, :10])
#     break