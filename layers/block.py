from torch import nn
from einops import rearrange

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