import torch
from torch import nn


class Patchify(nn.Module):
    """Patchify a tensor into smaller patches
    See: https://stackoverflow.com/a/68360020/5181304"""

    def __init__(self, kernel_size, stride=1, dilation=1, padding=0):
        super().__init__()

        self.padding = padding

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation, dilation)

    @staticmethod
    def _get_dim_blocks(dim_in, kernel_size, padding=0, stride=1, dilation=1):
        return (dim_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):

        x = x.contiguous()

        channels, depth, height, width = x.shape[-4:]
        d_blocks = Patchify._get_dim_blocks(
            depth,
            kernel_size=self.kernel_size[0],
            stride=self.stride[0],
            dilation=self.dilation[0],
        )
        h_blocks = Patchify._get_dim_blocks(
            height,
            kernel_size=self.kernel_size[1],
            stride=self.stride[1],
            dilation=self.dilation[1],
        )
        w_blocks = Patchify._get_dim_blocks(
            width,
            kernel_size=self.kernel_size[2],
            stride=self.stride[2],
            dilation=self.dilation[2],
        )
        shape = (
            channels,
            d_blocks,
            h_blocks,
            w_blocks,
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2],
        )
        strides = (
            width * height * depth,
            self.stride[0] * width * height,
            self.stride[1] * width,
            self.stride[2],
            self.dilation[0] * width * height,
            self.dilation[1] * width,
            self.dilation[2],
        )

        x = x.as_strided(shape, strides)

        # Permute to get the patches in the last dimension
        x = x.permute(1, 2, 3, 0, 4, 5, 6)
        return x
