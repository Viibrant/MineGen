"""Tests for custom layers."""

import pytest
import torch

from src.minegen.layers import (
    Attention,
    TransformerLayer,
    RearrangeModule,
    Patchify,
    ResidualBlock,
    NestedTransformer,
)


class TestAttention:
    """Tests for attention mechanism."""

    def test_attention_forward(self):
        """Test attention forward pass."""
        attention = Attention(dim=64, num_heads=8)
        
        # Input shape: (B, T, N, C)
        x = torch.randn(2, 4, 16, 64)
        output = attention(x)
        
        assert output.shape == x.shape


class TestRearrangeModule:
    """Tests for rearrange module."""

    def test_rearrange_forward(self):
        """Test rearrange module."""
        module = RearrangeModule(patch_size=4)
        
        # Input shape: (B, C, D, H, W)
        x = torch.randn(2, 3, 16, 16, 16)
        output = module(x)
        
        # Should split into patches
        expected_shape = (2, 3, 4, 4, 4, 4, 4, 4)
        assert output.shape == expected_shape


class TestResidualBlock:
    """Tests for residual block."""

    def test_residual_block_forward(self):
        """Test residual block forward pass."""
        block = ResidualBlock(in_channels=32, out_channels=64, kernel_size=3)
        
        x = torch.randn(2, 32, 8, 8, 8)
        output = block(x)
        
        assert output.shape == (2, 64, 8, 8, 8)


class TestNestedTransformer:
    """Tests for nested transformer."""

    def test_nested_transformer_forward(self):
        """Test nested transformer forward pass."""
        transformer = NestedTransformer(
            patch_size=4,
            embed_dim=64,
            num_heads=8,
            num_layers=2
        )
        
        x = torch.randn(2, 64, 16, 16, 16)
        output = transformer(x)
        
        # Output should be downsampled
        assert output.shape[0] == 2  # Batch size preserved
        assert output.shape[1] == 64  # Embed dim preserved