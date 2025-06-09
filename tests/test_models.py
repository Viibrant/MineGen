"""Tests for model architectures."""

import pytest
import torch

from src.minegen.models import HierarchicalTransformerModel, VAE


class TestHierarchicalTransformerModel:
    """Tests for the hierarchical transformer model."""

    def test_model_creation(self):
        """Test model can be created with default parameters."""
        model = HierarchicalTransformerModel()
        assert model is not None
        assert model.patch_size == 8
        assert model.c_embed == 16
        assert model.num_cat == 20

    def test_forward_pass(self, sample_batch):
        """Test forward pass produces expected output shapes."""
        model = HierarchicalTransformerModel(in_size=16, c_embed=16)
        
        # Forward pass
        recon_logits, category_logits = model(sample_batch)
        
        # Check shapes
        assert recon_logits.shape[0] == sample_batch.shape[0]  # Batch size
        assert category_logits.shape == (sample_batch.shape[0], 20)  # Categories

    def test_different_input_sizes(self):
        """Test model works with different input sizes."""
        for size in [8, 16, 32]:
            model = HierarchicalTransformerModel(in_size=size)
            x = torch.randint(0, 256, (2, size, size, size))
            
            recon_logits, category_logits = model(x)
            assert recon_logits.shape[0] == 2
            assert category_logits.shape == (2, 20)


class TestVAE:
    """Tests for the VAE model."""

    def test_vae_creation(self):
        """Test VAE can be created."""
        model = VAE(latent_dim=64, embedding_size=512)
        assert model is not None
        assert model.latent_dim == 64
        assert model.embedding_size == 512

    def test_encode_decode(self, sample_batch):
        """Test encode and decode functions."""
        model = VAE(latent_dim=64, embedding_size=512)
        
        # Test encoding
        mu, logvar = model.encode(sample_batch)
        assert mu.shape == (sample_batch.shape[0], 64)
        assert logvar.shape == (sample_batch.shape[0], 64)
        
        # Test reparameterization
        z = model.reparameterize(mu, logvar)
        assert z.shape == (sample_batch.shape[0], 64)
        
        # Test decoding
        recon_x = model.decode(z)
        assert recon_x.shape[0] == sample_batch.shape[0]

    def test_forward_pass(self, sample_batch, sample_categories):
        """Test full forward pass."""
        model = VAE(latent_dim=64, embedding_size=512, num_categories=20)
        
        recon_x, category_logits, mu, logvar = model(sample_batch)
        
        assert recon_x.shape[0] == sample_batch.shape[0]
        assert category_logits.shape == (sample_batch.shape[0], 20)
        assert mu.shape == (sample_batch.shape[0], 64)
        assert logvar.shape == (sample_batch.shape[0], 64)