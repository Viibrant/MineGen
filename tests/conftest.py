"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.minegen.config.models import Config


@pytest.fixture
def config():
    """Default test configuration."""
    return Config()


@pytest.fixture
def sample_schematic():
    """Sample 3D schematic tensor."""
    return torch.randint(0, 256, (16, 16, 16))


@pytest.fixture
def sample_batch():
    """Sample batch of schematics."""
    return torch.randint(0, 256, (4, 16, 16, 16))


@pytest.fixture
def sample_categories():
    """Sample one-hot encoded categories."""
    batch_size = 4
    num_categories = 20
    categories = torch.zeros(batch_size, num_categories)
    # Set random category for each sample
    for i in range(batch_size):
        categories[i, torch.randint(0, num_categories, (1,))] = 1
    return categories


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path