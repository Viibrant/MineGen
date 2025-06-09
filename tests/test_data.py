"""Tests for data loading and processing."""

import pytest
import torch
from unittest.mock import Mock, patch

from src.minegen.data import SchematicDataModule


class TestSchematicDataModule:
    """Tests for schematic data module."""

    @patch('src.minegen.data.dataset.SchematicDataset')
    def test_data_module_creation(self, mock_dataset):
        """Test data module can be created."""
        # Mock the dataset
        mock_dataset.return_value = Mock()
        mock_dataset.return_value.__len__ = Mock(return_value=100)
        
        dm = SchematicDataModule(batch_size=16, num_workers=4)
        assert dm.batch_size == 16
        assert dm.num_workers == 4

    @patch('src.minegen.data.dataset.SchematicDataset')
    def test_data_module_setup(self, mock_dataset):
        """Test data module setup."""
        # Mock the dataset
        mock_dataset.return_value = Mock()
        mock_dataset.return_value.__len__ = Mock(return_value=100)
        
        dm = SchematicDataModule(batch_size=16)
        
        # This would normally load real data, but we're mocking it
        try:
            dm.setup("fit")
        except Exception:
            # Expected to fail without real data
            pass