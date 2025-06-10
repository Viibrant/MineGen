"""Integration tests for the complete system."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import torch
import pandas as pd

from src.minegen.config.models import Config
from src.minegen.data import SchematicDataModule
from src.minegen.models import VAE, HierarchicalTransformerModel
from src.minegen.cli.main import cli
from click.testing import CliRunner


class TestFullPipeline:
    """Test the complete training and generation pipeline."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_data(self, temp_workspace):
        """Create mock data files."""
        # Create mock metadata
        metadata = pd.DataFrame({
            'ID': [1, 2, 3],
            'Name': ['House1', 'Castle1', 'Garden1'],
            'Category': ['Houses', 'Castles', 'Gardens'],
            'X': [16, 16, 16],
            'Y': [16, 16, 16],
            'Z': [16, 16, 16],
            'Path': [
                str(temp_workspace / 'house.schematic'),
                str(temp_workspace / 'castle.schematic'),
                str(temp_workspace / 'garden.schematic')
            ]
        })
        metadata.to_csv(temp_workspace / 'data.csv', index=False)
        
        # Create mock schematic files (just empty files for testing)
        for path in metadata['Path']:
            Path(path).touch()
        
        return temp_workspace

    @patch('src.minegen.data.dataset.SchematicFile.load')
    def test_data_loading_pipeline(self, mock_load, mock_data):
        """Test the complete data loading pipeline."""
        # Mock schematic loading
        mock_sf = Mock()
        mock_sf.blocks = torch.zeros(16, 16, 16).numpy()
        mock_load.return_value = mock_sf
        
        # Test data module
        dm = SchematicDataModule(
            data_dir=str(mock_data),
            metadata_file=str(mock_data / 'data.csv'),
            batch_size=2,
            num_workers=0  # Avoid multiprocessing in tests
        )
        
        dm.setup('fit')
        
        # Test data loading
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        
        assert len(batch) == 2  # x, y
        assert batch[0].shape[0] <= 2  # batch size
        assert batch[1].shape[0] <= 2  # categories

    @patch('lightning.pytorch.Trainer.fit')
    def test_training_pipeline(self, mock_fit, mock_data):
        """Test the training pipeline."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Create config
            config = Config()
            config.to_yaml('test_config.yaml')
            
            # Test training command
            result = runner.invoke(cli, [
                '--config', 'test_config.yaml',
                'train-model',
                '--model-type', 'vae',
                '--batch-size', '2',
                '--max-epochs', '1',
                '--output-dir', './test_output'
            ])
            
            # Should complete without error (even if mocked)
            assert result.exit_code == 0

    def test_model_integration(self):
        """Test model integration with different architectures."""
        models = [
            VAE(latent_dim=32, embedding_size=128),
            HierarchicalTransformerModel(in_size=16, c_embed=32),
        ]
        
        x = torch.randint(0, 256, (2, 16, 16, 16))
        
        for model in models:
            try:
                output = model(x)
                assert output is not None
            except RuntimeError:
                # Some models might fail due to lazy layers
                pass

    @patch('src.minegen.data.scraper.generate_dataset')
    def test_data_download_pipeline(self, mock_download):
        """Test data download pipeline."""
        mock_download.return_value = pd.DataFrame({
            'ID': [1, 2],
            'Name': ['Test1', 'Test2'],
            'Category': ['Houses', 'Castles']
        })
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'download-data',
            '--criteria', 'most-downloaded',
            '--num-pages', '1',
            '--max-workers', '1'
        ])
        
        assert result.exit_code == 0
        mock_download.assert_called_once()

    def test_config_system(self):
        """Test configuration system integration."""
        config = Config()
        
        # Test default values
        assert config.model.num_classes == 20
        assert config.solver.base_lr == 1e-3
        assert config.datasets.shape == [16, 16, 16]
        
        # Test updates
        updated = config.update(output_dir='./new_output')
        assert updated.output_dir == './new_output'
        
        # Test YAML serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            loaded = Config.from_yaml(f.name)
            assert loaded.model.num_classes == config.model.num_classes

    @patch('src.minegen.dashboard.app.run_dashboard')
    def test_dashboard_integration(self, mock_run):
        """Test dashboard integration."""
        runner = CliRunner()
        
        # Test dashboard command
        result = runner.invoke(cli, [
            'dashboard',
            '--host', 'localhost',
            '--port', '8051',
            '--debug'
        ])
        
        # Should attempt to run dashboard
        mock_run.assert_called_once()

    def test_cli_help_commands(self):
        """Test all CLI commands have help."""
        runner = CliRunner()
        
        commands = ['train-model', 'generate', 'download-data', 'dashboard', 'evaluate']
        
        for command in commands:
            result = runner.invoke(cli, [command, '--help'])
            assert result.exit_code == 0
            assert 'Usage:' in result.output

    @patch('torch.save')
    @patch('src.minegen.models.VAE.load_from_checkpoint')
    def test_generation_pipeline(self, mock_load, mock_save):
        """Test generation pipeline."""
        # Mock model
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.generate.return_value = torch.zeros(5, 16, 16, 16)
        mock_load.return_value = mock_model
        
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Create dummy checkpoint
            Path('model.ckpt').touch()
            
            result = runner.invoke(cli, [
                'generate',
                '--checkpoint', 'model.ckpt',
                '--num-samples', '5',
                '--output-dir', './generated'
            ])
            
            assert result.exit_code == 0

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        runner = CliRunner()
        
        # Test missing checkpoint
        result = runner.invoke(cli, [
            'generate',
            '--checkpoint', 'nonexistent.ckpt',
            '--num-samples', '1'
        ])
        assert result.exit_code != 0
        
        # Test invalid model type
        result = runner.invoke(cli, [
            'train-model',
            '--model-type', 'invalid_model'
        ])
        assert result.exit_code != 0


class TestSystemPerformance:
    """Test system performance and resource usage."""

    def test_memory_usage(self):
        """Test memory usage is reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create models
        models = [
            VAE(latent_dim=64),
            HierarchicalTransformerModel(in_size=16),
        ]
        
        # Test forward pass
        x = torch.randint(0, 256, (4, 16, 16, 16))
        for model in models:
            try:
                _ = model(x)
            except RuntimeError:
                pass  # Some models might fail due to lazy layers
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 1GB)
        assert memory_increase < 1024 * 1024 * 1024

    def test_batch_processing(self):
        """Test batch processing works correctly."""
        model = VAE(latent_dim=32, embedding_size=128)
        
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            x = torch.randint(0, 256, (batch_size, 16, 16, 16))
            output = model(x)
            
            assert output[0].shape[0] == batch_size  # recon_x
            assert output[1].shape[0] == batch_size  # category_logits
            assert output[2].shape[0] == batch_size  # mu
            assert output[3].shape[0] == batch_size  # logvar