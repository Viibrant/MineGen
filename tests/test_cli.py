"""Tests for CLI functionality."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch

from src.minegen.cli.main import cli


class TestCLI:
    """Tests for command line interface."""

    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "MineGen: Minecraft Schematic Generator" in result.output

    def test_train_command_help(self):
        """Test train command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train-model", "--help"])
        assert result.exit_code == 0
        assert "Train a model" in result.output

    @patch('src.minegen.cli.main.Trainer')
    @patch('src.minegen.cli.main.SchematicDataModule')
    def test_train_command_dry_run(self, mock_data, mock_trainer):
        """Test train command without actually training."""
        runner = CliRunner()
        
        # Mock the components
        mock_data.return_value = Mock()
        mock_trainer.return_value = Mock()
        
        result = runner.invoke(cli, [
            "train-model",
            "--model-type", "vae",
            "--batch-size", "16",
            "--max-epochs", "1"
        ])
        
        # Should complete without error
        assert result.exit_code == 0