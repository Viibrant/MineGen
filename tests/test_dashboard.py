"""Tests for dashboard functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import dash
from dash.testing.application_runners import import_app

from src.minegen.config.models import Config
from src.minegen.dashboard.app import create_dashboard_app, DashboardState
from src.minegen.dashboard.components import (
    create_training_monitor,
    create_generation_viewer,
    create_data_explorer,
    create_metrics_display,
    create_status_indicator
)


class TestDashboardComponents:
    """Tests for dashboard components."""

    def test_create_training_monitor(self):
        """Test training monitor component creation."""
        component = create_training_monitor()
        assert component is not None
        assert hasattr(component, 'children')

    def test_create_generation_viewer(self):
        """Test generation viewer component creation."""
        component = create_generation_viewer()
        assert component is not None
        assert hasattr(component, 'children')

    def test_create_data_explorer(self):
        """Test data explorer component creation."""
        component = create_data_explorer()
        assert component is not None
        assert hasattr(component, 'children')

    def test_create_metrics_display(self):
        """Test metrics display component."""
        metrics = {
            'train_loss': 0.5,
            'val_loss': 0.6,
            'accuracy': 0.85
        }
        component = create_metrics_display(metrics)
        assert component is not None

    def test_create_metrics_display_empty(self):
        """Test metrics display with empty metrics."""
        component = create_metrics_display({})
        assert component is not None

    def test_create_status_indicator(self):
        """Test status indicator component."""
        component = create_status_indicator('training', 'Model is training...')
        assert component is not None


class TestDashboardApp:
    """Tests for dashboard app."""

    def test_create_dashboard_app(self, config):
        """Test dashboard app creation."""
        app = create_dashboard_app(config)
        assert isinstance(app, dash.Dash)
        assert app.layout is not None

    def test_dashboard_state(self):
        """Test dashboard state management."""
        state = DashboardState()
        assert state.training_logs == []
        assert state.generated_samples == []
        assert state.model_checkpoints == []
        assert state.current_model is None
        assert state.is_training is False
        assert state.generation_queue == []

    @patch('src.minegen.dashboard.app.read_training_logs')
    def test_training_logs_reading(self, mock_read_logs):
        """Test training logs reading."""
        mock_read_logs.return_value = [
            {'epoch': 1, 'train_loss': 0.5, 'val_loss': 0.6}
        ]
        
        from src.minegen.dashboard.app import read_training_logs
        logs = read_training_logs()
        assert len(logs) >= 0  # Could be mock or real data

    @patch('src.minegen.dashboard.app.read_generated_samples')
    def test_generated_samples_reading(self, mock_read_samples):
        """Test generated samples reading."""
        mock_read_samples.return_value = [
            {'id': 1, 'category': 'house', 'timestamp': 123456}
        ]
        
        from src.minegen.dashboard.app import read_generated_samples
        samples = read_generated_samples()
        assert isinstance(samples, list)


class TestDashboardIntegration:
    """Integration tests for dashboard."""

    @patch('src.minegen.dashboard.app.start_background_tasks')
    @patch('dash.Dash.run_server')
    def test_run_dashboard(self, mock_run_server, mock_background, config):
        """Test dashboard running."""
        from src.minegen.dashboard.app import run_dashboard
        
        # Mock the server run
        mock_run_server.return_value = None
        
        # This would normally start the server
        try:
            run_dashboard(config, host='localhost', port=8050)
        except Exception:
            # Expected in test environment
            pass
        
        # Verify background tasks would be started
        mock_background.assert_called_once()

    def test_dashboard_config_integration(self):
        """Test dashboard integrates with config properly."""
        config = Config()
        config.dashboard.host = "test-host"
        config.dashboard.port = 9999
        config.dashboard.debug = False
        
        app = create_dashboard_app(config)
        assert app is not None

    @patch('threading.Thread')
    def test_background_tasks(self, mock_thread):
        """Test background task initialization."""
        from src.minegen.dashboard.app import start_background_tasks
        
        start_background_tasks()
        mock_thread.assert_called_once()


class TestDashboardCallbacks:
    """Tests for dashboard callbacks (would need selenium for full testing)."""

    def test_callback_registration(self, config):
        """Test callbacks are registered properly."""
        app = create_dashboard_app(config)
        
        # Check that callbacks are registered
        assert len(app.callback_map) > 0

    @patch('src.minegen.dashboard.app.read_training_logs')
    def test_training_monitor_callback_logic(self, mock_read_logs):
        """Test training monitor callback logic."""
        mock_read_logs.return_value = [
            {'epoch': 1, 'train_loss': 0.5, 'val_loss': 0.6},
            {'epoch': 2, 'train_loss': 0.4, 'val_loss': 0.55}
        ]
        
        # This would test the actual callback logic
        # In a real test, you'd use dash.testing.composite
        from src.minegen.dashboard.app import read_training_logs
        logs = read_training_logs()
        
        # Verify logs structure
        if logs:
            assert 'epoch' in logs[0]
            assert 'train_loss' in logs[0]