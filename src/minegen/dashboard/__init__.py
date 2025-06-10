"""Real-time dashboard for monitoring training and generation."""

from .app import create_dashboard_app, run_dashboard
from .components import create_training_monitor, create_generation_viewer, create_data_explorer

__all__ = [
    "create_dashboard_app",
    "run_dashboard", 
    "create_training_monitor",
    "create_generation_viewer",
    "create_data_explorer"
]