"""Main dashboard application using Dash."""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import torch
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import time

from ..config.models import Config
from .components import create_training_monitor, create_generation_viewer, create_data_explorer


class DashboardState:
    """Shared state for the dashboard."""
    
    def __init__(self):
        self.training_logs = []
        self.generated_samples = []
        self.model_checkpoints = []
        self.current_model = None
        self.is_training = False
        self.generation_queue = []


dashboard_state = DashboardState()


def create_dashboard_app(config: Config) -> dash.Dash:
    """Create the main dashboard application."""
    
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div([
            html.H1("🏗️ MineGen Dashboard", className="header-title"),
            html.Div([
                dcc.Link("Training Monitor", href="/training", className="nav-link"),
                dcc.Link("Generation Viewer", href="/generation", className="nav-link"),
                dcc.Link("Data Explorer", href="/data", className="nav-link"),
            ], className="nav-bar")
        ], className="header"),
        
        html.Div(id='page-content'),
        
        # Auto-refresh component
        dcc.Interval(
            id='interval-component',
            interval=config.dashboard.auto_refresh * 1000,  # in milliseconds
            n_intervals=0
        ),
        
        # Store components for sharing data
        dcc.Store(id='training-store'),
        dcc.Store(id='generation-store'),
        dcc.Store(id='model-store'),
    ], className="main-container")
    
    # Register callbacks
    register_callbacks(app, config)
    
    return app


def register_callbacks(app: dash.Dash, config: Config):
    """Register all dashboard callbacks."""
    
    @app.callback(
        Output('page-content', 'children'),
        Input('url', 'pathname')
    )
    def display_page(pathname):
        if pathname == '/training':
            return create_training_monitor()
        elif pathname == '/generation':
            return create_generation_viewer()
        elif pathname == '/data':
            return create_data_explorer()
        else:
            return create_home_page()
    
    @app.callback(
        [Output('training-store', 'data'),
         Output('training-status', 'children'),
         Output('training-progress', 'value'),
         Output('loss-graph', 'figure')],
        Input('interval-component', 'n_intervals'),
        prevent_initial_call=True
    )
    def update_training_monitor(n):
        # Read training logs
        logs = read_training_logs()
        
        # Update status
        status = "🟢 Training" if dashboard_state.is_training else "🔴 Idle"
        
        # Update progress
        progress = len(logs) % 100 if logs else 0
        
        # Create loss graph
        if logs:
            df = pd.DataFrame(logs)
            fig = px.line(df, x='epoch', y=['train_loss', 'val_loss'], 
                         title="Training Progress")
        else:
            fig = go.Figure()
            fig.add_annotation(text="No training data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
        
        return logs, status, progress, fig
    
    @app.callback(
        [Output('generation-store', 'data'),
         Output('generation-gallery', 'children')],
        Input('interval-component', 'n_intervals'),
        prevent_initial_call=True
    )
    def update_generation_viewer(n):
        # Read generated samples
        samples = read_generated_samples()
        
        # Create gallery
        gallery = create_sample_gallery(samples)
        
        return samples, gallery
    
    @app.callback(
        Output('generation-queue', 'children'),
        [Input('generate-button', 'n_clicks'),
         Input('num-samples', 'value'),
         Input('category-dropdown', 'value')],
        prevent_initial_call=True
    )
    def queue_generation(n_clicks, num_samples, category):
        if n_clicks and num_samples:
            # Add to generation queue
            dashboard_state.generation_queue.append({
                'num_samples': num_samples,
                'category': category,
                'timestamp': time.time()
            })
            return f"Queued {num_samples} samples"
        return "Ready to generate"


def create_home_page():
    """Create the home page layout."""
    return html.Div([
        html.H2("Welcome to MineGen Dashboard"),
        html.P("Monitor your Minecraft schematic generation in real-time."),
        
        html.Div([
            html.Div([
                html.H3("🎯 Quick Actions"),
                html.Button("Start Training", id="quick-train", className="action-button"),
                html.Button("Generate Samples", id="quick-generate", className="action-button"),
                html.Button("Download Data", id="quick-download", className="action-button"),
            ], className="quick-actions"),
            
            html.Div([
                html.H3("📊 System Status"),
                html.Div(id="system-status"),
            ], className="system-status"),
        ], className="home-grid"),
        
        html.Div([
            html.H3("📈 Recent Activity"),
            html.Div(id="recent-activity"),
        ], className="recent-activity"),
    ])


def create_sample_gallery(samples):
    """Create a gallery of generated samples."""
    if not samples:
        return html.Div("No samples generated yet.")
    
    gallery_items = []
    for i, sample in enumerate(samples[-12:]):  # Show last 12 samples
        gallery_items.append(
            html.Div([
                html.Img(src=sample.get('preview_url', '/assets/placeholder.png')),
                html.P(f"Sample {i+1}"),
                html.P(f"Category: {sample.get('category', 'Unknown')}"),
                html.Button("Download", className="download-btn"),
            ], className="gallery-item")
        )
    
    return html.Div(gallery_items, className="sample-gallery")


def read_training_logs() -> list:
    """Read training logs from file or wandb."""
    # This would integrate with your actual logging system
    # For now, return mock data
    if dashboard_state.is_training:
        return [
            {'epoch': i, 'train_loss': 2.5 - i*0.1 + np.random.normal(0, 0.1), 
             'val_loss': 2.3 - i*0.08 + np.random.normal(0, 0.1)}
            for i in range(len(dashboard_state.training_logs), len(dashboard_state.training_logs) + 1)
        ]
    return dashboard_state.training_logs


def read_generated_samples() -> list:
    """Read generated samples from storage."""
    # This would read from your actual sample storage
    return dashboard_state.generated_samples


def run_dashboard(config: Config, host: Optional[str] = None, port: Optional[int] = None):
    """Run the dashboard server."""
    app = create_dashboard_app(config)
    
    host = host or config.dashboard.host
    port = port or config.dashboard.port
    debug = config.dashboard.debug
    
    print(f"🚀 Starting MineGen Dashboard at http://{host}:{port}")
    app.run_server(host=host, port=port, debug=debug)


# Background task runner
def start_background_tasks():
    """Start background tasks for processing queues."""
    def process_generation_queue():
        while True:
            if dashboard_state.generation_queue:
                task = dashboard_state.generation_queue.pop(0)
                # Process generation task
                print(f"Processing generation: {task}")
                # Add mock generated sample
                dashboard_state.generated_samples.append({
                    'id': len(dashboard_state.generated_samples),
                    'category': task.get('category', 'unknown'),
                    'timestamp': task['timestamp'],
                    'preview_url': '/assets/sample.png'
                })
            time.sleep(1)
    
    # Start background thread
    thread = threading.Thread(target=process_generation_queue, daemon=True)
    thread.start()


# CSS styles
app_styles = """
.main-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
}

.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem 2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.header-title {
    margin: 0;
    font-size: 2rem;
    font-weight: 300;
}

.nav-bar {
    margin-top: 1rem;
}

.nav-link {
    color: white;
    text-decoration: none;
    margin-right: 2rem;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    transition: background-color 0.3s;
}

.nav-link:hover {
    background-color: rgba(255,255,255,0.2);
}

.home-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    padding: 2rem;
}

.quick-actions, .system-status {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.action-button {
    display: block;
    width: 100%;
    margin: 0.5rem 0;
    padding: 0.75rem;
    background: #667eea;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.action-button:hover {
    background: #5a6fd8;
}

.sample-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
    padding: 1rem;
}

.gallery-item {
    background: white;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    text-align: center;
}

.gallery-item img {
    width: 100%;
    height: 150px;
    object-fit: cover;
    border-radius: 4px;
}
"""