"""Dashboard components for different views."""

from dash import dcc, html
import plotly.graph_objs as go


def create_training_monitor():
    """Create the training monitor layout."""
    return html.Div([
        html.H2("🎯 Training Monitor"),
        
        html.Div([
            html.Div([
                html.H4("Training Status"),
                html.Div(id="training-status", className="status-indicator"),
                html.Progress(id="training-progress", value=0, max=100),
            ], className="status-panel"),
            
            html.Div([
                html.H4("Quick Controls"),
                html.Button("Start Training", id="start-training", className="control-button"),
                html.Button("Stop Training", id="stop-training", className="control-button"),
                html.Button("Save Checkpoint", id="save-checkpoint", className="control-button"),
            ], className="control-panel"),
        ], className="monitor-header"),
        
        html.Div([
            dcc.Graph(id="loss-graph"),
        ], className="graph-container"),
        
        html.Div([
            html.Div([
                html.H4("Training Metrics"),
                html.Div(id="training-metrics"),
            ], className="metrics-panel"),
            
            html.Div([
                html.H4("Model Configuration"),
                html.Div(id="model-config"),
            ], className="config-panel"),
        ], className="info-panels"),
    ])


def create_generation_viewer():
    """Create the generation viewer layout."""
    return html.Div([
        html.H2("🎨 Generation Viewer"),
        
        html.Div([
            html.Div([
                html.H4("Generate New Samples"),
                html.Label("Number of Samples:"),
                dcc.Input(id="num-samples", type="number", value=5, min=1, max=50),
                
                html.Label("Category:"),
                dcc.Dropdown(
                    id="category-dropdown",
                    options=[
                        {'label': 'Houses', 'value': 'houses'},
                        {'label': 'Castles', 'value': 'castles'},
                        {'label': 'Gardens', 'value': 'gardens'},
                        {'label': 'Random', 'value': 'random'},
                    ],
                    value='random'
                ),
                
                html.Button("Generate", id="generate-button", className="generate-button"),
                html.Div(id="generation-queue"),
            ], className="generation-controls"),
            
            html.Div([
                html.H4("Generation Settings"),
                html.Label("Temperature:"),
                dcc.Slider(id="temperature", min=0.1, max=2.0, value=1.0, step=0.1),
                
                html.Label("Seed:"),
                dcc.Input(id="seed", type="number", value=42),
                
                html.Label("Model Checkpoint:"),
                dcc.Dropdown(id="checkpoint-dropdown"),
            ], className="generation-settings"),
        ], className="generation-header"),
        
        html.Div([
            html.H4("Generated Samples"),
            html.Div(id="generation-gallery"),
        ], className="gallery-container"),
        
        html.Div([
            html.H4("Generation History"),
            html.Div(id="generation-history"),
        ], className="history-container"),
    ])


def create_data_explorer():
    """Create the data explorer layout."""
    return html.Div([
        html.H2("📊 Data Explorer"),
        
        html.Div([
            html.Div([
                html.H4("Dataset Statistics"),
                html.Div(id="dataset-stats"),
            ], className="stats-panel"),
            
            html.Div([
                html.H4("Data Actions"),
                html.Button("Download New Data", id="download-data", className="action-button"),
                html.Button("Refresh Dataset", id="refresh-dataset", className="action-button"),
                html.Button("Export Metadata", id="export-metadata", className="action-button"),
            ], className="data-actions"),
        ], className="data-header"),
        
        html.Div([
            dcc.Graph(id="size-distribution"),
            dcc.Graph(id="category-distribution"),
        ], className="distribution-graphs"),
        
        html.Div([
            html.H4("Sample Browser"),
            html.Div(id="sample-browser"),
        ], className="browser-container"),
        
        html.Div([
            html.H4("Data Quality"),
            html.Div(id="data-quality"),
        ], className="quality-container"),
    ])


def create_metrics_display(metrics):
    """Create a metrics display component."""
    if not metrics:
        return html.Div("No metrics available")
    
    metric_items = []
    for key, value in metrics.items():
        metric_items.append(
            html.Div([
                html.Span(key.replace('_', ' ').title(), className="metric-label"),
                html.Span(f"{value:.4f}" if isinstance(value, float) else str(value), 
                         className="metric-value"),
            ], className="metric-item")
        )
    
    return html.Div(metric_items, className="metrics-grid")


def create_model_config_display(config):
    """Create a model configuration display."""
    if not config:
        return html.Div("No configuration available")
    
    config_items = []
    for key, value in config.items():
        config_items.append(
            html.Div([
                html.Strong(f"{key}: "),
                html.Span(str(value)),
            ], className="config-item")
        )
    
    return html.Div(config_items, className="config-list")


def create_progress_bar(current, total, label="Progress"):
    """Create a progress bar component."""
    percentage = (current / total * 100) if total > 0 else 0
    
    return html.Div([
        html.Label(label),
        html.Progress(value=current, max=total),
        html.Span(f"{current}/{total} ({percentage:.1f}%)", className="progress-text"),
    ], className="progress-container")


def create_status_indicator(status, message=""):
    """Create a status indicator component."""
    status_colors = {
        'training': '#28a745',
        'idle': '#6c757d', 
        'error': '#dc3545',
        'generating': '#007bff',
    }
    
    color = status_colors.get(status.lower(), '#6c757d')
    
    return html.Div([
        html.Div(className="status-dot", style={'background-color': color}),
        html.Span(status.title(), className="status-text"),
        html.Span(message, className="status-message") if message else None,
    ], className="status-indicator")