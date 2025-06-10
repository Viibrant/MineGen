"""MineGen: Minecraft Schematic Generator using Deep Learning."""

__version__ = "0.1.0"
__author__ = "Viibrant"
__email__ = "your.email@example.com"

from .config import Config
from .models import VAE, HierarchicalTransformerModel, AutoEncoder
from .data import SchematicDataModule, generate_dataset

__all__ = [
    "Config",
    "VAE", 
    "HierarchicalTransformerModel",
    "AutoEncoder",
    "SchematicDataModule",
    "generate_dataset",
]