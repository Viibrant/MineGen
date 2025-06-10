"""Configuration models using Pydantic."""

from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration."""
    device: str = "auto"
    num_classes: int = 20
    latent_dim: int = 64
    embedding_size: int = 512
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 12
    patch_size: int = 8


class SolverConfig(BaseModel):
    """Solver/optimizer configuration."""
    optimizer_name: str = "Adam"
    base_lr: float = 1e-3
    weight_decay: float = 1e-5
    momentum: float = 0.9
    bias_lr_factor: float = 2.0
    weight_decay_bias: float = 0.0
    max_epochs: int = 100
    log_period: int = 10
    checkpoint_period: int = 50


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    data_dir: str = "schematics"
    metadata_file: str = "data.csv"
    shape: List[int] = [16, 16, 16]
    threshold: Optional[int] = None
    train_split: float = 0.8
    val_split: float = 0.1


class DataLoaderConfig(BaseModel):
    """DataLoader configuration."""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True


class ScraperConfig(BaseModel):
    """Scraper configuration."""
    criteria: str = "most-downloaded"
    num_pages: int = 10
    max_workers: int = 12
    schematics_dir: str = "schematics"
    errors_file: str = ".errors.log"
    cred_file: str = ".credentials.yml"


class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    host: str = "localhost"
    port: int = 8050
    debug: bool = True
    auto_refresh: int = 5  # seconds


class Config(BaseModel):
    """Main configuration."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    datasets: DatasetConfig = Field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    scraper: ScraperConfig = Field(default_factory=ScraperConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    output_dir: str = "./outputs"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def update(self, **kwargs) -> "Config":
        """Update config with new values."""
        data = self.model_dump()
        data.update(kwargs)
        return Config(**data)