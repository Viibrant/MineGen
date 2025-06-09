"""Main CLI entrypoint for MineGen."""

import logging
from pathlib import Path
from typing import Optional

import click
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from ..config.models import Config
from ..data import SchematicDataModule
from ..models import HierarchicalTransformerModel, VAE
from ..engine import train
from ..solver import make_optimiser


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: bool) -> None:
    """MineGen: Minecraft Schematic Generator"""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Load config
    if config:
        cfg = Config.from_yaml(config)
    else:
        cfg = Config()
    
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg


@cli.command()
@click.option("--model-type", type=click.Choice(["vae", "transformer"]), default="vae")
@click.option("--batch-size", type=int, default=32)
@click.option("--max-epochs", type=int, default=100)
@click.option("--learning-rate", type=float, default=1e-3)
@click.option("--output-dir", type=click.Path(), default="./outputs")
@click.option("--use-wandb", is_flag=True, help="Use Weights & Biases logging")
@click.pass_context
def train_model(
    ctx: click.Context,
    model_type: str,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    output_dir: str,
    use_wandb: bool,
) -> None:
    """Train a model."""
    cfg = ctx.obj["config"]
    
    # Update config with CLI args
    cfg.solver.max_epochs = max_epochs
    cfg.solver.base_lr = learning_rate
    cfg.output_dir = output_dir
    
    # Setup data
    data_module = SchematicDataModule(
        batch_size=batch_size,
        num_workers=cfg.dataloader.num_workers,
        threshold=cfg.datasets.threshold,
    )
    
    # Setup model
    if model_type == "vae":
        model = VAE(
            latent_dim=64,
            embedding_size=512,
            num_blocks=512,
            num_categories=cfg.model.num_classes,
        )
    else:  # transformer
        model = HierarchicalTransformerModel(
            in_size=16,
            c_embed=64,
            num_cat=cfg.model.num_classes,
            num_heads=8,
            num_layers=12,
        )
    
    # Setup trainer
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(output_dir) / "checkpoints",
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            save_top_k=3,
        ),
        EarlyStopping(monitor="val_loss", patience=10),
    ]
    
    logger = None
    if use_wandb:
        logger = WandbLogger(project="minegen", name=f"{model_type}_training")
    
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices="auto",
    )
    
    # Train
    trainer.fit(model, data_module)
    
    click.echo(f"Training completed! Model saved to {output_dir}")


@cli.command()
@click.option("--checkpoint", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(), default="./generated")
@click.option("--num-samples", type=int, default=10)
@click.option("--category", type=str, help="Target category for generation")
@click.pass_context
def generate(
    ctx: click.Context,
    checkpoint: str,
    output_dir: str,
    num_samples: int,
    category: Optional[str],
) -> None:
    """Generate schematics from a trained model."""
    cfg = ctx.obj["config"]
    
    # Load model from checkpoint
    if "vae" in checkpoint.lower():
        model = VAE.load_from_checkpoint(checkpoint)
    else:
        # For transformer models, we'd need to implement a Lightning wrapper
        click.echo("Transformer generation not yet implemented")
        return
    
    model.eval()
    
    # Generate samples
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i in range(num_samples):
            # Sample from latent space
            z = torch.randn(1, model.latent_dim)
            
            # Generate
            recon_x = model.decode(z)
            
            # Convert to schematic and save
            blocks = torch.argmax(recon_x, dim=1).squeeze().numpy()
            
            # Save as .npy for now (could integrate nbtschematic here)
            output_file = output_path / f"generated_{i:03d}.npy"
            torch.save(blocks, output_file)
    
    click.echo(f"Generated {num_samples} schematics in {output_dir}")


@cli.command()
@click.option("--data-dir", type=click.Path(), default="./schematics")
@click.option("--criteria", type=click.Choice(["most-downloaded", "top-rated", "latest"]), default="most-downloaded")
@click.option("--num-pages", type=int, default=10)
@click.option("--max-workers", type=int, default=12)
@click.pass_context
def download_data(
    ctx: click.Context,
    data_dir: str,
    criteria: str,
    num_pages: int,
    max_workers: int,
) -> None:
    """Download schematic data."""
    from ..data.scraper import generate_dataset
    
    click.echo(f"Downloading {criteria} schematics to {data_dir}...")
    
    # This would use the scraper from the notebooks
    # For now, just show what would happen
    click.echo(f"Would download {num_pages} pages of {criteria} schematics")
    click.echo("Note: Scraper integration pending")


def main() -> None:
    """Main entrypoint."""
    cli()


if __name__ == "__main__":
    main()