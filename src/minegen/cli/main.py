"""Main CLI entrypoint for MineGen."""

import logging
from pathlib import Path
from typing import Optional
import threading

import click
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from ..config.models import Config
from ..data import SchematicDataModule, generate_dataset
from ..models import (
    HierarchicalTransformerModel, 
    VAE, 
    AutoEncoder, 
    VoxelAutoencoder, 
    ExperimentalModel
)
from ..engine import train
from ..solver import make_optimiser
from ..dashboard import run_dashboard
from ..data.utils import write_schematics


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
@click.option("--model-type", type=click.Choice([
    "vae", "transformer", "autoencoder", "voxel", "experimental"
]), default="vae")
@click.option("--batch-size", type=int, default=32)
@click.option("--max-epochs", type=int, default=100)
@click.option("--learning-rate", type=float, default=1e-3)
@click.option("--output-dir", type=click.Path(), default="./outputs")
@click.option("--use-wandb", is_flag=True, help="Use Weights & Biases logging")
@click.option("--resume-from", type=click.Path(exists=True), help="Resume from checkpoint")
@click.pass_context
def train_model(
    ctx: click.Context,
    model_type: str,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    output_dir: str,
    use_wandb: bool,
    resume_from: Optional[str],
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
            latent_dim=cfg.model.latent_dim,
            embedding_size=cfg.model.embedding_size,
            hidden_dim=cfg.model.hidden_dim,
            learning_rate=learning_rate,
        )
    elif model_type == "transformer":
        model = HierarchicalTransformerModel(
            in_size=16,
            c_embed=64,
            num_cat=cfg.model.num_classes,
            num_heads=cfg.model.num_heads,
            num_layers=cfg.model.num_layers,
        )
    elif model_type == "autoencoder":
        model = AutoEncoder(
            num_categories=cfg.model.num_classes,
            learning_rate=learning_rate,
        )
    elif model_type == "voxel":
        model = VoxelAutoencoder(
            embedding_dim=cfg.model.latent_dim,
            learning_rate=learning_rate,
        )
    elif model_type == "experimental":
        model = ExperimentalModel(
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            learning_rate=learning_rate,
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
    if resume_from:
        trainer.fit(model, data_module, ckpt_path=resume_from)
    else:
        trainer.fit(model, data_module)
    
    click.echo(f"Training completed! Model saved to {output_dir}")


@cli.command()
@click.option("--checkpoint", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(), default="./generated")
@click.option("--num-samples", type=int, default=10)
@click.option("--category", type=str, help="Target category for generation")
@click.option("--save-schematics", is_flag=True, help="Save as .schematic files")
@click.pass_context
def generate(
    ctx: click.Context,
    checkpoint: str,
    output_dir: str,
    num_samples: int,
    category: Optional[str],
    save_schematics: bool,
) -> None:
    """Generate schematics from a trained model."""
    cfg = ctx.obj["config"]
    
    # Load model from checkpoint
    if "vae" in checkpoint.lower():
        model = VAE.load_from_checkpoint(checkpoint)
    elif "autoencoder" in checkpoint.lower():
        model = AutoEncoder.load_from_checkpoint(checkpoint)
    elif "voxel" in checkpoint.lower():
        model = VoxelAutoencoder.load_from_checkpoint(checkpoint)
    elif "experimental" in checkpoint.lower():
        model = ExperimentalModel.load_from_checkpoint(checkpoint)
    else:
        # Try to load as transformer
        model = HierarchicalTransformerModel.load_from_checkpoint(checkpoint)
    
    model.eval()
    
    # Generate samples
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i in range(num_samples):
            if hasattr(model, 'generate'):
                # Use model's generate method if available
                blocks = model.generate(1)
            else:
                # Sample from latent space for VAE-like models
                if hasattr(model, 'latent_dim'):
                    z = torch.randn(1, model.latent_dim)
                    if hasattr(model, 'decode'):
                        recon_x = model.decode(z)
                        blocks = torch.argmax(recon_x, dim=1).squeeze().numpy()
                    else:
                        blocks = torch.zeros(16, 16, 16).numpy()
                else:
                    # For other models, generate random input and get output
                    x = torch.randint(0, 256, (1, 16, 16, 16))
                    output = model(x)
                    if isinstance(output, tuple):
                        blocks = output[0].argmax(dim=1).squeeze().numpy()
                    else:
                        blocks = output.argmax(dim=1).squeeze().numpy()
            
            # Save as .npy or .schematic
            if save_schematics:
                from ..data.utils import to_schematic
                sf = to_schematic(blocks)
                output_file = output_path / f"generated_{i:03d}.schematic"
                sf.save(str(output_file))
            else:
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
    click.echo(f"Downloading {criteria} schematics to {data_dir}...")
    
    try:
        df = generate_dataset(
            criteria=criteria,
            num_pages=num_pages,
            max_workers=max_workers,
            schematics_dir=data_dir,
        )
        click.echo(f"Successfully downloaded {len(df)} schematics!")
    except Exception as e:
        click.echo(f"Error downloading data: {e}")


@cli.command()
@click.option("--host", default="localhost", help="Dashboard host")
@click.option("--port", type=int, default=8050, help="Dashboard port")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.pass_context
def dashboard(
    ctx: click.Context,
    host: str,
    port: int,
    debug: bool,
) -> None:
    """Launch the real-time dashboard."""
    cfg = ctx.obj["config"]
    
    # Update dashboard config
    cfg.dashboard.host = host
    cfg.dashboard.port = port
    cfg.dashboard.debug = debug
    
    click.echo(f"🚀 Launching MineGen Dashboard at http://{host}:{port}")
    
    # Start background tasks
    from ..dashboard.app import start_background_tasks
    start_background_tasks()
    
    # Run dashboard
    run_dashboard(cfg, host, port)


@cli.command()
@click.option("--checkpoint", type=click.Path(exists=True), required=True)
@click.option("--data-dir", type=click.Path(), default="./schematics")
@click.option("--output-dir", type=click.Path(), default="./evaluation")
@click.option("--num-samples", type=int, default=100)
@click.pass_context
def evaluate(
    ctx: click.Context,
    checkpoint: str,
    data_dir: str,
    output_dir: str,
    num_samples: int,
) -> None:
    """Evaluate a trained model."""
    cfg = ctx.obj["config"]
    
    # Load model
    if "vae" in checkpoint.lower():
        model = VAE.load_from_checkpoint(checkpoint)
    else:
        model = HierarchicalTransformerModel.load_from_checkpoint(checkpoint)
    
    # Load data
    data_module = SchematicDataModule(
        data_dir=data_dir,
        batch_size=32,
        num_workers=cfg.dataloader.num_workers,
    )
    data_module.setup("test")
    
    # Run evaluation
    trainer = Trainer(accelerator="auto", devices="auto")
    results = trainer.test(model, data_module)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_path / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    click.echo(f"Evaluation completed! Results saved to {output_dir}")


@cli.command()
@click.pass_context
def config_template(ctx: click.Context) -> None:
    """Generate a configuration template."""
    cfg = Config()
    cfg.to_yaml("config_template.yaml")
    click.echo("Configuration template saved to config_template.yaml")


def main() -> None:
    """Main entrypoint."""
    cli()


if __name__ == "__main__":
    main()