"""Utility functions for data processing and visualization."""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from nbtschematic import SchematicFile
from typing import Tuple, Optional


def to_schematic(obj: np.ndarray, shape: Tuple[int, int, int] = (16, 16, 16)) -> SchematicFile:
    """Convert numpy array to SchematicFile."""
    sf = SchematicFile(shape)
    sf.blocks = obj
    return sf


def plot_blockid_distribution(b_id: torch.Tensor, c: torch.Tensor, device: str = "cpu"):
    """
    Plot distribution of block IDs in generated schematics.
    
    Args:
        b_id: Block IDs tensor
        c: Counts tensor
        device: Device to use for computation
    """
    blockid_counts = torch.zeros(512, dtype=torch.int64, device=device)
    blockid_counts.scatter_add_(0, b_id.type(torch.int64), c)

    # Get count and percentage of air (block ID 0)
    air_count = blockid_counts[0].item()
    total_count = blockid_counts.sum().item()
    air_percentage = 100 * air_count / total_count

    # Remove block ID 0 from data
    non_air_indices = b_id != 0
    b_id_non_air = b_id[non_air_indices].cpu()
    c_non_air = c[non_air_indices].cpu()

    # Create pandas series without block ID 0
    blockid_counts_series = pd.Series(c_non_air.numpy(), index=b_id_non_air.numpy())

    # Create plot
    plt.figure(figsize=(12, 5))
    g = sns.barplot(x=blockid_counts_series.index, y=blockid_counts_series.values)
    g.set_title("Distribution of Block IDs in Generated Schematics (excluding Air)")
    g.set_xlabel("Block ID")
    g.set_ylabel("Count")
    _ = g.set_xticklabels(g.get_xticklabels(), rotation=90)

    # Add labels above each bar
    for i, v in enumerate(blockid_counts_series.values):
        g.text(i, v + 25, str(v), color="black", ha="center")

    # Add note about air
    note = f"Note: Air (Block ID 0) was excluded from this plot.\nThere were {air_count} Air blocks, comprising {air_percentage:.2f}% of total blocks."
    plt.figtext(0.5, -0.1, note, wrap=True, horizontalalignment="center", fontsize=12)

    return g


def analyze_size_distribution(df: pd.DataFrame, max_size: int = 256) -> None:
    """Analyze and plot size distribution of schematics."""
    # Create size categories
    df["Size"] = pd.cut(
        df[["X", "Y", "Z"]].max(axis=1),
        bins=[0, 32, 64, 128, max_size],
        labels=["Small", "Medium", "Large", "Huge"]
    )

    # Plot Y distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    g = sns.histplot(data=df, x="Y", hue="Size", multiple="stack", shrink=0.8, stat="probability", kde=True, bins=64)
    g.set_xticks(np.arange(0, max_size, 25))
    g.set_title("Y Distribution")
    plt.xlim(0, max_size)

    # Plot size distribution
    plt.subplot(2, 2, 2)
    g = sns.histplot(data=df, x="Size", shrink=0.8, stat="probability", hue="Size")
    g.set_title("Size Distribution")
    g.legend_.remove()

    # Plot theme distribution
    plt.subplot(2, 2, 3)
    theme_counts = df["Theme"].value_counts()
    g = sns.barplot(x=theme_counts.values, y=theme_counts.index)
    g.set_title("Theme Distribution")

    # Plot category distribution
    plt.subplot(2, 2, 4)
    category_counts = df["Category"].value_counts()
    g = sns.barplot(x=category_counts.values, y=category_counts.index)
    g.set_title("Category Distribution")
    
    plt.tight_layout()
    plt.show()


def filter_by_shape(df: pd.DataFrame, shape: Tuple[int, int, int]) -> pd.DataFrame:
    """Filter dataframe by maximum dimensions."""
    filtered = df[df["X"] <= shape[0]]
    filtered = filtered[filtered["Y"] <= shape[1]]
    filtered = filtered[filtered["Z"] <= shape[2]]
    
    print(f"Filtered from {len(df)} to {len(filtered)} schematics")
    return filtered


def get_category_names(enc) -> np.ndarray:
    """Get category names from encoder."""
    return enc.categories_[0]


def write_schematics(
    model, 
    x: torch.Tensor, 
    y: torch.Tensor, 
    cat_names: np.ndarray, 
    path: str = "generated/"
) -> None:
    """Write predicted schematics to files."""
    from pathlib import Path
    
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 2:
            # Model returns multiple outputs (like hierarchical transformer)
            x_hat, y_hat = model(x)
            x_hat = x_hat.argmax(dim=1) if x_hat.dim() > 4 else x_hat
        else:
            # Model returns single output
            x_hat = model(x)
            y_hat = torch.zeros(x.shape[0], len(cat_names))  # Dummy categories
    
    print(f"Generated blocks unique values: {torch.unique(x_hat)}")
    
    for i, _x in enumerate(x_hat):
        if y_hat.dim() > 1:
            gen_cat = cat_names[y_hat[i].argmax(dim=0)] if y_hat[i].sum() > 0 else "unknown"
            real_cat = cat_names[y[i].argmax(dim=0)] if y[i].sum() > 0 else "unknown"
        else:
            gen_cat = "unknown"
            real_cat = "unknown"
            
        gen_name = f"{i}_{gen_cat}.schematic"
        real_name = f"{i}_{real_cat}.schematic"
        
        # Save generated schematic
        schematic = _x.cpu().numpy() if isinstance(_x, torch.Tensor) else _x
        sf_gen = to_schematic(schematic)
        sf_gen.save(path / f"gen_{gen_name}")

        # Save real schematic
        sf_real = to_schematic(x[i].cpu().numpy())
        sf_real.save(path / f"real_{real_name}")

    print(f"Saved {len(x_hat)} schematic pairs to {path}")