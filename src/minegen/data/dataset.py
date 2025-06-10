"""Dataset implementation for schematic data."""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from sklearn.preprocessing import OneHotEncoder
from nbtschematic import SchematicFile
from pathlib import Path
from typing import Optional, Tuple, Union


class SchematicDataset(Dataset):
    """Dataset for loading Minecraft schematics."""

    def __init__(
        self,
        data_dir: str = "schematics",
        metadata_file: str = "data.csv",
        shape: Tuple[int, int, int] = (16, 16, 16),
        threshold: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.shape = shape
        self.threshold = threshold

        # Load metadata
        if os.path.exists(metadata_file):
            self.metadata = pd.read_csv(metadata_file)
        else:
            raise FileNotFoundError(f"Metadata file {metadata_file} not found")

        # Filter by size if threshold provided
        if threshold:
            self.metadata = self.metadata[
                (self.metadata["X"] <= threshold) &
                (self.metadata["Y"] <= threshold) &
                (self.metadata["Z"] <= threshold)
            ]

        # Filter by shape
        self.metadata = self.metadata[
            (self.metadata["X"] <= shape[0]) &
            (self.metadata["Y"] <= shape[1]) &
            (self.metadata["Z"] <= shape[2])
        ]

        # Setup category encoding
        self.enc = OneHotEncoder(sparse_output=False)
        categories = self.metadata["Category"].values.reshape(-1, 1)
        self.enc.fit(categories)

        # Filter existing files
        self.metadata = self.metadata[
            self.metadata["Path"].apply(lambda x: os.path.exists(x) if pd.notna(x) else False)
        ].reset_index(drop=True)

        print(f"Loaded {len(self.metadata)} schematics")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a schematic and its category."""
        row = self.metadata.iloc[idx]
        
        # Load schematic
        try:
            sf = SchematicFile.load(row["Path"])
            blocks = np.array(sf.blocks)
            
            # Pad or crop to target shape
            blocks = self._resize_blocks(blocks, self.shape)
            
        except Exception as e:
            print(f"Error loading {row['Path']}: {e}")
            # Return zeros if loading fails
            blocks = np.zeros(self.shape, dtype=np.int32)

        # Get category
        category = self.enc.transform([[row["Category"]]])[0]

        return torch.tensor(blocks, dtype=torch.float32), torch.tensor(category, dtype=torch.float32)

    def _resize_blocks(self, blocks: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Resize blocks to target shape by padding or cropping."""
        current_shape = blocks.shape
        result = np.zeros(target_shape, dtype=blocks.dtype)
        
        # Calculate copy region
        copy_shape = tuple(min(c, t) for c, t in zip(current_shape, target_shape))
        
        # Copy data
        result[:copy_shape[0], :copy_shape[1], :copy_shape[2]] = \
            blocks[:copy_shape[0], :copy_shape[1], :copy_shape[2]]
        
        return result


class SchematicDataModule(LightningDataModule):
    """Lightning data module for schematic data."""

    def __init__(
        self,
        data_dir: str = "schematics",
        metadata_file: str = "data.csv",
        batch_size: int = 32,
        num_workers: int = 4,
        shape: Tuple[int, int, int] = (16, 16, 16),
        threshold: Optional[int] = None,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shape = shape
        self.threshold = threshold
        self.train_split = train_split
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        # Create full dataset
        full_dataset = SchematicDataset(
            self.data_dir,
            self.metadata_file,
            self.shape,
            self.threshold
        )

        # Split dataset
        total_size = len(full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self) -> DataLoader:
        """Training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )