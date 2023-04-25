from fastapi import Path
from .scraper import main
from pathlib import PurePosixPath
from nbtschematic import SchematicFile
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset

# import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np


class SchematicDataset(Dataset):
    def __init__(
        self,
        threshold: int = 16,
        schematics_dir: str = "schematics",
        transform=None,
        metadata_file: str = "data.csv",
        download=False,
        **kwargs
    ):
        self.data_dir = schematics_dir
        self.transform = transform

        if download:
            main(**kwargs)

        self.metadata = pd.read_csv(metadata_file)
        self.enc = OneHotEncoder()
        self.enc.fit(self.metadata["Category"].values.reshape(-1, 1))

        if threshold:
            self.metadata = self.metadata[self.metadata["X"] <= threshold]
            self.metadata = self.metadata[self.metadata["Y"] <= threshold]
            self.metadata = self.metadata[self.metadata["Z"] <= threshold]

        self.threshold = (threshold, threshold, threshold)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx, return_metadata=False):
        # Handle slices
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        # Get and one-hot encode metadata
        metadata = self.metadata.iloc[idx].to_dict()
        path = PurePosixPath(metadata["Path"])
        category = self.enc.transform(
            np.array(metadata["Category"]).reshape(-1, 1)
        ).toarray()

        # Load schematic and convert to numpy array
        sf = SchematicFile.load(path)
        sf = np.array(sf.blocks)

        if self.transform:
            sf = self.transform(sf)

        # Pad to threshold
        sf = np.pad(
            sf, [(0, self.threshold[i] - sf.shape[i]) for i in range(len(sf.shape))]
        )

        if return_metadata is True:
            return torch.Tensor(sf), category, metadata

        return torch.Tensor(sf), category


class SchematicDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        threshold: int = 128,
        data_dir: str = "schematics",
        download=False,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.threshold = threshold
        self.data_dir = data_dir
        self.download = download
        self.kwargs = kwargs

    def prepare_data(self) -> None:
        if self.download:
            main()

    def setup(self, stage: str = ""):
        """
        TODO: `stage` here can be used for fit/test/predict stages.
        """
        self.dataset = SchematicDataset(
            threshold=self.threshold, schematics_dir=self.data_dir, **self.kwargs
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
