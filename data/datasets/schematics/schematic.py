from .scraper import main
from pathlib import PurePosixPath
from nbtschematic import SchematicFile
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset, random_split

import lightning.pytorch as pl
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
        # TODO! Hacky error handling
        while True:
            try:
                sf = SchematicFile.load(path)
                sf = torch.abs(torch.tensor(sf.blocks).long())
                break
            except Exception as e:
                # Sample a different schematic
                return self.__getitem__(
                    np.random.randint(0, len(self)), return_metadata=return_metadata
                )

        if self.transform:
            sf = self.transform(sf)

        if return_metadata is True:
            return sf, category, metadata

        return sf, category


def custom_collate_fn(batch, size=16):
    # extract the tensors and categories from the batch
    batch_sf, batch_category = zip(*batch)

    # compute the maximum shape of the tensors in batch_sf
    padded_sf = torch.stack(
        [
            torch.from_numpy(
                np.pad(
                    sf,
                    [(0, size - sf.shape[i]) for i in range(len(sf.shape))],
                )
            )
            for sf in batch_sf
        ]
    )

    batch_category = torch.cat([torch.from_numpy(cat) for cat in batch_category])

    # return padded_sf, torch.stack(batch_category)
    return padded_sf, batch_category


class SchematicDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64, num_workers: int = 4, **dataset_kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs

    def setup(self, stage=None):
        if stage in (None, "fit"):
            full_dataset = SchematicDataset(**self.dataset_kwargs)
            print(len(full_dataset))
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )

        if stage in (None, "test"):
            self.test_dataset = SchematicDataset(**self.dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )
