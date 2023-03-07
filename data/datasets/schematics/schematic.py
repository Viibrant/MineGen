from nbtschematic import SchematicFile
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import torch
from .scraper import main
import pandas as pd
import numpy as np


class SchematicDataset(Dataset):
    def __init__(
        self,
        threshold: int = 128,
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

        if threshold:
            self.metadata = self.metadata[self.metadata["X"] < threshold]
            self.metadata = self.metadata[self.metadata["Y"] < threshold]
            self.metadata = self.metadata[self.metadata["Z"] < threshold]

        self.threshold = (threshold, threshold, threshold)

        self.enc = OneHotEncoder()
        self.enc.fit(self.metadata["Category"].values.reshape(-1, 1))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx, return_metadata=False):

        # Handle slices
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        # Get and one-hot encode metadata
        metadata = self.metadata.iloc[idx].to_dict()
        category = self.enc.transform(
            np.array(metadata["Category"]).reshape(-1, 1)
        ).toarray()

        # Load schematic and convert to numpy array
        sf = SchematicFile.load(metadata["Path"])
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
