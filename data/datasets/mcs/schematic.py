from nbtschematic import SchematicFile
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
from .util import main
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

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx, metadata=False):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        metadata = self.metadata.iloc[idx].to_dict()
        sf = SchematicFile.load(metadata["Path"])
        sf = np.array(sf.blocks)

        if self.transform:
            sf = self.transform(sf)

        # Pad to threshold
        sf = np.pad(
            sf, [(0, self.threshold[i] - sf.shape[i]) for i in range(len(sf.shape))]
        )
        return torch.Tensor(sf)
