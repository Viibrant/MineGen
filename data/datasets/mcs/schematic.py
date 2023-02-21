from nbtschematic import SchematicFile
from torch.utils.data import DataLoader, Dataset
from .util import main
import pandas as pd
import numpy as np


class SchematicDataset(Dataset):
    def __init__(
        self,
        schematics_dir: str = "schematics",
        transform=None,
        metadata_file: str = "metadata.csv",
        download=False,
        **kwargs
    ):
        self.data_dir = schematics_dir
        self.transform = transform

        if download:
            main(**kwargs)

        self.metadata = pd.read_csv(metadata_file)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        metadata = self.metadata.iloc[idx]
        sf = SchematicFile.load(metadata["Path"])

        if self.transform:
            sf = self.transform(np.array(sf.blocks))

        sample = {"data": sf, "metadata": metadata}

        return sample
