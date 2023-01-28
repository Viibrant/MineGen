import numpy as np
from nbtschematic import SchematicFile
from torch.utils.data import DataLoader, Dataset

from .util.download import generate_dataset
from .util.metadata import Metadata


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
            generate_dataset(**kwargs)
        # load your data here, for example:
        # self.schematics = load_schematics_from_folder(data_dir)
        self.metadata = Metadata(
            self.data_dir,
            metadata_file=metadata_file,
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        metadata = self.metadata[idx]
        sf = SchematicFile.load(metadata["Path"])

        if self.transform:
            sf = self.transform(np.array(sf.blocks))

        sample = {"data": sf, "metadata": metadata}

        return sample
