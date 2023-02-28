from torch.utils.data import DataLoader
from .datasets.schematics import SchematicDataset
from .transforms import build_transforms


def build_dataset(transform, is_train=True, threshold=128, **kwargs):
    dataset = SchematicDataset(transform=transform, threshold=128, **kwargs)
    return dataset


def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False

    transforms = build_transforms(cfg, is_train)
    dataset = build_dataset(transforms, is_train, threshold=cfg.DATASETS.THRESHOLD)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader
