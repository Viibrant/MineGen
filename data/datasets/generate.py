from schematics import SchematicDataset
from mcs.util.download import generate_dataset

CRITERIA = "most-downloaded"
NUM_PAGES = 800
MAX_WORKERS = 1024
SCHEMATICS_DIR = "schematics"
ERRORS_FILE = "schematics/errors.txt"
AUTH_URL = "https://www.minecraft-schematics.com/login/action/"

if __name__ == "__main__":
    SchematicDataset(
        criteria=CRITERIA,
        num_pages=NUM_PAGES,
        max_workers=MAX_WORKERS,
        SCHEMATICS_DIR=SCHEMATICS_DIR,
        ERRORS_FILE=ERRORS_FILE,
        AUTH_URL=AUTH_URL,
        download=True,
    )
