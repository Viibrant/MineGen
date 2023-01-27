from .util import *

CRITERIA = "most-downloaded"
NUM_PAGES = 400
SCHEMATICS_DIR = "schematics"
ERRORS_FILE = "schematics/errors.txt"
AUTH_URL = "https://www.minecraft-schematics.com/login/action/"

if __name__ == "__main__":
    generate_dataset(
        criteria=CRITERIA,
        num_pages=NUM_PAGES,
        SCHEMATICS_DIR=SCHEMATICS_DIR,
        ERRORS_FILE=ERRORS_FILE,
        AUTH_URL=AUTH_URL,
    )
