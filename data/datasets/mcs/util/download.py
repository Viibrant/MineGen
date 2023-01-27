import gzip
import os
import requests
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from io import BytesIO
from typing import Dict, Optional
from bs4 import BeautifulSoup
from tqdm import tqdm, trange
from nbtschematic import SchematicFile
from . import Metadata


def download_schematic(
    url: str, COOKIES=None, SCHEMATICS_DIR="schematics", metadata: Optional[Dict] = None
) -> None:
    # Get ID from URL
    ID = url.split("/")[-1]

    # Request schematic file
    r = requests.get(
        f"{url}download/action/",
        cookies=COOKIES,
        params={"type": "schematic"},
    )

    # Get metadata if not provided
    if metadata is None:
        metadata = Metadata.get_metadata(url)

    # Create folder if it doesn't exist
    category_dir = os.path.join(SCHEMATICS_DIR, metadata["Category"])
    os.makedirs(category_dir, exist_ok=True)

    # File path
    file_path = os.path.join(category_dir, f"{ID}.schematic")

    # Check if file already exists
    if os.path.exists(file_path):
        return

    # Check if file is gzipped
    gzipped = BytesIO(r.content).read(3) == b"\x1f\x8b\x08"

    # Open file
    schem_gen = gzip.open(BytesIO(r.content)) if gzipped else BytesIO(r.content)

    # Write to file
    sf = SchematicFile.from_fileobj(schem_gen)

    # Integrity checks
    assert np.array(sf.shape) > (1, 1, 1)
    assert sf.blocks
    assert "BlockData" in sf.root.keys()
    sf.save(file_path)


def generate_dataset(
    criteria="most-downloaded",
    num_pages=5,
    interval=None,
    SCHEMATICS_DIR="schematics",
    ERRORS_FILE="schematics/errors.txt",
    CRED_FILE=".credentials.yml",
    AUTH_URL="https://www.minecraft-schematics.com/login/action/",
):
    """
    Generates a dataset of schematics from minecraft-schematics.com.

    Parameters
    ----------
    criteria : str, optional
        The criteria to use for sorting the schematics. Must be one of
        ["latest", "top-rated", "most-downloaded"], by default "most-downloaded"
    num_pages : int, optional
        The number of pages to scrape, by default 5
    interval : tuple, optional
        The interval of pages to scrape, by default None
    SCHEMATICS_DIR : str, optional
        The directory to save the schematics to, by default "schematics"
    ERRORS_FILE : str, optional
        The file to save errors to, by default "schematics/errors.txt"
    CRED_FILE : str, optional
        The file to load credentials from, by default ".credentials.yml"
    AUTH_URL : str, optional
        The URL to authenticate with, by default "https://www.minecraft-schematics.com/login/action/"
    """

    with open(CRED_FILE, "r") as credfile:
        cred = yaml.safe_load(credfile)

    s = requests.Session()
    s.post(AUTH_URL, data=cred)
    COOKIES = s.cookies.get_dict()

    # Check for interval
    a, b = 0, num_pages + 1
    if interval is not None:
        a, b = interval

    # Check for valid criteria
    valid_criteria = ["latest", "top-rated", "most-downloaded"]
    if criteria not in valid_criteria:
        raise ValueError(f"Criteria must be one of {valid_criteria}, got {criteria}.")

    # Set up variables for tqdm, loop
    kwargs_format = dict(
        bar_format="{desc:30.30}{percentage:3.0f}%|{bar:100}{r_bar:50.50}",
        leave=False,
    )

    TOTAL = num_pages * 18
    root = f"https://www.minecraft-schematics.com/{criteria}"
    list_metadata = []

    bar_page = trange(
        a, b, **kwargs_format, unit="page", desc=f"0 downloaded / {TOTAL} schematics"
    )

    for page in bar_page:
        # Get page content
        r = requests.get(f"{root}/{page}/", cookies=COOKIES)
        soup = BeautifulSoup(r.text, "html.parser")

        # Get schematic links from download buttons
        download_buttons = soup.find_all("a", class_="btn btn-primary")
        links = [button.get("href") for button in download_buttons]
        links = [f"https://www.minecraft-schematics.com{link}" for link in links]

        pbar = tqdm(links, **kwargs_format, unit="schem")

        # Iterate over all links for downloadable on page
        for link in pbar:

            # Grab metadata
            metadata = get_metadata(link)
            pbar.set_description(f"{metadata['Name']:30.30}")

            path = Path(
                "schematics", metadata["Category"], f"{metadata['ID']}.schematic"
            )

            if path.exists():
                pbar.set_description(f"{'Already exists (skipped)':30.30}")
                continue

            if "File Format" not in metadata:
                pbar.set_description(f"{pbar.desc:28.28}" + "\u274c")
                continue

            if metadata["File Format"] == ".schematic":
                try:
                    download_schematic(link, metadata=metadata)
                except Exception as e:
                    pbar.set_description(f"{pbar.desc:28.28}" + "\u274c")
                    with open(ERRORS_FILE, "a") as f:
                        f.write(
                            f"{metadata['ID']}, {metadata['Name']}, {metadata['Category']}, {e}\n"
                        )
                    continue

                list_metadata.append(metadata)

                pbar.set_description(f"{metadata['Name']:30.30}" + " " + "\u2713")
                bar_page.set_description(
                    f"{int(bar_page.desc.split(' downloaded')[0])+1} downloaded / {(page+1)*18} schematics"
                )

    df = pd.DataFrame(list_metadata)
    df.to_csv(f"schematics/{criteria}.csv", index=False)
