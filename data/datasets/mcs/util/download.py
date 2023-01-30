import gzip
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from csv import DictWriter
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from nbtschematic import SchematicFile
from tqdm import tqdm, trange

from .metadata import Metadata


def download_schematic(
    url: str, COOKIES=None, SCHEMATICS_DIR="schematics", metadata: Optional[Dict] = None
) -> Optional[Dict]:
    # Get ID from URL
    ID = url.split("/")[-2]
    # check if ID in any filenames
    p = Path(SCHEMATICS_DIR).glob("**/*.schematic")
    if any([ID in str(x) for x in p]):
        return

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
    assert sf.shape > (1, 1, 1)
    assert sf.blocks is not None
    assert "Blocks" in sf.root.keys()

    # Write shape
    metadata["Shape"] = sf.shape

    # Save schematic
    sf.save(file_path)

    return metadata


def find_urls(URL):
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, "html.parser")
    urls = soup.find_all("a", class_="btn btn-primary")
    urls = [url.get("href") for url in urls]
    urls = [url for url in urls if "/schematic/" in url]
    urls = [f"https://www.minecraft-schematics.com{url}" for url in urls]
    return urls


def generate_dataset(
    criteria="most-downloaded",
    num_pages=5,
    max_workers=256,
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
    # Initialise metadata file
    metadata_path = os.path.join(SCHEMATICS_DIR, "metadata.csv")

    # Create folder if it doesn't exist
    os.makedirs(SCHEMATICS_DIR, exist_ok=True)

    fields = [
        "Rating",
        "Category",
        "Theme",
        "Size",
        "File Format",
        "Submitted by",
        "Posted on",
        "Download(s)",
        "ID",
        "Name",
        "Path",
        "Shape",
    ]
    writer = DictWriter(open(metadata_path, "a"), fieldnames=fields)
    if os.path.getsize(metadata_path) == 0 or not os.path.exists(metadata_path):
        writer.writeheader()

    # Load credentials and authenticate
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

    list_metadata = []
    list_pages = [
        f"https://www.minecraft-schematics.com/{criteria}/{p+1}/" for p in range(a, b)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(
            total=len(list_pages), **kwargs_format, unit="page", desc="Finding URLs"
        ) as pbar:
            futures = []
            for page in list_pages:
                future = executor.submit(find_urls, page)
                future.add_done_callback(lambda p: pbar.update(1))
                futures.append(future)

            url_potential = []
            for future in futures:
                url_potential.extend(future.result())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(
            total=len(url_potential),
            **kwargs_format,
            unit="schematic",
            desc="Downloading",
        ) as pbar:
            futures = []

            for url in url_potential:
                future = executor.submit(
                    download_schematic, url, COOKIES, SCHEMATICS_DIR
                )
                future.add_done_callback(lambda p: pbar.update(1))
                futures.append(future)

            for future in futures:
                try:
                    metadata = future.result()
                    if isinstance(metadata, dict):
                        metadata["Path"] = os.path.join(
                            SCHEMATICS_DIR,
                            metadata["Category"],
                            f"{metadata['ID']}.schematic",
                        )
                        writer.writerow(metadata)
                except Exception as e:
                    traceback.print_exc(file=open(ERRORS_FILE, "a"), limit=1)
                    continue

    return list_metadata
