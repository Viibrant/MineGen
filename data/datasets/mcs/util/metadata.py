import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from csv import DictReader, DictWriter
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class Metadata:
    """A class for generating a CSV file containing metadata for all schematics in a directory."""

    @staticmethod
    def get_metadata(url=None, soup=None):
        """
        Scrapes the metadata from a schematic page.

        Returns
        metadata : dict
            A dictionary containing the metadata.
        """
        if not url and not soup:
            raise ValueError("Must provide either url or soup.")
        if url:
            soup = BeautifulSoup(requests.get(url).text, "html.parser")

        title = soup.h1.text
        table = soup.table

        raw = [p.strip() for p in table.text.split("\n") if p]
        keys = raw[0::2]
        values = raw[1::2]

        metadata = dict(zip(keys, values))
        metadata["ID"] = int(soup.title.text.split("#")[-1])
        metadata["Name"] = title
        metadata["Rating"] = float(metadata["Rating"].split(" ")[1])
        metadata["Download(s)"] = int(metadata["Download(s)"].split(" ")[-2])
        return metadata

    def __init__(
        self,
        schematics_dir: str = "schematics",
        metadata_file: str = "metadata.csv",
    ):
        self.schematics_dir = schematics_dir
        self.metadata_file = os.path.join(schematics_dir, metadata_file)
        self.metadata = {}
        self.fields = [
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
        ]

        if os.path.exists(self.metadata_file) and os.path.getsize(self.metadata_file):
            self.df = pd.read_csv(self.metadata_file)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def __len__(self):
        return len(self.df)

    def write_metadata(self, max_workers=256):
        """Append metadata to CSV file"""
        if self._get_schematic_ids() == []:
            print("No schematics found.")
            return
        self._scrape_metadata(max_workers=max_workers)

    def _get_schematic_ids(self):
        """Gets all schematic IDs from schematics directory"""

        Path(self.schematics_dir).mkdir(parents=True, exist_ok=True)

        schematic_ids = []
        for category_dir in os.listdir(self.schematics_dir):
            category_path = os.path.join(self.schematics_dir, category_dir)
            if os.path.isdir(category_path):
                for schematic_file in os.listdir(category_path):
                    schematic_id = schematic_file.split(".")[0]
                    schematic_ids.append(schematic_id)
        return schematic_ids

    def _get_existing_ids(self):
        """Gets existing schematic IDs from metadata file"""
        existing_ids = []
        with open(self.metadata_file, "r") as file:
            reader = DictReader(file)
            for row in reader:
                existing_ids.append(row["ID"])
        return existing_ids

    def _scrape_metadata(self, max_workers=256):
        """Scrapes metadata for all IDs not found in metadata file"""

        f = DictWriter(open(self.metadata_file, "w"), fieldnames=self.fields)

        # Get all schematic IDs from schematics directory
        schematic_ids = self._get_schematic_ids()

        # Get existing IDs from metadata file
        if os.path.exists(self.metadata_file):
            existing_ids = self._get_existing_ids()
        else:
            existing_ids = []
            f.writeheader()

        # Scrape metadata for new IDs
        new_ids = list(set(schematic_ids) - set(existing_ids))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(
                    self.get_metadata,
                    f"https://www.minecraft-schematics.com/schematic/{id}/",
                ): id
                for id in tqdm(new_ids)
            }
            for future in tqdm(as_completed(future_to_id), total=len(new_ids)):
                try:
                    metadata = future.result()
                    metadata["Path"] = os.path.join(
                        self.schematics_dir,
                        metadata["Category"],
                        f"{metadata['ID']}.schematic",
                    )
                    f.writerow(metadata)
                except Exception as exc:
                    pass
