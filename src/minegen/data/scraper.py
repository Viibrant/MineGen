"""
Schematic scraper functionality extracted from notebooks.
This integrates the scraper from data/datasets/schematics/scraper.py
"""

import asyncio
import gzip
import os
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any

import aiofiles
import aiohttp
import numpy as np
import pandas as pd
import yaml
from bs4 import BeautifulSoup
from httpx import AsyncClient
from nbtschematic import SchematicFile
from tqdm.asyncio import tqdm

BASE_URL = "https://www.minecraft-schematics.com"
AUTH_URL = BASE_URL + "/login/action/"
CRITERIA = ["most-downloaded", "top-rated", "latest"]


class CriteriaPage:
    """A page of schematics on minecraft-schematics.com."""

    def __init__(self, criteria: str, page: int, schematics_dir: str = "schematics"):
        assert criteria in CRITERIA, f"criteria must be one of {CRITERIA}"
        self.root = f"https://www.minecraft-schematics.com/{criteria}/{page}/"
        self.criteria = criteria
        self.page = page
        self.urls = None
        self.schematics_dir = f"{schematics_dir}/{criteria}"
        os.makedirs(self.schematics_dir, exist_ok=True)

    def __repr__(self):
        return f"CriteriaPage(criteria={self.criteria}, page={self.page})"

    async def get_candidates(self, session, tries: int = 5) -> Optional[List[str]]:
        """Get the URLs of the schematics on the page"""
        while tries > 0:
            try:
                response = await session.get(self.root, timeout=10)
            except Exception as e:
                tries -= 1
                if tries == 0:
                    return None
            else:
                self.soup = BeautifulSoup(response.text, "html.parser")
                self.urls = self._url_parse()
                return self.urls

    def _url_parse(self) -> List[str]:
        """Parse the URLs of the schematics on the page"""
        urls = self.soup.find_all("a", class_="btn btn-primary")
        urls = [url.get("href") for url in urls]
        urls = [url for url in urls if "/schematic/" in url]
        urls = [f"https://www.minecraft-schematics.com{url}" for url in urls]
        return urls

    async def get_metadata(self, session, url: str, sem, tries: int = 3) -> Dict[str, Any]:
        """Get the metadata of a schematic"""
        while tries > 0:
            try:
                async with sem:
                    response = await session.get(url, timeout=20)
            except Exception as e:
                print(e)
                tries -= 1
                if tries == 0:
                    return dict()
            else:
                soup = BeautifulSoup(response.text, "html.parser")
                metadata = self._metadata_parse(url, soup)
                return metadata

    def _metadata_parse(self, url: str, soup: BeautifulSoup) -> Dict[str, Any]:
        """Parse the metadata of a schematic"""
        assert soup.h1, "h1 tag not found in HTML"
        assert soup.table, "table tag not found in HTML"
        assert soup.title, "title tag not found in HTML"

        # Parse metadata and convert to dict
        table = soup.table.text
        raw = [part.strip() for part in table.split("\n") if part]
        keys = raw[0::2]
        values = raw[1::2]
        metadata = dict(zip(keys, values))

        # Convert to correct types
        metadata["ID"] = int(url.split("/")[-2])
        metadata["Name"] = soup.h1.text
        metadata["Rating"] = float(metadata["Rating"].split(" ")[1])
        metadata["Download(s)"] = int(metadata["Download(s)"].split(" ")[-2])

        # Parse file format
        file_format = metadata.get("File Format")
        if file_format is None:
            metadata["File Format"] = None
            metadata["Path"] = None
        else:
            metadata["Path"] = os.path.join(
                self.schematics_dir,
                str(metadata["ID"]) + file_format,
            )

        metadata["URL"] = url
        metadata["Page"] = self.page
        return metadata

    @staticmethod
    async def download(
        metadata: Dict[str, Any], 
        session, 
        sem, 
        path: Optional[str] = None, 
        skip: bool = True
    ) -> Dict[str, Any]:
        """Download a schematic"""
        if skip and os.path.exists(metadata.get("Path")):
            return metadata

        try:
            async with sem:
                url = metadata.get("URL")
                path = metadata.get("Path") if path is None else path

                # Get download link
                response = await session.get(
                    url + "download/action/",
                    params={"type": "schematic"},
                    timeout=20,
                )
                try:
                    sf = SchematicFile.from_fileobj(BytesIO(response.content))
                except KeyError:
                    # file is gzipped
                    sf = SchematicFile.from_fileobj(
                        gzip.open(BytesIO(response.content))
                    )
                assert sf.shape != (1, 1, 1), "Schematic is empty"
                assert np.asarray(sf.blocks) is not None, "Blocks is empty"
                assert "Blocks" in sf.root.keys(), "Blocks not in root"
                assert path is not None, "Path is None"

                # Write schematic
                async with aiofiles.open(path, "wb") as f:
                    await f.write(response.content)

                metadata["Y"] = int(sf.shape[0])
                metadata["Z"] = int(sf.shape[1])
                metadata["X"] = int(sf.shape[2])
                return metadata

        except Exception as e:
            return (metadata.get("URL"), e)


async def generate_dataset(
    criteria: str = "most-downloaded",
    num_pages: int = 5,
    max_workers: int = 10,
    interval: Optional[tuple] = None,
    schematics_dir: str = "schematics",
    errors_file: str = ".errors.log",
    cred_file: str = ".credentials.yml",
    auth_url: str = "https://www.minecraft-schematics.com/login/action/",
) -> pd.DataFrame:
    """Generate a dataset of schematics from minecraft-schematics.com."""
    # Create folder if it doesn't exist
    os.makedirs(schematics_dir, exist_ok=True)

    # Check interval
    a, b = (1, num_pages) if interval is None else interval

    # Check for valid criteria
    valid_criteria = ["latest", "top-rated", "most-downloaded"]
    if criteria not in valid_criteria:
        raise ValueError(f"Criteria must be one of {valid_criteria}, got {criteria}.")

    async with AsyncClient() as session:
        sem = asyncio.Semaphore(max_workers)
        
        # Load credentials if available
        if os.path.exists(cred_file):
            creds = yaml.safe_load(open(cred_file, "r"))
            await session.post(auth_url, data=creds)

        found_urls = []

        # Execute all requests
        print("Getting candidates...")
        list_obj = [CriteriaPage(criteria, p + 1) for p in range(a, b)]
        tasks = [asyncio.create_task(page.get_candidates(session)) for page in list_obj]

        # Check if any schematics already exist
        existing_schematics = []
        if os.path.exists(schematics_dir):
            existing_schematics = os.listdir(schematics_dir)
            # Get files that are in form <int>.<format>
            existing_schematics = [
                x for x in existing_schematics if "." in x and x.split(".")[0].isdigit()
            ]
            # Get IDs of existing schematics
            existing_schematics = [int(x.split(".")[0]) for x in existing_schematics]

        # For each discovered URL, get metadata
        for result in await tqdm.gather(*tasks):
            if result is not None:
                found_urls.extend(result)

        # Filter out existing schematics
        found_urls = [
            x for x in found_urls 
            if int(x.split("/")[-2]) not in existing_schematics
        ]

        print("Getting metadata...")
        # Get metadata
        tasks = [
            asyncio.create_task(
                CriteriaPage(criteria, 0).get_metadata(session, url, sem)
            )
            for url in found_urls
        ]

        metadata_list = await tqdm.gather(*tasks)
        metadata_list = [
            metadata
            for metadata in metadata_list
            if metadata.get("File Format") == ".schematic"
        ]

        # Download schematics
        print(f"Found {len(metadata_list)} suitable schematics.")
        print("Downloading schematics...")
        tasks = [
            asyncio.create_task(CriteriaPage.download(metadata, session, sem))
            for metadata in metadata_list
        ]

        metadata_list = await tqdm.gather(*tasks)
        valid_list = []

        for metadata in metadata_list:
            if isinstance(metadata, tuple):
                with open(errors_file, "a") as f:
                    f.write(f"{metadata[0]}: {metadata[1]}\n")
            elif metadata is not None:
                valid_list.append(metadata)

    # Generate dataframe
    df = pd.DataFrame(valid_list)
    if os.path.exists("data.csv"):
        df = pd.concat([df, pd.read_csv("data.csv")])
    df.to_csv("data.csv", index=False)
    return df


def main(**kwargs) -> pd.DataFrame:
    """Main function to run the scraper."""
    return asyncio.run(generate_dataset(**kwargs))