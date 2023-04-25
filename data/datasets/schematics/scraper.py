"""
File that scrapes the minecraft-schematics.com website for schematics and
metadata. This scraper is multithreaded and asynchronous.
TODO! Check if the schematic is already downloaded before downloading it
TODO! Add logging
TODO! Add error handling
TODO! Refactor and simplify code
"""


import asyncio
import gzip
import os
from io import BytesIO
from typing import Optional

import aiofiles
import numpy as np
import pandas as pd
import yaml
from bs4 import BeautifulSoup
from httpx import AsyncClient
from nbtschematic import SchematicFile
from tqdm.asyncio import tqdm

BASE_URL = "https://www.minecraft-schematics.com"
AUTH_URL = BASE_URL + "/login/action/"
CRED_FILE = ".credentials.yml"
ERROR_FILE = ".errors.log"
SCHEMATICS_DIR = "schematics"
CRITERIA = ["most-downloaded", "top-rated", "latest"]
NUM_WORKERS = 12


class CriteriaPage:
    """A page of schematics on minecraft-schematics.com.

    Parameters
    ----------
    criteria : str
        The criteria to use for sorting the schematics. Must be one of
        ["latest", "top-rated", "most-downloaded"]
    page : int
        The page number to scrape

    Attributes
    ----------
    root : str
        The root URL of the page
    criteria : str
        The criteria to use for sorting the schematics. Must be one of
        ["latest", "top-rated", "most-downloaded"]
    page : int
        The page number to scrape
    urls : list[str]
        The URLs of the schematics on the page

    Methods
    -------
    get_urls(session, tries=3)
        Get the URLs of the schematics on the page
    get_metadata(url, session)
        Get the metadata of a schematic
    """

    def __init__(self, criteria: str, page: int):
        assert criteria in CRITERIA, f"criteria must be one of {CRITERIA}"
        self.root = f"https://www.minecraft-schematics.com/{criteria}/{page}/"
        self.criteria = criteria
        self.page = page
        self.urls = None
        self.SCHEMATICS_DIR = f"{SCHEMATICS_DIR}/{criteria}"
        os.makedirs(self.SCHEMATICS_DIR, exist_ok=True)

    def __repr__(self):
        return f"CriteriaPage(criteria={self.criteria}, page={self.page})"

    async def get_candidates(self, session, tries=5):
        """Get the URLs of the schematics on the page

        Parameters
        ----------
        session : httpx.AsyncClient
            The session to use for making HTTP requests
        tries : int, optional
            The number of times to retry the request, by default 3

        Returns
        -------
        list[str]
            The URLs of the schematics on the page
        """
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

    def _url_parse(self) -> list[str]:
        """Parse the URLs of the schematics on the page

        Returns
        -------
        list[str]
            The URLs of the schematics on the page
        """
        urls = self.soup.find_all("a", class_="btn btn-primary")
        urls = [url.get("href") for url in urls]
        urls = [url for url in urls if "/schematic/" in url]
        urls = [f"https://www.minecraft-schematics.com{url}" for url in urls]
        return urls

    async def get_metadata(self, session, url, sem, tries=3):
        """Get the metadata of a schematic

        Parameters
        ----------
        url : str
            The URL of the schematic
        session : httpx.AsyncClient
            The session to use for making HTTP requests

        Returns
        -------
        dict
            The metadata of the schematic
        """
        while tries > 0:
            try:
                async with sem:
                    async with asyncio.timeout(20):
                        response = await session.get(url)
            except Exception as e:
                tries -= 1
                if tries == 0:
                    return dict()
            else:
                soup = BeautifulSoup(response.text, "html.parser")
                metadata = self.__metadata_parse(url, soup)
                return metadata

    def __metadata_parse(self, url: str, soup: BeautifulSoup):
        """Parse the metadata of a schematic

        Parameters
        ----------
        url : str
            The URL of the schematic
        soup : BeautifulSoup
            The BeautifulSoup object of the schematic page

        Returns
        -------
        dict
            The metadata of the schematic
        """

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
                self.SCHEMATICS_DIR,
                str(metadata["ID"]) + file_format,
            )

        metadata["URL"] = url
        metadata["Page"] = self.page
        return metadata

    @staticmethod
    async def download(metadata, session, sem, path: Optional[str] = None, skip=True):
        """Download a schematic

        Parameters
        ----------
        metadata : dict
            The metadata of the schematic
        session : httpx.AsyncClient
            The session to use for making HTTP requests
        path : str, optional
            The path to save the schematic to, by default None

        Returns
        -------
        tuple
            The URL of the schematic and whether the download was successful
        """

        if skip and os.path.exists(metadata.get("Path")):
            return metadata

        try:
            async with sem:
                url = metadata.get("URL")
                path = metadata.get("Path") if path is None else path

                # Get download link
                async with asyncio.timeout(20):
                    response = await session.get(
                        url + "download/action/",
                        params={"type": "schematic"},
                        timeout=10,
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
    criteria="most-downloaded",
    num_pages=5,
    max_workers=10,
    interval=None,
    SCHEMATICS_DIR=SCHEMATICS_DIR,
    ERRORS_FILE=ERROR_FILE,
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
    # Create folder if it doesn't exist
    os.makedirs(SCHEMATICS_DIR, exist_ok=True)

    # Check interval
    a, b = (1, num_pages) if interval is None else interval

    # Check for valid criteria
    valid_criteria = ["latest", "top-rated", "most-downloaded"]
    if criteria not in valid_criteria:
        raise ValueError(f"Criteria must be one of {valid_criteria}, got {criteria}.")

    async with AsyncClient() as session:
        sem = asyncio.Semaphore(max_workers)
        creds = yaml.safe_load(open(CRED_FILE, "r"))
        await session.post(AUTH_URL, data=creds)

        found_urls = []

        # Execute all requests
        print("Getting candidates...")
        list_obj = [CriteriaPage(criteria, p + 1) for p in range(a, b)]
        tasks = [asyncio.create_task(page.get_candidates(session)) for page in list_obj]

        # Check if any schematics already exist
        existing_schematics = os.listdir(SCHEMATICS_DIR)
        # Get files that are in form <int>.<format>
        existing_schematics = [
            x for x in existing_schematics if "." in x and x.split(".")[0].isdigit()
        ]
        # Get IDs of existing schematics
        existing_schematics = [int(x.split(".")[0]) for x in existing_schematics]
        # Get URLs of existing schematics
        existing_schematics = [
            f"{BASE_URL}/schematic/{_id}" for _id in existing_schematics
        ]

        # For each discovered URL, get metadata
        # Skip all URLs that have already been scraped
        # TODO: Validate local copy of metadata for those URLs.
        for result in await tqdm.gather(*tasks):
            if result in existing_schematics:
                continue
            found_urls.append(result)

        # Flatten list and drop None
        found_urls = sum([x for x in found_urls if x is not None], [])

        found_urls = [
            x for x in found_urls if int(x.split("/")[-2]) not in existing_schematics
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
        print(len(metadata_list))
        print(metadata_list[0])
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
                with open(ERRORS_FILE, "a") as f:
                    f.write(f"{metadata[0]}: {metadata[1]}")
            elif metadata is not None:
                valid_list.append(metadata)

    # Generate dataframe
    df = pd.DataFrame(valid_list)
    if os.path.exists("data.csv"):
        df = pd.concat([df, pd.read_csv("data.csv")])
    df.to_csv("data.csv", index=False)
    return df


if __name__ == "__main__":
    df = asyncio.run(generate_dataset(interval=(0, 835), criteria="latest"))
