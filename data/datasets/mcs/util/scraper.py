from concurrent.futures import ProcessPoolExecutor
from nbtschematic import SchematicFile
from typing import Optional
from io import BytesIO
import asyncio
import gzip
import os
import time
from bs4 import BeautifulSoup
from httpx import AsyncClient
import pandas as pd
import numpy as np
import yaml
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

    def __repr__(self):
        return f"CriteriaPage(criteria={self.criteria}, page={self.page})"

    async def get_urls(self, session, tries=3):
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
                break
            except Exception as e:
                print(e)
                tries -= 1
                if tries == 0:
                    return None

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

    async def get_metadata(self, url, session):
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

        try:
            response = await session.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            metadata = self.__metadata_parse(url, soup)
            return metadata

        except Exception as e:
            with open(ERROR_FILE, "a") as f:
                f.write(f"{url}: {e}")
            return None

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
                SCHEMATICS_DIR,
                str(metadata["ID"]) + file_format,
            )

        metadata["URL"] = url
        metadata["Page"] = self.page
        return metadata

    async def download(self, metadata, session, path: Optional[str] = None):
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

        try:
            url = metadata.get("URL")
            path = metadata.get("Path") if path is None else path

            response = await session.get(url + "download/action/")
            content = response.content

            # Open content as file-like object
            gzipped = BytesIO(content).read(3) == b"\x1f\x8b\x08"
            schem_gen = gzip.open(BytesIO(content)) if gzipped else BytesIO(content)

            # Read schematic
            sf = SchematicFile.from_fileobj(schem_gen)

            # Integrity checks
            assert sf.shape != (1, 1, 1)
            assert np.asarray(sf.blocks) is not None
            assert "Blocks" in sf.root.keys()
            assert path is not None

            # Write schematic
            sf.save(path)
            return url, True
        except Exception as e:
            return metadata.get("URL"), False

    async def execute(self):
        """Execute the page

        Returns
        -------
        CriteriaPage
            The page with the schematics
        """

        async with AsyncClient(timeout=20) as session:
            # Authenticate session
            creds = yaml.safe_load(open(CRED_FILE, "r"))
            await session.post(AUTH_URL, data=creds)
            await self.get_urls(session)

            # Download schematics in parallel using asyncio
            if self.urls is not None:
                tasks = [self.get_metadata(url, session) for url in self.urls]
                self.schematics = []

                # For each task, add the metadata to the list of schematics
                for f in tqdm.as_completed(tasks):
                    metadata = await f
                    if metadata is None:
                        continue
                    self.schematics.append(metadata)

                # Download the schematics
                tasks = [
                    self.download(metadata, session) for metadata in self.schematics
                ]

                # Log errors
                for f in tqdm.as_completed(tasks):
                    url, success = await f
                    if not success:
                        with open(ERROR_FILE, "a") as f:
                            f.write(f"{url}: Failed to download")

        return self


async def generate_dataset(
    criteria="most-downloaded",
    num_pages=5,
    max_workers=10,
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

    # Create folder if it doesn't exist
    os.makedirs(SCHEMATICS_DIR, exist_ok=True)

    # Check interval
    a, b = 0, num_pages + 1
    if interval is not None:
        a, b = interval

    # Check for valid criteria
    valid_criteria = ["latest", "top-rated", "most-downloaded"]
    if criteria not in valid_criteria:
        raise ValueError(f"Criteria must be one of {valid_criteria}, got {criteria}.")

    # Create list of CriteriaPage objects
    list_obj = [CriteriaPage(criteria, p + 1) for p in range(a, b)]

    # Execute all requests
    scraped_pages = await asyncio.gather(
        *[obj.execute() for obj in list_obj],
    )

    # Get all metadata
    scraped_metadata = []
    for page in scraped_pages:
        scraped_metadata.extend(page.schematics)

    # Generate dataframe
    df = pd.DataFrame(scraped_metadata)
    return df


def main(**kwargs):
    df = asyncio.run(generate_dataset(**kwargs))
    return df


if __name__ == "__main__":
    df = asyncio.run(generate_dataset(interval=(0, 400)))
    df.to_csv("data.csv", index=False)
