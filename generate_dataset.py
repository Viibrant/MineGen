from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm, trange
from nbtschematic import SchematicFile
import requests
import pandas as pd
import re
import yaml


def get_metadata(url=None, soup=None):
    """
    Scrapes the metadata from a schematic page.

    Returns
    metadata : dict
        A dictionary containing the metadata.
    """

    if soup is None:
        r = requests.get(url, cookies=COOKIES)
        soup = BeautifulSoup(r.text, "html.parser")

    # Extract the name from the page
    title = soup.find_all("h1")[0].text

    # Find table containing metadata
    table = soup.find("table")

    # Extract rows, splitting into columns
    raw = [r.text for r in table.find_all("tr")]
    raw = [r.split("\n") for r in raw]

    # Remove empty strings
    raw = [p.strip() for r in raw for p in r[1:-1]]

    # Iniitalise dictionary, key-value pairs
    keys = [p for i, p in enumerate(raw) if i % 2 == 0]
    values = [p for p in raw if p not in keys]

    metadata = dict(zip(keys, values))
    metadata["Name"] = title
    metadata["ID"] = int(re.search(r"schematic/(\d+)", url).group(1))

    # Data sanitisation
    rating = re.search(r"\d+\.\d+", metadata["Rating"])
    if rating is not None:
        metadata["Rating"] = float(rating.group(0))

    downloads = re.search(r"\d+", metadata["Download(s)"])
    if downloads is not None:
        metadata["Download(s)"] = int(downloads.group(0))

    return metadata


def download_schematic(url, metadata=None):
    ID = re.search(r"schematic/(\d+)", url).group(1)
    r = requests.get(
        f"{url}download/action/", cookies=COOKIES, params={"type": "schematic"}
    )

    if metadata is None:
        metadata = get_metadata(url)

    # Create folder if it doesn't exist
    path = Path("schematics", metadata["Category"])
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / f"{ID}.schematic"

    if file_path.exists():
        return

    sf = SchematicFile().from_fileobj(r.raw)
    sf.save(file_path, gzipped=True)


def generate_dataset(criteria="most-downloaded", num_pages=5, interval=None):
    """
    Generates a dataset of schematics from the Minecraft Schematics website.

    Parameters
    ----------
    criteria : str, optional
        The criteria to sort the schematics by. Must be one of "latest", "top-rated", "most-downloaded".
        Defaults to "most-downloaded".
    num_pages : int, optional
        The number of pages to scrape. Defaults to 5.
    interval : iterable of shape (2,), optional
        The interval of pages to scrape. Defaults to None.
    """
    root = f"https://www.minecraft-schematics.com/{criteria}"
    list_metadata = []
    bar_page = trange(1, num_pages + 1, leave=False, unit="page", desc="Page")

    if criteria not in ["latest", "top-rated", "most-downloaded"]:
        raise ValueError(
            "Criteria must be one of 'latest', 'top-rated', 'most-downloaded'"
        )
    if interval is not None:
        num_pages = interval[1] - interval[0] + 1
        bar_page = trange(
            interval[0], interval[1] + 1, leave=False, unit="page", desc="Page"
        )

    for page in bar_page:
        bar_page.set_description(f"Page {page}")
        url = f"{root}/{page}/"
        r = requests.get(url, cookies=COOKIES)
        soup = BeautifulSoup(r.text, "html.parser")

        # Get schematic links from download buttons
        download_buttons = soup.find_all("a", class_="btn btn-primary")
        links = [button.get("href") for button in download_buttons]
        links = [f"https://www.minecraft-schematics.com{link}" for link in links]

        pbar = tqdm(links, unit="schem", leave=False)
        for link in pbar:

            metadata = get_metadata(link)
            pbar.set_description(metadata["Name"])

            path = Path(
                "schematics", metadata["Category"], f"{metadata['ID']}.schematic"
            )

            if path.exists():
                pbar.set_description("Already exists (skipped)")
                continue

            if "File Format" not in metadata:
                desc = "".join(["\u0336{}".format(c) for c in pbar.desc])
                pbar.set_description(desc)
                continue

            if metadata["File Format"] == ".schematic":
                download_schematic(link, metadata=metadata)
                list_metadata.append(metadata)
                pbar.set_description(metadata["Name"] + " " + "\u2713")

    df = pd.DataFrame(list_metadata)
    df.to_csv(f"schematics/{criteria}.csv", index=False)


NUM_PAGES = 2
# use headers from cookies
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:68.0) Gecko/20100101 Firefox/68.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.minecraft-schematics.com/",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
}

auth_url = "https://www.minecraft-schematics.com/login/action/"


if __name__ == "__main__":
    with open(".credentials.yml", "r") as f:
        cred = yaml.safe_load(f)
    s = requests.Session()
    s.post(auth_url, data=cred)
    COOKIES = s.cookies.get_dict()
    generate_dataset(interval=(60, 250))
