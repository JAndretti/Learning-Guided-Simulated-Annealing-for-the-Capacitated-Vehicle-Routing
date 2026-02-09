import os
import ssl
import urllib.request

import py7zr
from tqdm import tqdm

# unexpected/unsafe: creates an unverified SSL context globally
ssl._create_default_https_context = ssl._create_unverified_context

DOWNLOAD_DB = True

for folder in ["wandb", "res", "bdd"]:
    os.makedirs(folder, exist_ok=True)

link_bdd = [
    "https://galgos.inf.puc-rio.br/cvrplib/en/download/instance-set/17.7z",
    "https://galgos.inf.puc-rio.br/cvrplib/en/download/instance-set/20.7z",
    "https://galgos.inf.puc-rio.br/cvrplib/en/download/instance-set/21.7z",
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/files/xml100/solutions.7z",
]


if DOWNLOAD_DB:
    # Download the datasets
    print("Downloading datasets...")
    for url in tqdm(link_bdd, desc="Downloading datasets"):
        filename = os.path.basename(url)
        zip_path = os.path.join("bdd", filename)
        # Download the file
        urllib.request.urlretrieve(url, zip_path)
        # Unzip the file
        with py7zr.SevenZipFile(zip_path, mode="r") as z:
            z.extractall(path="bdd")
        # Remove the zip file after extraction
        os.remove(zip_path)
    print("Datasets downloaded and extracted.")
