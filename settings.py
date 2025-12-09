import os
from tqdm import tqdm
import py7zr
import urllib.request

DOWNLOAD_DB = False

for folder in ["wandb", "res", "bdd"]:
    os.makedirs(folder, exist_ok=True)

link_bdd = [
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/A.7z",
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/B.7z",
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/E.7z",
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/F.7z",
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/M.7z",
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/P.7z",
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/CMT.7z",
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/tai.7z",
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Golden.7z",
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/Li.7z",
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/X.7z",
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/AGS.7z",
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/DIMACS.7z",
    "https://galgos.inf.puc-rio.br/cvrplib/uploads/instances/CVRP/XML.7z",
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
