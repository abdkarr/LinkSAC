import os
import zipfile
import tarfile

from pathlib import Path

import requests
import pandas as pd
import networkx as nx


def load_wikipedia(
    root: str | os.PathLike, ds: str = "chameleon", download: bool = True
) -> nx.Graph:

    download_url = "https://snap.stanford.edu/data/wikipedia.zip"
    dataset_dir = Path(root, "wikipedia")

    # Download the dataset
    if (not dataset_dir.exists()) and download:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        zipped_file = Path(dataset_dir, "wikipedia.zip")

        # Download the dataset from SNAP as zip
        with requests.get(download_url, stream=True) as response:
            if response.status_code == requests.codes.ok:
                with open(zipped_file, mode="wb") as file:
                    for chunk in response.iter_content(chunk_size=10 * 1024):
                        file.write(chunk)
            else:
                raise Exception("Remote not available to download Wikipedia dataset.")

        # Unzip the data
        with zipfile.ZipFile(zipped_file, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)

        Path(dataset_dir, "wikipedia").rename(Path(dataset_dir, "inputs"))
        zipped_file.unlink()

    # Read the graph
    try:
        edge_file = Path(root, "wikipedia", "inputs", ds, f"musae_{ds}_edges.csv")
    except:
        raise Exception(f"Wikipedia `musae_{ds}_edges.csv` file not found.")

    edge_list = pd.read_csv(edge_file)
    graph = nx.from_pandas_edgelist(edge_list, source="id1", target="id2")

    # Remove self-loops
    graph.remove_edges_from(nx.selfloop_edges(graph))

    return graph


def load_facebook(
    root: str | os.PathLike, ds: str = "politician", download: bool = True
) -> nx.Graph:

    download_url = "https://snap.stanford.edu/data/gemsec_facebook_dataset.tar.gz"
    dataset_dir = Path(root, "facebook")

    # Download the dataset
    if (not dataset_dir.exists()) and download:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        zipped_file = Path(dataset_dir, "facebook.tar.gz")

        # Download the dataset from SNAP as zip
        with requests.get(download_url, stream=True) as response:
            if response.status_code == requests.codes.ok:
                with open(zipped_file, mode="wb") as file:
                    for chunk in response.iter_content(chunk_size=10 * 1024):
                        file.write(chunk)
            else:
                raise Exception("Remote not available to download Facebook dataset.")

        # Unzip the data
        with tarfile.open(zipped_file, "r:gz") as tar:
            tar.extractall(dataset_dir, filter="data")

        Path(dataset_dir, "facebook_clean_data").rename(Path(dataset_dir, "inputs"))
        zipped_file.unlink()

    # Read the graph
    try:
        edge_file = Path(root, "facebook", "inputs", f"{ds}_edges.csv")
    except:
        raise Exception(f"Facebook `{ds}_edges.csv` file not found.")

    edge_list = pd.read_csv(edge_file)
    graph = nx.from_pandas_edgelist(edge_list, source="node_1", target="node_2")

    # Remove self-loops
    graph.remove_edges_from(nx.selfloop_edges(graph))

    return graph


def load_ucimessage(
    root: str | os.PathLike, ds: str = "politician", download: bool = True
) -> nx.Graph:

    download_url = "http://konect.cc/files/download.tsv.opsahl-ucsocial.tar.bz2"
    dataset_dir = Path(root, "ucimessage")

    # Download the dataset
    if (not dataset_dir.exists()) and download:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        zipped_file = Path(dataset_dir, "ucimessage.tar.bz2")

        # Download the dataset from SNAP as zip
        with requests.get(download_url, stream=True) as response:
            if response.status_code == requests.codes.ok:
                with open(zipped_file, mode="wb") as file:
                    for chunk in response.iter_content(chunk_size=10 * 1024):
                        file.write(chunk)
            else:
                raise Exception("Remote not available to download UCIMessage dataset.")

        # Unzip the data
        with tarfile.open(zipped_file, "r:bz2") as tar:
            tar.extractall(dataset_dir, filter="data")

        Path(dataset_dir, "opsahl-ucsocial").rename(Path(dataset_dir, "inputs"))
        zipped_file.unlink()

    # Read the graph
    try:
        edge_file = Path(root, "ucimessage", "inputs", "out.opsahl-ucsocial")
    except:
        raise Exception(f"UCIMessage `out.opsahl-ucsocial` file not found.")

    edge_list = pd.read_csv(
        edge_file,
        sep=" ",
        comment="%",
        names=["source", "target", "weight", "timestamp"],
    )
    graph = nx.from_pandas_edgelist(edge_list, source="source", target="target")

    # Remove self-loops
    graph.remove_edges_from(nx.selfloop_edges(graph))

    return graph
