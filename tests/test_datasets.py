from pathlib import Path

from linksac import datasets
from linksac import PROJECT_DIR

def test_load_wikipedia():
    datasets.load_wikipedia(Path(PROJECT_DIR, "data"))

def test_load_facebook():
    datasets.load_facebook(Path(PROJECT_DIR, "data"))

def test_load_ucimessage():
    datasets.load_ucimessage(Path(PROJECT_DIR, "data"))