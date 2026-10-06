"""Download the PhiUSIIL dataset from UCI into data/raw/.

Run: python scripts/download_data.py [--force]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phishing_detection.data import download_dataset  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    download_dataset(force=parser.parse_args().force)
