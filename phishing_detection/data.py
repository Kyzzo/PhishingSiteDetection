"""Downloading and loading the PhiUSIIL dataset."""
import io
import urllib.request
import zipfile

import pandas as pd

from phishing_detection import config


def download_dataset(force: bool = False) -> None:
    """Download the dataset zip from UCI and extract the CSV into data/raw/."""
    if config.RAW_CSV.exists() and not force:
        print(f"Dataset already present at {config.RAW_CSV}")
        return

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {config.DATASET_URL} ...")
    with urllib.request.urlopen(config.DATASET_URL) as resp:
        payload = resp.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found in archive: {zf.namelist()}")
        with zf.open(csv_names[0]) as src:
            config.RAW_CSV.write_bytes(src.read())
    print(f"Saved {config.RAW_CSV}")


def load_raw() -> pd.DataFrame:
    """Load the raw CSV as distributed by UCI."""
    if not config.RAW_CSV.exists():
        raise FileNotFoundError(
            f"{config.RAW_CSV} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(config.RAW_CSV)
