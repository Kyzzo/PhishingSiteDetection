"""Cleaning, target encoding, and train/test splitting.

Owner: Joshua (Week 1). Tests: tests/test_pipeline.py::test_clean_*, test_split_is_stratified
"""
import pandas as pd

from phishing_detection import config


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Turn the raw UCI frame into a modeling-ready frame.

    Should:
      - drop the columns listed in config.DROP_COLUMNS (skip any that are missing)
      - remove duplicate rows (print how many when verbose)
      - fail loudly if any values are missing
      - replace config.RAW_TARGET (1 = legitimate, 0 = phishing) with
        config.TARGET where 1 = phishing, so phishing is the positive class
      - fail loudly if any non-numeric columns remain

    Returns numeric features plus the config.TARGET column.
    """

    if config.RAW_TARGET not in df.columns:
        raise KeyError(f"Expected raw target column '{config.RAW_TARGET}'" 
                       "(1 = legitimate, 0 = phishing).")

    to_drop = [col for col in config.DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=to_drop)

    n_before = len(df)
    df = df.drop_duplicates()

    if verbose:
        print(f"clean: removed {n_before - len(df)} duplicate rows")

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        raise ValueError(f"clean: found {len(missing)} columns with missing values:\n{missing}")

    if not set(df[config.RAW_TARGET].unique()) <= {0, 1}:
        raise ValueError(f"clean: expected only 0/1 values in '{config.RAW_TARGET}'")
    df[config.TARGET] = 1 - df[config.RAW_TARGET]
    df = df.drop(columns=[config.RAW_TARGET])

    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"clean: expected all numeric columns, but found {len(non_numeric)} non-numeric:\n{non_numeric}")
    return df

def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y): all columns except config.TARGET, and config.TARGET."""
    return df.drop(columns=[config.TARGET]), df[config.TARGET]


def train_test(df: pd.DataFrame):
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test.

    Use config.TEST_SIZE and config.RANDOM_STATE so everyone gets the same split.
    Hint: sklearn.model_selection.train_test_split(..., stratify=y)
    """
    x, y = split_features_target(df)
    return train_test_split(x, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y)


def load_split():
    """Convenience: load raw data, clean it, and return the stratified split."""
    from phishing_detection.data import load_raw

    return train_test(clean(load_raw()))
