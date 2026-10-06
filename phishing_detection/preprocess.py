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
    # TODO: implement
    raise NotImplementedError


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y): all columns except config.TARGET, and config.TARGET."""
    # TODO: implement
    raise NotImplementedError


def train_test(df: pd.DataFrame):
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test.

    Use config.TEST_SIZE and config.RANDOM_STATE so everyone gets the same split.
    Hint: sklearn.model_selection.train_test_split(..., stratify=y)
    """
    # TODO: implement
    raise NotImplementedError


def load_split():
    """Convenience: load raw data, clean it, and return the stratified split."""
    from phishing_detection.data import load_raw

    return train_test(clean(load_raw()))
