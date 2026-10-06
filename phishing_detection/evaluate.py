"""Metrics, cross-validation, and result persistence. Phishing (1) is the positive class.

Owner: Talan (Week 1). Tests: tests/test_pipeline.py::test_models_fit_and_evaluate
"""
from pathlib import Path

import pandas as pd

from phishing_detection import config

CV_METRICS = ["accuracy", "precision", "recall", "f1"]


def cv_splitter():
    """Return a shuffled StratifiedKFold using config.CV_FOLDS and config.RANDOM_STATE."""
    # TODO: implement
    raise NotImplementedError


def cross_validate_model(model, X, y) -> dict[str, float]:
    """Cross-validate on the training data.

    Returns {"cv_<metric>_mean": ..., "cv_<metric>_std": ...} for each metric in CV_METRICS.
    """
    # TODO: implement
    raise NotImplementedError


def holdout_metrics(y_true, y_pred) -> dict[str, float]:
    """Scores on the held-out test set.

    Returns accuracy, precision, recall, f1, and the confusion-matrix counts tn, fp, fn, tp.
    """
    # TODO: implement
    raise NotImplementedError


def fit_and_evaluate(name, model, X_train, y_train, X_test, y_test, run_cv=True) -> dict:
    """One results row for one model.

    Should: optionally cross-validate on train, fit on all of train (record
    "fit_seconds"), then add holdout_metrics on test. Include "model" and "n_features".
    """
    # TODO: implement
    raise NotImplementedError


def plot_confusion_matrix(model, X_test, y_test, title: str, path: Path) -> None:
    """Save a confusion-matrix figure with classes labeled Legitimate / Phishing."""
    # TODO: implement
    raise NotImplementedError


def save_table(rows: list[dict] | pd.DataFrame, filename: str) -> pd.DataFrame:
    """Write rows to results/tables/<filename> as CSV and return them as a DataFrame."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.TABLES_DIR / filename, index=False)
    return df
