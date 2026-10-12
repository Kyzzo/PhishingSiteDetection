"""Metrics, cross-validation, and result persistence. Phishing (1) is the positive class.

Owner: Talan (Week 1). Tests: tests/test_pipeline.py::test_models_fit_and_evaluate
"""
from pathlib import Path

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import StratifiedKFold, cross_validate

from phishing_detection import config



CV_METRICS = ["accuracy", "precision", "recall", "f1"]


def cv_splitter():
    """Return a shuffled StratifiedKFold using config.CV_FOLDS and config.RANDOM_STATE."""
    return StratifiedKFold(config.CV_FOLDS, shuffle= True, random_state=config.RANDOM_STATE)


def cross_validate_model(model, X, y) -> dict[str, float]:
    """Cross-validate on the training data.

    Returns {"cv_<metric>_mean": ..., "cv_<metric>_std": ...} for each metric in CV_METRICS.
    """
    scores = cross_validate(model, X, y, cv=cv_splitter(), scoring=CV_METRICS, n_jobs=config.N_JOBS)
    out = {}
    for m in CV_METRICS:
        fold_scores = scores[f"test_{m}"]
        out[f"cv_{m}_mean"] = fold_scores.mean()
        out[f"cv_{m}_std"] = fold_scores.std()
    return out


def holdout_metrics(y_true, y_pred) -> dict[str, float]:
    """Scores on the held-out test set.

    Returns accuracy, precision, recall, f1, and the confusion-matrix counts tn, fp, fn, tp.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels= [0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def fit_and_evaluate(name, model, X_train, y_train, X_test, y_test, run_cv=True) -> dict:
    """One results row for one model.

    Should: optionally cross-validate on train, fit on all of train (record
    "fit_seconds"), then add holdout_metrics on test. Include "model" and "n_features".
    """
    row = {"model": name, "n_features": X_train.shape[1]}

    if run_cv:
        row.update(cross_validate_model(model, X_train, y_train))

    start = time.perf_counter()
    model.fit(X_train, y_train)
    row["fit_seconds"] = time.perf_counter() - start

    y_pred = model.predict(X_test)
    row.update(holdout_metrics(y_test, y_pred))

    return row


def plot_confusion_matrix(model, X_test, y_test, title: str, path: Path) -> None:
    """Save a confusion-matrix figure with classes labeled Legitimate / Phishing."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["Legitimate", "Phishing"], ax=ax, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_table(rows: list[dict] | pd.DataFrame, filename: str) -> pd.DataFrame:
    """Write rows to results/tables/<filename> as CSV and return them as a DataFrame."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.TABLES_DIR / filename, index=False)
    return df
