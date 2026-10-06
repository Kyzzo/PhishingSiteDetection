"""Feature ranking and selection. Always fit on training data only.

Owner: Paul (Week 3). Tests: tests/test_pipeline.py::test_select_top_k
"""
import pandas as pd


def rank_features(X_train: pd.DataFrame, y_train: pd.Series, method: str = "rf") -> pd.Series:
    """Score every feature's relevance to the target, sorted descending.

    Support at least two methods (e.g. "rf" for random-forest importance and
    "mutual_info" for mutual information) so Experiment 2 can compare them.
    """
    # TODO: implement
    raise NotImplementedError


def select_top_k(X_train: pd.DataFrame, y_train: pd.Series, k: int, method: str = "rf") -> list[str]:
    """Return the names of the k highest-ranked features."""
    # TODO: implement
    raise NotImplementedError
