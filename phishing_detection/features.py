"""Feature ranking and selection. Always fit on training data only.

Owner: Paul (Week 3). Tests: tests/test_pipeline.py::test_select_top_k
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from phishing_detection import config

METHODS = ("rf", "mutual_info")


def rank_features(X_train: pd.DataFrame, y_train: pd.Series, method: str = "rf") -> pd.Series:
    """Score every feature's relevance to the target, sorted descending.

    Support at least two methods (e.g. "rf" for random-forest importance and
    "mutual_info" for mutual information) so Experiment 2 can compare them.
    """
    if method == "rf":
        forest = RandomForestClassifier(
            random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS)
        forest.fit(X_train, y_train)
        scores = forest.feature_importances_
    elif method == "mutual_info":
        scores = mutual_info_classif(X_train, y_train, random_state=config.RANDOM_STATE)
    else:
        raise ValueError(f"Unknown method {method}. Must be one of {METHODS}.")

    return pd.Series(scores, index=X_train.columns).sort_values(ascending=False)

def select_top_k(X_train: pd.DataFrame, y_train: pd.Series, k: int, method: str = "rf") -> list[str]:
    """Return the names of the k highest-ranked features."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    return rank_features(X_train, y_train, method).head(k).index.tolist()
