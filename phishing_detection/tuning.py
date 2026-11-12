"""Hyperparameter search spaces and CV-based tuning.

Owner: Talan (Week 3).
"""
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from phishing_detection import config
from phishing_detection.evaluate import cv_splitter

# Search space per model name from models.get_models().
# Keys are prefixed with "clf__" because every model is a Pipeline with a "clf" step.
PARAM_GRIDS = {
    "logistic_regression": {
        "clf__C": loguniform(1e-3, 1e2),  # inverse regularization strength
        "clf__class_weight": [None, "balanced"],
    },
    "decision_tree": {
        "clf__max_depth": [None, 5, 10, 15, 20, 30],
        "clf__min_samples_split": randint(2, 50),
        "clf__min_samples_leaf": randint(1, 20),
        "clf__criterion": ["gini", "entropy"],
        "clf__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "clf__n_estimators": randint(100, 400),
        "clf__max_depth": [None, 10, 20, 30],
        "clf__max_features": ["sqrt", "log2", 0.3, 0.5],
        "clf__min_samples_leaf": randint(1, 10),
        "clf__class_weight": [None, "balanced"],
    },
}


def tune(name: str, pipeline, X_train, y_train, n_iter: int = 20, sample_frac: float | None = None):
    """Search PARAM_GRIDS[name] with cross-validation, optimizing config.SCORING.

    Uses evaluate.cv_splitter() for the folds and refits the best model on all of
    X_train. Returns the fitted search object (best_params_, best_score_, best_estimator_).

    sample_frac: run the search on a stratified sample of X_train (e.g. 0.3) to save
    time, then refit the best settings on all of X_train. best_score_ is then the CV
    score on the sample.
    """
    X_search, y_search = X_train, y_train
    if sample_frac is not None and sample_frac < 1:
        X_search, _, y_search, _ = train_test_split(
            X_train, y_train, train_size=sample_frac, stratify=y_train, random_state=config.RANDOM_STATE)

    search = RandomizedSearchCV(
        pipeline, PARAM_GRIDS[name], n_iter=n_iter, scoring=config.SCORING, cv=cv_splitter(),
        n_jobs=config.N_JOBS, random_state=config.RANDOM_STATE, refit=X_search is X_train)
    search.fit(X_search, y_search)

    if X_search is not X_train:
        # Refit on all of X_train; set the attribute RandomizedSearchCV would have set.
        search.best_estimator_ = clone(pipeline).set_params(**search.best_params_).fit(X_train, y_train)
    return search
