"""Hyperparameter search spaces and CV-based tuning.

Owner: Talan (Week 3).
"""
from phishing_detection import config  # noqa: F401  (use config.SCORING, N_JOBS, RANDOM_STATE)

# Search space per model name from models.get_models().
# Keys are prefixed with "clf__" because every model is a Pipeline with a "clf" step.
PARAM_GRIDS = {
    "logistic_regression": {},  # TODO: e.g. regularization strength, class weighting
    "decision_tree": {},  # TODO: e.g. depth, min samples per split/leaf, criterion
    "random_forest": {},  # TODO: e.g. number of trees, depth, max features
}


def tune(name: str, pipeline, X_train, y_train, n_iter: int = 20):
    """Search PARAM_GRIDS[name] with cross-validation, optimizing config.SCORING.

    Use evaluate.cv_splitter() for the folds and refit the best model on all of
    X_train. Return the fitted search object (best_params_, best_score_, best_estimator_).
    """
    # TODO: implement
    raise NotImplementedError
