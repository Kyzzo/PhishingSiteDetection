"""Model definitions.

Owner: Paul (Week 1). Tests: tests/test_pipeline.py::test_models_fit_and_evaluate
"""
from sklearn.pipeline import Pipeline

from phishing_detection import config  # noqa: F401  (use config.RANDOM_STATE / N_JOBS)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from phishing_detection import config

def get_models() -> dict[str, Pipeline]:
    """Return fresh, default (untuned) models keyed by name.

    Keys must be "logistic_regression", "decision_tree", "random_forest";
    tuning.py and the experiment scripts look models up by these names.

    Wrap each model in a sklearn Pipeline with the estimator step named "clf",
    so tuning parameters can be addressed as "clf__<param>". Logistic Regression
    also needs a scaling step inside its pipeline, so scaling is fit only on
    training folds during cross-validation.
    """
    # TODO: implement
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("clf", DecisionTreeClassifier(random_state=config.RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("clf", RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS)),
            ]
        ),
    }
