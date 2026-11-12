"""Experiment 3: tune each model's hyperparameters with CV and compare against defaults.

Owner: Talan (Weeks 3-4).
Run: python -m experiments.exp3_tuning [--n-iter 20] [--sample-frac 0.3] [--feature-sets full ablated]

For each feature set and model:
  - default: the untuned pipeline from models.get_models()
  - tuned:   RandomizedSearchCV over tuning.PARAM_GRIDS on a stratified sample of the
             training split (--sample-frac), best settings refit on all of it
Both versions get 5-fold CV on the full training split (same folds) and one score on the
held-out test set, so the default-vs-tuned comparison is like for like.

Outputs:
  results/tables/exp3_tuning.csv          default vs tuned, CV and test metrics
  results/tables/exp3_best_params.json    best settings per feature set and model
  results/models/exp3_<set>_<model>.joblib  tuned, fitted models (for Experiment 4)
  results/figures/exp3_cm_<set>_<model>.png confusion matrices of the tuned models
"""
import argparse
import json
import time

import joblib
import pandas as pd
from sklearn.base import clone

from phishing_detection import config
from phishing_detection.data import load_raw
from phishing_detection.evaluate import fit_and_evaluate, plot_confusion_matrix, save_table
from phishing_detection.models import get_models
from phishing_detection.preprocess import clean, train_test
from phishing_detection.tuning import tune

COLUMNS = ["feature_set", "model", "variant", "n_features",
           "cv_f1_mean", "cv_f1_std", "cv_recall_mean", "cv_precision_mean",
           "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp",
           "search_cv_f1", "search_seconds", "fit_seconds", "predict_seconds"]


def _plain(value):
    """Make numpy scalars JSON-serializable."""
    return value.item() if hasattr(value, "item") else value


def main(feature_sets=("full", "ablated"), n_iter=20, sample_frac=0.3):
    cleaned = clean(load_raw())
    rows, best_params = [], {}
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for feature_set in feature_sets:
        X_train, X_test, y_train, y_test = train_test(cleaned, feature_set)
        best_params[feature_set] = {}

        for name, default_model in get_models().items():
            print(f"[{feature_set}] {name}: default", flush=True)
            row = fit_and_evaluate(name, default_model, X_train, y_train, X_test, y_test)
            rows.append({**row, "feature_set": feature_set, "variant": "default"})

            print(f"[{feature_set}] {name}: searching {n_iter} settings on {sample_frac:.0%} of train",
                  flush=True)
            start = time.perf_counter()
            search = tune(name, get_models()[name], X_train, y_train, n_iter=n_iter, sample_frac=sample_frac)
            search_seconds = time.perf_counter() - start
            params = {k: _plain(v) for k, v in search.best_params_.items()}
            best_params[feature_set][name] = params
            print(f"    best {params}  (search CV F1 {search.best_score_:.4f})", flush=True)

            tuned = clone(search.best_estimator_)
            row = fit_and_evaluate(name, tuned, X_train, y_train, X_test, y_test)
            rows.append({**row, "feature_set": feature_set, "variant": "tuned",
                         "search_cv_f1": search.best_score_, "search_seconds": search_seconds})

            joblib.dump(tuned, config.MODELS_DIR / f"exp3_{feature_set}_{name}.joblib")
            plot_confusion_matrix(tuned, X_test, y_test, f"{name}, tuned ({feature_set} features)",
                                  config.FIGURES_DIR / f"exp3_cm_{feature_set}_{name}.png")

    df = save_table(pd.DataFrame(rows).reindex(columns=COLUMNS), "exp3_tuning.csv")
    (config.TABLES_DIR / "exp3_best_params.json").write_text(json.dumps(best_params, indent=2))

    summary = df.pivot_table(index=["feature_set", "model"], columns="variant",
                             values=["cv_f1_mean", "f1", "recall"])
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feature-sets", nargs="+", choices=list(config.FEATURE_SETS),
                        default=["full", "ablated"])
    parser.add_argument("--n-iter", type=int, default=20, help="settings tried per model")
    parser.add_argument("--sample-frac", type=float, default=0.3,
                        help="fraction of the training split used for the search (1 = all)")
    args = parser.parse_args()
    main(args.feature_sets, args.n_iter, args.sample_frac)
