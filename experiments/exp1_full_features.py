"""Experiment 1: compare default LR / DT / RF on the full set of usable features.

Owner: Paul (Week 2).
Run: python -m experiments.exp1_full_features
"""

import argparse
import pandas as pd
from phishing_detection import config
from phishing_detection.data import load_raw
from phishing_detection.evaluate import fit_and_evaluate, plot_confusion_matrix, save_table
from phishing_detection.models import get_models
from phishing_detection.preprocess import clean, train_test

COLUMNS = ["feature_set", "model", "n_features", "cv_f1_mean", "cv_f1_std", "cv_recall_mean", "cv_precision_mean", "accuracy"
           , "precision", "recall", "f1", "tn", "fp", "fn", "tp", "fit_seconds", "predict_seconds"]

def main(feature_sets=("full","ablated")):
    cleaned = clean(load_raw())
    rows = []

    for feature_set in feature_sets:
        x_train, x_test, y_train, y_test = train_test(cleaned, feature_set)
        for name, model in get_models().items():
            print(f"[{feature_set}] {name}", flush=True)
            row = fit_and_evaluate(name, model, x_train, y_train, x_test, y_test)
            rows.append({**row, "feature_set": feature_set})
            plot_confusion_matrix(model, x_test, y_test, f"{name} ({feature_set} features)",
                                  config.figures_dir("exp1") / f"exp1_cm_{feature_set}_{name}.png")

    df = save_table(pd.DataFrame(rows).reindex(columns=COLUMNS), "exp1_full_features.csv")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(df[["feature_set", "model", "f1", "recall", "precision", "fn", "fit_seconds"]])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feature-sets", nargs="+", choices=list(config.FEATURE_SETS), default=["full", "ablated"])
    args = parser.parse_args()
    main(args.feature_sets)
