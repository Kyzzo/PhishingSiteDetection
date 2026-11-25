"""Experiment 2: retrain each model on reduced feature subsets and compare to full features.

Owner: Paul (Week 3).
Run: python -m experiments.exp2_feature_selection
"""

import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from phishing_detection import config
from phishing_detection.data import load_raw
from phishing_detection.evaluate import fit_and_evaluate, save_table
from phishing_detection.features import METHODS, rank_features
from phishing_detection.models import get_models
from phishing_detection.preprocess import clean, train_test

COLUMNS = ["feature_set", "ranking","k", "model", "n_features", "f1", "recall", "precision", "accuracy", "tn",
           "fp", "fn", "tp", "fit_seconds", "predict_seconds", "tree_depth", "total_nodes", "cv_f1_mean", "cv_f1_std"]

def model_size(pipeline) -> dict:
    clf = pipeline.named_steps["clf"]
    if isinstance(clf, DecisionTreeClassifier):
        return {"tree_depth": clf.get_depth(), "total_nodes": clf.tree_.node_count}
    if isinstance(clf, RandomForestClassifier):
        return {"tree_depth": None, "total_nodes": sum(t.tree_.node_count for t in clf.estimators_)}
    return {"tree_depth": None, "total_nodes": None}

def evaluate_models(X_train, X_test, y_train, y_test, columns, run_cv) -> list[dict]:
    rows = []
    for name, model in get_models().items():
        row = fit_and_evaluate(name, model, X_train[columns], y_train, X_test[columns], y_test,
                               run_cv=run_cv)
        rows.append({**row, **model_size(model)})
    return rows

def main(k_values=(10,15,20), feature_sets=("full","ablated"), run_cv=False):
    cleaned = clean(load_raw())
    rows, rankings = [], []
    for feature_set in feature_sets:
        X_train, X_test, y_train, y_test = train_test(cleaned, feature_set)
        all_columns = list(X_train.columns)

        print(f"[{feature_set}] reference: all {len(all_columns)} features", flush=True)
        for row in evaluate_models(X_train, X_test, y_train, y_test, all_columns, run_cv):
            rows.append({**row, "feature_set": feature_set, "ranking": "all", "k": len(all_columns)})
        for method in METHODS:
            ranking = rank_features(X_train, y_train, method)
            rankings += [{"feature_set": feature_set, "ranking": method, "rank": i, "feature": feature, "score": score}
                         for i, (feature, score) in enumerate(ranking.items(), start=1)]
            for k in k_values:
                if k >= len(all_columns):
                    print(f"[{feature_set}] {method}: skipping k={k} (set has {len(all_columns)} features)")
                    continue
                print(f"[{feature_set}] {method}: top {k} features", flush=True)
                columns = ranking.head(k).index.tolist()
                for row in evaluate_models(X_train, X_test, y_train, y_test, columns, run_cv):
                    rows.append({**row, "feature_set": feature_set, "ranking": method, "k": k})
    df = save_table(pd.DataFrame(rows).reindex(columns=COLUMNS), "exp2_feature_selection.csv")
    save_table(rankings, "exp2_feature_ranking.csv")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        for feature_set, part in df.groupby("feature_set", sort=False):
            print(f"\nTest F1, {feature_set} features (columns: ranking method, k)")
            print(part.pivot_table(index="model", columns=["ranking", "k"], values="f1"))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, nargs="+", default=[10,15,20],
                        help="numbers of top features to keep")
    parser.add_argument("--feature-sets", nargs="+", choices=list(config.FEATURE_SETS),
                        default=["full", "ablated"])
    parser.add_argument("--cv", action="store_true", help="also run 5-fold CV (slower)")
    args = parser.parse_args()
    main(args.k, args.feature_sets, args.cv)
