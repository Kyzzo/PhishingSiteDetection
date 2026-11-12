"""Experiment 5: a deep learning model on raw URL text vs. the classical models.

Owner: Talan.
Run: python -m experiments.exp5_deep_learning [--url-mode normalized raw] [--skip-tabular] [--retrain]

The first run trains each CNN (~25 min each on CPU) and saves it to results/models/
(gitignored). Later runs reload it in seconds unless the data, split or CNN settings changed.

Every model is trained and scored on the same stratified split:
  - LR / DT / RF on the engineered numeric features (models.get_models())
  - TF-IDF char n-grams + Logistic Regression on the URL text (classical text baseline)
  - Character-level CNN on the URL text (the deep learning method)

URL modes: "normalized" strips the scheme, "www." and trailing "/" (see deep.normalize_url),
because in this dataset those characters alone give away the label. "raw" is kept to show
how large that shortcut is.

Requires preprocess.clean to keep the raw row index, so URLs can be looked up for the split rows.
"""
import argparse
import hashlib
import time

import pandas as pd

from phishing_detection import config
from phishing_detection.data import load_raw
from phishing_detection.deep import CharCNNClassifier, tfidf_logreg, url_frame
from phishing_detection.evaluate import fit_and_evaluate, holdout_metrics, plot_confusion_matrix, save_table
from phishing_detection.models import get_models
from phishing_detection.preprocess import clean, train_test

COLUMNS = ["model", "family", "input", "n_features", "accuracy", "precision", "recall", "f1",
           "tn", "fp", "fn", "tp", "fit_seconds", "predict_seconds", "n_parameters"]


def split_fingerprint(U_train, y_train) -> str:
    """Changes whenever the training rows, URL text, labels or CNN settings change."""
    h = hashlib.sha256(pd.util.hash_pandas_object(U_train, index=True).values.tobytes())
    h.update(pd.util.hash_pandas_object(y_train, index=False).values.tobytes())
    h.update(repr(sorted(CharCNNClassifier().get_params().items())).encode())
    return h.hexdigest()[:16]


def fit_or_load_cnn(mode, U_train, y_train, U_test, y_test, retrain=False):
    """Train the CNN once and save it to results/models/; later runs reload it if nothing changed."""
    path = config.MODELS_DIR / f"exp5_char_cnn_{mode}.pt"
    fingerprint = split_fingerprint(U_train, y_train)

    if path.exists() and not retrain:
        cnn, meta = CharCNNClassifier.load(path)
        if meta.get("fingerprint") == fingerprint:
            print(f"  loaded saved model {path.name} (pass --retrain to train again)")
            start = time.perf_counter()
            y_pred = cnn.predict(U_test)
            row = {"model": "char_cnn", "n_features": U_train.shape[1],
                   "fit_seconds": meta["fit_seconds"],  # from the original training run
                   "predict_seconds": time.perf_counter() - start}
            return cnn, {**row, **holdout_metrics(y_test, y_pred)}
        print(f"  {path.name} was trained on different data or settings; retraining")

    cnn = CharCNNClassifier()
    row = fit_and_evaluate("char_cnn", cnn, U_train, y_train, U_test, y_test, run_cv=False)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cnn.save(path, fingerprint=fingerprint, fit_seconds=row["fit_seconds"])
    print(f"  saved {path.name} to results/models/")
    return cnn, row


def main(url_modes=("normalized", "raw"), skip_tabular=False, retrain=False):
    raw = load_raw()
    X_train, X_test, y_train, y_test = train_test(clean(raw))
    urls = raw.loc[X_train.index.union(X_test.index), "URL"]
    rows = []

    if not skip_tabular:
        for name, model in get_models().items():
            print(f"[tabular] {name}")
            row = fit_and_evaluate(name, model, X_train, y_train, X_test, y_test, run_cv=False)
            clf = model.named_steps["clf"]
            if hasattr(clf, "tree_"):
                row["n_parameters"] = clf.tree_.node_count
            elif hasattr(clf, "estimators_"):
                row["n_parameters"] = sum(t.tree_.node_count for t in clf.estimators_)
            else:
                row["n_parameters"] = clf.coef_.size + clf.intercept_.size
            rows.append({**row, "family": "classical", "input": "engineered features"})

    for mode in url_modes:
        U_train = url_frame(urls.loc[X_train.index], normalize=mode == "normalized")
        U_test = url_frame(urls.loc[X_test.index], normalize=mode == "normalized")

        print(f"[text/{mode}] tfidf_logreg")
        tfidf = tfidf_logreg()
        row = fit_and_evaluate("tfidf_logreg", tfidf, U_train, y_train, U_test, y_test, run_cv=False)
        row["n_features"] = len(tfidf.named_steps["tfidf"].get_feature_names_out())
        row["n_parameters"] = row["n_features"] + 1
        rows.append({**row, "family": "classical", "input": f"URL text ({mode})"})

        print(f"[text/{mode}] char_cnn")
        cnn, row = fit_or_load_cnn(mode, U_train, y_train, U_test, y_test, retrain)
        row["n_parameters"] = cnn.n_parameters()
        rows.append({**row, "family": "deep learning", "input": f"URL text ({mode})"})
        pd.DataFrame(cnn.history_).to_csv(
            config.TABLES_DIR / f"exp5_char_cnn_history_{mode}.csv", index=False)

        for name, model in [("tfidf_logreg", tfidf), ("char_cnn", cnn)]:
            plot_confusion_matrix(model, U_test, y_test, f"{name} (URL {mode})",
                                  config.FIGURES_DIR / f"exp5_cm_{name}_{mode}.png")

    df = save_table(pd.DataFrame(rows)[COLUMNS], "exp5_deep_learning.csv")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(df.drop(columns=["tn", "fp", "tp"]).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url-mode", nargs="+", choices=["normalized", "raw"],
                        default=["normalized", "raw"])
    parser.add_argument("--skip-tabular", action="store_true")
    parser.add_argument("--retrain", action="store_true",
                        help="train the CNN again even if a saved model in results/models/ matches")
    args = parser.parse_args()
    main(args.url_mode, args.skip_tabular, args.retrain)
