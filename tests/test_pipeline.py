"""Smoke tests on a small synthetic frame shaped like PhiUSIIL (no download needed)."""
import numpy as np
import pandas as pd
import pytest

from phishing_detection import config
from phishing_detection.evaluate import fit_and_evaluate
from phishing_detection.features import select_top_k
from phishing_detection.models import get_models
from phishing_detection.preprocess import clean, train_test


@pytest.fixture
def raw_df():
    rng = np.random.default_rng(0)
    n = 200
    label = rng.integers(0, 2, n)
    return pd.DataFrame({
        "FILENAME": [f"f{i}.txt" for i in range(n)],
        "URL": [f"https://site{i}.com" for i in range(n)],
        "Domain": [f"site{i}.com" for i in range(n)],
        "TLD": ["com"] * n,
        "Title": ["t"] * n,
        "URLLength": rng.integers(10, 100, n) + label * 20,
        "IsHTTPS": label,
        "NoOfImage": rng.integers(0, 50, n),
        "label": label,
    })


def test_clean_flips_label_and_drops_text(raw_df):
    df = clean(raw_df, verbose=False)
    assert config.TARGET in df.columns
    assert "label" not in df.columns
    assert not set(config.DROP_COLUMNS) & set(df.columns)
    # Raw label 0 (phishing) must become is_phishing == 1.
    assert (df[config.TARGET] == (raw_df["label"] == 0).astype(int)).all()


def test_clean_drops_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df], ignore_index=True)
    assert len(clean(doubled, verbose=False)) == len(clean(raw_df, verbose=False))


def test_split_is_stratified(raw_df):
    X_train, X_test, y_train, y_test = train_test(clean(raw_df, verbose=False))
    assert abs(y_train.mean() - y_test.mean()) < 0.05
    assert len(X_test) == pytest.approx(len(raw_df) * config.TEST_SIZE, abs=1)


def test_models_fit_and_evaluate(raw_df):
    X_train, X_test, y_train, y_test = train_test(clean(raw_df, verbose=False))
    for name, model in get_models().items():
        row = fit_and_evaluate(name, model, X_train, y_train, X_test, y_test)
        assert 0 <= row["f1"] <= 1
        assert row["tp"] + row["fn"] == y_test.sum()


def test_select_top_k(raw_df):
    X_train, _, y_train, _ = train_test(clean(raw_df, verbose=False))
    assert len(select_top_k(X_train, y_train, 2)) == 2


def test_feature_sets_share_the_split_and_ablated_keeps_url_features():
    rng = np.random.default_rng(1)
    n = 100
    label = rng.integers(0, 2, n)
    cols = {c: rng.integers(0, 50, n) for c in config.URL_FEATURES}
    raw = pd.DataFrame({**cols, "NoOfImage": rng.integers(0, 9, n), "label": label})
    df = clean(raw, verbose=False)

    full_train, full_test, _, _ = train_test(df, "full")
    abl_train, abl_test, _, _ = train_test(df, "ablated")
    assert list(abl_train.columns) == config.FEATURE_SETS["ablated"]
    assert not set(config.ABLATED_EXCLUDE) & set(abl_train.columns)
    assert "NoOfImage" in full_train.columns and "NoOfImage" not in abl_train.columns
    assert full_train.index.equals(abl_train.index) and full_test.index.equals(abl_test.index)


def test_unknown_feature_set_fails_loudly(raw_df):
    with pytest.raises(ValueError):
        train_test(clean(raw_df, verbose=False), "nope")
