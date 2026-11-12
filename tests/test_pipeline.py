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


@pytest.mark.parametrize("sample_frac", [None, 0.5])
def test_tune_refits_best_model_on_all_training_rows(raw_df, sample_frac):
    from phishing_detection.tuning import PARAM_GRIDS, tune

    X_train, X_test, y_train, _ = train_test(clean(raw_df, verbose=False))
    for name, model in get_models().items():
        search = tune(name, model, X_train, y_train, n_iter=2, sample_frac=sample_frac)
        assert set(search.best_params_) <= set(PARAM_GRIDS[name])
        best = search.best_estimator_
        assert len(best.predict(X_test)) == len(X_test)
        if name == "decision_tree":  # fitted on all of X_train, not just the sample
            assert best.named_steps["clf"].tree_.n_node_samples[0] == len(X_train)


def test_exp3_end_to_end_on_synthetic_data(raw_df, tmp_path, monkeypatch):
    import json
    from experiments import exp3_tuning

    for attr in ("TABLES_DIR", "MODELS_DIR", "FIGURES_DIR"):
        monkeypatch.setattr(config, attr, tmp_path)
    monkeypatch.setattr(exp3_tuning, "load_raw", lambda: raw_df)
    exp3_tuning.main(feature_sets=("full",), n_iter=2, sample_frac=0.5)

    table = pd.read_csv(tmp_path / "exp3_tuning.csv")
    assert len(table) == 6 and set(table["variant"]) == {"default", "tuned"}
    assert set(json.loads((tmp_path / "exp3_best_params.json").read_text())["full"]) == set(get_models())
    assert len(list(tmp_path.glob("exp3_full_*.joblib"))) == 3
    assert len(list((tmp_path / "exp3").glob("exp3_cm_full_*.png"))) == 3
