"""Smoke tests for the URL text models (no download needed)."""
import numpy as np
import pandas as pd
import pytest

from phishing_detection.deep import CharCNNClassifier, normalize_url, tfidf_logreg, url_frame
from phishing_detection.evaluate import fit_and_evaluate


@pytest.mark.parametrize("url, expected", [
    ("https://www.example.com", "example.com"),
    ("http://login-paypal.example.com/verify/", "login-paypal.example.com/verify"),
    ("HTTPS://www2.site.org", "www2.site.org"),
    ("example.com", "example.com"),
])
def test_normalize_url(url, expected):
    assert normalize_url(url) == expected


@pytest.fixture
def url_split():
    rng = np.random.default_rng(0)
    legit = [f"https://www.site{i}.com" for i in range(120)]
    phish = [f"http://secure-login{i}.verify-account.xyz/{'é' if i % 7 == 0 else ''}" for i in range(120)]
    urls = pd.Series(legit + phish)
    y = pd.Series([0] * 120 + [1] * 120)
    idx = rng.permutation(len(urls))
    train, test = idx[:180], idx[180:]
    X = url_frame(urls)
    return X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]


def test_url_frame_keeps_index():
    urls = pd.Series(["https://www.a.com/"], index=[42])
    assert url_frame(urls).loc[42, "url"] == "a.com"


@pytest.mark.parametrize("make", [tfidf_logreg, lambda: CharCNNClassifier(max_epochs=2, verbose=False)])
def test_text_models_fit_and_evaluate(url_split, make):
    X_train, X_test, y_train, y_test = url_split
    row = fit_and_evaluate("m", make(), X_train, y_train, X_test, y_test, run_cv=False)
    assert 0 <= row["f1"] <= 1
    assert row["tp"] + row["fn"] == y_test.sum()


def test_char_cnn_proba_shape(url_split):
    X_train, X_test, y_train, _ = url_split
    cnn = CharCNNClassifier(max_epochs=1, verbose=False).fit(X_train, y_train)
    proba = cnn.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_char_cnn_save_load_roundtrip(url_split, tmp_path):
    X_train, X_test, y_train, _ = url_split
    cnn = CharCNNClassifier(max_epochs=1, verbose=False).fit(X_train, y_train)
    cnn.save(tmp_path / "cnn.pt", note="hi")
    loaded, meta = CharCNNClassifier.load(tmp_path / "cnn.pt")
    assert meta == {"note": "hi"}
    assert loaded.threshold_ == cnn.threshold_
    assert np.allclose(loaded.predict_proba(X_test), cnn.predict_proba(X_test), atol=1e-6)
    assert (loaded.predict(X_test) == cnn.predict(X_test)).all()


def test_char_cnn_threshold_auto_or_fixed(url_split):
    X_train, X_test, y_train, _ = url_split
    auto = CharCNNClassifier(max_epochs=1, verbose=False).fit(X_train, y_train)
    assert 0.05 <= auto.threshold_ <= 0.95
    p = auto.predict_proba(X_test)[:, 1]
    assert (auto.predict(X_test) == (p >= auto.threshold_)).all()

    fixed = CharCNNClassifier(max_epochs=1, threshold=0.5, verbose=False).fit(X_train, y_train)
    assert fixed.threshold_ == 0.5


def test_exp5_reuses_saved_cnn_only_for_same_data(url_split, tmp_path, monkeypatch):
    from experiments import exp5_deep_learning as exp5
    from phishing_detection import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(exp5, "CharCNNClassifier", type(
        "QuickCNN", (CharCNNClassifier,), {"__init__": lambda self, **kw: CharCNNClassifier.__init__(
            self, **{"max_epochs": 1, "verbose": False, **kw})}))
    X_train, X_test, y_train, y_test = url_split

    _, first = exp5.fit_or_load_cnn("normalized", X_train, y_train, X_test, y_test)
    saved = (tmp_path / "exp5_char_cnn_normalized.pt").stat().st_mtime_ns
    _, second = exp5.fit_or_load_cnn("normalized", X_train, y_train, X_test, y_test)
    assert (tmp_path / "exp5_char_cnn_normalized.pt").stat().st_mtime_ns == saved  # reused, not retrained
    assert second["f1"] == first["f1"] and second["fit_seconds"] == first["fit_seconds"]

    exp5.fit_or_load_cnn("normalized", X_train.iloc[1:], y_train.iloc[1:], X_test, y_test)
    assert (tmp_path / "exp5_char_cnn_normalized.pt").stat().st_mtime_ns != saved  # data changed -> retrained


def test_exp5_end_to_end_has_rows_for_each_feature_set(tmp_path, monkeypatch):
    from experiments import exp5_deep_learning as exp5
    from phishing_detection import config

    rng = np.random.default_rng(0)
    n = 200
    label = rng.integers(0, 2, n)  # 1 = legitimate, 0 = phishing
    raw = pd.DataFrame({c: rng.integers(0, 50, n) for c in config.URL_FEATURES})
    raw["NoOfImage"] = rng.integers(0, 9, n)
    raw["URL"] = [f"https://www.site{i}.com" if l else f"http://login{i}.verify.xyz/" for i, l in enumerate(label)]
    raw["label"] = label

    for attr in ("TABLES_DIR", "MODELS_DIR", "FIGURES_DIR"):
        monkeypatch.setattr(config, attr, tmp_path)
    monkeypatch.setattr(exp5, "load_raw", lambda: raw)
    monkeypatch.setattr(exp5, "CharCNNClassifier", type(
        "QuickCNN", (CharCNNClassifier,), {"__init__": lambda self, **kw: CharCNNClassifier.__init__(
            self, **{"max_epochs": 1, "verbose": False, **kw})}))
    exp5.main(url_modes=("normalized",))

    table = pd.read_csv(tmp_path / "exp5_deep_learning.csv")
    assert set(table["input"]) == {"engineered features (full)", "engineered features (ablated)",
                                   "URL text (normalized)"}
    ablated = table[table["input"] == "engineered features (ablated)"]
    assert (ablated["n_features"] == len(config.FEATURE_SETS["ablated"])).all()
    cnn_rows = table[table["family"] == "deep learning"]
    assert list(cnn_rows["model"]) == ["char_cnn", "char_cnn (val threshold)"]
    assert cnn_rows["threshold"].iloc[0] == 0.5
    assert (table.loc[table["family"] == "classical", "threshold"] == 0.5).all()
    # one confusion matrix per row, all in results/figures/exp5/
    images = sorted(p.name for p in (tmp_path / "exp5").glob("*.png"))
    assert len(images) == len(table) == 9
    assert "exp5_cm_char_cnn_normalized_t0.50.png" in images
    assert "exp5_cm_char_cnn_normalized_val_threshold.png" in images
    assert not list(tmp_path.glob("*.png"))  # nothing left in the top-level figures folder
