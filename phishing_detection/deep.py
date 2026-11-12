"""Deep learning and text baselines that read the raw URL string instead of engineered features.

Owner: Talan. Used by experiments/exp5_deep_learning.py.

Both models take a one-column DataFrame (column "url") so they work with
evaluate.fit_and_evaluate and plot_confusion_matrix like the tabular models.
"""
import re
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from phishing_detection import config

URL_COLUMN = "url"

# In PhiUSIIL every legitimate URL starts with "https://www." and every URL ending in "/"
# is phishing, so a text model can score ~99% from those characters alone. Normalizing
# removes that collection artifact and makes the model look at the domain and path.
_PREFIX = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://(www\.)?")


def normalize_url(url: str) -> str:
    """Strip the scheme, a leading "www." and trailing slashes."""
    return _PREFIX.sub("", url).rstrip("/")


def url_frame(urls: pd.Series, normalize: bool = True) -> pd.DataFrame:
    """Wrap a Series of URLs as the one-column frame the text models expect."""
    if normalize:
        urls = urls.map(normalize_url)
    return pd.DataFrame({URL_COLUMN: urls.values}, index=urls.index)


def _texts(X) -> list[str]:
    if isinstance(X, pd.DataFrame):
        X = X[URL_COLUMN]
    return [str(s) for s in X]


def tfidf_logreg() -> Pipeline:
    """Classical text baseline: character 1-5-gram TF-IDF into Logistic Regression."""
    return Pipeline(steps=[
        ("tfidf", ColumnTransformer([(
            "chars",
            TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=3,
                            max_features=200_000, sublinear_tf=True, lowercase=False),
            URL_COLUMN,
        )])),
        ("clf", LogisticRegression(max_iter=2000, C=10.0, random_state=config.RANDOM_STATE)),
    ])


class CharCNNClassifier(ClassifierMixin, BaseEstimator):
    """Character-level CNN over the URL string (Zhang et al. 2015 / Le et al. 2018 URLNet style).

    Each character is embedded, three parallel 1-D convolutions (widths 3, 5, 7) learn
    n-gram-like detectors, global max pooling keeps the strongest match of each detector
    anywhere in the URL, and a small dense head outputs P(phishing).

    10% of the training data is held out for early stopping on F1, so fit() never
    sees the test set. The same held-out slice picks the decision threshold: with
    threshold="auto", fit() chooses the cutoff on P(phishing) that maximizes F1 there
    (stored as threshold_); a number fixes it instead (0.5 = the usual cutoff).
    """

    def __init__(self, max_len=200, embed_dim=32, n_filters=128, kernel_sizes=(3, 5, 7),
                 hidden=64, dropout=0.5, lr=1e-3, batch_size=512, max_epochs=15,
                 patience=2, val_fraction=0.1, threshold="auto", random_state=config.RANDOM_STATE,
                 verbose=True):
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.threshold = threshold
        self.random_state = random_state
        self.verbose = verbose

    # Printable ASCII 32..126 -> ids 2..96; 0 = padding, 1 = any other character.
    def _encode(self, X) -> np.ndarray:
        texts = _texts(X)
        ids = np.zeros((len(texts), self.max_len), dtype=np.int64)
        for i, s in enumerate(texts):
            codes = np.frombuffer(s[: self.max_len].encode("utf-32-le"), dtype=np.uint32)
            ids[i, : len(codes)] = np.where((codes >= 32) & (codes <= 126), codes - 30, 1)
        return ids

    def _build(self):
        import torch
        from torch import nn

        ks, nf = self.kernel_sizes, self.n_filters

        class Net(nn.Module):
            def __init__(net):
                super().__init__()
                net.embed = nn.Embedding(97, self.embed_dim, padding_idx=0)
                net.convs = nn.ModuleList(
                    nn.Conv1d(self.embed_dim, nf, k, padding=k // 2) for k in ks)
                net.head = nn.Sequential(
                    nn.Dropout(self.dropout), nn.Linear(nf * len(ks), self.hidden),
                    nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(self.hidden, 1))

            def forward(net, x):
                e = net.embed(x).transpose(1, 2)  # (batch, embed, chars)
                pooled = [torch.relu(c(e)).amax(dim=2) for c in net.convs]
                return net.head(torch.cat(pooled, dim=1)).squeeze(1)

        return Net()

    def _batches(self, ids, y=None, shuffle=False, rng=None):
        import torch

        order = rng.permutation(len(ids)) if shuffle else np.arange(len(ids))
        for start in range(0, len(ids), self.batch_size):
            idx = order[start:start + self.batch_size]
            batch = ids[idx]
            # Trim padding columns no URL in this batch uses (most URLs are < 40 chars).
            width = max(int((batch != 0).sum(axis=1).max()), max(self.kernel_sizes))
            xb = torch.from_numpy(batch[:, :width]).to(self.device_)
            yield (xb, None) if y is None else (xb, torch.from_numpy(y[idx]).to(self.device_))

    def _proba(self, ids) -> np.ndarray:
        import torch

        self.model_.eval()
        out = []
        with torch.no_grad():
            for xb, _ in self._batches(ids):
                out.append(torch.sigmoid(self.model_(xb)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def fit(self, X, y):
        import torch

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        self.classes_ = np.array([0, 1])

        ids, y = self._encode(X), np.asarray(y, dtype=np.float32)
        tr, va = train_test_split(np.arange(len(ids)), test_size=self.val_fraction,
                                  stratify=y, random_state=self.random_state)
        self.model_ = self._build().to(self.device_)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        best_f1, best_state, bad, self.history_ = -1.0, None, 0, []
        for epoch in range(1, self.max_epochs + 1):
            start = time.perf_counter()
            self.model_.train()
            total = 0.0
            for xb, yb in self._batches(ids[tr], y[tr], shuffle=True, rng=rng):
                opt.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            val_f1 = f1_score(y[va], self._proba(ids[va]) >= 0.5)
            self.history_.append({"epoch": epoch, "train_loss": total / len(tr), "val_f1": val_f1})
            if self.verbose:
                print(f"  epoch {epoch:2d}  loss {total / len(tr):.4f}  val F1 {val_f1:.4f}"
                      f"  ({time.perf_counter() - start:.0f}s)")
            if val_f1 > best_f1:
                best_f1, bad = val_f1, 0
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.best_val_f1_ = best_f1
        self.threshold_ = self._choose_threshold(y[va], self._proba(ids[va]))
        if self.verbose:
            print(f"  decision threshold {self.threshold_:.2f} (chosen on the validation slice)")
        return self

    def _choose_threshold(self, y_val, p_val) -> float:
        """The fixed threshold, or with "auto" the cutoff that maximizes F1 on the validation slice."""
        if self.threshold != "auto":
            return float(self.threshold)
        candidates = np.round(np.arange(0.05, 0.96, 0.01), 2)
        scores = [f1_score(y_val, p_val >= t, zero_division=0) for t in candidates]
        return float(candidates[int(np.argmax(scores))])

    def predict_proba(self, X) -> np.ndarray:
        p = self._proba(self._encode(X))
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model_.parameters())

    def save(self, path, **meta) -> None:
        """Save the trained weights and settings; extra keyword args are stored as metadata."""
        import torch

        torch.save({"params": self.get_params(), "state_dict": self.model_.state_dict(),
                    "history": self.history_, "best_val_f1": self.best_val_f1_,
                    "threshold": self.threshold_, "meta": meta}, path)

    @classmethod
    def load(cls, path):
        """Rebuild a trained model from save(). Returns (model, metadata)."""
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)  # our own file, trusted
        model = cls(**checkpoint["params"])
        model.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        model.classes_ = np.array([0, 1])
        model.model_ = model._build().to(model.device_)
        model.model_.load_state_dict(checkpoint["state_dict"])
        model.history_ = checkpoint["history"]
        model.best_val_f1_ = checkpoint["best_val_f1"]
        model.threshold_ = checkpoint.get("threshold", 0.5)
        return model, checkpoint["meta"]
