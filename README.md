# Machine Learning-Based Phishing Website Detection

**The First Group:** Joshua Groh, Paul O'brien, Talan Tran, Wilfred Vergara

We compare Logistic Regression, Decision Tree, and Random Forest on the
[PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset).
We test whether feature selection and hyperparameter tuning help, and look at which URL and
page features drive the predictions. We focus on **recall** and **F1** with **phishing as the positive class**.

> **Status:** the modeling code is stubbed out for the team to build. Each stub's docstring says what
> to implement and who owns it. [SCHEDULE.md](SCHEDULE.md) has the week-by-week plan. `pytest` fails
> until the stubs are implemented. Getting it to pass is the first milestone.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python scripts/download_data.py # saves data/raw/PhiUSIIL_Phishing_URL_Dataset.csv
pytest                          # tests on synthetic data (fail until the stubs are implemented)
```

## Running the experiments

Run each command from the repo root:

| # | Command | Question | Outputs |
|---|---------|----------|---------|
| 1 | `python -m experiments.exp1_full_features` | Which model works best on all usable features? | `results/tables/exp1_full_features.csv`, confusion matrices |
| 2 | `python -m experiments.exp2_feature_selection --k 10 15 20` | Does a smaller feature set keep performance and cut training time? | `results/tables/exp2_feature_selection_rf.csv` |
| 3 | `python -m experiments.exp3_tuning --n-iter 20` | How much does CV hyperparameter tuning help over the defaults? | `results/tables/exp3_tuning.csv`, `exp3_best_params.json`, `results/models/*.joblib` |
| 4 | `python -m experiments.exp4_feature_importance` | Which features matter most? | `results/tables/exp4_feature_importance.csv`, importance plots |
| 5 | `python -m experiments.exp5_deep_learning` | Can a deep learning model on the raw URL text match models on engineered features? | `results/tables/exp5_deep_learning.csv`, CNN training history, confusion matrices |

## Project layout

```
phishing_detection/     reusable package
  config.py             paths, seed, split size, CV folds, columns to drop
  data.py               download and load the raw CSV
  preprocess.py         drop columns, dedupe, flip label, stratified split
  models.py             default LR / DT / RF pipelines (LR includes StandardScaler)
  evaluate.py           CV and held-out metrics, confusion matrix plots, CSV output
  features.py           feature ranking (RF importance, mutual information)
  tuning.py             hyperparameter search spaces and RandomizedSearchCV
experiments/            one script per planned experiment
scripts/download_data.py
notebooks/              EDA and scratch work
tests/                  pytest smoke tests
data/raw, data/processed    (gitignored)
results/figures, results/tables, results/models
```

## Preprocessing decisions (the spec for `preprocess.py`)

- **Target:** the raw `label` is 1 = legitimate and 0 = phishing. `preprocess.clean` replaces it
  with `is_phishing` (1 = phishing), so the default sklearn precision, recall, and F1 score phishing.
- **Dropped columns:** `FILENAME` (an identifier), plus the raw-text columns `URL`, `Domain`, `TLD`, and `Title`.
  Their signal is already in the engineered numeric features, such as `TLDLegitimateProb` and `DomainTitleMatchScore`.
  To change this, edit `DROP_COLUMNS` in `config.py`. This is provisional until the Week 1 EDA
  decides, per model, whether to encode or exclude each text and categorical column.
- **Duplicates** are removed before splitting, so the same row can't end up in both train and test.
- **Scaling** happens inside the Logistic Regression pipeline, so it is fit only on the training folds during CV.
- **Held-out test set (20%, stratified):** we use it only for the final evaluation. All tuning and
  feature selection use the training split.

## Deep learning comparison (Experiment 5)

Experiment 5 adds one deep learning method, a **character-level CNN** (`deep.CharCNNClassifier`). It reads
the URL string directly and learns its own features, so it needs no hand-engineering. Each character
is embedded, and parallel 1-D convolutions of width 3, 5 and 7 learn substring detectors (for example
`login`, `-secure`, `.xyz`). Global max pooling keeps each detector's strongest match anywhere in the URL.
Training uses Adam, with early stopping on a 10% validation slice of the training set.

To separate *input* from *method*, the experiment also trains a classical text baseline: TF-IDF
character 1–5-grams fed to Logistic Regression. All five models use the same stratified split:

| Model | Input | Family |
|-------|-------|--------|
| Logistic Regression, Decision Tree, Random Forest | engineered numeric features | classical |
| TF-IDF + Logistic Regression | URL text | classical |
| Char-CNN | URL text | deep learning |

**URL normalization.** In PhiUSIIL every legitimate URL starts with `https://www.`, and every URL that
ends in `/` is phishing. On raw URLs, a text model can score about 99% from those characters alone.
By default the text models therefore see URLs with the scheme, a leading `www.` and trailing slashes
removed (`deep.normalize_url`). The raw-URL scores are reported alongside to show the size of that shortcut.

Install note: `pip install -r requirements.txt` gets the CPU build of PyTorch, which is enough here.

## Feature sets

Pick one with `load_split("full")` / `load_split("ablated")`, or change `config.FEATURE_SET`. Both use the
same rows and the same train/test split.

| Set | Features | Why |
|-----|----------|-----|
| `full` | all 50 | Comparable to the PhiUSIIL paper, but every model scores ~100% F1 |
| `ablated` | 20 URL-string features, without `URLSimilarityIndex` and `IsHTTPS` | First set where the models differ |

How the ablated set was chosen (5-fold CV on the training split only; the test set was not used):

- `URLSimilarityIndex` alone, with one threshold, separates the classes at 99.7% accuracy.
- Dropping it, or it plus the strongest page-content counts, barely helps: every model stays at 99.7–100% F1,
  because the remaining features overlap so much.
- Splitting by feature source does help. URL-string features without `URLSimilarityIndex` and `IsHTTPS`
  give LR / DT / RF F1 of 0.973 / 0.990 / 0.994 (LR recall 0.951). This is also a realistic scenario,
  classifying a link before loading the page, and it matches the input of the Experiment 5 text models.

**Caveat.** Every legitimate URL starts with `https://www.`, and the remaining URL features still
partly encode that prefix (a Random Forest predicts the prefix from them at 99.5% accuracy). Restricted
to URLs that start with `https://www.`, where the prefix gives nothing away, a Random Forest on the URL
features reaches only 0.83 F1. Report that subset as the hard evaluation; it is in line with the 0.87 F1
the Experiment 5 text models reach on normalized URLs.

## Notes and caveats

- Published results on PhiUSIIL report near-perfect accuracy. The differences between classes come
  partly from how the data was collected (legitimate sites are mostly top-ranked homepages), so no
  feature list removes the shortcut completely. See "Feature sets" above.

## Reference

Prasad, A., & Chandra, S. (2024). PhiUSIIL: A diverse security profile empowered phishing URL
detection framework based on similarity index and incremental learning. *Computers & Security, 136*, 103545.
https://doi.org/10.1016/j.cose.2023.103545
