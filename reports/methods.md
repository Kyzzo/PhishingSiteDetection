# Methods

All models are trained on one fixed 80/20 stratified split of the cleaned PhiUSIIL data and scored once on
the held-out 20%; every choice of features or settings uses the training split only.

## Data and preprocessing

PhiUSIIL contains 235,795 URLs: 134,850 legitimate and 100,945 phishing. We drop the identifier column
`FILENAME` and the raw-text columns `URL`, `Domain`, `TLD` and `Title`, whose signal is already in
engineered features such as `TLDLegitimateProb`. That leaves 50 numeric features. We then remove 808
duplicate rows, so no identical row can appear in both training and test data, and check that no values
are missing.

The raw label is 1 for legitimate and 0 for phishing. We flip it to `is_phishing`, so phishing is the
positive class and precision, recall and F1 all describe phishing detection.

## Split and cross-validation

| Set | Rows | Phishing share |
| --- | --- | --- |
| Training (80%) | 187,989 | 42.6% |
| Test (20%) | 46,998 (26,970 legitimate, 20,028 phishing) | 42.6% |

The split is stratified on the label with random seed 42, so every experiment sees the same rows. Model
selection and tuning use stratified 5-fold cross-validation (shuffled, seed 42) inside the training set.
The test set is used only for the final scores.

## Models and metrics

We compare Logistic Regression, a Decision Tree and a Random Forest from scikit-learn. Each is a pipeline;
Logistic Regression standardizes its inputs inside the pipeline, so the scaler is fit only on training
folds. We report accuracy, precision, recall and F1. F1 is the primary metric for model selection and
tuning. Recall gets extra weight in the discussion, because a missed phishing site costs more than a false
alarm.

## Feature sets

With all 50 features every model scores about 100% F1, so the comparison cannot tell models apart. We
therefore also evaluate an ablated set, chosen with 5-fold cross-validation on the training split only:

| Feature set | Features | LR / DT / RF cross-validated F1 |
| --- | --- | --- |
| Full | all 50 | 0.9998 / 1.0000 / 1.0000 |
| Full minus `URLSimilarityIndex` | 49 | 0.9991 / 0.9988 / 0.9999 |
| Also minus the 6 strongest page-content counts | 43 | 0.9983 / 0.9970 / 0.9993 |
| **Ablated: 20 URL-string features, without `URLSimilarityIndex` and `IsHTTPS`** | 20 | **0.9727 / 0.9898 / 0.9940** |

- `URLSimilarityIndex` alone, with a single threshold, separates the classes at 99.7% accuracy, and every
  legitimate URL uses HTTPS. Both give the label away.
- Dropping single features barely helps, because the remaining features overlap heavily.
- Restricting to features computed from the URL string does separate the models. It also matches a
  realistic use case, judging a link before loading the page, and it is the same input the deep learning
  model in Experiment 5 reads.

Caveat: every legitimate URL in PhiUSIIL starts with `https://www.`, and the URL features still partly
encode that prefix. A Random Forest predicts the prefix from them at 99.5% accuracy. On the URLs that start
with `https://www.`, where the prefix gives nothing away, a Random Forest on the URL features reaches only
0.83 F1. Scores near 100% therefore reflect how the dataset was collected as much as how well the models
detect phishing.
