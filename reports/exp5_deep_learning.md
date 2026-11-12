# Experiment 5: Deep learning vs. classical models

The deep learning model, a character-level CNN, only tied a simple TF-IDF baseline on F1 (0.871 vs. 0.874)
and caught fewer phishing sites (recall 0.792 vs. 0.828); training longer and tuning its decision threshold
did not change that.

## Models

All models use the same train/test split as the other experiments.

- **Character-level CNN (deep learning).** The CNN reads the URL one character at a time, so it needs no
  hand-made features. Each character becomes a learned 32-dimensional vector. Three parallel 1-D
  convolutions of width 3, 5 and 7 (128 filters each) learn substring detectors such as `login` or
  `-secure`, and global max pooling keeps each detector's strongest match anywhere in the URL. A dense layer
  (64 units, dropout 0.5) outputs P(phishing). It has 89,633 parameters, trains with Adam for up to 15
  epochs, and stops early when F1 on a 10% validation slice of the training data stops improving. The design
  follows character-level text CNNs (Zhang et al., 2015) and URLNet, a CNN built for malicious URL detection
  (Le et al., 2018).
- **TF-IDF + Logistic Regression (classical text baseline).** This model counts character 1- to 5-grams in
  the URL (up to 200,000 n-grams, TF-IDF weighted) and fits a linear model on them. It sees exactly the same
  input as the CNN, so comparing the two isolates the method from the input.
- **LR, Decision Tree, Random Forest on engineered features**, on both feature sets from Methods, for
  reference.

## The URL-prefix shortcut

Every legitimate URL in PhiUSIIL starts with `https://www.`, and every URL ending in `/` is phishing. A
one-line rule, "phishing unless the URL starts with `https://www.`", reaches 98.96% accuracy. A text model
given raw URLs learns that rule instead of anything about phishing. By default we therefore strip the
scheme, a leading `www.` and trailing slashes (normalized URLs), and we report raw-URL scores separately to
show the size of the shortcut.

## Results (test set)

| Input | Model | Precision | Recall | F1 | Missed phishing | Training time |
| --- | --- | --- | --- | --- | --- | --- |
| URL text, normalized | TF-IDF + LR | 0.925 | **0.828** | **0.874** | 3,451 | 22 s |
| URL text, normalized | Char-CNN, threshold 0.5 | 0.966 | 0.792 | 0.871 | 4,160 | 102 s (GPU) |
| URL text, normalized | Char-CNN, validation threshold (0.49) | 0.965 | 0.794 | 0.871 | 4,129 | 102 s (GPU) |
| URL text, raw | TF-IDF + LR | 1.000 | 0.994 | 0.997 | 114 | 22 s |
| URL text, raw | Char-CNN, threshold 0.5 | 1.000 | 0.996 | 0.998 | 82 | 41 s (GPU) |
| Engineered, ablated (20) | Random Forest | 0.994 | 0.995 | 0.995 | 94 | 4 s |
| Engineered, ablated (20) | Logistic Regression | 0.997 | 0.964 | 0.980 | 715 | 1 s |
| Engineered, full (50) | Random Forest | 1.000 | 1.000 | 1.000 | 0 | 4 s |

The test set holds 20,028 phishing URLs. Raw-URL scores are inflated by the prefix shortcut.

Source: `results/tables/exp5_deep_learning.csv` (all 12 rows), training histories in
`results/tables/exp5_char_cnn_history_*.csv`; one confusion matrix per row in `results/figures/exp5/`.

## What we tried to improve the CNN

- **More epochs.** We raised the limit from 10 to 15 epochs. Early stopping ended training at epoch 10,
  with the best validation F1 (0.8725) at epoch 8, so the model was not undertrained.
- **Decision threshold.** The CNN's precision is well above its recall, and this project weighs recall, so
  we let it choose the cutoff on P(phishing) that maximizes F1 on its validation slice, never on the test
  set. The chosen cutoff was 0.49, essentially the default 0.5, and recall moved only from 0.792 to 0.794.

Neither change closed the gap, so the CNN's lower recall comes from what it learned, not from where the
cutoff sits.

## Compute cost

On a CPU the CNN took 1,312 seconds to train, about 60 times as long as TF-IDF (22 seconds). On an NVIDIA
RTX 2060 GPU it took 102 seconds, still about 5 times as long. The classical models train in 1 to 22
seconds on a CPU.

## Why each method wins or loses

- **TF-IDF beats the CNN on normalized URLs.** Both learn from character patterns, but TF-IDF counts every
  1- to 5-gram explicitly, 200,000 of them, and a linear model on that many features is hard to beat with
  about 170,000 training URLs. The CNN's 384 filters learn similar substring detectors, but max pooling keeps
  only each detector's strongest match and discards how often a pattern occurs. URLs are short (median 17
  characters after normalization), so the CNN's advantage, building longer patterns out of shorter ones, has little to work
  with.
- **The CNN trades recall for precision.** It flags a URL only when a strong learned pattern fires, which
  keeps false alarms low (558 vs. 1,340) but misses more phishing (4,160 vs. 3,451). For a phishing filter,
  where a miss costs more than a false alarm, that is the wrong trade.
- **Engineered features score higher than any text model on normalized URLs** (Random Forest 0.995 vs.
  0.874 F1), but the comparison is not fully fair: the engineered URL features were computed from the raw
  URL and still partly encode the `https://www.` prefix. The fair pairing is with the raw-URL text models,
  and there all three land between 0.995 and 0.998 F1. Learned and engineered features perform about the
  same once both see the same information.
- **Random Forest beats Logistic Regression on the default settings** because it captures interactions
  between features, for example a long URL that also has an unusual top-level domain, while LR can only add
  up weighted features. Experiment 3 showed that tuned LR catches up.

On this dataset deep learning did not pay off: a simple text model matched the CNN at a fraction of the
cost, and classical models on engineered features did as well or better. We would recommend a Random
Forest, or a tuned Logistic Regression, for this task.

## References

- Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification.
  *Advances in Neural Information Processing Systems 28*.
- Le, H., Pham, Q., Sahoo, D., & Hoi, S. C. H. (2018). URLNet: Learning a URL representation with deep
  learning for malicious URL detection. *arXiv:1802.03162*.
- Prasad, A., & Chandra, S. (2024). PhiUSIIL: A diverse security profile empowered phishing URL detection
  framework based on similarity index and incremental learning. *Computers & Security, 136*, 103545.
