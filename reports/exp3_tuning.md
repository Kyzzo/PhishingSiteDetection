# Experiment 3: Hyperparameter tuning

Tuning mattered only for Logistic Regression: on the ablated features its test F1 rose from 0.980 to
0.997, while the Decision Tree and Random Forest changed by less than 0.002.

## Setup

For each model and feature set we ran a randomized search over 20 settings, scored by F1 with 5-fold
cross-validation. To keep the Random Forest search to minutes, the search ran on a stratified 30% sample of
the training split; the best settings were then refit on the full training split. Default and tuned
versions were compared on the same cross-validation folds and the same test set.

| Model | Settings searched |
| --- | --- |
| Logistic Regression | regularization strength C (0.001 to 100, log scale), class weighting |
| Decision Tree | maximum depth, minimum samples per split and per leaf, split criterion, class weighting |
| Random Forest | number of trees (100 to 400), maximum depth, features per split, minimum samples per leaf, class weighting |

## Results (ablated features, test set)

| Model | Default F1 | Tuned F1 | Default recall | Tuned recall | Missed phishing (default to tuned) | Settings chosen |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.980 | **0.997** | 0.964 | **0.995** | 715 to 102 | C = 74, no class weighting |
| Decision Tree | 0.990 | 0.992 | 0.991 | 0.993 | 187 to 137 | entropy, depth 30, min 2 per leaf, min 22 per split, balanced classes |
| Random Forest | 0.995 | 0.996 | 0.995 | 0.995 | 94 to 106 | 149 trees, depth 20, log2 features per split, min 1 per leaf |

On the full 50 features every model scores between 0.9999 and 1.0000 F1 with or without tuning, so tuning
cannot show an effect there.

Source: `results/tables/exp3_tuning.csv`, `results/tables/exp3_best_params.json`; confusion matrices in
`results/figures/exp3/`.

## Why only Logistic Regression improved

Regularization explains it. C is the inverse of the regularization strength, so the default C = 1 shrinks
the feature weights toward zero. The URL features are heavy-tailed counts and ratios, and that shrinkage
kept the linear model from fitting them closely: it underfit, missing 715 phishing sites. The search chose
C = 74, roughly 74 times weaker regularization, and missed phishing dropped to 102.

The trees had little room to improve. scikit-learn's defaults already grow them fully, so they fit these
features well out of the box. Tuning traded a little recall for precision in the Random Forest (106 missed
instead of 94, but 70 false alarms instead of 112), because it optimized F1 rather than recall.

After tuning, the linear model ties the Random Forest (0.997 vs. 0.996 F1). On these features, a
well-regularized linear model is as good as an ensemble of 149 trees, and it trains in under a second
instead of about 10 seconds. The searches took between 8 seconds (Decision Tree) and 3.5 minutes (Random
Forest).
