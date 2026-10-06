# Phishing Detection Project Schedule

## Overview

We finish in 7 weeks, Oct 5 to Nov 22, 2026. All results are frozen by the end of Week 5, and the report and presentation are submitted in Week 7. If the course deadline differs, shift the weeks and keep the order.

**Already in the repo (setup only, no modeling code):**

- The folder layout, `requirements.txt`, the README and the shared settings in `config.py` (seed, split size, CV folds, columns to drop).
- `scripts/download_data.py`, which downloads the dataset into `data/raw/`.
- Stubs for every module and experiment script. Each one's docstring says what to build and who owns it.
- Tests in `tests/test_pipeline.py` that define "done" for the core modules. They fail until the stubs are implemented.

**We build:** preprocessing, the models, evaluation, feature selection, tuning, all four experiments, the EDA and error analysis, the report and the slides.

**Known issue to plan for:** published results on this dataset are near 100%. A few features, especially `URLSimilarityIndex`, nearly separate the classes on their own. Week 2 checks this in our data and decides on an ablated feature set, so the experiments can tell the models apart.

**Deliverables:** a final report, a presentation, and a repo anyone can rerun from a clean clone.

## Roles

Each person owns one area from start to finish, including the code, the runs and the report section for it. Each also reviews one teammate's pull requests. Owners are a starting proposal, so swap them freely.

| Member | Builds | Writes report section | Reviews |
| --- | --- | --- | --- |
| Talan Tran | `evaluate.py`, `tuning.py`, Experiment 3 (tuning), feature-set option, final reproducibility run | Methods, Experiment 3 | Paul |
| Joshua Groh | `preprocess.py`, EDA notebook, the ablation decision, error-analysis notebook | Dataset and Preprocessing, Error Analysis | Wilfred |
| Paul O'brien | `models.py`, `features.py`, Experiments 1 and 2, comparison with Prasad and Chandra | Experiments 1 and 2, Related Work | Joshua |
| Wilfred Vergara | Experiment 4 (feature importance), shared plot style, report and slide integration | Experiment 4, Introduction, Discussion and Conclusion | Talan |

## Week-by-week plan

Each week has one goal, one task per person and a milestone checked at the weekly sync.

### Week 1 (Oct 5–11): Setup and core modules

Everyone clones the repo, follows the README setup and downloads the data. Weeks 2–5 depend on the three core modules, so those come first.

| Member | Tasks |
| --- | --- |
| Joshua | Implement `preprocess.py` (`clean`, `split_features_target`, `train_test`). Start the EDA notebook: class balance and feature types. Examine the text and categorical columns (`URL`, `Domain`, `TLD`, `Title`) and decide per model whether to encode or exclude each one (for example, `TLD` could be encoded for the tree models). Record the decision in `config.DROP_COLUMNS` and the README. |
| Talan | Add teammates as collaborators and protect `main`. Implement `evaluate.py`: CV splitter, cross-validation, held-out metrics with phishing as positive, `fit_and_evaluate`, confusion-matrix plot. |
| Paul | Implement `models.py` with the three default pipelines. Read the PhiUSIIL paper and note its models, metrics and scores. |
| Wilfred | Set up the report outline and a shared plot style (colors, fonts, labels). Draft the Motivation from the proposal. |

**Done when:** `pytest` passes for the cleaning, split and model tests, and the team agrees at the sync on how to handle the text columns.

### Week 2 (Oct 12–18): EDA, ablation decision and Experiment 1

| Member | Tasks |
| --- | --- |
| Joshua | Finish the EDA notebook: distributions by class, correlations, and how well each feature separates the classes on its own. Propose an ablated feature set. |
| Paul | Implement and run `exp1_full_features.py` on both feature sets. Save the tables and confusion matrices. Draft the Experiment 1 results paragraph. |
| Talan | Add a feature-set option (full or ablated) to `config.py` and the data-loading path, with a test. |
| Wilfred | Draft the Dataset and Preprocessing section using Joshua's EDA, and hand it to Joshua to own. |

**Done when:** the team agrees on the ablated set at the sync and the Experiment 1 tables are pushed.

### Week 3 (Oct 19–25): Experiment 2 and tuning setup

| Member | Tasks |
| --- | --- |
| Paul | Implement `features.py` with at least two ranking methods. Implement and run `exp2_feature_selection.py` for several values of k, on both feature sets. Report F1, recall, fit time and model complexity against the full-feature models. For complexity, use the Decision Tree's depth and node count and the Random Forest's total node count. |
| Talan | Implement `tuning.py` (search spaces and CV search) and `exp3_tuning.py`. Start the tuning runs, which are the longest compute job. |
| Joshua | Compare which features each ranking method picks and flag redundant pairs. Start the error-analysis notebook. |
| Wilfred | Implement `exp4_feature_importance.py` on a default Random Forest to prototype the importance plots in the shared style. |

**Done when:** all tests pass, the Experiment 2 tables are pushed and the tuning runs are under way.

### Week 4 (Oct 26–Nov 1): Experiment 3, tuning

| Member | Tasks |
| --- | --- |
| Talan | Finish tuning all three models on both feature sets. Save the best parameters and models, then build the tuned-versus-default table. |
| Paul | Write the Experiment 1 and 2 results sections. |
| Joshua | Check the tuning setup for leakage (selection and scaling inside CV only). Analyze the test-set false negatives (phishing sites the models missed). |
| Wilfred | Switch Experiment 4 to the tuned best model once Talan publishes it. |

**Done when:** the best model is chosen and saved in `results/models/`.

### Week 5 (Nov 2–8): Experiment 4 and results freeze

| Member | Tasks |
| --- | --- |
| Wilfred | Finalize impurity importance, permutation importance and Logistic Regression coefficients on both feature sets. Export the top rules of the Decision Tree. |
| Joshua | Write up the error analysis, explaining what the missed phishing sites have in common. |
| Paul | Compare our results with Prasad and Chandra and build the summary results figure. |
| Talan | Write the Experiment 3 section. Add a `run_all` script and check that it reproduces every table. |

**Done when:** results are frozen on Friday, Nov 6. No reruns after that unless a bug is found.

### Week 6 (Nov 9–15): Report draft and slides

| Member | Tasks |
| --- | --- |
| All | Finish your own report sections (see Roles) and make 2–3 slides for your area. |
| Wilfred | Write the Introduction, Discussion and Conclusion. Merge everything into one draft and one slide deck. |

**Done when:** a full draft and deck exist by Friday, Nov 13.

### Week 7 (Nov 16–22): Review and submit

| Member | Tasks |
| --- | --- |
| All | Review your assigned teammate's section and their code (see Roles). Rehearse the presentation twice. |
| Talan | Run the whole pipeline from a fresh clone and fix anything that breaks. Tag the release. |
| Wilfred | Apply the review edits and check that figures, references and formatting are consistent. Submit. |

**Done when:** the report, slides and tagged repo are submitted.

## How we work

- **Weekly sync:** 30 minutes, same day each week. Each person shows their milestone and names anything blocking them.
- **Branches:** one branch per task (for example `preprocess-clean`). Merge to `main` through a pull request that your reviewer approves.
- **Interfaces:** keep the function names and signatures in the stubs. Other people's code calls them. If one has to change, say so at the sync first.
- **Results:** generate tables and figures only with the scripts in `experiments/`, never by hand. Commit them to `results/` so the report cites one source.
- **Shared settings:** don't change the seed (42), the test split (20%) or the CV folds (5) without telling the team, because that changes everyone's numbers.
- **Report:** write in one shared document. Each section names its owner at the top until the final pass.

## Risks

| Risk | Fallback |
| --- | --- |
| The core modules slip past Week 1 and block everyone | They are small. Pair up at the Week 1 sync to finish whichever one is late. |
| Every model scores near 100%, so the comparison is uninformative | Report both feature sets and lead with the ablated one (Week 2 decision). |
| Random Forest tuning on about 188k training rows runs for hours | Start in Week 3. Tune on a stratified 30% sample of the training set, then refit the best settings on all of it. |
| A teammate falls behind | Their reviewer picks up the task at the next sync. The Week 5 freeze leaves 2 weeks of buffer. |
| Results change after the writing has started | Freeze on Nov 6. Any later rerun must update every affected table and figure in the same pull request. |
