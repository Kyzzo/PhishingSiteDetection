"""Experiment 2: retrain each model on reduced feature subsets and compare to full features.

Owner: Paul (Week 3).
Run: python -m experiments.exp2_feature_selection
"""


def main():
    # TODO:
    #   1. Load the stratified split.
    #   2. Evaluate every model on the full feature set as the reference.
    #   3. For several k values and ranking methods, pick the top-k features
    #      using the TRAINING split only (features.select_top_k), then evaluate
    #      every model on just those columns.
    #   4. Save one table comparing recall, F1, n_features, fit time and model
    #      complexity (Decision Tree depth + node count, Random Forest total nodes).
    raise NotImplementedError


if __name__ == "__main__":
    main()
