"""Experiment 4: analyze which features drive the strongest model's predictions.

Owner: Wilfred (Weeks 3-5).
Run: python -m experiments.exp4_feature_importance
"""

def main(feature_sets=("full", "ablated"), n_repeats=10, top_n=TOP_N):
    apply_style()
    cleaned_data = clean(load_raw())

    all_tables = []
    all_agreements = []

    for feature_set in feature_sets:
        table, agreement = run_feature_set(
            cleaned_data,
            feature_set,
            n_repeats,
            top_n,
        )

        all_tables.append(table)
        all_agreements.append(agreement)

        columns_to_display = [
            "feature",
            "rf_impurity",
            "perm_f1_mean",
            "lr_coef",
        ]

        with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
            print(table.head(10)[columns_to_display].to_string(index=False))

    # Save the feature importance results
    save_table(
        pd.concat(all_tables, ignore_index=True),
        "exp4_feature_importance.csv",
    )

    # Save the method agreement results
    save_table(
        pd.concat(all_agreements, ignore_index=True),
        "exp4_method_agreement.csv",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=list(config.FEATURE_SETS),
        default=["full", "ablated"],
    )

    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="permutation repeats per feature",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N,
        help="features shown in the plots",
    )

    args = parser.parse_args()
    main(args.feature_sets, args.n_repeats, args.top_n)
