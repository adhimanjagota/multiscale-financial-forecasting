"""Paired HAC forecast-loss comparisons for decomposition features."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


PAIRS = [
    ("Ridge_Baseline", "Ridge_With_EMD"),
    ("Ridge_Baseline", "Ridge_With_CEEMDAN"),
    ("GradientBoosting_Baseline", "GradientBoosting_With_EMD"),
    ("GradientBoosting_Baseline", "GradientBoosting_With_CEEMDAN"),
]


def analyze(df, maxlags=5):
    if maxlags < 0 or maxlags >= len(df):
        raise ValueError("Invalid HAC lag")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate prediction dates")

    df = df.sort_values("date").reset_index(drop=True)
    actual = df["actual_next_day_log_return"].to_numpy(dtype=float)

    rows = []

    for baseline_name, augmented_name in PAIRS:
        baseline = df[baseline_name].to_numpy(dtype=float)
        augmented = df[augmented_name].to_numpy(dtype=float)

        if not np.isfinite(
            np.column_stack([actual, baseline, augmented])
        ).all():
            raise ValueError("Non-finite prediction values")

        baseline_loss = (actual - baseline) ** 2
        augmented_loss = (actual - augmented) ** 2

        # Positive difference: decomposition has higher squared error.
        diff = augmented_loss - baseline_loss

        fit = sm.OLS(diff, np.ones((len(diff), 1))).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": maxlags},
        )

        mean_diff = float(fit.params[0])
        se = float(fit.bse[0])

        if not np.isfinite(se) or se <= 0:
            raise ValueError("Invalid HAC standard error")

        z = mean_diff / se
        p = float(2 * norm.sf(abs(z)))

        rows.append({
            "baseline": baseline_name,
            "decomposition_model": augmented_name,
            "N": len(diff),
            "mean_loss_difference": mean_diff,
            "hac_standard_error": se,
            "z_statistic": z,
            "p_value": p,
            "ci_95_lower": mean_diff - 1.96 * se,
            "ci_95_upper": mean_diff + 1.96 * se,
            "baseline_RMSE": float(np.sqrt(baseline_loss.mean())),
            "decomposition_RMSE": float(np.sqrt(augmented_loss.mean())),
        })

    output = pd.DataFrame(rows)

    output["holm_adjusted_p"] = multipletests(
        output["p_value"],
        method="holm",
    )[1]

    output["significant_at_0_05"] = output["holm_adjusted_p"] < 0.05

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="docs/walk_forward_predictions.csv",
    )
    parser.add_argument(
        "--output",
        default="results/decomposition_significance",
    )
    parser.add_argument("--maxlags", type=int, default=5)
    args = parser.parse_args()

    results = analyze(
        pd.read_csv(args.csv),
        maxlags=args.maxlags,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "results.csv", index=False)

    print("\nDirect decomposition comparisons:")
    print(results.to_string(
        index=False,
        float_format=lambda value: f"{value:.8g}",
    ))
    print("\nPositive loss difference = decomposition performs worse.")
    print("HAC lag:", args.maxlags)


if __name__ == "__main__":
    main()
