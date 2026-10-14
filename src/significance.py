
"""Paired forecast-loss comparisons using HAC uncertainty."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.sandwich_covariance import cov_hac
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


BENCHMARK = "HistoricalMean"
ACTUAL = "actual_next_day_log_return"


def analyze(predictions: pd.DataFrame, maxlags: int = 5):
    predictions = predictions.sort_values("date").reset_index(drop=True)

    if predictions["date"].duplicated().any():
        raise ValueError("Duplicate prediction dates")

    models = [
        col for col in predictions.columns
        if col not in {"date", "fold_year", ACTUAL, BENCHMARK}
    ]

    numeric = predictions[[ACTUAL, BENCHMARK] + models].apply(
        pd.to_numeric, errors="raise"
    )

    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Predictions contain missing or infinite values")

    actual = numeric[ACTUAL].to_numpy()
    benchmark = numeric[BENCHMARK].to_numpy()
    benchmark_loss = (actual - benchmark) ** 2

    results = []

    for model in models:
        forecast = numeric[model].to_numpy()
        loss = (actual - forecast) ** 2
        difference = loss - benchmark_loss

        # Intercept-only regression: coefficient = mean loss difference.
        fit = sm.OLS(
            difference,
            np.ones((len(difference), 1))
        ).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

        mean_diff = float(fit.params[0])
        se = float(fit.bse[0])

        if se > 0:
            statistic = mean_diff / se
            p_value = float(2 * norm.sf(abs(statistic)))
        else:
            statistic = np.nan
            p_value = np.nan

        results.append({
            "model": model,
            "N": len(difference),
            "mean_loss_difference": mean_diff,
            "hac_standard_error": se,
            "z_statistic": statistic,
            "p_value": p_value,
            "ci_95_lower": mean_diff - 1.96 * se,
            "ci_95_upper": mean_diff + 1.96 * se,
            "model_RMSE": float(np.sqrt(np.mean(loss))),
            "benchmark_RMSE": float(np.sqrt(np.mean(benchmark_loss))),
        })

    output = pd.DataFrame(results)

    if output["p_value"].isna().any():
        raise ValueError("Cannot compute valid p-values for all models")

    _, adjusted, _, _ = multipletests(
        output["p_value"].to_numpy(),
        method="holm"
    )

    output["holm_adjusted_p"] = adjusted
    output["significant_at_0_05"] = adjusted < 0.05

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="docs/walk_forward_predictions.csv"
    )
    parser.add_argument(
        "--output",
        default="results/significance"
    )
    parser.add_argument("--maxlags", type=int, default=5)
    args = parser.parse_args()

    if args.maxlags < 0:
        parser.error("--maxlags must be nonnegative")

    predictions = pd.read_csv(args.csv)

    if args.maxlags >= len(predictions):
        parser.error("--maxlags must be smaller than the sample size")

    results = analyze(predictions, maxlags=args.maxlags)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_dir / "significance_results.csv", index=False)

    print("\nPaired squared-error comparisons against HistoricalMean:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.8g}"))
    print("\nPositive mean difference = model has higher average squared error.")
    print("HAC lag:", args.maxlags)
    print("Two-sided normal-approximation tests; Holm correction applied.")


if __name__ == "__main__":
    main()
