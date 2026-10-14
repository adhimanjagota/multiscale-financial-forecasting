# Statistical evaluation of walk-forward SPY forecasts

## Scope and question

This exploratory analysis uses **1,197 paired out-of-sample predictions** from 2022 through October 2026. The response is the next-day SPY adjusted-price log return. Two separate comparison families address (1) each model versus a historical-mean benchmark (six comparisons) and (2) EMD/CEEMDAN models versus their respective non-decomposition versions (four comparisons).

Daily squared-error loss differential is `d[t] = loss(candidate, t) - loss(reference, t)`. Positive average values indicate that the candidate has **higher average squared prediction error**, not better forecasting performance. Tests fit an intercept-only regression with HAC covariance and two-sided normal-approximation p-values; Holm adjustments are applied *within each comparison family*. These are approximate, DM-style tests, **not a comprehensive Diebold–Mariano implementation or a Clark–West adjustment**.

## Main benchmark results

Across all six comparisons with historical mean, the Holm-adjusted p-values exceed 0.05 at every examined HAC lag: **0, 5, 10 and 20**. The historical mean has the lowest observed RMSE (0.010869), but **failure to reject is not evidence of equivalence**.

| Model | Holm p (lag 0) | Holm p (lag 5) | Holm p (lag 10) | Holm p (lag 20) |
|---|---:|---:|---:|---:|
| `Ridge_Baseline` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `GradientBoosting_Baseline` | 0.2198 | 0.2037 | 0.2312 | 0.2997 |
| `Ridge_With_EMD` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `GradientBoosting_With_EMD` | 0.1739 | 0.2037 | 0.2312 | 0.2997 |
| `Ridge_With_CEEMDAN` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `GradientBoosting_With_CEEMDAN` | 0.0516 | 0.0596 | 0.0745 | 0.1766 |

## Direct feature-addition comparisons (HAC lag 5)

Directly comparing decomposed models to their corresponding base learners also finds **no statistically significant evidence of a difference** after Holm correction.

| Base model | Augmented model | Base RMSE | Augmented RMSE | Raw p | Holm-adjusted p |
|---|---|---:|---:|---:|---:|
| `Ridge_Baseline` | `Ridge_With_EMD` | 0.010919 | 0.010922 | 0.8474 | 1.0000 |
| `Ridge_Baseline` | `Ridge_With_CEEMDAN` | 0.010919 | 0.010920 | 0.9548 | 1.0000 |
| `GradientBoosting_Baseline` | `GradientBoosting_With_EMD` | 0.011124 | 0.011228 | 0.3493 | 1.0000 |
| `GradientBoosting_Baseline` | `GradientBoosting_With_CEEMDAN` | 0.011124 | 0.011246 | 0.3506 | 1.0000 |

## Interpretation and limitations

- EMD and CEEMDAN feature additions do not improve observed aggregate RMSE in these experiments; the direct paired tests also fail to establish a statistically detectable difference.
- The test uses an **exploratory, fixed set of models and four selected HAC bandwidths**. Its p-values depend on serial-dependence assumptions, bandwidth, normal approximation and the fixed evaluation design. The four bandwidths are sensitivity analyses, not four separate independent studies.
- The comparisons involve related/nested predictive models. Standard unadjusted mean-loss-difference tests can have nonstandard properties for nested-model forecast comparisons; dedicated tests (e.g. Clark–West where appropriate) would be follow-up work.
- Holm correction covers comparisons **within** each stated family, not every choice, specification, bandwidth, or exploratory analysis conducted. Confidence intervals shown in CSVs are unadjusted 95% intervals.
- Five annual test folds share one time series; 1,197 daily predictions are not 1,197 independent market environments. Model selection, hyperparameter sensitivity, volatility regimes, end-of-sample dependence and stochastic CEEMDAN seed robustness remain open.
- Statistical evidence of forecast-loss differences is not evidence of trading profitability. No trading P&L, transaction costs, or risk-adjusted returns are evaluated.

## Reproduce

Install `statsmodels` (and ensure `pandas`, `numpy`, `scipy` are installed), then run from the repository root:

```bash
python -m pip install statsmodels
python -m pytest -q
for lag in 0 5 10 20; do
  python -m src.significance --csv docs/walk_forward_predictions.csv --maxlags "$lag" --output "results/significance_lag_$lag"
done
python -m src.decomposition_significance --csv docs/walk_forward_predictions.csv --maxlags 5 --output results/decomposition_significance
```

Input: [`docs/walk_forward_predictions.csv`](../walk_forward_predictions.csv). This script depends on the previously committed walk-forward prediction table; the ZIP update does not replace that table. CSV outputs live in [`docs/significance/`](significance/).
