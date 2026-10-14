# Walk-forward evaluation — frozen SPY series (October 2026)

This report is based on user-generated model outputs inspected against their accompanying metadata. **Seven** models are evaluated through five expanding-window, annually retrained out-of-sample folds (2022, 2023, 2024, 2025, and partial 2026). Hyperparameters are fixed and not independently tuned.

## Audited dataset alignment

- Adjusted daily SPY closing prices: **4,219**, 2010-01-04 through 2026-10-12.
- Price SHA-256 fingerprint in all three runs: `aee5568aa2d37585077b69a894f044513891adf6d535c7362887fac0299d7df5`.
- Out-of-sample dates: **1,197** matching dates, first 2022-01-03 and last 2026-10-09 (feature dates).
- Across all three prediction CSVs, date, annual fold, actual next-day log return, and all three baseline predictions match **exactly**.
- Annual fold sizes: 2022 = 251; 2023 = 250; 2024 = 252; 2025 = 250; 2026 (partial) = 194.
- Training data expands each year. Metadata records that the next observed close for any training label must predate the first test feature date.

## Aggregate out-of-sample results

| Model | RMSE | MAE | Directional accuracy | N |
|---|---|---|---|---|
| HistoricalMean | 0.010869 | 0.007658 | 53.38% | 1197 |
| Ridge_Baseline | 0.010919 | 0.007779 | 50.79% | 1197 |
| Ridge_With_EMD | 0.010922 | 0.007787 | 49.96% | 1197 |
| Ridge_With_CEEMDAN | 0.010920 | 0.007792 | 49.54% | 1197 |
| GradientBoosting_Baseline | 0.011124 | 0.007927 | 53.72% | 1197 |
| GradientBoosting_With_EMD | 0.011228 | 0.007983 | 52.55% | 1197 |
| GradientBoosting_With_CEEMDAN | 0.011246 | 0.007921 | 53.22% | 1197 |

**Result:** the historical-mean model has the lowest aggregate RMSE (**0.010869**) and MAE (**0.007658**). Neither EMD nor CEEMDAN lowers the aggregate RMSE of its corresponding baseline model. Boosting has the highest directional accuracy (about 53.72%), which alone does not establish trading value or significance.

![Walk-forward overall RMSE](../figures/walk_forward_rmse.png)

## Annual fold comparison

![Walk-forward annual RMSE](../figures/walk_forward_annual_rmse.png)

![Cumulative loss relative to historical mean](../figures/walk_forward_cumulative_loss.png)

Positive cumulative loss difference indicates more squared forecasting error than the historical mean. Individual annual improvements do not establish regime-specific performance or statistical significance.

## Reproducing the runs

Using the frozen adjusted-close CSV created locally, execute:

```bash
python -m src.walk_forward --csv data/spy_adjusted_frozen.csv --output results/walk_forward_baseline
python -m src.walk_forward --csv data/spy_adjusted_frozen.csv --emd emd --output results/walk_forward_emd
python -m src.walk_forward --csv data/spy_adjusted_frozen.csv --emd ceemdan --output results/walk_forward_ceemdan
```

The CSV is excluded from this update and has to be recreated or supplied by the user. Provider data can be revised. Results were computed with **12 CEEMDAN trials** as shown in metadata.

## Limitations

- This is an annual expanding-window experiment, not an intrayear daily-retrained or continuously tuned strategy.
- The training-period labels were cutoff-aware, but verifying those constraints in source code is separate from this output-level audit.
- The uploaded files do not demonstrate repeated-seed CEEMDAN robustness or independent hyperparameter selection.
- Statistical significance, trading costs, portfolio construction, risk management and profitability were not evaluated.
- Price-adjustment revisions and the small number of annual folds limit generalization.

Machine-readable outputs are in `docs/walk_forward_overall_metrics.csv`, `docs/walk_forward_fold_metrics.csv`, and `docs/walk_forward_predictions.csv`.
