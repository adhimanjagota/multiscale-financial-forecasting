# Frozen-data SPY forecast benchmark

**Status:** Controlled single-test-window comparison, October 2026. Values directly calculated from the three `metrics.csv` files saved from the same frozen adjusted-close CSV. This is a methodological prototype, **not** a demonstration of profitability or a replication of Leung–Zhao's complete HHT system.

## Data and protocol

- Asset: SPY, adjusted daily closing prices, 4,219 observations from **2010-01-04 through 2026-10-12**.
- Feature-date test period: **2022-01-03 through 2026-10-09**; **1,197** test predictions. Features at a date forecast the next close-to-close log return.
- Training through 2018-12-31; nominal validation from 2019 through 2021. **The implementation combines the train and validation periods for final fitting, so validation is not an independently assessed holdout.**
- Trailing 80-observation EMD or CEEMDAN windows; baseline modes use the same warmup.
- Dataset SHA-256 fingerprint across all three `run_metadata.json` files: `aee5568aa2d37585077b69a894f044513891adf6d535c7362887fac0299d7df5`.
- Dates and numerical targets match exactly across all three exported prediction CSVs; all baseline metrics also match exactly.

## Verified results

| Model | RMSE | MAE | Directional accuracy |
|---|---:|---:|---:|
| Historical mean | 0.010868 | 0.007656 | 53.38% |
| Ridge | 0.010924 | 0.007794 | 50.88% |
| Gradient boosting | 0.011135 | 0.007953 | 52.38% |
| Ridge + EMD | 0.010929 | 0.007801 | 51.13% |
| Gradient boosting + EMD | 0.011409 | 0.008022 | 50.46% |
| Ridge + CEEMDAN | 0.010923 | 0.007803 | 51.13% |
| Gradient boosting + CEEMDAN | 0.011288 | 0.008015 | 51.55% |


**Interpretation.** The historical mean obtains the lowest RMSE (0.010868). Adding EMD/CEEMDAN features does not show a convincing RMSE improvement against the historical-mean benchmark. Ridge + CEEMDAN's tiny improvement over Ridge alone does not establish significance. No strategy returns, transaction costs, risk measures or formal forecasting significance tests are included.

## Figures

![RMSE comparison](../figures/frozen_rmse_comparison.png)

![Actual vs forecasts](../figures/frozen_actual_vs_predictions.png)

![Cumulative excess squared errors](../figures/frozen_cumulative_error_difference.png)

**Plot interpretation:** positive cumulative excess squared error means a model accumulated more squared prediction error than the historical-mean comparator. The actual-vs-predicted figure uses the final 160 observations from the CEEMDAN experiment.

## Important limitations and next experiments

1. The reported scores arise from one fixed 2022–2026 test window, not independently repeated rolling-origin assessments or nested hyperparameter tuning.
2. Training and validation are combined for final fitting; a true validation-based selection process is outstanding.
3. CEEMDAN is stochastic. A matched dataset and deterministic baseline outputs do not establish ensemble stability across repeated CEEMDAN seeds.
4. Rolling-window decomposition avoids directly including future prices within a feature window, but terminal IMF features are susceptible to edge effects, mode-mixing, and unstable component ordering.
5. Forecast performance is not the same as economic profitability. No fee-adjusted backtest is performed.
6. Full HHT instantaneous-frequency/amplitude feature extraction and original-study replication are out of scope.
7. Provider datasets can be revised. The frozen CSV is deliberately not redistributed; fingerprints and date constraints aid auditing but do not grant re-download identity.

## Reproduce locally (with your frozen CSV)

```bash
python -m src.experiment --csv data/spy_adjusted_frozen.csv --output results/frozen_baseline
python -m src.experiment --csv data/spy_adjusted_frozen.csv --emd emd --output results/frozen_emd
python -m src.experiment --csv data/spy_adjusted_frozen.csv --emd ceemdan --output results/frozen_ceemdan
```

The raw adjusted-price CSV is intentionally excluded from this GitHub update. Numerical result reproduction requires precisely the same saved CSV; a fresh Yahoo download may differ.
