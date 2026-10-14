# SPY out-of-sample forecasting: October 13, 2026 runs

## Status and data

These are the **actual terminal metrics reported by the project operator** after running v0.2.2 locally, not measurements reproduced in this build environment. Price loader reported **4,219 valid adjusted SPY closes** from **2010-01-04 to 2026-10-12**, after excluding one missing/nonfinite `Close` record dated 2026-10-13. All evaluations reported **1,197 test predictions**. The configured cutoffs were training through 2018-12-31, validation through 2021-12-31, test from 2022 onward. Model fitting currently combines training and validation observations after fixing settings; this is **not an independent validation/tuning study**.

* RMSE/MAE are in **log-return units**, not dollar prices.
* Directional accuracy is correct positive-versus-nonpositive sign classification; it does not establish trading profitability.
* The following three commands were run **separately** against live Yahoo Finance data, not against a stored immutable CSV. Dataset identity across runs has **not been verified**. Results should be considered preliminary.

## Baselines-only run

`python -m src.experiment --output results/spy_baselines_v2`

| Model | RMSE ↓ | MAE ↓ | Directional accuracy ↑ | N |
|---|---:|---:|---:|---:|
| HistoricalMean | **0.010868** | **0.007656** | **53.3835%** | 1,197 |
| Ridge_Baseline | 0.010924 | 0.007794 | 50.8772% | 1,197 |
| GradientBoosting_Baseline | 0.011147 | 0.007952 | 52.4645% | 1,197 |

## EMD run

`python -m src.experiment --emd emd --output results/spy_emd_v2`

| Model | RMSE ↓ | MAE ↓ | Directional accuracy ↑ | N |
|---|---:|---:|---:|---:|
| HistoricalMean | **0.010868** | **0.007656** | **53.3835%** | 1,197 |
| Ridge_Baseline | 0.010924 | 0.007794 | 50.8772% | 1,197 |
| GradientBoosting_Baseline | 0.011124 | 0.007946 | 53.2999% | 1,197 |
| Ridge_With_EMD | 0.010928 | 0.007800 | 51.1278% | 1,197 |
| GradientBoosting_With_EMD | 0.011467 | 0.008066 | 52.4645% | 1,197 |

## CEEMDAN run

`python -m src.experiment --emd ceemdan --output results/spy_ceemdan_v2`

| Model | RMSE ↓ | MAE ↓ | Directional accuracy ↑ | N |
|---|---:|---:|---:|---:|
| HistoricalMean | **0.010868** | **0.007656** | **53.3835%** | 1,197 |
| Ridge_Baseline | 0.010924 | 0.007794 | 50.8772% | 1,197 |
| GradientBoosting_Baseline | 0.011106 | 0.007958 | 52.5480% | 1,197 |
| Ridge_With_CEEMDAN | 0.010923 | 0.007803 | 51.1278% | 1,197 |
| GradientBoosting_With_CEEMDAN | 0.011303 | 0.008005 | 49.8747% | 1,197 |

## Interpretation

The **historical-mean reference forecast had the lowest displayed test RMSE** in each run (0.010868). The Ridge + CEEMDAN RMSE was nearly identical to Ridge alone (0.010923 versus 0.010924), but the difference is too small to infer a genuine advantage. In these runs, adding decomposition features worsened gradient boosting's RMSE. The preliminary evidence therefore **does not show a convincing improvement** from EMD/CEEMDAN for this feature set and test interval.

## Reproducibility caveats and next audit

1. **Gradient-boosting baseline varies:** RMSE **0.011147** (baseline-only), **0.011124** (EMD), **0.011106** (CEEMDAN). Despite matching reported data coverage and test sample sizes, these results are *not numerically identical*. Do not average these values or select one as a universal baseline; compare each feature-augmented model to the corresponding baseline **within its own run**.
2. The complete input prices and their hashes were **not supplied** with these terminal outputs. The repository has fingerprint metadata functionality; preserve the three `run_metadata.json` outputs and compare recorded hashes, start/end dates, row counts, package versions, and model settings. Freeze one adjusted-close CSV locally and rerun all modes against that **same** CSV for the final research report.
3. CEEMDAN uses noise-assisted decomposition. Repeat with specified seeds and multiple seed sensitivity runs before treating differences this small as stable.
4. No rolling-origin refits, formal uncertainty intervals, economic backtest, trading-cost assessment, or independent hyperparameter selection have been completed. The current prototype is **not** a faithful Leung–Zhao HHT replication.

## Reproduce using a locally frozen data snapshot

The project's `--csv` option expects `Date,Close` where `Close` is **adjusted close**. Once a vetted local dataset is saved at `data/spy_adjusted.csv`, rerun:

```bash
python -m src.experiment --csv data/spy_adjusted.csv --output results/frozen_baselines
python -m src.experiment --csv data/spy_adjusted.csv --emd emd --output results/frozen_emd
python -m src.experiment --csv data/spy_adjusted.csv --emd ceemdan --output results/frozen_ceemdan
```

Avoid committing raw provider data unless redistribution is permitted. Save the hashes/metadata and publish verified aggregate metrics and selected figures instead.
