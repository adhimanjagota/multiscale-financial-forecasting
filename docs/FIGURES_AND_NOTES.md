## Visual summary

### Model comparison
![RMSE comparison](figures/rmse_comparison.png)

### Actual vs predicted returns
![Actual vs predicted returns](figures/actual_vs_predictions_ceemdan.png)

### Cumulative error difference vs. historical mean
![Cumulative error difference](figures/cumulative_error_difference.png)

### Notes on interpretation

- The prediction **dates align across runs**, but the three runs were executed with slightly different downloaded SPY datasets. The corresponding `run_metadata.json` files have different dataset fingerprints:
  - baseline: `79017975608528a2...`
  - emd: `ff2c0d5ff2b0c792...`
  - ceemdan: `c607cf191646315f...`
- Because of those dataset differences, the baseline gradient-boosting metrics vary slightly between runs.
- These figures are therefore best interpreted as a **preliminary visual summary**, not a fully frozen scientific benchmark.
