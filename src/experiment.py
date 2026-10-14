"""Forecast next-day SPY log returns with time-safe baselines and optional EMD features.

Not investment advice. This is a methodological research prototype, not a reproduction
of Leung & Zhao (2021). See README for important limits.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_COLS = ['ret_lag_0', 'ret_lag_1', 'ret_lag_2', 'ret_lag_4', 'ret_lag_9',
             'mean_5', 'mean_20', 'vol_5', 'vol_20', 'momentum_5']
EMD_COLS = ['emd_fast', 'emd_mid', 'emd_slow', 'emd_remainder']


def download_spy(start='2010-01-01', end=None):
    """Fetch adjusted SPY close; explicitly report/remove missing market rows.

    Missing bars cannot be used for log returns. Zero/negative prices and
    duplicated dates are rejected rather than silently 'repaired'.
    """
    import yfinance as yf
    df = yf.download('SPY', start=start, end=end, auto_adjust=True,
                     progress=False, multi_level_index=False)
    if df.empty:
        raise ValueError('SPY download returned no rows. Try --csv with a local price file.')
    prices = df['Close']
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError('Expected one SPY Close column, got several.')
        prices = prices.iloc[:, 0]
    close = pd.Series(pd.to_numeric(prices, errors='coerce').to_numpy(dtype=float),
                      index=pd.to_datetime(df.index), name='Close').sort_index()
    if close.index.has_duplicates:
        raise ValueError('Duplicate dates returned for SPY; inspect the data source.')
    invalid = ~np.isfinite(close.to_numpy(dtype=float))
    if invalid.any():
        sample = ', '.join(str(d.date()) for d in close.index[invalid][:5])
        print(f'WARNING: SPY source has {int(invalid.sum())} missing/nonfinite Close rows'
              f' (first dates: {sample}); dropping those rows.')
        close = close.loc[~invalid]
    if close.empty or (close <= 0).any():
        raise ValueError('SPY price data are empty or contain nonpositive prices.')
    print(f'Loaded {len(close)} valid SPY adjusted daily closing prices '
          f'({close.index.min().date()} to {close.index.max().date()}).')
    return close


def load_csv(path):
    df = pd.read_csv(path)
    if 'Date' not in df or 'Close' not in df:
        raise ValueError('CSV must contain Date and Close columns (adjusted Close preferred).')
    dates = pd.to_datetime(df['Date'], errors='raise')
    px = pd.to_numeric(df['Close'], errors='raise').astype(float)
    out = pd.Series(px.to_numpy(), index=dates, name='Close').sort_index()
    if out.index.has_duplicates: raise ValueError('Duplicate dates in input CSV')
    return out


def make_features(close):
    close = close.astype(float).sort_index()
    if (close <= 0).any() or not np.isfinite(close.to_numpy(dtype=float)).all():
        raise ValueError('Close prices must be finite, positive and nonmissing')
    ret = np.log(close).diff()
    # Feature row t uses data through close[t]; target is close[t+1]/close[t].
    x = pd.DataFrame(index=close.index)
    for lag in [0, 1, 2, 4, 9]:
        x[f'ret_lag_{lag}'] = ret.shift(lag)
    x['mean_5'] = ret.rolling(5).mean()
    x['mean_20'] = ret.rolling(20).mean()
    x['vol_5'] = ret.rolling(5).std()
    x['vol_20'] = ret.rolling(20).std()
    x['momentum_5'] = np.log(close / close.shift(5))
    x['target_next_day'] = ret.shift(-1)
    return x.dropna().copy()


def rolling_emd_features(close, index, window=80, trials=0, seed=42):
    """Compute decomposition using only trailing prices at *each feature date*.

    Default EMD is deterministic/fast; trials>0 switches to noise-assisted CEEMDAN.
    Mode ordering of IMFs can vary by window; these are exploratory summary features.
    """
    try:
        from PyEMD import EMD, CEEMDAN
    except ImportError as exc:
        raise ImportError('Install EMD-signal (pip install -r requirements.txt)') from exc
    prices = np.log(close.to_numpy(dtype=float))
    indexer = {dt: i for i, dt in enumerate(close.index)}
    records = []
    for dt in index:
        i = indexer[dt]
        if i + 1 < window:
            records.append([np.nan] * 4)
            continue
        hist = prices[i-window+1:i+1]
        # Relative log prices ensure translation invariance. Differencing is not needed
        # for this descriptive representation, but nonstationarity remains.
        hist = hist - hist[0]
        if trials > 0:
            decomposer = CEEMDAN(trials=trials, parallel=False)
            decomposer.noise_seed(seed)
            imfs = decomposer.ceemdan(hist)
        else:
            imfs = EMD().emd(hist)
        imfs = np.atleast_2d(imfs)
        # Aggregate by IMF order; first components capture faster oscillations.
        last = imfs[:, -1]
        fast = float(last[0]) if len(last) >= 1 else 0.
        mid = float(last[1]) if len(last) >= 2 else 0.
        slow = float(last[2:-1].sum()) if len(last) > 3 else 0.
        remainder = float(last[-1]) if len(last) >= 3 else 0.
        records.append([fast, mid, slow, remainder])
    return pd.DataFrame(records, index=index, columns=EMD_COLS)


def split_by_date(frame, train_end='2018-12-31', valid_end='2021-12-31'):
    tr = frame.loc[frame.index <= train_end]
    va = frame.loc[(frame.index > train_end) & (frame.index <= valid_end)]
    te = frame.loc[frame.index > valid_end]
    if min(len(tr), len(va), len(te)) < 50:
        raise ValueError('Need >=50 samples in each period; adjust date boundaries.')
    # Remove last training and validation row: target next-day may fall in next split.
    tr = tr.iloc[:-1]
    va = va.iloc[:-1]
    return tr, va, te


def metrics(y, yhat):
    a = np.asarray(y)
    b = np.asarray(yhat)
    return {'RMSE': float(np.sqrt(mean_squared_error(a, b))),
            'MAE': float(mean_absolute_error(a, b)),
            'DirectionalAccuracy': float(np.mean((a > 0) == (b > 0))),
            'N': int(len(a))}


def models():
    return {'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            'GradientBoosting': HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=12,
                                                              l2_regularization=5.0, random_state=42)}


def run(close, output, train_end='2018-12-31', valid_end='2021-12-31', emd='none',
        emd_window=80, ceemdan_trials=12):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if emd_window < 20:
        raise ValueError('emd_window must be at least 20 trading observations')
    if emd == 'ceemdan' and ceemdan_trials < 1:
        raise ValueError('ceemdan_trials must be positive')
    close = close.sort_index()
    if len(close) < emd_window + 200:
        raise ValueError('Insufficient price history for aligned comparisons')
    features = make_features(close)
    # IMPORTANT: apply the exact same 80-day (configurable) warmup in EVERY
    # experiment, including `--emd none`, so baselines share training rows.
    features = features.loc[features.index >= close.index[emd_window - 1]].copy()
    cols = list(BASE_COLS)
    if emd != 'none':
        more = rolling_emd_features(close, features.index, window=emd_window,
                                    trials=ceemdan_trials if emd == 'ceemdan' else 0)
        features = features.join(more).dropna()
        cols += EMD_COLS
    tr, va, te = split_by_date(features, train_end, valid_end)
    # Fixed hyperparameters. Training + validation are combined for fitting;
    # validation is NOT an independent holdout in the final reported scores.
    # Test should not be used to select/tune hyperparameters repeatedly.
    results = []
    preds = pd.DataFrame({'date': te.index, 'actual_next_day_log_return': te['target_next_day'].to_numpy()})
    combined = pd.concat([tr, va])
    y = te['target_next_day']
    historical = np.repeat(float(combined['target_next_day'].mean()), len(y))
    results.append({'model': 'HistoricalMean', **metrics(y, historical)})
    preds['HistoricalMean'] = historical
    # Optional decomposition comparison: train on same sample and compare regular
    # feature-only models against feature-plus-EMD models on aligned dates.
    configurations = {'Baseline': BASE_COLS}
    if emd != 'none': configurations[f'With_{emd.upper()}'] = cols
    for label, used in configurations.items():
        for name, estimator in models().items():
            estimator.fit(combined[used], combined['target_next_day'])
            predicted = estimator.predict(te[used])
            key = f'{name}_{label}'
            results.append({'model': key, **metrics(y, predicted)})
            preds[key] = predicted
    pd.DataFrame(results).to_csv(output / 'metrics.csv', index=False)
    preds.to_csv(output / 'predictions.csv', index=False)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(te.index, y.to_numpy(), color='black', linewidth=.6, alpha=.65, label='Actual')
    best = min(results, key=lambda z: z['RMSE'])['model']
    ax.plot(te.index, preds[best].to_numpy(), linewidth=.9, label=f'Forecast: {best}')
    ax.set_title('SPY next-day log-return forecasting (test period)')
    ax.set_xlabel('Date'); ax.set_ylabel('Log return'); ax.legend()
    fig.tight_layout(); fig.savefig(output/'test_predictions.png', dpi=160); plt.close(fig)
    price_bytes = pd.util.hash_pandas_object(close, index=True).to_numpy().tobytes()
    fingerprint = hashlib.sha256(price_bytes).hexdigest()
    metadata = {'asset': 'SPY or user-supplied CSV',
                'first_price_date': str(close.index.min().date()),
                'last_price_date': str(close.index.max().date()),
                'price_rows': len(close),
                'price_fingerprint_sha256': fingerprint,
                'last_test_feature_date': str(te.index.max().date()),
                'first_test_feature_date': str(te.index.min().date()),
                'baseline_warmup_rows': emd_window,
                'validation_used_for_final_fit': True, 'splits': {'train_end': train_end,
                'validation_end': valid_end, 'test_start_after': valid_end},
                'emd': emd, 'emd_window': emd_window, 'ceemdan_trials': ceemdan_trials,
                'train_rows': len(tr), 'validation_rows': len(va), 'test_rows': len(te),
                'warning': 'Research prototype; test metrics not a profitability claim.'}
    (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(results).to_string(index=False))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', help='Local Date,Close CSV. If omitted, downloads SPY.')
    parser.add_argument('--start', default='2010-01-01')
    parser.add_argument('--end', default=None, help='Exclusive end date')
    parser.add_argument('--output', default='results/spy')
    parser.add_argument('--train-end', default='2018-12-31')
    parser.add_argument('--valid-end', default='2021-12-31')
    parser.add_argument('--emd', choices=['none', 'emd', 'ceemdan'], default='none')
    parser.add_argument('--emd-window', type=int, default=80)
    parser.add_argument('--ceemdan-trials', type=int, default=12)
    args = parser.parse_args()
    close = load_csv(args.csv) if args.csv else download_spy(args.start, args.end)
    run(close, args.output, args.train_end, args.valid_end,
        args.emd, args.emd_window, args.ceemdan_trials)

if __name__ == '__main__':
    main()
