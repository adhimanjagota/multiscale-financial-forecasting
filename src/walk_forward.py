"""Expanding-window annual walk-forward forecast evaluation (research only).

Run: python -m src.walk_forward --csv data/spy_adjusted_frozen.csv --output results/walk_forward_baseline
The feature date t uses prices through t and forecasts the next available price
observation. A fold's training targets must be observed before its first test
feature date. Models are re-fitted at the beginning of each calendar test year.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import (BASE_COLS, EMD_COLS, load_csv, make_features, metrics,
                         models, rolling_emd_features, download_spy)


def build_folds(close, features, first_test_year=2022, min_train=252):
    """Yield (year, train index, test index) without crossing training targets.

    A feature at t predicts the return from t to the NEXT observed close.
    Its target becomes known on close.index[position(t) + 1]. Training targets
    must be strictly earlier than the first test feature date for a fold.
    """
    close_dates = pd.DatetimeIndex(close.index)
    if close_dates.has_duplicates or not close_dates.is_monotonic_increasing:
        raise ValueError('Price dates must be unique and increasing')
    if not features.index.isin(close_dates).all():
        raise ValueError('Feature dates must belong to close prices')
    if first_test_year < close_dates.min().year:
        raise ValueError('First test year precedes the price series')
    target_positions = close_dates.get_indexer(features.index) + 1
    if (target_positions >= len(close_dates)).any():
        raise ValueError('Features contain target beyond last available close')
    target_dates = pd.Series(close_dates[target_positions], index=features.index)
    for year in range(first_test_year, close_dates.max().year + 1):
        first_feature = pd.Timestamp(year, 1, 1)
        test_index = features.index[(features.index >= first_feature) & (features.index.year == year)]
        if len(test_index) == 0:
            continue
        first_actual_feature = test_index[0]
        train_index = features.index[target_dates < first_actual_feature]
        if len(train_index) < min_train:
            raise ValueError(f'Fold {year} needs at least {min_train} training samples, got {len(train_index)}')
        assert target_dates.loc[train_index].max() < first_actual_feature
        assert train_index.intersection(test_index).empty
        yield year, train_index, test_index


def run_walk_forward(close, output, first_test_year=2022, emd='none', emd_window=80,
                     ceemdan_trials=12, min_train=252):
    if emd not in ('none', 'emd', 'ceemdan'):
        raise ValueError('emd must be none, emd or ceemdan')
    if emd_window < 20 or min_train < 2 or (emd == 'ceemdan' and ceemdan_trials < 1):
        raise ValueError('Invalid window, minimum training rows or CEEMDAN trials')
    close = close.astype(float).sort_index()
    if close.index.has_duplicates or (not np.isfinite(close.to_numpy()).all()) or (close <= 0).any():
        raise ValueError('Close prices must be unique by date, finite and positive')
    features = make_features(close)
    features = features.loc[features.index >= close.index[emd_window - 1]].copy()
    if emd != 'none':
        extra = rolling_emd_features(close, features.index, window=emd_window,
                                      trials=ceemdan_trials if emd == 'ceemdan' else 0)
        features = features.join(extra).dropna()
    # All modes should have identical feature dates; decomposition must produce finite columns.
    columns = {'HistoricalMean': [], 'Ridge_Baseline': BASE_COLS,
               'GradientBoosting_Baseline': BASE_COLS}
    if emd != 'none':
        suffix = emd.upper()
        columns[f'Ridge_With_{suffix}'] = BASE_COLS + EMD_COLS
        columns[f'GradientBoosting_With_{suffix}'] = BASE_COLS + EMD_COLS
    prediction_parts = []
    fold_records = []
    fold_descriptions = []
    for year, train_idx, test_idx in build_folds(close, features, first_test_year, min_train):
        train, test = features.loc[train_idx], features.loc[test_idx]
        pred = pd.DataFrame({'date': test.index, 'fold_year': year,
                             'actual_next_day_log_return': test.target_next_day.to_numpy()})
        for key, used in columns.items():
            if key == 'HistoricalMean':
                yhat = np.full(len(test), float(train.target_next_day.mean()))
            else:
                model_name = 'Ridge' if key.startswith('Ridge') else 'GradientBoosting'
                estimator = models()[model_name]
                estimator.fit(train[used], train.target_next_day)
                yhat = estimator.predict(test[used])
            pred[key] = yhat
            fold_records.append({'fold_year': year, 'model': key,
                                 **metrics(test.target_next_day, yhat)})
        fold_descriptions.append({'year': year, 'train_rows': len(train), 'test_rows': len(test),
                                  'last_training_feature_date': str(train.index.max().date()),
                                  'first_test_feature_date': str(test.index.min().date()),
                                  'last_test_feature_date': str(test.index.max().date())})
        prediction_parts.append(pred)
    if not prediction_parts:
        raise ValueError('No test folds created. Check --first-test-year')
    predictions = pd.concat(prediction_parts, ignore_index=True)
    if predictions.date.duplicated().any():
        raise AssertionError('Overlapping test folds')
    per_fold = pd.DataFrame(fold_records)
    overall = pd.DataFrame([
        {'model': key, **metrics(predictions.actual_next_day_log_return, predictions[key])}
        for key in columns
    ]).sort_values('RMSE')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(output / 'fold_metrics.csv', index=False)
    overall.to_csv(output / 'overall_metrics.csv', index=False)
    predictions.to_csv(output / 'predictions.csv', index=False, date_format='%Y-%m-%d')
    price_bytes = pd.util.hash_pandas_object(close, index=True).to_numpy().tobytes()
    metadata = {'asset': 'SPY or user-supplied adjusted-close series',
                'price_fingerprint_sha256': hashlib.sha256(price_bytes).hexdigest(),
                'price_rows': len(close), 'first_price_date': str(close.index.min().date()),
                'last_price_date': str(close.index.max().date()),
                'first_test_year': first_test_year, 'method': 'expanding training, annual retraining',
                'training_target_rule': 'next observed close must predate first test feature date',
                'emd': emd, 'emd_window': emd_window, 'ceemdan_trials': ceemdan_trials,
                'folds': fold_descriptions,
                'limitations': 'No independent tuning/validation, no financial backtest, no uncertainty test.'}
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    print('\nAnnual walk-forward fold metrics:')
    print(per_fold.to_string(index=False))
    print('\nAggregate out-of-sample metrics:')
    print(overall.to_string(index=False))
    return overall, per_fold, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', help='Frozen local Date,Close adjusted-price CSV')
    parser.add_argument('--start', default='2010-01-01')
    parser.add_argument('--end', default=None)
    parser.add_argument('--output', default='results/walk_forward_baseline')
    parser.add_argument('--first-test-year', type=int, default=2022)
    parser.add_argument('--emd', choices=['none', 'emd', 'ceemdan'], default='none')
    parser.add_argument('--emd-window', type=int, default=80)
    parser.add_argument('--ceemdan-trials', type=int, default=12)
    parser.add_argument('--min-train', type=int, default=252)
    args = parser.parse_args()
    close = load_csv(args.csv) if args.csv else download_spy(args.start, args.end)
    run_walk_forward(close, args.output, args.first_test_year, args.emd,
                     args.emd_window, args.ceemdan_trials, args.min_train)

if __name__ == '__main__':
    main()
