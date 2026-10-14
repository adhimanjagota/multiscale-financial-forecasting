"""Regression and validation tests for paired forecast-loss analysis."""
import numpy as np
import pandas as pd
import pytest

from src.significance import analyze as benchmark_analyze
from src.decomposition_significance import analyze as decomposition_analyze, PAIRS


def _fixture(n=30):
    rng = np.random.default_rng(77)
    actual = rng.normal(0, 0.01, n)
    data = {
        "date": pd.date_range("2024-01-01", periods=n, freq="B").astype(str),
        "fold_year": [2024] * n,
        "actual_next_day_log_return": actual,
        "HistoricalMean": np.full(n, 0.0001),
    }
    for i, name in enumerate(sorted(set(x for pair in PAIRS for x in pair))):
        data[name] = actual + rng.normal(0, 0.01 + i*0.0001, n)
    return pd.DataFrame(data)


def test_benchmark_produces_six_valid_comparisons():
    r = benchmark_analyze(_fixture(), maxlags=2)
    assert len(r) == 6
    assert ((r.holm_adjusted_p >= 0) & (r.holm_adjusted_p <= 1)).all()
    assert r.N.eq(30).all()


def test_direct_produces_four_valid_comparisons():
    r = decomposition_analyze(_fixture(), maxlags=2)
    assert len(r) == 4
    assert ((r.holm_adjusted_p >= 0) & (r.holm_adjusted_p <= 1)).all()


def test_sign_convention_matches_squared_error():
    d = _fixture()
    r = decomposition_analyze(d, maxlags=0)
    for row in r.itertuples():
        y = d.actual_next_day_log_return.to_numpy()
        expected = np.mean((y - d[row.decomposition_model].to_numpy())**2
                           - (y - d[row.baseline].to_numpy())**2)
        assert np.isclose(row.mean_loss_difference, expected)


def test_invalid_and_duplicate_data_rejected():
    d = _fixture()
    d.loc[1, 'date'] = d.loc[0, 'date']
    with pytest.raises(ValueError, match='Duplicate'):
        decomposition_analyze(d)
    with pytest.raises(ValueError, match='Duplicate'):
        benchmark_analyze(d)


def test_nonfinite_direct_data_rejected():
    d = _fixture()
    d.loc[0, 'Ridge_With_EMD'] = np.nan
    with pytest.raises(ValueError, match='Non-finite'):
        decomposition_analyze(d)


def test_invalid_hac_lags_rejected():
    d = _fixture()
    with pytest.raises(ValueError, match='Invalid'):
        decomposition_analyze(d, maxlags=len(d))
