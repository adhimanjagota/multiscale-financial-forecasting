import numpy as np
import pandas as pd
import pytest
from src.experiment import make_features, split_by_date, metrics, run


def dummy_price():
    rng = np.random.default_rng(12)
    dates = pd.bdate_range('2016-01-01', '2024-06-30')
    returns = rng.normal(.0002, .01, len(dates))
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=dates, name='Close')


def test_no_future_information_in_features():
    close = dummy_price()
    earlier = make_features(close)
    edited = close.copy()
    edited.iloc[-100:] *= 1.6
    updated = make_features(edited)
    cutoff = close.index[-110]
    pd.testing.assert_frame_equal(earlier.loc[:cutoff], updated.loc[:cutoff])


def test_label_is_next_day():
    close = dummy_price()
    frame = make_features(close)
    dt = frame.index[60]
    position = close.index.get_loc(dt)
    expected = np.log(close.iloc[position+1]/close.iloc[position])
    assert np.isclose(frame.loc[dt, 'target_next_day'], expected)


def test_split_is_disjoint():
    tr, va, te = split_by_date(make_features(dummy_price()), '2018-12-31', '2021-12-31')
    assert tr.index.max() < va.index.min() < te.index.min()
    assert tr.index.max() < pd.Timestamp('2018-12-31')


def test_metrics():
    out = metrics([.01, -.02], [.015, -.01])
    assert out['N'] == 2 and out['DirectionalAccuracy'] == 1


def test_end_to_end(tmp_path):
    res = run(dummy_price(), tmp_path)
    assert len(res) == 3
    for f in ['metrics.csv', 'predictions.csv', 'test_predictions.png', 'run_metadata.json']:
        assert (tmp_path/f).exists()


def test_missing_or_bad_prices_rejected():
    x = dummy_price()
    x.iloc[7] = -1
    with pytest.raises(ValueError): make_features(x)


def test_baselines_match_across_modes(tmp_path, monkeypatch):
    import src.experiment as exp
    close = dummy_price()
    def fake_decomposition(close, index, window=80, trials=0, seed=42):
        # Simulate valid trailing features; compare baseline sample alignment.
        positions = close.index.get_indexer(index)
        values = np.where(positions >= window - 1, 0.0, np.nan)
        return pd.DataFrame({c: values for c in exp.EMD_COLS}, index=index)
    monkeypatch.setattr(exp, 'rolling_emd_features', fake_decomposition)
    a = exp.run(close, tmp_path / 'none')
    b = exp.run(close, tmp_path / 'emd', emd='emd')
    assert a[0] == b[0]
    for x, y in zip(a[1:], b[1:3]):
        assert x == y
    assert 'Ridge_With_EMD' in [r['model'] for r in b]
    meta = __import__('json').loads((tmp_path/'emd'/'run_metadata.json').read_text())
    assert meta['baseline_warmup_rows'] == 80
