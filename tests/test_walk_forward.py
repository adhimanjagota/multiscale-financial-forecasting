import json
import numpy as np
import pandas as pd
import pytest
from src.experiment import make_features
from src.walk_forward import build_folds, run_walk_forward


def sample_prices():
    rng = np.random.default_rng(99)
    idx = pd.bdate_range('2018-01-01', '2024-04-30')
    ret = rng.normal(.00015, .008, len(idx))
    return pd.Series(100 * np.exp(np.cumsum(ret)), index=idx, name='Close')


def test_train_targets_precede_test_features():
    prices = sample_prices()
    features = make_features(prices)
    folds = list(build_folds(prices, features, first_test_year=2022))
    assert [x[0] for x in folds] == [2022, 2023, 2024]
    for year, tr, te in folds:
        observed_close = prices.index[prices.index.get_indexer(tr) + 1]
        assert observed_close.max() < te.min()
        assert tr.intersection(te).empty
        assert (te.year == year).all()


def test_training_label_does_not_cross_new_year():
    prices = sample_prices()
    features = make_features(prices)
    year, train, test = next(iter(build_folds(prices, features, first_test_year=2022)))
    next_close = prices.index[prices.index.get_indexer(train) + 1]
    assert (next_close < test.min()).all()


def test_walk_forward_outputs_are_consistent(tmp_path):
    prices = sample_prices()
    overall, folds, preds = run_walk_forward(prices, tmp_path, first_test_year=2022)
    assert preds.date.is_unique
    assert len(preds) == sum(folds.loc[folds.model == 'HistoricalMean', 'N'])
    assert (folds.fold_year.unique() == [2022, 2023, 2024]).all()
    for model in overall.model:
        row = overall.set_index('model').loc[model]
        assert row.N == len(preds)
        np.testing.assert_allclose(row.RMSE, np.sqrt(np.mean((preds[model] - preds.actual_next_day_log_return) ** 2)))
    meta = json.loads((tmp_path/'run_metadata.json').read_text())
    assert meta['price_rows'] == len(prices)
    assert len(meta['folds']) == 3


def test_mode_alignment(tmp_path, monkeypatch):
    import src.walk_forward as wf
    from src.experiment import EMD_COLS
    prices = sample_prices()
    def fake_emd(close, index, window, trials):
        return pd.DataFrame({col: np.zeros(len(index)) for col in EMD_COLS}, index=index)
    monkeypatch.setattr(wf, 'rolling_emd_features', fake_emd)
    a, _, p_a = run_walk_forward(prices, tmp_path/'a', first_test_year=2024)
    b, _, p_b = run_walk_forward(prices, tmp_path/'b', first_test_year=2024, emd='emd')
    for name in ('HistoricalMean', 'Ridge_Baseline', 'GradientBoosting_Baseline'):
        np.testing.assert_array_equal(p_a[name], p_b[name])
    np.testing.assert_array_equal(p_a.actual_next_day_log_return, p_b.actual_next_day_log_return)
