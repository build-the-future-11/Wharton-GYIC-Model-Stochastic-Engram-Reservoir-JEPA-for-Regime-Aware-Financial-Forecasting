import csv
from pathlib import Path
import numpy as np
import pytest
from frem.benchmark import make_windows, expanding_folds, read_csv, FourierReservoirFeatures, evaluate


def test_windows_are_strictly_causal_even_multi_horizon():
    v = np.arange(40, dtype=float)
    w = make_windows(v, context=5, horizon=3)
    assert w.target_indices[0] == 7
    assert np.array_equal(w.features[0], np.arange(5))
    assert w.latest_observed_indices[0] == 4
    assert np.all(w.latest_observed_indices < w.target_indices)


def test_reject_invalid_series_and_context():
    with pytest.raises(ValueError): make_windows(np.ones((10, 2)), 3)
    with pytest.raises(ValueError): make_windows(np.array([1., float('nan'), 3.]), 2)
    with pytest.raises(ValueError): make_windows(np.arange(20.), 1)


def test_ascending_data_required(tmp_path):
    p = tmp_path / 'data.csv'
    p.write_text('timestamp,value\n2026-01-01,1\n2026-01-02,2\n2026-01-02,3\n')
    with pytest.raises(ValueError, match='strictly increasing'):
        read_csv(p, 'timestamp', 'value')
    p.write_text('timestamp,value\n2026-01-01,1\n2026-01-02,2\n')
    data, digest = read_csv(p, 'timestamp', 'value')
    assert data.tolist() == [1., 2.]
    assert len(digest) == 64


def test_folds_never_overlap_targets():
    f = list(expanding_folds(130, min_train=60, test_size=20, gap=3))
    assert f == [(60, 63, 83), (80, 83, 103), (100, 103, 123)]
    assert all(train_end <= test_start for train_end,test_start,_ in f)


def test_engineered_features_are_deterministic_finite():
    x = np.arange(32.,dtype=float).reshape(4,8)
    a = FourierReservoirFeatures(state_dim=5,seed=17).transform(x, 7.5, 3.0)
    b = FourierReservoirFeatures(state_dim=5,seed=17).transform(x, 7.5, 3.0)
    assert a.shape == (4, 3+4+9+5)
    assert np.array_equal(a,b)
    assert np.all(np.isfinite(a))


def test_expanding_baselines_record_all_folds_and_are_deterministic():
    v = np.sin(np.arange(120)*.2) + np.arange(120)*.005
    a = evaluate(v,context=8,min_train=50,test_size=20,reservoir_dim=6,seed=19)
    b = evaluate(v,context=8,min_train=50,test_size=20,reservoir_dim=6,seed=19)
    assert a == b
    assert len(a['folds']) == 3
    assert set(a['folds'][0]['models']) == {'historical_mean','persistence','ridge_context','fourier_reservoir_ridge'}
    assert all(np.isfinite(m['mse']) and np.isfinite(m['mae'])
               for fold in a['folds'] for m in fold['models'].values())


def test_future_change_does_not_change_earlier_fold_predictions():
    v = np.sin(np.arange(140)*.2)
    a = evaluate(v,context=8,min_train=50,test_size=20,reservoir_dim=6)
    tampered = v.copy()
    tampered[110:] = 1000.0
    b = evaluate(tampered,context=8,min_train=50,test_size=20,reservoir_dim=6)
    assert a['folds'][0]['models'] == b['folds'][0]['models']
    assert a['folds'][1]['models'] == b['folds'][1]['models']


def test_too_short_fold_or_bad_alpha_rejected():
    with pytest.raises(ValueError,match='no valid chronological'):
        evaluate(np.arange(35.),context=8,min_train=30,test_size=20)
    with pytest.raises(ValueError,match='Ridge alpha'):
        evaluate(np.arange(90.),alpha=0)
