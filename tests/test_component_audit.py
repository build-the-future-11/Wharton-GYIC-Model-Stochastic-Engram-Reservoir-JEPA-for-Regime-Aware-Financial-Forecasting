"""Synthetic component audit checks; these do not qualify a market experiment."""
import numpy as np
import pytest
from frem.component_audit import windows, components, evaluate, synthetic_series, latent_features

@pytest.mark.parametrize('horizon', [1, 3, 8])
def test_causal_context_and_training_targets(horizon):
    v = np.arange(120.)
    X, y, t, origins, future_context = windows(v, 12, horizon)
    assert np.all(X[:, -1] == origins)
    assert np.all(t-origins == horizon)
    assert np.all(future_context[:, -1] == t)
    result = evaluate(synthetic_series(180), horizon=horizon)
    for fold in result['folds']:
        assert fold['last_train_target'] <= fold['first_forecast_origin']

@pytest.mark.parametrize('bad', [np.array([1, np.nan, 2]), np.ones((5, 2))])
def test_invalid_series(bad):
    with pytest.raises(ValueError):
        windows(bad)

def test_future_mutation_leaves_earlier_fold_unchanged():
    v = synthetic_series(180)
    first = evaluate(v)
    v[140:] += 1e6
    second = evaluate(v)
    for name, model in first['folds'][0]['models'].items():
        assert model['predictions'] == second['folds'][0]['models'][name]['predictions']

def test_target_cannot_enter_its_own_prediction():
    v = synthetic_series(180)
    first = evaluate(v)
    v[100] += 1e6
    second = evaluate(v)
    for name, model in first['folds'][0]['models'].items():
        assert model['predictions'][0] == second['folds'][0]['models'][name]['predictions'][0]

def test_components_are_separate_and_reproducible():
    X = windows(synthetic_series(80))[0]
    a = components(X, 0, 1, seed=7)
    b = components(X, 0, 1, seed=7)
    c = components(X, 0, 1, seed=8)
    for key in a:
        assert np.array_equal(a[key], b[key])
    assert not np.array_equal(a['reservoir'], c['reservoir'])
    for key in ['raw', 'fourier', 'ngram']:
        assert np.array_equal(a[key], c[key])
    assert np.allclose(a['ngram'].sum(axis=1), 1)

def test_thirteen_arms_when_optional_prototype_enabled():
    pytest.importorskip('torch')
    result = evaluate(synthetic_series(180), include_jepa=True, epochs=3)
    for fold in result['folds']:
        assert len(fold['models']) == 13
        assert fold['latent_diagnostics']['prototype'] is True
        assert all(np.isfinite(m['mse']) for m in fold['models'].values())

def test_optional_latent_is_deterministic():
    pytest.importorskip('torch')
    X, _, _, _, Z = windows(synthetic_series(90))
    a = latent_features(X[:50], Z[:50], X[50:], epochs=3)
    b = latent_features(X[:50], Z[:50], X[50:], epochs=3)
    assert np.array_equal(a[0], b[0])
    assert np.isfinite(a[1]).all()
