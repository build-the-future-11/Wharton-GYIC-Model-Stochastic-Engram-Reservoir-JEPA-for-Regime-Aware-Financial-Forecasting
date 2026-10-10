"""Regression checks for component fitting's process-wide side effects."""
import numpy as np
import pytest

from frem.component_audit import latent_features, synthetic_series, windows

torch = pytest.importorskip('torch')


@pytest.fixture
def latent_inputs():
    x, _, _, _, y = windows(synthetic_series(90))
    return x[:50], y[:50], x[50:]


@pytest.fixture(autouse=True)
def preserve_test_process():
    """Keep regression tests from changing subsequent test cases."""
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        yield
    finally:
        torch.set_rng_state(rng)
        torch.set_num_threads(threads)


def test_fit_leaves_caller_random_stream_and_threads_unchanged(latent_inputs):
    torch.manual_seed(910)
    before = torch.get_rng_state().clone()
    expected_next = torch.rand(12)
    torch.set_rng_state(before)

    result = latent_features(*latent_inputs, seed=7, epochs=2)

    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(torch.rand(12), expected_next)
    assert torch.get_num_threads() == 1
    assert np.isfinite(result[0]).all()
    assert np.isfinite(result[1]).all()


def test_fit_preserves_state_after_optimizer_failure(latent_inputs, monkeypatch):
    torch.manual_seed(911)
    before = torch.get_rng_state().clone()

    def fail_after_random_draw(self, *args, **kwargs):
        torch.rand(8)
        raise RuntimeError('injected optimizer failure')

    monkeypatch.setattr(torch.optim.AdamW, 'step', fail_after_random_draw)
    with pytest.raises(RuntimeError, match='injected optimizer failure'):
        latent_features(*latent_inputs, seed=7, epochs=2)

    assert torch.equal(torch.get_rng_state(), before)
    assert torch.get_num_threads() == 1


def test_cpu_fit_does_not_call_accelerator_seeders(latent_inputs, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU prototype reseeded an accelerator')

    monkeypatch.setattr(torch.cuda, 'manual_seed_all', forbidden)
    latent_features(*latent_inputs, seed=7, epochs=1)


def test_fit_seed_controls_features_without_caller_seed_dependency(latent_inputs):
    torch.manual_seed(912)
    a = latent_features(*latent_inputs, seed=7, epochs=2)
    torch.manual_seed(913)
    b = latent_features(*latent_inputs, seed=7, epochs=2)
    c = latent_features(*latent_inputs, seed=8, epochs=2)

    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] == b[2]
    assert not np.array_equal(a[0], c[0])


def test_cpu_fit_overrides_and_restores_caller_default_device(latent_inputs):
    # meta exercises device routing on a CPU host without requiring or touching
    # an accelerator. Missing CPU pinning fails when the feature array is read.
    before = torch.get_rng_state().clone()
    with torch.device('meta'):
        result = latent_features(*latent_inputs, seed=7, epochs=1)
        assert torch.empty(0).device.type == 'meta'
    assert torch.equal(torch.get_rng_state(), before)
    assert np.isfinite(result[0]).all()
    assert np.isfinite(result[1]).all()
