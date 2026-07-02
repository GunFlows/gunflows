"""Unit tests for gunflows.losses (no ROOT/GUNDAM needed)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gunflows.losses import LOSS_MAP


class _FakeModel:
    """Minimal stand-in exposing what importance_losses._common needs."""

    def __init__(self):
        self.log_norm = torch.tensor(0.0)
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def log_prob(self, zb, ctx):
        return -0.5 * (zb**2).sum(dim=1)


class _FakeDataset:
    """log_prob returns the (data_spline, data_cond, -log_q, -log_p) contract."""

    def log_prob(self, idx):
        n = len(idx)
        zb = torch.randn(n, 2)
        ctx = torch.randn(n, 1)
        log_g = -0.5 * (zb**2).sum(dim=1)
        log_p = log_g + 0.01 * torch.randn(n)
        return zb, ctx, log_g, log_p


@pytest.fixture
def model_and_dataset():
    return _FakeModel(), _FakeDataset()


@pytest.mark.parametrize("name", ["exp_forward", "exp_reverse"])
def test_single_direction_losses_are_finite(model_and_dataset, name):
    model, dataset = model_and_dataset
    loss = LOSS_MAP[name](model, dataset, list(range(32)))
    assert torch.isfinite(loss)


@pytest.mark.parametrize("name", ["exp_symmetric", "kl_symmetric", "absolute_kl_symmetric"])
def test_symmetric_losses_return_extras(model_and_dataset, name):
    model, dataset = model_and_dataset
    loss, extras = LOSS_MAP[name](model, dataset, list(range(32)), 0, return_extra=True)
    assert torch.isfinite(loss)
    assert extras["ess"] > 0
    assert "mean_w_forward" in extras and "mean_w_reverse" in extras


@pytest.mark.parametrize("name", ["exp_forward", "exp_reverse"])
def test_validation_diag_plot_does_not_crash(tmp_path, model_and_dataset, name):
    # Regression test: _diag_plot used to require a positional `stage` that
    # exp_forward/exp_reverse never passed, so validation=True crashed.
    model, dataset = model_and_dataset
    loss = LOSS_MAP[name](model, dataset, list(range(32)), validation=True, save_dir=tmp_path)
    assert torch.isfinite(loss)
    assert any(tmp_path.iterdir())


def test_dead_loss_names_were_removed():
    assert "kl_forward" not in LOSS_MAP
    assert "kl_reverse" not in LOSS_MAP
