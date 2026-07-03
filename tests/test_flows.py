"""Unit tests for the GUNDAM-independent flow components (no ROOT/GUNDAM needed)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from gunflows.flows.cov_flow import CovFlow
from gunflows.utils.build_flow import build_base, build_flow_layers, build_model


def _fake_target(ndim=4, phase_space_dim=(0, 1)):
    cond = [i for i in range(ndim) if i not in phase_space_dim]
    cov = torch.eye(ndim) + 0.1 * torch.rand(ndim, ndim).tril()
    cov = cov @ cov.T  # SPD
    return SimpleNamespace(
        cholesky=torch.linalg.cholesky(cov),
        list_dim_conditionnal=cond,
        phase_space_dim=list(phase_space_dim),
    )


def test_cov_flow_round_trip():
    target = _fake_target()
    flow = CovFlow(target, device="cpu")
    x = torch.randn(16, 4)

    z, ctx, log_det = flow(x)
    x_rec, log_det_inv = flow.inverse(z, ctx)

    assert torch.allclose(x, x_rec, atol=1e-4)
    assert torch.allclose(log_det, -log_det_inv, atol=1e-4)


def test_systematic_flow_sample_and_log_prob_are_finite():
    target = _fake_target()
    base = build_base(target.cholesky.shape[0])
    flows = build_flow_layers(
        nflows=2,
        dim_spline=len(target.phase_space_dim),
        hidden=8,
        nlayers=1,
        nbins=4,
        tail_bounds=torch.ones(len(target.phase_space_dim)) * 3.0,
        n_context=len(target.list_dim_conditionnal),
    )
    model = build_model(base, flows, target, context_transform=False)

    x, log_q_sample = model.sample(num_samples=32)
    assert x.shape == (32, target.cholesky.shape[0])
    assert torch.isfinite(log_q_sample).all()

    log_q = model.log_prob(x[:, target.phase_space_dim], x[:, target.list_dim_conditionnal])
    assert torch.isfinite(log_q).all()
