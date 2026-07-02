"""Unit tests for SystematicDatasetFile / BaseSystematicDataset (no ROOT/GUNDAM needed)."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from gunflows.dataset.systematic_dataset_file import SystematicDatasetFile


@pytest.fixture
def npz_batch(tmp_path):
    rng = np.random.default_rng(0)
    ndim, nsample = 4, 200
    cov = np.eye(ndim, dtype=np.float32)
    mean = np.zeros(ndim, dtype=np.float32)
    data = rng.multivariate_normal(mean, cov, size=nsample).astype(np.float32)
    log_p = -0.5 * (data**2).sum(axis=1)
    log_q = log_p + 0.01 * rng.standard_normal(nsample).astype(np.float32)
    np.savez(
        tmp_path / "batch0.npz",
        data=data,
        log_p=log_p.astype(np.float32),
        log_q=log_q.astype(np.float32),
        cov=cov,
        mean=mean,
        par_names=np.array([f"p{i}" for i in range(ndim)]),
        bestfit_nll=np.array(0.0, dtype=np.float32),
    )
    return tmp_path


def test_load_and_len(npz_batch):
    ds = SystematicDatasetFile(str(npz_batch), phase_space_dim=[0, 1], plot_grid=False)
    assert len(ds) == 200


def test_log_prob_contract(npz_batch):
    ds = SystematicDatasetFile(str(npz_batch), phase_space_dim=[0, 1], plot_grid=False)
    idx = list(range(8))
    data_spline, data_cond, neg_log_q, neg_log_p = ds.log_prob(idx)
    assert data_spline.shape == (8, 2)
    assert data_cond.shape == (8, 2)
    assert neg_log_q.shape == (8,)
    assert neg_log_p.shape == (8,)
    # __getitem__ (inherited from BaseSystematicDataset) must delegate to log_prob
    assert ds[idx][0].shape == data_spline.shape


def test_base_dataset_getters(npz_batch):
    ds = SystematicDatasetFile(str(npz_batch), phase_space_dim=[0, 1], plot_grid=False)
    assert ds.get_cov().shape == (4, 4)
    assert torch.equal(ds.get_true_cov(), ds.get_cov())
    assert ds.get_mean().shape == (4,)

    x = torch.randn(5, 4)
    x_data_space = ds.transform_eigen_space_to_data_space(x)
    assert torch.allclose(x_data_space, x * ds.std_per_dim + ds.mean)
