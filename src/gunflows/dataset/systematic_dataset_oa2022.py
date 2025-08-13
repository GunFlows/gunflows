#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: SystematicDatasetStream Class
Author: Mathias El Baz
Date: 28/01/2025
Description:
  Dataset class that can start from nothing and stream lots from a background sampler
  (covariance-first, then NF when available), or fall back to reading batch*.npz files.
  Standardization is done with (x - mean) / sqrt(diag(cov)); log_p/log_q are aligned
  with the same shifts you use in the file-based dataset.
"""

import glob
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from torch.utils.data import Dataset
from pathlib import Path
import multiprocessing as mp, queue as _q

from gunflows.likelihood_sampler.nf_llh_sampler import NFSamplerProcess

__all__ = ["SystematicDatasetOA2022"]


def _plot_grid(samples, mean, weights, cov, names, n, out_dir, phase_dims, reweight=False):
    samples = samples[:, phase_dims]
    if not reweight:
        weights = np.ones(samples.shape[0]) / samples.shape[0]
    mean = mean[phase_dims]
    cov = cov[np.ix_(phase_dims, phase_dims)]
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                x = samples[:, i]
                ax.hist(x, bins=60, weights=weights, density=True, histtype="step")
                xs = np.linspace(x.min(), x.max(), 200)
                mean_i = 0
                std_i = 1
                pdf = (1 / (np.sqrt(2 * np.pi * std_i**2))) * np.exp(
                    -0.5 * ((xs - mean_i) ** 2) / std_i**2
                )
                ax.plot(xs, pdf, color="r")
            else:
                ax.hist2d(samples[:, j], samples[:, i], weights=weights, bins=60, norm=LogNorm())
            if i == n - 1:
                ax.set_xlabel(names[j], fontsize=7)
            if j == 0:
                ax.set_ylabel(names[i], fontsize=7)
            ax.tick_params(axis="both", labelsize=6)
    plt.tight_layout()
    if reweight:
        plt.suptitle("Reweighted Grid", fontsize=10)
        out_dir = out_dir / "reweighted"
    else:
        plt.suptitle("Grid", fontsize=10)
        out_dir = out_dir / "grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(top=0.9)
    plt.savefig(out_dir / "grid_reweighted.png" if reweight else out_dir / "grid.png", dpi=150)
    plt.close(fig)


class SystematicDatasetOA2022(Dataset):
    def __init__(
        self,
        phase_space_dim: list[int],
        starting_folder: str | None = None,
        batch_index: int | None = None,
        with_sampler: bool = False,
        nf_ckpt: str | None = None,
        gen_batch_size: int = 1000000,
        llh_config: str | None = None,
        llh_overrides: list[str] | None = None,
        llh_cwd: str | None = None,
        seed: int = 42,
        queue_size: int = 2,
        plot_grid: bool = False,
        out_dir: str | os.PathLike = "plots",
        load_data: bool = True,
        timeout: float = 6000.0,
        shift_log_p: bool = True,
    ):
        super().__init__()
        self.phase_space_dim = phase_space_dim
        self.out_dir = Path(out_dir)
        self._gen_proc: mp.Process | None = None
        self._data_q: mp.Queue | None = None
        self._cmd_q: mp.Queue | None = None
        self._stop_evt: mp.Event | None = None
        self._last_payload: dict | None = None
        self.timeout = timeout
        self.shift_log_p = shift_log_p

        if starting_folder:
            self._load_from_files(starting_folder, batch_index, plot_grid, load_data)
        elif with_sampler:
            if llh_config is None:
                raise ValueError("llh_config must be provided when with_sampler=True")
            self._start_sampler(
                nf_ckpt=nf_ckpt,
                n_points=gen_batch_size,
                llh_config=llh_config,
                llh_overrides=llh_overrides or [],
                llh_cwd=llh_cwd,
                seed=seed,
                queue_size=queue_size,
            )
            self._wait_and_swap(plot_grid=plot_grid, timeout=self.timeout)
        else:
            raise ValueError("Either starting_folder must be set or with_sampler=True")

    def __len__(self) -> int:
        return self.nsample

    def __getitem__(self, idx: int):
        return (
            self.data_spline[idx, :],
            self.data_cond[idx, :],
            -self.log_q[idx],
            -self.log_p[idx],
        )

    def log_prob(self, idx):
        return (
            self.data_spline[idx, :],
            self.data_cond[idx, :],
            -self.log_q[idx],
            -self.log_p[idx],
        )

    def get_cov(self) -> torch.Tensor:
        return self.cov

    def get_true_cov(self) -> torch.Tensor:
        return self.true_cov

    def get_cov_sub(self, dims: list[int]) -> torch.Tensor:
        return self.cov[np.ix_(dims, dims)]

    def get_mean(self) -> torch.Tensor:
        return self.mean

    def transform_eigen_space_to_data_space(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std_per_dim + self.mean

    def refresh_if_ready(self, plot_grid: bool = False) -> bool:
        if self._data_q is None:
            return False
        try:
            payload = self._data_q.get_nowait()
        except _q.Empty:
            return False
        self._last_payload = payload
        self._finalize_from_dict(payload, plot_grid=plot_grid)
        return True

    def request_switch_to_nf(self, nf_ckpt: str) -> None:
        if self._cmd_q is not None:
            self._cmd_q.put(f"reload:{nf_ckpt}")

    def close(self) -> None:
        if self._stop_evt is not None:
            self._stop_evt.set()
        if self._gen_proc is not None:
            self._gen_proc.join(timeout=5.0)
            if self._gen_proc.is_alive():
                self._gen_proc.terminate()
        self._gen_proc = None
        self._data_q = None
        self._cmd_q = None
        self._stop_evt = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _load_from_files(self, data_dir: str, batch_index: int | None, plot_grid: bool, load_data: bool):
        if batch_index is None:
            file_list = sorted(glob.glob(os.path.join(data_dir, "batch*.npz")))
            if not file_list:
                raise FileNotFoundError(f"No files found in {data_dir} matching 'batch*.npz'")
        else:
            p = os.path.join(data_dir, f"batch{batch_index}.npz")
            if not os.path.isfile(p):
                raise FileNotFoundError(f"File {p} does not exist.")
            file_list = [p]

        data_list, log_p_list, log_q_list = [], [], []
        bestfit_nll = 0.0
        for path in file_list:
            ld = np.load(path, allow_pickle=True)
            data_list.append(torch.tensor(ld["data"], dtype=torch.float32))
            log_p_list.append(torch.tensor(ld["log_p"], dtype=torch.float32))
            log_q_list.append(torch.tensor(ld["log_q"], dtype=torch.float32))
            cov_ref = torch.tensor(ld["cov"], dtype=torch.float32)
            mean_ref = torch.tensor(ld["mean"], dtype=torch.float32)
            titles_ref = ld["par_names"]
            if "bestfit_nll" in ld:
                bestfit_nll = float(ld["bestfit_nll"].item() if hasattr(ld["bestfit_nll"], "item") else ld["bestfit_nll"])

        data_phys = torch.cat(data_list, dim=0)
        log_p_phys = torch.cat(log_p_list, dim=0)
        log_q_phys = torch.cat(log_q_list, dim=0)
        payload = dict(data=data_phys.numpy(), log_p=log_p_phys.numpy(), log_q=log_q_phys.numpy(),
                       cov=cov_ref.numpy(), mean=mean_ref.numpy(), par_names=titles_ref, bestfit_nll=bestfit_nll)
        self._finalize_from_dict(payload, plot_grid=plot_grid)

    def _start_sampler(self, nf_ckpt, n_points, llh_config, llh_overrides, llh_cwd, seed, queue_size):
        self._data_q = mp.Queue(maxsize=queue_size)
        self._cmd_q = mp.Queue()
        self._stop_evt = mp.Event()
        self._gen_proc = NFSamplerProcess(
            nf_ckpt=nf_ckpt,
            n_points=n_points,
            llh_config=llh_config,
            llh_overrides=llh_overrides,
            phase_space_dim=self.phase_space_dim,
            data_q=self._data_q,
            cmd_q=self._cmd_q,
            stop_evt=self._stop_evt,
            seed=seed,
            llh_cwd=llh_cwd,
        )
        self._gen_proc.start()

    def _wait_and_swap(self, timeout: float = 600.0, plot_grid: bool = False):
        assert self._data_q is not None and self._gen_proc is not None
        try:
            payload = self._data_q.get(timeout=timeout)
        except Exception as e:
            if not self._gen_proc.is_alive():
                raise RuntimeError(f"Sampler exited with code {self._gen_proc.exitcode}") from e
            raise
        self._last_payload = payload
        self._finalize_from_dict(payload, plot_grid=plot_grid)

    def _finalize_from_dict(self, d: dict, plot_grid: bool = False):
        data_phys = torch.tensor(np.asarray(d["data"], dtype=np.float32))
        log_p_phys = torch.tensor(np.asarray(d["log_p"], dtype=np.float32))
        cov_ref = torch.tensor(np.asarray(d["cov"], dtype=np.float32))
        mean_ref = torch.tensor(np.asarray(d["mean"], dtype=np.float32))
        titles_ref = d.get("par_names", getattr(self, "titles", None))
        bestfit_nll = float(d.get("bestfit_nll", 0.0))

        if "log_q" in d and d["log_q"] is not None:
            log_q_phys = torch.tensor(np.asarray(d["log_q"], dtype=np.float32))
        else:
            L = torch.linalg.cholesky(cov_ref + 1e-6 * torch.eye(cov_ref.shape[0], dtype=cov_ref.dtype))
            diff = data_phys - mean_ref
            y = torch.linalg.solve_triangular(L, diff.T, upper=False)
            quad = (y * y).sum(dim=0)
            log_det_cov = 2.0 * torch.log(torch.diag(L)).sum()
            const = data_phys.shape[1] * np.log(2.0 * np.pi)
            log_q_phys = -0.5 * (quad + log_det_cov + const)

        std_per_dim = torch.sqrt(torch.diag(cov_ref))
        data = (data_phys - mean_ref) / std_per_dim
        d_inv = torch.diag(1.0 / std_per_dim)
        cov = d_inv @ cov_ref @ d_inv
        log_det_D = torch.logdet(cov)
        chol = torch.linalg.cholesky(cov)

        self.data = data
        self.log_p = log_p_phys - bestfit_nll
        print(f" Mean log_p : {self.log_p.mean().item()}")
        print()
        self.log_q = log_q_phys + 0.5 * log_det_D
        print(f" Mean log_q : {self.log_q.mean().item()}")
        self.log_p = self.log_p + torch.median(self.log_q - self.log_p)
        print(f" Mean log_p after shift: {self.log_p.mean().item()}")
        dtmp = torch.exp(self.log_q - self.log_p)
        dtmp = dtmp[dtmp <= torch.quantile(dtmp, 0.999)].clamp_min(1e-40)
        logshift = torch.log(dtmp.mean())
        print(f"Log shift applied: {dtmp.mean().item()}")
        print(f"Log shift applied: {logshift.item()}")
        if self.shift_log_p:
            self.log_p += logshift
        print(f" Mean log_p after final shift: {self.log_p.mean().item()}")

        self.cov = cov
        self.true_cov = cov
        self.mean = mean_ref
        self.titles = titles_ref
        self.cholesky = chol
        self.std_per_dim = std_per_dim

        self.nsample, self.ndim = data.shape
        self.list_dim_conditionnal = [i for i in range(self.ndim) if i not in self.phase_space_dim]
        self.data_cond = self.data[:, self.list_dim_conditionnal]
        self.data_spline = self.data[:, self.phase_space_dim]

        print(f"Dataset loaded with {self.nsample} samples and {self.ndim} dimensions.")

        if plot_grid:
            logw = (self.log_q - self.log_p).reshape(-1)
            q = torch.quantile(logw, 0.999)
            mask = logw <= q
            logw = logw[mask]
            logw = logw - torch.logsumexp(logw, dim=0)
            samples_np = self.data[mask, :].detach().cpu().numpy()
            weights_np = torch.exp(logw).detach().cpu().numpy()
            self.out_dir.mkdir(parents=True, exist_ok=True)
            n = 20
            mean_np = self.mean.detach().cpu().numpy()
            cov_np = self.cov.detach().cpu().numpy()
            _plot_grid(samples_np, mean_np, weights_np, cov_np, self.titles, n, self.out_dir, self.phase_space_dim)
            _plot_grid(samples_np, mean_np, weights_np, cov_np, self.titles, n, self.out_dir, self.phase_space_dim, reweight=True)
