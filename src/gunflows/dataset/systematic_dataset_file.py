#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: SystematicDataset Class
Author: Mathias El Baz
Date: 28/01/2025
Description:
  Dataset class for training a Normalizing Flow model. It loads one or more .npz
  files containing:
    - Nsample x Ndim data in the eigenspace of the covariance
    - True probability (not assuming Gaussianity)
    - Covariance matrix & mean of the data 
"""

from __future__ import annotations

import glob
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

from .base import BaseSystematicDataset

__all__ = ["SystematicDatasetFile"]


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
    # Save the figure with name reweight or not
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

class SystematicDatasetFile(BaseSystematicDataset):
    def __init__(
        self,
        data_dir: str,
        phase_space_dim: list[int],
        batch_index: int | None = None,
        load_data: bool = True,
        plot_grid: bool = True,             
        out_dir: str | os.PathLike = "plots"
    ):
        super().__init__()
        self.phase_space_dim = phase_space_dim
        self.out_dir = Path(out_dir)

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
        for path in file_list:
            ld = np.load(path)
            d = torch.tensor(ld["data"], dtype=torch.float32)
            data_list.append(d)
            log_p_list.append(torch.tensor(ld["log_p"], dtype=torch.float32))
            log_q_list.append(torch.tensor(ld["log_q"], dtype=torch.float32))
            cov_ref = torch.tensor(ld["cov"], dtype=torch.float32)  
            mean_ref = torch.tensor(ld["mean"], dtype=torch.float32)
            titles_ref = ld["par_names"]
            bestfit_nll = ld["bestfit_nll"].item() 
            print(f"Best fit NLL: {bestfit_nll}")  # Debugging information
        data = torch.cat(data_list, dim=0)
        log_p = torch.cat(log_p_list, dim=0)
        self.log_q = torch.cat(log_q_list, dim=0) 

        chol = torch.linalg.cholesky(cov_ref)
        std_per_dim = torch.sqrt(torch.diag(cov_ref))
        data = (data - mean_ref) / std_per_dim
        d_inv = torch.diag(1.0 / std_per_dim)
        cov = d_inv @ cov_ref @ d_inv
        log_det_D = torch.logdet(cov)
        chol = torch.linalg.cholesky(cov)

        self.data = data
        self.log_p = log_p - bestfit_nll 
        print(f" Mean log_p : {self.log_p.mean().item()}")  # Debugging information
        self.log_q = self.log_q + 0.5* log_det_D
        print(f" Mean log_q : {self.log_q.mean().item()}")
        self.log_p = self.log_p + torch.median(self.log_q - self.log_p)
        print(f" Mean log_p after shift: {self.log_p.mean().item()}")  # Debugging information
        logshift = torch.log((d := torch.exp(self.log_q - self.log_p))[d <= torch.quantile(d, 0.999)].clamp_min(1e-40).mean())
        print(f"Log shift applied: {(d := torch.exp(self.log_q - self.log_p))[d <= torch.quantile(d, 0.999)].clamp_min(1e-40).mean()}")  # Debugging information
        print(f"Log shift applied: {logshift.item()}")  # Debugging information
        self.log_p += logshift
        print(f" Mean log_p after final shift: {self.log_p.mean().item()}")  # Debugging information
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

        
        ## Debug information
        print(f"Dataset loaded with {self.nsample} samples and {self.ndim} dimensions.")
        
        if load_data and plot_grid:
            logw = (self.log_q - self.log_p).reshape(-1)
            q = torch.quantile(logw, 0.999)
            mask = logw <= q
            logw = logw[mask]
            logw = logw - torch.logsumexp(logw, dim=0)  
            
            samples_np = self.data[mask, :].detach().cpu().numpy()
            weights_np = torch.exp(logw).detach().cpu().numpy()
            self.out_dir.mkdir(parents=True, exist_ok=True)
            n = 20
            mean_np    = self.mean.detach().cpu().numpy()
            cov_np     = self.cov.detach().cpu().numpy()

            _plot_grid(
                samples=samples_np,
                mean=mean_np,
                weights=weights_np,
                cov=cov_np,
                names=self.titles,
                n=n,
                out_dir=self.out_dir,
                phase_dims=self.phase_space_dim,
            )
            _plot_grid(
                samples=samples_np,
                mean=mean_np,
                weights=weights_np,
                cov=cov_np,
                names=self.titles,
                n=n,
                out_dir=self.out_dir,
                phase_dims=self.phase_space_dim,
                reweight=True,
            )

    def __len__(self) -> int:
        return self.nsample

    def log_prob(self, idx):
        return (
            self.data_spline[idx, :],
            self.data_cond[idx, :],
            -self.log_q[idx],
            -self.log_p[idx],
        )
