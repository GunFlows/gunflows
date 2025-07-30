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

import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["SystematicDataset"]


class SystematicDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        phase_space_dim: list[int],
        batch_index: int | None = None,
        load_data: bool = True,
    ):
        super().__init__()
        self.phase_space_dim = phase_space_dim

        if batch_index is None:
            file_list = sorted(glob.glob(os.path.join(data_dir, "batch*.npz")))
            if not file_list:
                raise FileNotFoundError(f"No files found in {data_dir} matching 'batch*.npz'")
        else:
            p = os.path.join(data_dir, f"batch{batch_index}.npz")
            if not os.path.isfile(p):
                raise FileNotFoundError(f"File {p} does not exist.")
            file_list = [p]

        data_list, log_p_list = [], []
        for path in file_list:
            ld = np.load(path)
            d = torch.tensor(ld["data"], dtype=torch.float32)
            std_eig = torch.median(d.std(dim=0))         
            d = d / std_eig
            data_list.append(d)
            log_p_list.append(torch.tensor(ld["log_p"], dtype=torch.float32))
            cov_ref = torch.tensor(ld["cov"], dtype=torch.float32)  
            mean_ref = torch.tensor(ld["mean"], dtype=torch.float32)
            titles_ref = ld["par_names"]

        if not load_data:
            self.cov = cov_ref
            self.true_cov = cov_ref
            self.mean = mean_ref
            self.titles = titles_ref
            self.std_per_dim = torch.sqrt(torch.diag(cov_ref))
            self.list_dim_conditionnal = [i for i in range(cov_ref.shape[0]) if i not in self.phase_space_dim]
            self.nsample = 0
            self.ndim = cov_ref.shape[0]
            self.data = self.log_p = None
            return

        data = torch.cat(data_list, dim=0)
        log_p = torch.cat(log_p_list, dim=0)

        chol = torch.linalg.cholesky(cov_ref)
        data = (chol @ data.T).T
        std_per_dim = data.std(dim=0)
        data = data / std_per_dim
        d_inv = torch.diag(1.0 / std_per_dim)
        cov = d_inv @ cov_ref @ d_inv
        chol = torch.linalg.cholesky(cov)

        self.data = data
        self.log_p = log_p
        self.cov = cov
        self.true_cov = cov
        self.mean = mean_ref
        self.titles = titles_ref
        self.cholesky = chol
        self.std_per_dim = std_per_dim

        self.nsample, self.ndim = data.shape
        self.list_dim_conditionnal = [i for i in range(self.ndim) if i not in self.phase_space_dim]

        self.nll_all = self._n_log_g_sub(self.data, list(range(self.ndim)))
        self.nll_cond = self._n_log_g_sub(self.data, self.list_dim_conditionnal)

        shift = torch.median(self.nll_all) - torch.median(self.log_p)
        self.log_p = self.log_p + shift

        self.data_cond = self.data[:, self.list_dim_conditionnal]
        self.data_spline = self.data[:, self.phase_space_dim]

        diff = self.nll_all - self.log_p
        q_thresh = torch.quantile(diff, 0.99)
        self.shift = torch.log(torch.mean(torch.exp(diff[diff <= q_thresh])))
        self.log_p = self.log_p + self.shift

    def __len__(self) -> int:
        return self.nsample

    def __getitem__(self, idx: int):
        return (
            self.data_spline[idx, :],
            self.data_cond[idx, :],
            -self.nll_all[idx],
            -self.nll_cond[idx],
            -self.log_p[idx],
        )

    def log_prob(self, idx):
        return (
            self.data_spline[idx, :],
            self.data_cond[idx, :],
            -self.nll_all[idx],
            -self.nll_cond[idx],
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

    def _n_log_g_sub(self, x: torch.Tensor, dims: list[int]) -> torch.Tensor:
        x_sub = x[:, dims]
        s_sub = self.get_cov_sub(dims)
        s_inv = torch.linalg.inv(s_sub)
        ld = torch.logdet(s_sub)
        quad = torch.sum((x_sub @ s_inv) * x_sub, dim=1)
        cst = len(dims) * np.log(2.0 * np.pi)
        return 0.5 * (cst + ld + quad)

    def transform_eigen_space_to_data_space(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std_per_dim + self.mean
