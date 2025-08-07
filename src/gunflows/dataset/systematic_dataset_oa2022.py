#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: SystematicDataset Class
Author: Mathias El Baz
Date: 28/01/2025
Description:
  Dataset class for training a Normalizing Flow model. It can start from:
    - a set of .npz files (starting_folder), or
    - sampler metadata only (with_sampler=True) and be filled later in-RAM.
  Assumes input data are in physical space and applies only:
    x_std = (x - mean) / sigma, with sigma = sqrt(diag(cov)).
"""

import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from gunflows.likelihood_sampler.likelihoodSampler import LikelihoodSampler

__all__ = ["SystematicDatasetOA2022"]


class SystematicDatasetOA2022(Dataset):
    def __init__(
        self,
        phase_space_dim: list[int],
        starting_folder: str | None = None,
        batch_index: int | None = None,
        with_sampler: bool = False,
        config_file: str | None = None,
        overrides: list[str] | None = None,
        load_data: bool = True,
    ):
        super().__init__()
        self.phase_space_dim = phase_space_dim
        self.with_sampler = with_sampler

        self.data = None
        self.log_p = None
        self.cov = None
        self.true_cov = None
        self.mean = None
        self.titles = None
        self.std_per_dim = None
        self.nsample = 0
        self.ndim = 0
        self.list_dim_conditionnal = []
        self.nll_all = None
        self.nll_cond = None
        self.data_cond = None
        self.data_spline = None
        self.shift = None

        sampler = None
        if with_sampler:
            if config_file is None:
                raise ValueError("with_sampler=True requires config_file.")
            sampler = LikelihoodSampler(config_file, override_files=overrides or [], threads=1, data_is_asimov=True)

        if starting_folder is None and not with_sampler:
            raise ValueError("Provide starting_folder or set with_sampler=True.")

        if starting_folder is not None:
            self._load_from_files(starting_folder, batch_index, load_data)
        else:
            cov_ref = torch.tensor(np.asarray(sampler.postfit_covariance_matrix, dtype=np.float32))
            mean_ref = torch.tensor(np.asarray(sampler.postfit_parameter_values, dtype=np.float32))
            titles_ref = sampler.get_parameter_names()
            self._set_metadata_only(cov_ref, mean_ref, titles_ref)

    def _load_from_files(self, folder: str, batch_index: int | None, load_data: bool):
        if batch_index is None:
            file_list = sorted(glob.glob(os.path.join(folder, "batch*.npz")))
            if not file_list:
                raise FileNotFoundError(f"No files found in {folder} matching 'batch*.npz'")
        else:
            p = os.path.join(folder, f"batch{batch_index}.npz")
            if not os.path.isfile(p):
                raise FileNotFoundError(f"File {p} does not exist.")
            file_list = [p]

        data_list, logp_list = [], []
        cov_ref = mean_ref = titles_ref = None

        for path in file_list:
            ld = np.load(path, allow_pickle=True)
            d = torch.tensor(ld["data"], dtype=torch.float32)      
            lp = torch.tensor(ld["log_p"], dtype=torch.float32)
            data_list.append(d)
            logp_list.append(lp)
            if cov_ref is None:
                cov_ref = torch.tensor(ld["cov"], dtype=torch.float32)
                mean_ref = torch.tensor(ld["mean"], dtype=torch.float32)
                titles_ref = ld["par_names"]

        if not load_data:
            self._set_metadata_only(cov_ref, mean_ref, titles_ref)
            return

        data = torch.cat(data_list, dim=0)
        log_p = torch.cat(logp_list, dim=0)
        self._finalize_from_raw(data, log_p, cov_ref, mean_ref, titles_ref)

    def _set_metadata_only(self, cov_ref: torch.Tensor, mean_ref: torch.Tensor, titles_ref):
        self.true_cov = cov_ref
        self.mean = mean_ref
        self.titles = titles_ref
        self.std_per_dim = torch.sqrt(torch.diag(cov_ref))
        self.cov = self._std_cov(cov_ref, self.std_per_dim)
        self.nsample = 0
        self.ndim = cov_ref.shape[0]
        self.list_dim_conditionnal = [i for i in range(self.ndim) if i not in self.phase_space_dim]
        self.data = self.log_p = None
        self.nll_all = self.nll_cond = None
        self.data_cond = self.data_spline = None

    def _finalize_from_raw(self, data_phys: torch.Tensor, log_p: torch.Tensor,
                           cov_ref: torch.Tensor, mean_ref: torch.Tensor, titles_ref):
        std_per_dim = torch.sqrt(torch.diag(cov_ref))
        x = (data_phys - mean_ref) / std_per_dim
        cov_std = self._std_cov(cov_ref, std_per_dim)

        self.data = x
        self.log_p = log_p
        self.cov = cov_std
        self.true_cov = cov_ref
        self.mean = mean_ref
        self.titles = titles_ref
        self.std_per_dim = std_per_dim

        self.nsample, self.ndim = x.shape
        self.list_dim_conditionnal = [i for i in range(self.ndim) if i not in self.phase_space_dim]

        self.nll_all = self._n_log_g_sub(self.data, list(range(self.ndim)))
        self.nll_cond = self._n_log_g_sub(self.data, self.list_dim_conditionnal)

        shift0 = torch.median(self.nll_all) - torch.median(self.log_p)
        self.log_p = self.log_p + shift0
        self.data_cond = self.data[:, self.list_dim_conditionnal]
        self.data_spline = self.data[:, self.phase_space_dim]

        diff = self.nll_all - self.log_p
        q = torch.quantile(diff, 0.99)
        self.shift = torch.log(torch.mean(torch.exp(diff[diff <= q])))
        self.log_p = self.log_p + self.shift

    @staticmethod
    def _std_cov(cov_ref: torch.Tensor, std_per_dim: torch.Tensor) -> torch.Tensor:
        d_inv = torch.diag(1.0 / std_per_dim)
        return d_inv @ cov_ref @ d_inv

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

    @classmethod
    def from_dict(cls, phase_space_dim: list[int], d: dict):
        data = torch.tensor(np.asarray(d["data"], dtype=np.float32))
        log_p = torch.tensor(np.asarray(d["log_p"], dtype=np.float32))
        cov_ref = torch.tensor(np.asarray(d["cov"], dtype=np.float32))
        mean_ref = torch.tensor(np.asarray(d["mean"], dtype=np.float32))
        titles_ref = d.get("par_names", None)

        obj = cls(phase_space_dim=phase_space_dim, starting_folder=None, with_sampler=True, config_file="dummy")
        obj._finalize_from_raw(data, log_p, cov_ref, mean_ref, titles_ref)
        return obj

    def replace_from_dict(self, d: dict):
        data = torch.tensor(np.asarray(d["data"], dtype=np.float32))
        log_p = torch.tensor(np.asarray(d["log_p"], dtype=np.float32))
        cov_ref = torch.tensor(np.asarray(d["cov"], dtype=np.float32))
        mean_ref = torch.tensor(np.asarray(d["mean"], dtype=np.float32))
        titles_ref = d.get("par_names", self.titles)
        self._finalize_from_raw(data, log_p, cov_ref, mean_ref, titles_ref)
