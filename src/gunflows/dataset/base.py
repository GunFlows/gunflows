#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title : Base Systematic Dataset
Author: Mathias El Baz
Date  : 2026-07-02

Common interface shared by every SystematicDataset* implementation.

Concrete datasets standardize samples into an eigenspace with a reference
(mean, cov) and split dimensions into a "spline" (phase-space) part and a
"conditional" (context) part. This base class fixes the contract that
gunflows.losses and gunflows.trainer rely on:
  - log_prob(idx) -> (data_spline, data_cond, -log_q, -log_p)
  - __getitem__ delegates to log_prob
  - get_cov / get_true_cov / get_cov_sub / get_mean / transform_eigen_space_to_data_space
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["BaseSystematicDataset"]


class BaseSystematicDataset(Dataset, ABC):
    cov: torch.Tensor
    true_cov: torch.Tensor
    mean: torch.Tensor
    std_per_dim: torch.Tensor

    @abstractmethod
    def log_prob(self, idx):
        """Return (data_spline, data_cond, -log_q, -log_p) for the given index/indices."""

    def __getitem__(self, idx):
        return self.log_prob(idx)

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
