#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title : ToyLLH
Author: Mathias El Baz
Date  : 2026-07-02

ToyLLH -- a GUNDAM-free likelihood-sampler backend, for demonstrating that
gunflows works against any likelihood implementing
gunflows.likelihood_sampler.base.LikelihoodSamplerProtocol, not just GUNDAM.

Target distribution, 60 independent dimensions ("50 plus 10", matching the
GUNDAM configs' naming convention):
  - dims  0-49: i.i.d. standard Gaussian N(0, 1)
  - dims 50-59: i.i.d. skew-normal (scipy.stats.skewnorm) with a small shape
    parameter, i.e. a Gaussian perturbed by a small skewness

To point gunflows at this instead of GUNDAM, set (in a dataset config):
    sampler_target: apps.toyllh.likelihood.ToyLLH
    llh_config: toy        # unused, but a truthy value is required
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skewnorm

N_GAUSS = 50
N_SKEW = 10
SKEW_A = 5.0  # skewnorm shape parameter; a few units = a small, visible skew


class ToyLLH:
    def __init__(
        self,
        config_file: str,
        override_files: list[str] | None = None,
        threads: int = 1,
        data_is_asimov: bool = False,
        seed: int = 42,
    ) -> None:
        del config_file, override_files, threads, data_is_asimov
        self.dim = N_GAUSS + N_SKEW
        self._rng = np.random.default_rng(seed)

        xs = np.linspace(-6, 6, 200_001)
        pdf = skewnorm.pdf(xs, SKEW_A)
        skew_mode = float(xs[np.argmax(pdf)])
        skew_var = float(skewnorm.var(SKEW_A))

        self.postfit_parameter_values = np.array(
            [0.0] * N_GAUSS + [skew_mode] * N_SKEW, dtype=np.float64
        )
        self.postfit_covariance_matrix = np.diag(
            [1.0] * N_GAUSS + [skew_var] * N_SKEW
        ).astype(np.float64)

        self.likelihood_at_bestfit = self._nll(self.postfit_parameter_values)

    def get_parameter_names(self) -> list[str]:
        return [f"gauss_{i}" for i in range(N_GAUSS)] + [f"skew_{i}" for i in range(N_SKEW)]

    def _nll(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        gauss_part = x[:N_GAUSS]
        skew_part = x[N_GAUSS:]
        log_pdf = (
            -0.5 * gauss_part**2 - 0.5 * np.log(2.0 * np.pi)
        ).sum() + skewnorm.logpdf(skew_part, SKEW_A).sum()
        return float(-log_pdf)

    def inject_params_and_compute_likelihood(
        self, params: list[float], extend_continue: bool = False
    ) -> tuple[float, object, object]:
        del extend_continue
        return self._nll(params), None, None
