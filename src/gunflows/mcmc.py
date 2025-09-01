#!/usr/bin/env python3
# =============================================================================
#  Title: mcmc.py
#  Author: Mathias El Baz
#  Date: 31/08/2025
#  Description:
#       Metropolis-Hastings MCMC implementation. This script evaluates the true likelihood through the
#       class gunflows.likelihood_sampler.LikelihoodSampler and runs a Metropolis-Hastings chain.
#       The proposal distribution is a multivariate normal centred at the current state with a covariance
#       taken from the likelihood's post-fit covariance matrix scaled by a user defined factor. The chain
#       is run for burn_in + n_steps iterations but only the final n_steps samples are stored.
#       The resulting chain along with the negative log-likelihood values is stored as a NumPy .npz file.
# =============================================================================
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager

import hydra
import numpy as np
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from gunflows.likelihood_sampler import LikelihoodSampler


@contextmanager
def pushd(path: str):
    prev = os.getcwd()
    if path:
        os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _corner_plot(samples: np.ndarray, names: list[str], out_file: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(samples[:, i], bins=60, density=True, histtype="step")
            else:
                ax.hist2d(samples[:, j], samples[:, i], bins=60, norm=LogNorm())
            if i == n - 1:
                ax.set_xlabel(names[j], fontsize=7)
            if j == 0:
                ax.set_ylabel(names[i], fontsize=7)
            ax.tick_params(axis="both", labelsize=6)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


@dataclass
class LikelihoodConfig:
    config: str
    overrides: Optional[list[str]] = None
    cwd: Optional[str] = None
    threads: int = 1
    asimov: bool = False


@dataclass
class PlotConfig:
    dims: list[int]
    every: int = 1000


@hydra.main(config_path="../../configs", config_name="mcmc", version_base=None)
def main(cfg: DictConfig) -> None:
    llh_cfg = LikelihoodConfig(**cfg.likelihood)
    plot_cfg = PlotConfig(**cfg.plot)

    try:
        base_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
    except Exception:
        base_dir = os.path.abspath(os.getcwd())

    cwd = llh_cfg.cwd or os.path.dirname(os.path.abspath(llh_cfg.config))
    with pushd(cwd):
        sampler = LikelihoodSampler(
            config_file=llh_cfg.config,
            override_files=llh_cfg.overrides,
            data_is_asimov=llh_cfg.asimov,
            threads=llh_cfg.threads,
        )

    rng = np.random.default_rng(cfg.seed)

    current = (
        np.array(cfg.initial_point)
        if cfg.initial_point is not None
        else sampler.postfit_parameter_values
    )
    dim = len(current)

    current_nll, _, _ = sampler.inject_params_and_compute_likelihood(
        current, extend_continue=False
    )
    print(f"Initial point: {current}")
    if current_nll == -1:
        raise RuntimeError("Initial point outside parameter domain")

    proposal_cov = np.asarray(sampler.postfit_covariance_matrix, dtype=float)
    proposal_cov *= cfg.proposal_scale ** 2

    total_steps = cfg.n_steps + cfg.burn_in
    chain = np.zeros((cfg.n_steps, dim))
    nlls = np.zeros(cfg.n_steps)

    for i in range(total_steps):
        proposal = rng.multivariate_normal(current, proposal_cov)
        nll, _, _ = sampler.inject_params_and_compute_likelihood(
            proposal, extend_continue=False
        )
        print(f"Step {i+1}/{total_steps}: Current NLL = {current_nll:.2f}, Proposal NLL = {nll:.2f}")
        if nll != -1:
            accept_prob = np.exp(-(nll - current_nll))
            print(f"  Acceptance probability: {accept_prob:.2f}")
            if rng.random() < accept_prob:
                current = proposal
                current_nll = nll
                print("  Accepted")

        if i >= cfg.burn_in:
            j = i - cfg.burn_in
            chain[j] = current
            nlls[j] = current_nll

    out_parent = base_dir
    if cfg.save_dir is not None:
        out_parent = os.path.join(base_dir, cfg.save_dir)
    os.makedirs(out_parent, exist_ok=True)
    out_file = os.path.join(out_parent, cfg.out_file)

    np.savez(
        out_file,
        chain=chain,
        nll=nlls,
        par_names=sampler.get_parameter_names(),
        bestfit_nll=sampler.likelihood_at_bestfit,
    )
    print(f"Saved chain to {out_file}")

    if plot_cfg.dims:
        plot_dir = os.path.join(out_parent, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        names = sampler.get_parameter_names()
        dims = plot_cfg.dims
        plot_names = [names[i] for i in dims]
        for step in range(plot_cfg.every, cfg.n_steps + 1, plot_cfg.every):
            path = os.path.join(plot_dir, f"corner_{step:06d}.png")
            _corner_plot(chain[:step, dims], plot_names, path)
            print(f" Plot at path {path}")


if __name__ == "__main__":
    main()
