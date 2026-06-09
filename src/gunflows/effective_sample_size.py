#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Title: sample_mcmc.py
#  Author: Lorenzo Giannessi
#  Date: 29/01/2026
#  Description:
#   Compute the ESS of the NF model as a function of time, then compare to the MCMC throws
# =============================================================================

from __future__ import annotations
import math, time, os, sys, json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import multiprocessing as mp

import re
import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import kstest
from concurrent.futures import ThreadPoolExecutor, as_completed
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra.utils import instantiate
from matplotlib.colors import LogNorm
from sample_mcmc_toy import _abspath, _strip_common_prefixes
from sample_mcmc import check_parameters_limits, sample_check_append
from ess_plotting import (
    make_ess_plots,
    epoch_to_time_hours,
    epoch_to_n_samplings,
    sum_generated_from_progress,
)


NF_LOCAL = os.path.join(os.path.dirname(__file__), "..", "normalizing-flows")
sys.path.append(os.path.abspath(NF_LOCAL))

from gunflows.utils.build_flow import build_base, build_flow_layers, build_model
from gunflows.likelihood_sampler import LikelihoodSampler, pygundam_utils

import ROOT # to read the MCMC chain ROOT file

def redirect_fds(path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)

    # redirect stdout (1) and stderr (2)
    os.dup2(fd, 1)
    os.dup2(fd, 2)

    os.close(fd)

    # also update Python wrappers
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)

def init_worker(cfg,logdir):
    global _sampler
    worker_index = mp.current_process()._identity[0] - 1

    pid = os.getpid()
    logfile = os.path.join(logdir, f"worker_{worker_index}.log")

    # redirect stdout / stderr
    redirect_fds(logfile)

    print(f"Worker {worker_index} starting")
    _sampler = LikelihoodSampler(config_file=cfg.experiment.dataset.llh_config,
        override_files=cfg.experiment.dataset.llh_overrides,
        data_is_asimov=cfg.experiment.dataset.data_is_asimov,
        threads=cfg.experiment.sampler.threads,
        llh_cwd=cfg.experiment.dataset.llh_cwd,
        light_mode=False,
    )
    print(f"Worker {worker_index} initialized.")

def worker(v):
    t0 = time.perf_counter()
    logp, _, _ = _sampler.inject_params_and_compute_likelihood(values=v, extend_continue=False, verbose=0)
    # print(f"computed LH in {time.perf_counter()-t0:.2f} s. NLL/2: {logp}", flush=True)
    return logp


def _stable_weights_from_logratio(log_ratio: np.ndarray) -> np.ndarray:
    lr = np.asarray(log_ratio, dtype=np.float64).reshape(-1)
    if lr.size == 0:
        return lr
    m = np.max(lr)
    w = np.exp(lr - m)
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(lr, dtype=np.float64)
    return w / s


def _ks_stat_against_gaussian(x: np.ndarray, mu: float, sigma: float) -> float:
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    z = (np.asarray(x, dtype=np.float64).reshape(-1) - mu) / sigma
    z = z[np.isfinite(z)]
    if z.size < 5:
        return float("nan")
    stat, _ = kstest(z, "norm")
    return float(stat)


def _mass_to_threshold_levels(hist: np.ndarray, mass_levels: tuple[float, ...]) -> list[float]:
    h = np.asarray(hist, dtype=np.float64)
    if h.size == 0:
        return []
    flat = h.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return []
    total = float(np.sum(flat))
    if total <= 0:
        return []

    sorted_desc = np.sort(flat)[::-1]
    cdf = np.cumsum(sorted_desc) / total
    thresholds = []
    for m in mass_levels:
        m = float(m)
        if m <= 0 or m >= 1:
            continue
        idx = int(np.searchsorted(cdf, m, side="left"))
        idx = min(max(idx, 0), sorted_desc.size - 1)
        thresholds.append(float(sorted_desc[idx]))

    levels = sorted(set([lv for lv in thresholds if np.isfinite(lv) and lv > 0]))
    return levels


def _plot_marginal_nf_vs_reweighted(
    x_nf: np.ndarray,
    w_reweighted: np.ndarray,
    mu: float,
    sigma: float,
    label: str,
    out_path: Path,
    bins: int = 60,
) -> None:
    x = np.asarray(x_nf, dtype=np.float64).reshape(-1)
    w = np.asarray(w_reweighted, dtype=np.float64).reshape(-1)
    if x.size == 0 or w.size == 0:
        return

    p1, p99 = np.percentile(x, [1.0, 99.0])
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        p1, p99 = float(np.min(x)), float(np.max(x))
    if np.isfinite(sigma) and sigma > 0 and np.isfinite(mu):
        p1 = min(p1, mu - 5.0 * sigma)
        p99 = max(p99, mu + 5.0 * sigma)
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        p1, p99 = 0.0, 1.0
    span = p99 - p1
    xmin, xmax = p1 - 0.05 * span, p99 + 0.05 * span

    edges = np.linspace(xmin, xmax, bins + 1)
    xgrid = np.linspace(xmin, xmax, 400)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(x, bins=edges, density=True, histtype="step", linewidth=1.3, label="NF", color="C0")
    ax.hist(x, bins=edges, density=True, histtype="step", linewidth=1.3, label="NF reweighted", color="C3", weights=w)

    if np.isfinite(sigma) and sigma > 0 and np.isfinite(mu):
        gauss = np.exp(-0.5 * ((xgrid - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        ax.plot(xgrid, gauss, color="0.25", linewidth=1.0, linestyle="--", label="Gaussian ref")

    ax.set_xlabel(label)
    ax.set_ylabel("a.u.")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_contours_nf_vs_reweighted(
    x_nf: np.ndarray,
    y_nf: np.ndarray,
    w_reweighted: np.ndarray,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    bins: int = 60,
) -> None:
    x = np.asarray(x_nf, dtype=np.float64).reshape(-1)
    y = np.asarray(y_nf, dtype=np.float64).reshape(-1)
    w = np.asarray(w_reweighted, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0 or w.size == 0:
        return

    x_low, x_high = np.percentile(x, [1.0, 99.0])
    y_low, y_high = np.percentile(y, [1.0, 99.0])
    if not np.isfinite(x_low) or not np.isfinite(x_high) or x_high <= x_low:
        x_low, x_high = float(np.min(x)), float(np.max(x))
    if not np.isfinite(y_low) or not np.isfinite(y_high) or y_high <= y_low:
        y_low, y_high = float(np.min(y)), float(np.max(y))
    if not np.isfinite(x_low) or not np.isfinite(x_high) or x_high <= x_low:
        x_low, x_high = 0.0, 1.0
    if not np.isfinite(y_low) or not np.isfinite(y_high) or y_high <= y_low:
        y_low, y_high = 0.0, 1.0

    x_span = x_high - x_low
    y_span = y_high - y_low
    xrange = [x_low - 0.05 * x_span, x_high + 0.05 * x_span]
    yrange = [y_low - 0.05 * y_span, y_high + 0.05 * y_span]

    h_nf, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xrange, yrange], density=True)
    h_rw, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=w, density=True)

    levels_nf = _mass_to_threshold_levels(h_nf, (0.50, 0.80, 0.95))
    levels_rw = _mass_to_threshold_levels(h_rw, (0.50, 0.80, 0.95))

    xcent = 0.5 * (xedges[:-1] + xedges[1:])
    ycent = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xcent, ycent, indexing="xy")

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    if len(levels_nf) > 0 and np.max(h_nf) > 0:
        ax.contour(X, Y, h_nf.T, levels=levels_nf, colors="C0", linewidths=1.2, linestyles="-")
    if len(levels_rw) > 0 and np.max(h_rw) > 0:
        ax.contour(X, Y, h_rw.T, levels=levels_rw, colors="C3", linewidths=1.2, linestyles="--")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.20)
    legend_lines = [
        Line2D([0], [0], color="C0", lw=1.4, linestyle="-", label="NF"),
        Line2D([0], [0], color="C3", lw=1.4, linestyle="--", label="NF reweighted"),
    ]
    ax.legend(handles=legend_lines, fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_marginal_epoch_panel(
    epoch_data: list,
    mu: float,
    sigma: float,
    label: str,
    out_path: Path,
    bins: int = 60,
) -> None:
    """One figure for a single parameter; one subpanel per training epoch.

    Each subpanel overlays:
      - NF marginal (unweighted)
      - NF reweighted marginal (IS weights = exp(log_ratio))
      - Gaussian reference (best-fit mean, postfit sigma)
      - dashed vertical line at the best-fit value

    Parameters
    ----------
    epoch_data : list of (epoch, x_nf, log_ratio) sorted by epoch ascending.
    """
    if len(epoch_data) == 0:
        return

    n = len(epoch_data)
    ncols = min(5, n)
    nrows = int(math.ceil(n / ncols))

    # Shared x range across panels: percentile of pooled samples, padded by
    # +/- 4 sigma so the Gaussian reference is always visible.
    all_x = np.concatenate(
        [np.asarray(x, dtype=np.float64).reshape(-1) for _, x, _ in epoch_data]
    )
    if all_x.size == 0:
        return
    p1, p99 = np.percentile(all_x, [0.5, 99.5])
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        p1, p99 = float(np.min(all_x)), float(np.max(all_x))
    if np.isfinite(sigma) and sigma > 0 and np.isfinite(mu):
        p1 = min(p1, mu - 4.0 * sigma)
        p99 = max(p99, mu + 4.0 * sigma)
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        p1, p99 = 0.0, 1.0
    span = p99 - p1
    xmin, xmax = p1 - 0.05 * span, p99 + 0.05 * span
    edges = np.linspace(xmin, xmax, bins + 1)
    xgrid = np.linspace(xmin, xmax, 400)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.0 * ncols, 2.6 * nrows),
        sharex=True, sharey=False, squeeze=False,
    )

    for k, (ep, x, lr) in enumerate(epoch_data):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        lr = np.asarray(lr, dtype=np.float64).reshape(-1)
        if x.size == 0:
            ax.set_visible(False)
            continue
        w = _stable_weights_from_logratio(lr)
        ax.hist(x, bins=edges, density=True, histtype="step", linewidth=1.2,
                color="C2", label="NF")
        ax.hist(x, bins=edges, density=True, histtype="step", linewidth=1.2,
                color="C3", weights=w, label="NF rw")
        if np.isfinite(sigma) and sigma > 0 and np.isfinite(mu):
            gauss = np.exp(-0.5 * ((xgrid - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
            ax.plot(xgrid, gauss, color="C4", linewidth=1.0, label="Gauss")
            ax.axvline(mu, color="C0", linestyle="--", linewidth=0.9, label="best fit")
        ax.set_title(f"epoch {ep}", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.2)
        if k == 0:
            ax.legend(fontsize=7, loc="best")

    # Hide unused subplots
    for k in range(len(epoch_data), nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].set_visible(False)

    # Common x-axis label on the bottom row
    for c in range(ncols):
        axes[nrows - 1][c].set_xlabel(label, fontsize=9)

    fig.suptitle(f"Marginal evolution: {label}", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@hydra.main(config_path="../../configs", config_name="effective_sample_size", version_base=None)
def main(cfg: DictConfig) -> None:
    training_folder = _abspath(str(cfg.training_folder))
    run_without_mcmc = bool(getattr(cfg, "run_without_mcmc", False))
    mcmc_chain_cfg = getattr(cfg, "mcmc_chain", None)
    mcmc_root = None
    if mcmc_chain_cfg is not None and str(mcmc_chain_cfg).strip() != "":
        mcmc_root = _abspath(str(mcmc_chain_cfg))
    if not run_without_mcmc and mcmc_root is None:
        raise RuntimeError("mcmc_chain must be provided when run_without_mcmc is false.")
    save_dir = _abspath(str(cfg.save_dir))

    print(f"PWD (hydra chdir): {os.getcwd()}", flush=True)
    print(f"training_folder: {training_folder}", flush=True)
    print(f"run_without_mcmc: {run_without_mcmc}", flush=True)
    print(f"mcmc_chain: {mcmc_root}", flush=True)
    print(f"save_dir: {save_dir}", flush=True)

    train_cfg_path = os.path.join(training_folder, ".hydra", "config.yaml")
    if not os.path.isfile(train_cfg_path):
        raise RuntimeError(f"Training config not found: {train_cfg_path}")

    train_cfg = OmegaConf.load(train_cfg_path)
    cfg = OmegaConf.merge(train_cfg, cfg)

    cfg.experiment.dataset.max_batches = 1
    cfg.experiment.dataset.with_sampler = False
    cfg.experiment.dataset.plot_grid = False

    seed = int(getattr(cfg, "seed", 0))
    torch.manual_seed(seed)


    # create output directories
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "marginals"
    img_dir.mkdir(parents=True, exist_ok=True)
    corr2d_dir = out_dir / "corr2d"
    corr2d_dir.mkdir(parents=True, exist_ok=True)

    # initialize likelihood interface
    print("Initializing likelihood interface...", flush=True)
    likelihood_sampler = LikelihoodSampler(
        config_file=cfg.experiment.dataset.llh_config,
        override_files=cfg.experiment.dataset.llh_overrides,
        data_is_asimov=cfg.experiment.dataset.data_is_asimov,
        threads=cfg.experiment.sampler.threads,
        llh_cwd=cfg.experiment.dataset.llh_cwd,
        light_mode=False,
    )

    nf_param_names = list(likelihood_sampler.get_parameter_names())
    nf_param_names_short = [_strip_common_prefixes(n) for n in nf_param_names]
    parameter_limits: dict[str, tuple[float, float]] = {n: likelihood_sampler.get_parameter_limits(n) for n in nf_param_names}

    bestfit_parameter_values = np.asarray(likelihood_sampler.postfit_parameter_values, dtype=np.float64).reshape(-1)
    postfit_covariance = np.asarray(likelihood_sampler.postfit_covariance_matrix, dtype=np.float64)

    dataset = instantiate(cfg.experiment.dataset)
    dim_spline = len(dataset.phase_space_dim)


    ess_list = []
    ess_filtered_list = []
    epoch_list = []
    latest_epoch = -1
    latest_payload = None

    # ---- per-epoch marginal scan -------------------------------------------------
    # User-configurable list of parameter indices to plot one figure per param,
    # with one subpanel per checkpoint epoch. Default = empty (feature off).
    _scan_cfg = getattr(cfg, "marginal_scan_param_indices", None)
    if _scan_cfg is None:
        marginal_scan_indices: list[int] = []
    else:
        marginal_scan_indices = [int(i) for i in _scan_cfg]
    marginal_scan_bins = int(getattr(cfg, "marginal_scan_bins", 60))
    # epoch_marginal_payloads[idx] = list of (epoch, x_arr, log_ratio_arr)
    epoch_marginal_payloads: dict = {i: [] for i in marginal_scan_indices}
    if marginal_scan_indices:
        print(
            f"Marginal-evolution scan enabled for param indices: {marginal_scan_indices} "
            f"(bins={marginal_scan_bins})",
            flush=True,
        )

    if (cfg.llh_workers > 0):
        print(f"Initializing {cfg.llh_workers} ({mp.cpu_count()}) workers to compute LH values in parallel.", flush=True)
        # compute LH with multiple threads
        workers_log_dir = out_dir / "llh_workers_logs"
        workers_log_dir.mkdir(parents=True, exist_ok=True)
        pool = mp.Pool(processes=cfg.llh_workers, initializer=init_worker, initargs=(cfg, workers_log_dir))

    # start a loop where at each iteration you pickup a checkpoint, sample from it, compute ESS, make some plots, then store the results
    # Checkpoint files are named `sampler_epoch<EPOCH>.pt` where <EPOCH> is the
    # REAL (zero-padded) epoch number, e.g. sampler_epoch00500.pt -> epoch 500.
    ckpt_folder = os.path.join(training_folder, "checkpoints")
    pattern = re.compile(r"sampler_epoch(\d+)\.pt$")
    all_ckpts: list[tuple[int, str]] = []
    for fname in os.listdir(ckpt_folder):
        m = pattern.match(fname)
        if m:
            all_ckpts.append((int(m.group(1)), fname))
    all_ckpts.sort(key=lambda t: t[0])
    print(f"Total NF checkpoints found: {len(all_ckpts)}", flush=True)
    # Maximum available epoch -> used as the training span for the
    # epoch -> #LH-samplings linear conversion (see below).
    max_ckpt_epoch = max((e for e, _ in all_ckpts), default=0)

    # --- Checkpoint selection -------------------------------------------------
    # If `custom_epochs` is provided (a list of REAL epoch numbers), evaluate
    # exactly those checkpoints (epoch 0 is the Gaussian point, handled above).
    # Otherwise fall back to the previous behaviour: `n_checkpoints` roughly
    # equally-spaced checkpoints within [0, max_epoch].
    _custom_cfg = getattr(cfg, "custom_epochs", None)
    custom_epochs = None
    if _custom_cfg is not None:
        custom_epochs = [int(e) for e in _custom_cfg]

    if custom_epochs is not None and len(custom_epochs) > 0:
        avail = {e: f for e, f in all_ckpts}
        selected = []
        for e in sorted(set(custom_epochs)):
            if e == 0:
                continue  # epoch 0 == Gaussian baseline, no checkpoint file
            if e in avail:
                selected.append((e, avail[e]))
            else:
                print(f"  WARNING: requested custom epoch {e} has no checkpoint "
                      f"sampler_epoch{e:05d}.pt -- skipping.", flush=True)
        print(f"Selected {len(selected)} checkpoints from custom_epochs "
              f"(epochs: {[e for e, _ in selected]})", flush=True)
    else:
        # Optionally ignore checkpoints beyond `max_epoch` (real epoch units).
        max_epoch = getattr(cfg, "max_epoch", None)
        if max_epoch is not None and int(max_epoch) > 0:
            max_epoch = int(max_epoch)
            kept = [(e, f) for e, f in all_ckpts if e <= max_epoch]
            if not kept:
                raise RuntimeError(
                    f"max_epoch={max_epoch} excludes every checkpoint "
                    f"(available epochs: {[e for e, _ in all_ckpts]})"
                )
            all_ckpts = kept
            print(f"Applied max_epoch={max_epoch}: {len(all_ckpts)} checkpoints "
                  f"remain (epochs: {[e for e, _ in all_ckpts]})", flush=True)

        # Sub-sample to `n_checkpoints` roughly equally-spaced (default 10).
        n_pick = int(getattr(cfg, "n_checkpoints", 10))
        if n_pick > 0 and len(all_ckpts) > n_pick:
            idx = np.linspace(0, len(all_ckpts) - 1, n_pick).round().astype(int)
            idx = sorted(set(int(i) for i in idx))
            selected = [all_ckpts[i] for i in idx]
        else:
            selected = all_ckpts
        print(f"Selected {len(selected)} checkpoints (epochs): "
              f"{[e for e, _ in selected]}", flush=True)
    tot_models = len(selected)

    # --- Epoch -> time / #LH-samplings linear conversions ---------------------
    # time(epoch)  = epoch * (time_ref_hours / time_ref_epochs)        [b = 0]
    # nsamp(epoch) = samplings_base + (sum_generated/ref_epoch)*epoch  [b = base]
    time_ref_epochs = float(getattr(cfg, "time_ref_epochs", 56000) or 0)
    time_ref_hours = float(getattr(cfg, "time_ref_hours", 23.0 + 20.0 / 60.0))
    samplings_base = float(getattr(cfg, "samplings_base", 2_500_000))
    samplings_ref_epoch_cfg = getattr(cfg, "samplings_ref_epoch", None)
    samplings_ref_epoch = (
        float(samplings_ref_epoch_cfg)
        if samplings_ref_epoch_cfg is not None and float(samplings_ref_epoch_cfg) > 0
        else float(max_ckpt_epoch)
    )
    _prog_glob = getattr(cfg, "samplings_progress_glob", None)
    if _prog_glob is None or str(_prog_glob).strip() == "":
        _prog_glob = os.path.join(training_folder, "generated_samples", "progress_*.json")
    else:
        _prog_glob = _abspath(str(_prog_glob))
    sum_generated = sum_generated_from_progress(_prog_glob)
    print(f"Epoch->time: {time_ref_hours:.4f} h over {time_ref_epochs:.0f} epochs.", flush=True)
    print(f"Epoch->#samplings: base={samplings_base:.0f} + "
          f"(sum_generated={sum_generated:.0f} / ref_epoch={samplings_ref_epoch:.0f})*epoch "
          f"(glob={_prog_glob}).", flush=True)

    # ── Gaussian approximation ESS (plotted at epoch 0) ──────────────────────
    num_samples = int(cfg.num_samples)
    batch_size  = int(cfg.batch_size)
    print("\nComputing Gaussian approximation ESS (epoch 0)...", flush=True)
    _rng_g = np.random.default_rng(42)
    _mu_g  = bestfit_parameter_values.copy()
    _cov_g = postfit_covariance.copy()
    _ndim  = len(_mu_g)
    _L_g   = np.linalg.cholesky(_cov_g)
    _logdet_g = 2.0 * np.sum(np.log(np.diag(_L_g)))
    _log_norm_g = -0.5 * (_ndim * np.log(2.0 * np.pi) + _logdet_g)
    _cov_inv_g  = np.linalg.solve(_cov_g, np.eye(_ndim))

    _g_samples: list[np.ndarray] = []
    _g_logqs:   list[float]      = []
    while len(_g_samples) < num_samples:
        take = min(batch_size, (num_samples - len(_g_samples)) * 3)
        _z = _rng_g.standard_normal((take, _ndim))
        _xs = _mu_g + _z @ _L_g.T
        for _x in _xs:
            if not check_parameters_limits(_x, parameter_limits):
                continue
            _diff = _x - _mu_g
            _g_logqs.append(float(_log_norm_g - 0.5 * float(_diff @ _cov_inv_g @ _diff)))
            _g_samples.append(_x.copy())
            if len(_g_samples) >= num_samples:
                break

    _g_samples = np.asarray(_g_samples[:num_samples])
    _g_logqs   = np.asarray(_g_logqs[:num_samples], dtype=np.float64)

    print(f"Computing LH for {len(_g_samples)} Gaussian samples...", flush=True)
    _t0_g = time.time()
    if cfg.llh_workers > 0:
        _g_lh = pool.map(worker, _g_samples, chunksize=32)
        _g_rw = np.array([-lq - lp for lp, lq in zip(_g_lh, _g_logqs)])
    else:
        _g_rw_list = []
        for _gx, _glq in zip(_g_samples, _g_logqs):
            _glp, _, _ = likelihood_sampler.inject_params_and_compute_likelihood(_gx, extend_continue=False)
            _g_rw_list.append(-_glq - _glp)
        _g_rw = np.array(_g_rw_list)
    print(f"Gaussian LH done in {time.time()-_t0_g:.1f}s", flush=True)

    _g_rw -= np.median(_g_rw)
    _g_w   = np.exp(_g_rw)
    _g_ess = float(np.sum(_g_w)**2 / np.sum(_g_w**2))
    _g_lo, _g_hi = np.quantile(_g_rw, 0.001), np.quantile(_g_rw, 0.999)
    _g_fmask = (_g_rw >= _g_lo) & (_g_rw <= _g_hi)
    _g_fw    = np.exp(_g_rw[_g_fmask])
    _g_ess_f = float(np.sum(_g_fw)**2 / np.sum(_g_fw**2))
    print(f"Gaussian ESS: {_g_ess:.1f}/{num_samples}  "
          f"filtered: {_g_ess_f:.1f}/{_g_fmask.sum()}", flush=True)

    epoch_list.append(0)
    ess_list.append(_g_ess / num_samples)
    ess_filtered_list.append(_g_ess_f / float(_g_fmask.sum()))
    tot_models += 1
    # ── end Gaussian ESS ─────────────────────────────────────────────────────

    for ep, fname in selected:
        print(f"Found NF model file: {fname}", flush=True)
        ckpt_path = Path(os.path.join(ckpt_folder, fname))
        print("Using NF model:", ckpt_path, flush=True)

        base = build_base(cfg.experiment.model.total_dim)
        tail_bounds = torch.ones(dim_spline) * cfg.experiment.model.tail_bound
        flows = build_flow_layers(
            cfg.experiment.model.nflows,
            dim_spline,
            cfg.experiment.model.hidden,
            cfg.experiment.model.nlayers,
            cfg.experiment.model.nbins,
            tail_bounds,
            n_context=cfg.experiment.model.total_dim - dim_spline,
        )
        model = build_model(
            base,
            flows,
            dataset,
            cfg.experiment.model.context_transform,
            cfg.experiment.model.freeze_covflow,
            n_context_flows=cfg.experiment.model.n_context_flows,
            hidden_dim=cfg.experiment.model.hidden_dim,
            n_hidden_layers=cfg.experiment.model.n_hidden_layers,
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
        model = model.to(cfg.device).eval()
        print("NF model loaded.", flush=True)

        # sample from NF in physical space
        num_samples = int(cfg.num_samples)
        batch_size = int(cfg.batch_size)

        print(f"Sampling {num_samples} events from NF model...",flush=True)    
        t0 = time.time()
        # Sample from NF (vectorized): sample batches at once, filter vectorially
        batches = math.ceil(num_samples / batch_size)
        samples_nf, logqs = [], []
        start = time.time()
        need_total = int(num_samples)
        with torch.no_grad():
            while len(samples_nf) < need_total:
                need = need_total - len(samples_nf)
                b = min(int(batch_size), need)
                if (cfg.verbose>=1): print(f" NF sampling. {need} throws to go. Sampling {b} now ...", flush=True)            
                remain = b
                while remain > 5:
                    take = sample_check_append(sample_from_nf=True, batch_size=remain, model=model, dataset=dataset, parameter_limits=parameter_limits, samples=samples_nf, return_probs=True, logqs=logqs, mean=None, cov=None)
                    remain -= take
                    if take == 0:
                        break  # avoid infinite loop if no samples accepted
                # for any remaining samples not accepted, fall back to single-sample retry
                if remain > 0:
                    if (cfg.verbose>=2): print(f" Need {remain} more samples after batch filtering. Sampling individually...", flush=True)
                    for _ in range(remain):
                        z, logq = model.sample(1)
                        z = z.to('cpu')
                        phys_z = dataset.transform_eigen_space_to_data_space(z).detach().cpu().numpy()[0]
                        logq_np = float(logq.detach().cpu().numpy()[0])
                        while not check_parameters_limits(phys_z, parameter_limits):
                            # print(f"  -debug- single sample not physical, resampling...", flush=True)
                            z, logq = model.sample(1)
                            z = z.to('cpu')
                            phys_z = dataset.transform_eigen_space_to_data_space(z).detach().cpu().numpy()[0]
                            logq_np = float(logq.detach().cpu().numpy()[0])
                        samples_nf.append(phys_z)
                        logqs.append(logq_np)
                        if (cfg.verbose>=2): print(f" Sampled individual throw. {_+1}/{remain}", flush=True)
                if (cfg.verbose>=1): print(f"Total samples collected: {len(samples_nf)}/{need_total}", flush=True)

        # samples_nf already contains physical-space numpy arrays (appended above);
        # just convert the list to a single NumPy array instead of transforming again.
        samples_nf = np.asarray(samples_nf)

        # Prepare optional weights for NF histogram
        logq_nf = None
        w = np.asarray(logqs)
        if w.ndim > 1:
            w = w.reshape(-1)
        # Ensure weights length matches number of NF samples
        logq_nf = w[: samples_nf.shape[0]]        
        
        print(f"NF sampling done: {samples_nf.shape} in {time.time()-t0:.1f} s", flush=True)

        # scan through the nf samples, compute the nll and compare it to the nf weight (logq_nf)
        iter = 0
        reweight_nf_to_lh = []
        lh_values = []
        start = time.time()
        print("Computing reweighting factors from NF to LH...",flush=True)

        if (cfg.llh_workers > 0):
            print(f"Using {cfg.llh_workers} ({mp.cpu_count()}) workers to compute LH values in parallel.", flush=True)
            # compute LH with multiple threads
            lh_values = pool.map(worker, samples_nf, chunksize=32)
            reweight_nf_to_lh = [-logq - logp for logp, logq in zip(lh_values, logq_nf)]
        else:
            print("Computing LH values sequentially.", flush=True)
            for nf_vector, logq in zip(samples_nf, logq_nf):
                logp,nll_stat,nll_syst = likelihood_sampler.inject_params_and_compute_likelihood(nf_vector,extend_continue=False)
                if (iter % max(1, num_samples // 100) == 0):
                        if (cfg.verbose >= 3):    
                            print(f"iter {iter} NLL/2: {logp}, log_q_nf: {logq}", flush=True)
                iter += 1
                reweight_nf_to_lh.append(-logq - logp)
                lh_values.append(-logp)

        print(f"Computed reweighting factors for {len(reweight_nf_to_lh)} NF samples.",flush=True)
        end = time.time()
        print(f"Time to compute LH values: {end - start:.1f}s", flush=True)

        # Normalize reweighting factors
        if reweight_nf_to_lh:
            median_reweight = np.median(reweight_nf_to_lh)
            reweight_nf_to_lh = (np.array(reweight_nf_to_lh)-median_reweight)
            # shift the median of the likelihood values and log_q_nf accordingly
            median_lh = np.median(lh_values)
            lh_values = np.array(lh_values) - median_lh
            median_logq = np.median(logq_nf)
            logq_nf = logq_nf - median_logq
        # compute variance
        variance_reweight = np.var(reweight_nf_to_lh)
        # compute variance after removing 0.001 quantiles
        lower_bound = np.quantile(reweight_nf_to_lh, 0.001)
        upper_bound = np.quantile(reweight_nf_to_lh, 0.999)
        outlier_mask = (reweight_nf_to_lh >= lower_bound) & (reweight_nf_to_lh <= upper_bound)
        filtered_reweights = reweight_nf_to_lh[outlier_mask]
        variance_filtered = np.var(filtered_reweights)
        # compute effective sample size
        weights = np.exp(reweight_nf_to_lh)
        effective_sample_size = np.sum(weights) ** 2 / np.sum(weights ** 2)
        filtered_weights = np.exp(filtered_reweights)
        effective_sample_size_filtered = np.sum(filtered_weights) ** 2 / np.sum(filtered_weights ** 2)
        print(f"Effective sample size (NF to LH): {effective_sample_size} / {len(reweight_nf_to_lh)}", flush=True)
        print(f"Effective sample size (NF to LH, filtered): {effective_sample_size_filtered} / {len(filtered_reweights)}", flush=True)
        epoch_list.append(ep)
        ess_list.append(effective_sample_size/len(reweight_nf_to_lh))
        ess_filtered_list.append(effective_sample_size_filtered/len(filtered_reweights))

        d = min(
            int(samples_nf.shape[1]),
            len(nf_param_names_short),
            int(bestfit_parameter_values.shape[0]),
            int(postfit_covariance.shape[0]),
        )
        if d > 0 and ep > latest_epoch:
            latest_epoch = ep
            latest_payload = {
                "epoch": int(ep),
                "samples_nf": np.asarray(samples_nf[:, :d], dtype=np.float64),
                "log_ratio": np.asarray(reweight_nf_to_lh[:samples_nf.shape[0]], dtype=np.float64),
                "labels": list(nf_param_names_short[:d]),
                "mu": np.asarray(bestfit_parameter_values[:d], dtype=np.float64),
                "sigma": np.sqrt(np.clip(np.diag(postfit_covariance[:d, :d]), 0.0, np.inf)),
            }

        # Stash this epoch's column slice + log-ratio for the marginal-evolution scan.
        if marginal_scan_indices:
            _lr_arr = np.asarray(reweight_nf_to_lh[:samples_nf.shape[0]], dtype=np.float64)
            for _idx in marginal_scan_indices:
                if 0 <= _idx < int(samples_nf.shape[1]):
                    epoch_marginal_payloads[_idx].append(
                        (int(ep),
                         np.asarray(samples_nf[:, _idx], dtype=np.float64).copy(),
                         _lr_arr.copy())
                    )

        # sort lists by epoch
        sorted_indices = np.argsort(epoch_list)
        epoch_list = [epoch_list[i] for i in sorted_indices]
        ess_list = [ess_list[i] for i in sorted_indices]
        ess_filtered_list = [ess_filtered_list[i] for i in sorted_indices]

        # epoch -> time [hours] and epoch -> #LH-samplings (linear conversions)
        _time_hours = epoch_to_time_hours(
            epoch_list, time_ref_epochs, time_ref_hours).tolist()
        _n_samplings = epoch_to_n_samplings(
            epoch_list, samplings_base, sum_generated, samplings_ref_epoch).tolist()

        # save intermediate results to json (includes everything needed to
        # re-plot offline via plot_ess_from_json.py)
        results = {
            "epochs": epoch_list,
            "ess": ess_list,
            "ess_filtered": ess_filtered_list,
            "time_hours": _time_hours,
            "n_samplings": _n_samplings,
            "num_samples": int(num_samples),
            "conversion": {
                "time_ref_epochs": time_ref_epochs,
                "time_ref_hours": time_ref_hours,
                "samplings_base": samplings_base,
                "samplings_ref_epoch": samplings_ref_epoch,
                "sum_generated": sum_generated,
            },
        }
        json_path = out_dir / "ess_vs_epoch.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)

        # 6 ESS plots: {non-filtered, filtered} x {epoch, time, #samplings}
        make_ess_plots(results, out_dir, num_samples=num_samples)

        print(f"Completed {len(epoch_list)}/{tot_models} models", flush=True)

    



    if cfg.llh_workers > 0:
        pool.close()
        pool.join()

    if latest_payload is not None:
        print(
            f"Producing least-Gaussian parameter plots from epoch {latest_payload['epoch']}.",
            flush=True,
        )

        least_dir = out_dir / "least_gaussian"
        least_marg_dir = least_dir / "marginals"
        least_corr_dir = least_dir / "corr2d_contours"
        least_dir.mkdir(parents=True, exist_ok=True)
        least_marg_dir.mkdir(parents=True, exist_ok=True)
        least_corr_dir.mkdir(parents=True, exist_ok=True)

        samples_plot = latest_payload["samples_nf"]
        log_ratio = latest_payload["log_ratio"]
        labels_plot = latest_payload["labels"]
        mu_plot = latest_payload["mu"]
        sigma_plot = latest_payload["sigma"]

        w_reweighted = _stable_weights_from_logratio(log_ratio)

        ks_records = []
        for idx in range(samples_plot.shape[1]):
            ks_val = _ks_stat_against_gaussian(samples_plot[:, idx], float(mu_plot[idx]), float(sigma_plot[idx]))
            if np.isfinite(ks_val):
                ks_records.append((idx, float(ks_val), labels_plot[idx]))

        ks_records = sorted(ks_records, key=lambda t: t[1], reverse=True)
        n_select = min(10, len(ks_records))
        selected = ks_records[:n_select]
        selected_indices = [x[0] for x in selected]

        with open(least_dir / "least_gaussian_ks.json", "w") as f:
            json.dump(
                {
                    "epoch": int(latest_payload["epoch"]),
                    "top_k": int(n_select),
                    "selected": [
                        {
                            "index": int(i),
                            "name": str(name),
                            "ks_stat": float(score),
                        }
                        for i, score, name in selected
                    ],
                },
                f,
                indent=2,
            )

        marginal_bins = int(getattr(cfg, "least_gaussian_marginal_bins", 60))
        contour_bins = int(getattr(cfg, "least_gaussian_corr2d_bins", 70))

        for i in selected_indices:
            out_path = least_marg_dir / f"least_gaussian_marginal_{i:03d}.png"
            _plot_marginal_nf_vs_reweighted(
                samples_plot[:, i],
                w_reweighted,
                float(mu_plot[i]),
                float(sigma_plot[i]),
                labels_plot[i],
                out_path,
                bins=marginal_bins,
            )

        if len(selected_indices) >= 2:
            for a_pos in range(len(selected_indices)):
                for b_pos in range(a_pos + 1, len(selected_indices)):
                    a = selected_indices[a_pos]
                    b = selected_indices[b_pos]
                    out_path = least_corr_dir / f"least_gaussian_corr2d_{a:03d}_{b:03d}.png"
                    _plot_contours_nf_vs_reweighted(
                        samples_plot[:, a],
                        samples_plot[:, b],
                        w_reweighted,
                        labels_plot[a],
                        labels_plot[b],
                        out_path,
                        bins=contour_bins,
                    )

        print(
            f"Least-Gaussian plots saved in {least_dir} (selected={len(selected_indices)}).",
            flush=True,
        )
    else:
        print("No valid checkpoint payload available for least-Gaussian plotting.", flush=True)

    print("Finished looping over checkpoints.")

    # -------------------------------------------------------------------------
    # Marginal-evolution panels: one figure per scanned parameter, one subpanel
    # per training epoch. Shows NF unweighted vs NF-reweighted marginal so the
    # build-up (or absence) of the reweighting-induced shift is visible across
    # epochs.
    # -------------------------------------------------------------------------
    if marginal_scan_indices:
        scan_dir = out_dir / "marginal_epoch_scan"
        scan_dir.mkdir(parents=True, exist_ok=True)

        # Resolve labels / mu / sigma from any successful checkpoint
        if latest_payload is not None:
            labels_ref = latest_payload["labels"]
            mu_ref = latest_payload["mu"]
            sigma_ref = latest_payload["sigma"]
        else:
            labels_ref = nf_param_names_short
            mu_ref = bestfit_parameter_values
            sigma_ref = np.sqrt(np.clip(np.diag(postfit_covariance), 0.0, np.inf))

        for _idx in marginal_scan_indices:
            data = epoch_marginal_payloads.get(_idx, [])
            if not data:
                print(f"  scan: no data collected for index {_idx}, skipping.", flush=True)
                continue
            data_sorted = sorted(data, key=lambda t: t[0])
            label = labels_ref[_idx] if 0 <= _idx < len(labels_ref) else f"param_{_idx}"
            mu_i = float(mu_ref[_idx]) if 0 <= _idx < len(mu_ref) else float("nan")
            sig_i = float(sigma_ref[_idx]) if 0 <= _idx < len(sigma_ref) else float("nan")
            safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label))
            out_path = scan_dir / f"param_{_idx:03d}_{safe_label}.png"
            _plot_marginal_epoch_panel(
                data_sorted,
                mu_i,
                sig_i,
                str(label),
                out_path,
                bins=marginal_scan_bins,
            )
            print(f"  scan: wrote {out_path}  (epochs={[e for e,_,_ in data_sorted]})", flush=True)

        print(f"Marginal-evolution scan plots saved in {scan_dir}.", flush=True)



if __name__ == "__main__":
    main()



