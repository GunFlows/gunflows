#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from gunflows.dataset.streaming_dataset import StreamingDataset as SystematicDataset


def _plot_grid(samples, names, out_dir, start_dim=0, ndim=8):
    phase = range(start_dim, start_dim + ndim)
    sub = samples[:, phase]
    mean = sub.mean(axis=0)
    cov = np.cov(sub, rowvar=False)

    fig, ax = plt.subplots(ndim, ndim, figsize=(3 * ndim, 3 * ndim))
    for i in range(ndim):
        for j in range(ndim):
            a = ax[i, j]
            if i == j:
                x = sub[:, i]
                a.hist(x, bins=60, density=True, histtype="step")
                xs = np.linspace(x.min(), x.max(), 200)
                s = np.sqrt(cov[i, i])
                a.plot(xs, np.exp(-0.5 * (xs - mean[i]) ** 2 / s**2) / (np.sqrt(2 * np.pi) * s))
            else:
                a.hist2d(sub[:, j], sub[:, i], bins=60)
            if i == ndim - 1:
                a.set_xlabel(str(names[phase[j]]), fontsize=7)
            if j == 0:
                a.set_ylabel(str(names[phase[i]]), fontsize=7)
            a.tick_params(axis="both", labelsize=6)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "grid.png", dpi=150)
    plt.close(fig)


def _plot_nll(log_p, log_q, out_dir):
    x = (-log_q).reshape(-1)
    y = (-log_p).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    fig, ax = plt.subplots(figsize=(6, 5))
    h = ax.hist2d(x, y, bins=120, norm=LogNorm())
    ax.set_xlabel("-log q")
    ax.set_ylabel("-log p")
    cb = fig.colorbar(h[3], ax=ax)
    cb.set_label("count")
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, "k--", linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "nll_hist.png", dpi=150)
    plt.close(fig)


def _dummy_work(seconds: float = 2.0):
    t0 = time.time()
    x = torch.randn(2048, 512)
    w = torch.randn(512, 512)
    while time.time() - t0 < seconds:
        x = torch.tanh(x @ w)
        _ = float(x.mean())


def _wait_next(ds: SystematicDataset, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if ds.refresh_if_ready(plot_grid=False):
            return True
        time.sleep(0.05)
    return False
# Useful override : "/workspace/config/GundamInputOA2021/override/onlyRun4and5.yaml" -> 5 times less data I guess because 5 times faster

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llh-config", type=str, default="/workspace/config/GundamInputOA2021/output/gundamFitter_configOa2021_With_allowEigenDecompWithBounds_Asimov.root")
    p.add_argument("--llh-overrides", type=str, default=["/workspace/config/GundamInputOA2021/override/onlyRun4and5.yaml"])
    p.add_argument("--llh-cwd", type=str, default="/workspace/config/GundamInputOA2021")
    p.add_argument("--nf-ckpt", type=str, default=None)
    p.add_argument("--gen-batch-size", type=int, default=1000)
    p.add_argument("--phase-space-dim", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", type=str, default="tests/outputs/streaming_cov_first")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-dim", type=int, default=0)
    p.add_argument("--ndim-plot", type=int, default=6)
    p.add_argument("--timeout", type=float, default=6000.0)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = SystematicDataset(
        phase_space_dim=args.phase_space_dim,
        starting_folder=None,
        with_sampler=True,
        nf_ckpt=None,
        gen_batch_size=args.gen_batch_size,
        llh_config=args.llh_config,
        llh_overrides=args.llh_overrides,
        llh_cwd=args.llh_cwd,
        seed=args.seed,
        queue_size=2,
        plot_grid=False,
        out_dir=out_dir / "plots",
        load_data=False,
        timeout=args.timeout,
        shift_log_p=False,
    )

    _plot_grid(ds.data.cpu().numpy(), ds.titles, out_dir / "plot_cov",
               start_dim=args.start_dim, ndim=args.ndim_plot)
    _plot_nll(ds.log_p.detach().cpu().numpy(), ds.log_q.detach().cpu().numpy(), out_dir / "plot_cov")
    print(f"[test] covariance batch: {len(ds)}")

    _dummy_work(2.0)

    if args.nf_ckpt:
        ds.request_switch_to_nf(args.nf_ckpt)

    if not _wait_next(ds, args.timeout):
        raise TimeoutError("Timed out waiting for next batch from sampler.")

    tag = "nf" if args.nf_ckpt else "cov2"
    _plot_grid(ds.data.cpu().numpy(), ds.titles, out_dir / f"plot_{tag}",
               start_dim=args.start_dim, ndim=args.ndim_plot)
    _plot_nll(ds.log_p.detach().cpu().numpy(), ds.log_q.detach().cpu().numpy(), out_dir / f"plot_{tag}")
    print(f"[test] {tag} batch: {len(ds)}")

    ds.close()
    print("[test] done.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()

