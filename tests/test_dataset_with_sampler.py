#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, time
from pathlib import Path
import multiprocessing as mp, queue as _q
import numpy as np
import torch
import matplotlib.pyplot as plt

from gunflows.dataset.systematic_dataset_oa2022 import SystematicDatasetOA2022 as SystematicDataset
from gunflows.likelihood_sampler.nf_llh_sampler import NFSamplerProcess



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


<<<<<<< HEAD
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


=======
>>>>>>> 650826f (test files)
def _dummy_work(seconds: float = 2.0):
    t0 = time.time()
    x = torch.randn(2048, 512)
    w = torch.randn(512, 512)
    while time.time() - t0 < seconds:
        x = torch.tanh(x @ w)
        _ = float(x.mean())


<<<<<<< HEAD
def _wait_next(ds: SystematicDataset, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if ds.refresh_if_ready(plot_grid=False):
            return True
        time.sleep(0.05)
    return False
=======
def _recv(q: mp.Queue, timeout: float):
    t0 = time.time()
    while True:
        try:
            return q.get_nowait()
        except _q.Empty:
            if time.time() - t0 > timeout:
                raise TimeoutError("Timed out waiting for sampler payload.")
            time.sleep(0.05)
>>>>>>> 650826f (test files)


def main():
    p = argparse.ArgumentParser()
<<<<<<< HEAD
    p.add_argument("--llh-config", type=str, required=True)
    p.add_argument("--llh-overrides", type=str, nargs="*", default=[])
    p.add_argument("--llh-cwd", type=str, required=True)
    p.add_argument("--nf-ckpt", type=str, default=None)
    p.add_argument("--gen-batch-size", type=int, default=1000)
    p.add_argument("--phase-space-dim", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", type=str, default="tests/outputs/streaming_cov_first")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-dim", type=int, default=0)
    p.add_argument("--ndim-plot", type=int, default=6)
    p.add_argument("--timeout", type=float, default=6000.0)
=======
    p.add_argument("--llh-config", type=str, default="/workspace/config/GundamInputOA2021/output/gundamFitter_configOa2021_With_allowEigenDecompWithBounds_Asimov_ToyFit.root")    
    p.add_argument("--llh-overrides", type=str, nargs="*", default=[])
    p.add_argument("--nf-ckpt", type=str, default=None)      # optional: switch to NF later
    p.add_argument("--gen-batch-size", type=int, default=1000)
    p.add_argument("--phase-space-dim", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", type=str, default="/workspace/work/GuNFlows/tests/outputs/streaming_cov_first")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-dim", type=int, default=660)
    p.add_argument("--ndim-plot", type=int, default=6)
    p.add_argument("--timeout", type=float, default=3000.0)
    p.add_argument("--llh-cwd", type=str, default="/workspace/config/GundamInputOA2021")
>>>>>>> 650826f (test files)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

<<<<<<< HEAD
=======
    data_q, cmd_q, stop_evt = mp.Queue(maxsize=2), mp.Queue(), mp.Event()

    gen = NFSamplerProcess(
        nf_ckpt=args.nf_ckpt,                   # None → covariance first
        n_points=args.gen_batch_size,
        llh_config=args.llh_config,
        llh_overrides=args.llh_overrides,
        phase_space_dim=args.phase_space_dim,
        data_q=data_q,
        cmd_q=cmd_q,
        stop_evt=stop_evt,
        seed=args.seed,
        llh_cwd=args.llh_cwd,
    )
    gen.start()

    payload1 = _recv(data_q, args.timeout)      # covariance batch
>>>>>>> 650826f (test files)
    ds = SystematicDataset(
        phase_space_dim=args.phase_space_dim,
        starting_folder=None,
        with_sampler=True,
<<<<<<< HEAD
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
=======
        config_file=args.llh_config,
        overrides=args.llh_overrides,
        load_data=False,
    )
    ds.replace_from_dict(payload1)
    _plot_grid(ds.data.cpu().numpy(), ds.titles, out_dir / "plot_cov", start_dim=args.start_dim, ndim=args.ndim_plot)
    print(f"[test] covariance batch: {len(ds)} samples")
>>>>>>> 650826f (test files)

    _dummy_work(2.0)

    if args.nf_ckpt:
<<<<<<< HEAD
        ds.request_switch_to_nf(args.nf_ckpt)

    if not _wait_next(ds, args.timeout):
        raise TimeoutError("Timed out waiting for next batch from sampler.")

    tag = "nf" if args.nf_ckpt else "cov2"
    _plot_grid(ds.data.cpu().numpy(), ds.titles, out_dir / f"plot_{tag}",
               start_dim=args.start_dim, ndim=args.ndim_plot)
    _plot_nll(ds.log_p.detach().cpu().numpy(), ds.log_q.detach().cpu().numpy(), out_dir / f"plot_{tag}")
    print(f"[test] {tag} batch: {len(ds)}")

    ds.close()
=======
        cmd_q.put(f"reload:{args.nf_ckpt}")

    payload2 = _recv(data_q, args.timeout)      # next batch (NF if ckpt provided, else covariance again)
    ds.replace_from_dict(payload2)
    tag = "nf" if args.nf_ckpt else "cov2"
    _plot_grid(ds.data.cpu().numpy(), ds.titles, out_dir / f"plot_{tag}", start_dim=args.start_dim, ndim=args.ndim_plot)
    print(f"[test] {tag} batch: {len(ds)} samples")

    stop_evt.set()
    gen.join(timeout=10.0)
    if gen.is_alive():
        gen.terminate()
>>>>>>> 650826f (test files)
    print("[test] done.")


if __name__ == "__main__":
<<<<<<< HEAD
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()

=======
    mp.set_start_method("spawn", force=True)
    main()
>>>>>>> 650826f (test files)
