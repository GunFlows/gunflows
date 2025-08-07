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


def _dummy_work(seconds: float = 2.0):
    t0 = time.time()
    x = torch.randn(2048, 512)
    w = torch.randn(512, 512)
    while time.time() - t0 < seconds:
        x = torch.tanh(x @ w)
        _ = float(x.mean())


def _recv(q: mp.Queue, timeout: float):
    t0 = time.time()
    while True:
        try:
            return q.get_nowait()
        except _q.Empty:
            if time.time() - t0 > timeout:
                raise TimeoutError("Timed out waiting for sampler payload.")
            time.sleep(0.05)


def main():
    p = argparse.ArgumentParser()
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
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    ds = SystematicDataset(
        phase_space_dim=args.phase_space_dim,
        starting_folder=None,
        with_sampler=True,
        config_file=args.llh_config,
        overrides=args.llh_overrides,
        load_data=False,
    )
    ds.replace_from_dict(payload1)
    _plot_grid(ds.data.cpu().numpy(), ds.titles, out_dir / "plot_cov", start_dim=args.start_dim, ndim=args.ndim_plot)
    print(f"[test] covariance batch: {len(ds)} samples")

    _dummy_work(2.0)

    if args.nf_ckpt:
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
    print("[test] done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
