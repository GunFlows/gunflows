#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Title: sample.py
#  Author: Mathias El Baz
#  Date: 28/01/2025
#  Description:
#    Sample from a Normalizing Flow model and save the samples.
#    Optionally, plot the least Gaussian and grid of samples.
#    Requires a trained model checkpoint.
# =============================================================================

from __future__ import annotations
import math, time, os, sys, json
from pathlib import Path
from datetime import datetime

import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from omegaconf import DictConfig
from hydra.utils import instantiate
from matplotlib.colors import LogNorm

NF_LOCAL = os.path.join(os.path.dirname(__file__), "..", "normalizing-flows")
sys.path.append(os.path.abspath(NF_LOCAL))

from gunflows.utils.build_flow import build_base, build_flow_layers, build_model


def _plot_least_gaussian(samples, mean, cov, names, out_dir, phase_dims):
    ks_stats = [
        kstest(samples[:, i],
               "norm",
               args=(samples[:, i].mean(), samples[:, i].std()))[0]
        for i in range(samples.shape[1])
    ]
    worst = np.argsort(ks_stats)[-10:]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, idx in zip(axes.ravel(), worst):
        print(f"Least Gaussian dim {idx}")
        data = samples[:, idx]
        ax.hist(data, bins=60, density=True, alpha=0.6)
        xs = np.linspace(data.min(), data.max(), 200)
        mean_i = np.mean(data)
        cov_i = np.std(data) ** 2
        pdf = (1 / (np.sqrt(2 * np.pi * cov_i))) * np.exp(
            -0.5 * ((xs - mean_i) ** 2) / cov_i
        )
        ax.plot(xs, pdf, lw=2, color="r")
        ax.set_title(names[idx], fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "least_gaussian.png", dpi=150)
    plt.close(fig)


def _plot_grid(samples, mean, cov, names, n, out_dir, phase_dims):
    samples = samples[:, phase_dims]
    mean = mean[phase_dims]
    cov = cov[np.ix_(phase_dims, phase_dims)]
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                x = samples[:, i]
                ax.hist(x, bins=60, density=True, histtype="step")
                xs = np.linspace(x.min(), x.max(), 200)
                mean_i = np.mean(x)
                std_i = np.std(x)
                pdf = (1 / (np.sqrt(2 * np.pi * std_i**2))) * np.exp(
                    -0.5 * ((xs - mean_i) ** 2) / std_i**2
                )
                ax.plot(xs, pdf, color="r")
            else:
                ax.hist2d(samples[:, j], samples[:, i], bins=60, norm=LogNorm())
            if i == n - 1:
                ax.set_xlabel(names[j], fontsize=7)
            if j == 0:
                ax.set_ylabel(names[i], fontsize=7)
            ax.tick_params(axis="both", labelsize=6)
    plt.tight_layout()
    plt.savefig(out_dir / "grid.png", dpi=150)
    plt.close(fig)


@hydra.main(config_path="../../configs", config_name="sample", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)

    ckpt_path = Path(cfg.ckpt).expanduser().resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(cfg.save_dir).expanduser()
        if cfg.save_dir is not None
        else ckpt_path.parent.parent / "samples" / ts
    )
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    dataset = instantiate(cfg.dataset)
    phase_dims = dataset.phase_space_dim
    dim_spline = len(phase_dims)
    names = [dataset.titles[i].split("/")[-1] for i in range(cfg.model.total_dim)]

    base = build_base(cfg.model.total_dim)
    tail_bounds = torch.ones(dim_spline) * cfg.model.tail_bound
    flows = build_flow_layers(
        cfg.model.nflows,
        dim_spline,
        cfg.model.hidden,
        cfg.model.nlayers,
        cfg.model.nbins,
        tail_bounds,
        n_context=cfg.model.total_dim - dim_spline,
    )
    model = build_model(base, flows, dataset, cfg.model.context_transform)
    model.load_state_dict(torch.load(cfg.ckpt, map_location=cfg.device))
    model = model.to(cfg.device).eval()

    batches = math.ceil(cfg.num_samples / cfg.batch_size)
    samples, logqs = [], []
    start = time.time()
    with torch.no_grad():
        for _ in range(batches):
            z, logq = model.sample(cfg.batch_size)
            samples.append(z.cpu().numpy())
            if cfg.return_probs:
                logqs.append(logq.cpu().numpy())
    samples = np.concatenate(samples, 0)[: cfg.num_samples]
    if cfg.return_probs:
        logqs = np.concatenate(logqs, 0)[: cfg.num_samples]

    samples = dataset.transform_eigen_space_to_data_space(torch.from_numpy(samples)).numpy()
    mean_sample = dataset.mean.numpy()
    cov_sample = dataset.get_true_cov().numpy()

    dur = time.time() - start
    print(f"Done: {cfg.num_samples} samples in {dur:.1f}s")

    _plot_least_gaussian(samples, mean_sample, cov_sample, names, img_dir, phase_dims)
    _plot_grid(samples, mean_sample, cov_sample, names, cfg.grid, img_dir, phase_dims)

    np.save(out_dir / "samples.npy", samples)
    if cfg.return_probs:
        np.save(out_dir / "logq.npy", logqs)

    meta = {"checkpoint": cfg.ckpt, "n_samples": int(cfg.num_samples), "time_s": dur}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
