#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import ROOT, GUNDAM

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC))

from gunflows.likelihood_sampler.likelihoodSampler import LikelihoodSampler


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run LikelihoodSampler throws and save basic diagnostics."
    )
    p.add_argument("-cfg", type=str, default="/workspace/config/GundamInputOA2021/output/gundamFitter_configOa2021_With_allowEigenDecompWithBounds_Asimov_ToyFit.root")    
    p.add_argument("config")
    p.add_argument("-o", "--override", action="append", default=[])
    p.add_argument("-n", type=int, default=1_000)
    p.add_argument("--asimov", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    return p.parse_args()


def _plot_grid(samples, names, out_dir, start_dim=660, ndim=8):
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
                a.plot(xs, np.exp(-0.5 * (xs - mean[i]) ** 2 / s**2) / (np.sqrt(2 * np.pi) * s), c="r")
            else:
                a.hist2d(sub[:, j], sub[:, i], bins=60)
            if i == ndim - 1:
                a.set_xlabel(names[phase[j]], fontsize=7)
            if j == 0:
                a.set_ylabel(names[phase[i]], fontsize=7)
            a.tick_params(axis="both", labelsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "grid.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_cli()

    os.chdir("/workspace/config/GundamInputOA2021")

    lh = LikelihoodSampler(
<<<<<<< HEAD
        args.cfg,
=======
        args.config,
>>>>>>> 650826f (test files)
        override_files=args.override,
        threads=args.threads,
        data_is_asimov=args.asimov,
    )

    params, weights, nll = lh.throw_n_from_covariance(args.n)
    weights = np.array(weights)
    nll = np.array(nll)
    params = np.array(params)
    weights = np.sum(weights, axis=1) if weights.ndim > 1 else weights
    weights = np.exp(-nll + weights)

    outdir = Path("/workspace/work/GuNFlows/tests/img")
    outdir.mkdir(exist_ok=True)

    plt.figure()
    plt.hist(nll, bins="auto")
    plt.xlabel("−logLLH")
    plt.ylabel("entries")
    plt.tight_layout()
    plt.savefig(outdir / "nll_hist.png", dpi=150)
    plt.close()

    if params.shape[1] >= 2:
        names = lh.get_parameter_names()

        plt.figure()
        plt.scatter(params[:, 0], params[:, 1], s=5, alpha=0.5)
        plt.xlabel(names[0])
        plt.ylabel(names[1])
        plt.tight_layout()
        plt.savefig(outdir / "param_scatter.png", dpi=150)
        plt.close()

        plt.figure()
        plt.hist2d(params[:, 0], params[:, 1], bins=50, weights=weights, cmap="Blues")
        plt.colorbar(label="Weighted counts")
        plt.xlabel(names[0])
        plt.ylabel(names[1])
        plt.tight_layout()
        plt.savefig(outdir / "param_2d_hist.png", dpi=150)
        plt.close()

        _plot_grid(params, names, outdir)  # default start_dim=660, ndim=8


if __name__ == "__main__":
    main()
