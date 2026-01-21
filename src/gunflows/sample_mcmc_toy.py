#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Title: sample_nf_mcmc_toy.py
#  Author: Mathias El Baz (adapted version of sample_mcmc.py for the Toy OA configuration)
#  Date: 21/01/2026
#  Description:
#    Sample from a trained Normalizing Flow model and compare MCMC samples.
#    This uses GUNDAM format for the MCMC output.
# =============================================================================

from __future__ import annotations

import os
import sys
import re
import time
from pathlib import Path
from typing import List, Optional

import hydra
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import ROOT

NF_LOCAL = os.path.join(os.path.dirname(__file__), "..", "normalizing-flows")
sys.path.append(os.path.abspath(NF_LOCAL))

from gunflows.utils.build_flow import build_base, build_flow_layers, build_model
from gunflows.likelihood_sampler import LikelihoodSampler


def _abspath(p: str) -> str:
    return str(Path(p).expanduser().resolve())


def _strip_common_prefixes(s: str) -> str:
    s2 = str(s).strip()
    s2 = re.sub(r"^Linear Systematics\/", "", s2)
    s2 = re.sub(r"^Systematics\/", "", s2)
    s2 = re.sub(r"^Parameters\/", "", s2)
    return s2


def _short(s: str, n: int = 65) -> str:
    s = str(s)
    return s if len(s) <= n else (s[: n - 3] + "...")


def check_parameters_array_limits(array_of_param_vector: np.ndarray, limits_dictionary: dict[str, tuple[float, float]]) -> np.ndarray:
    par_names = list(limits_dictionary.keys())
    limits_vector = [limits_dictionary[name] for name in par_names]
    mask = np.ones(array_of_param_vector.shape[0], dtype=bool)
    for i, limits in enumerate(limits_vector):
        low, high = limits
        if np.isnan(low):
            low = -np.inf
        if np.isnan(high):
            high = np.inf
        vals = array_of_param_vector[:, i]
        mask &= (vals >= low) & (vals <= high)
    return mask


def sample_nf_physical(model, dataset, parameter_limits: dict[str, tuple[float, float]], num_samples: int, batch_size: int) -> np.ndarray:
    samples_nf: list[np.ndarray] = []
    need_total = int(num_samples)

    with torch.no_grad():
        while len(samples_nf) < need_total:
            need = need_total - len(samples_nf)
            b = min(int(batch_size), need)
            z_batch, _ = model.sample(b)
            z_batch = z_batch.detach().to(dtype=torch.float32, device="cpu")
            phys_batch = dataset.transform_eigen_space_to_data_space(z_batch)
            phys_np = phys_batch.detach().cpu().numpy().astype(np.float32)

            mask = check_parameters_array_limits(phys_np, parameter_limits)
            accepted = phys_np[mask]
            for row in accepted:
                samples_nf.append(row)

            print(f" NF: accepted {int(mask.sum())}/{b}  -> total {len(samples_nf)}/{need_total}", flush=True)

            if int(mask.sum()) == 0 and b <= 16:
                for _ in range(128):
                    if len(samples_nf) >= need_total:
                        break
                    z1, _ = model.sample(1)
                    z1 = z1.detach().to(dtype=torch.float32, device="cpu")
                    phys1 = dataset.transform_eigen_space_to_data_space(z1).detach().cpu().numpy()[0].astype(np.float32)
                    m1 = check_parameters_array_limits(phys1[None, :], parameter_limits)[0]
                    if m1:
                        samples_nf.append(phys1)

    return np.asarray(samples_nf[:need_total], dtype=np.float32)


def walk_dirs(tfile: ROOT.TFile) -> list[str]:
    dirs: list[str] = []

    def rec(d, path: str):
        keys = d.GetListOfKeys()
        if not keys:
            return
        for i in range(keys.GetSize()):
            k = keys.At(i)
            if str(k.GetClassName()).startswith("TDirectory"):
                sub = d.Get(k.GetName())
                if sub:
                    subpath = f"{path}/{k.GetName()}" if path else str(k.GetName())
                    dirs.append(subpath)
                    rec(sub, subpath)

    rec(tfile, "")
    return dirs


def list_dir_keys(d) -> list[tuple[str, int, str]]:
    keys = d.GetListOfKeys()
    out: list[tuple[str, int, str]] = []
    if not keys:
        return out
    for i in range(keys.GetSize()):
        k = keys.At(i)
        out.append((str(k.GetName()), int(k.GetCycle()), str(k.GetClassName())))
    return out


def find_dirs_containing(tfile: ROOT.TFile, objname: str) -> list[str]:
    out: list[str] = []
    for dpath in walk_dirs(tfile):
        d = tfile.GetDirectory(dpath)
        if not d:
            continue
        for name, _, _ in list_dir_keys(d):
            if name == objname:
                out.append(dpath)
                break
    return out


def infer_base_dir(tfile: ROOT.TFile) -> str:
    pdirs = set(find_dirs_containing(tfile, "parameterSets"))
    mdirs = set(find_dirs_containing(tfile, "MCMC"))
    inter = sorted(list(pdirs & mdirs), key=lambda s: s.count("/"), reverse=True)
    if not inter:
        raise RuntimeError("Could not find a directory that contains both 'parameterSets' and 'MCMC'.")
    return inter[0]


def get_latest_obj_in_dir(tfile: ROOT.TFile, dirname: str, objname: str):
    d = tfile.GetDirectory(dirname)
    if not d:
        raise RuntimeError(f"Directory not found in ROOT file: {dirname}")
    best = -1
    for name, cyc, _ in list_dir_keys(d):
        if name == objname:
            best = max(best, cyc)
    if best < 0:
        raise RuntimeError(f"Object '{objname}' not found in directory '{dirname}'.")
    obj = d.Get(f"{objname};{best}")
    return obj, int(best)


def read_param_names_parameterSets(tfile: ROOT.TFile, base_dir: str) -> tuple[list[str], int]:
    t, cyc = get_latest_obj_in_dir(tfile, base_dir, "parameterSets")
    t.GetEntry(0)
    v = getattr(t, "parameterName")
    names = [str(v.at(i)) for i in range(int(v.size()))]
    return names, cyc


def read_points_vector_tree(t_mcmc, max_steps: int | None) -> np.ndarray:
    n = int(t_mcmc.GetEntries())
    if max_steps is not None:
        n = min(n, int(max_steps))
    if not t_mcmc.GetBranch("Points"):
        raise RuntimeError("MCMC tree has no 'Points' branch.")
    t_mcmc.GetEntry(0)
    d = int(getattr(t_mcmc, "Points").size())
    pts = np.empty((n, d), dtype=np.float64)
    for i in range(n):
        t_mcmc.GetEntry(i)
        v = getattr(t_mcmc, "Points")
        for j in range(d):
            pts[i, j] = float(v.at(j))
    return pts


def load_mcmc_gundamworkspace(input_root: str, max_steps: int | None) -> tuple[str, list[str], np.ndarray, dict]:
    tf = ROOT.TFile.Open(input_root, "READ")
    if not tf or tf.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {input_root}")

    base_dir = infer_base_dir(tf)
    names, pcyc = read_param_names_parameterSets(tf, base_dir)
    t_mcmc, mcyc = get_latest_obj_in_dir(tf, base_dir, "MCMC")
    pts = read_points_vector_tree(t_mcmc, max_steps)

    meta = {"base_dir": base_dir, "parameterSets_cycle": pcyc, "MCMC_cycle": mcyc}
    return base_dir, names, pts, meta


def apply_burnin_thin(pts: np.ndarray, burnin_frac: float, thin: int) -> tuple[np.ndarray, int]:
    nsteps = int(pts.shape[0])
    burn = int(max(0, min(nsteps, burnin_frac * nsteps)))
    thin = max(1, int(thin))
    post = pts[burn:nsteps:thin]
    return post, burn


def parse_dim_list(cfg_val, ndim: int) -> list[int]:
    if cfg_val is None:
        return []
    if isinstance(cfg_val, (list, tuple)):
        out = []
        for x in cfg_val:
            try:
                out.append(int(x))
            except Exception:
                pass
        out = [i for i in out if 0 <= i < ndim]
        seen = set()
        out2 = []
        for i in out:
            if i not in seen:
                seen.add(i)
                out2.append(i)
        return out2
    s = str(cfg_val).strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":")
            a = int(a) if a else 0
            b = int(b) if b else ndim
            out.extend(list(range(max(0, a), min(ndim, b))))
        else:
            out.append(int(part))
    out = [i for i in out if 0 <= i < ndim]
    seen = set()
    out2 = []
    for i in out:
        if i not in seen:
            seen.add(i)
            out2.append(i)
    return out2


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0 or not np.isfinite(sigma):
        return np.zeros_like(x, dtype=np.float64)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def plot_2d_hist_side_by_side(x_nf: np.ndarray, y_nf: np.ndarray, x_mc: np.ndarray, y_mc: np.ndarray,
                              xlabel: str, ylabel: str, outpath: Path, bins: int = 60) -> None:
    xmin = float(min(np.min(x_nf), np.min(x_mc)))
    xmax = float(max(np.max(x_nf), np.max(x_mc)))
    ymin = float(min(np.min(y_nf), np.min(y_mc)))
    ymax = float(max(np.max(y_nf), np.max(y_mc)))

    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        xmin, xmax = 0.0, 1.0
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin:
        ymin, ymax = 0.0, 1.0

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.hist2d(x_nf, y_nf, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    ax1.set_title("NF")
    ax1.set_xlabel(_short(xlabel, 45))
    ax1.set_ylabel(_short(ylabel, 45))

    ax2.hist2d(x_mc, y_mc, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    ax2.set_title("MCMC")
    ax2.set_xlabel(_short(xlabel, 45))
    ax2.set_ylabel(_short(ylabel, 45))

    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


@hydra.main(config_path="../../configs", config_name="sample_mcmc_nf_toyOA", version_base=None)
def main(cfg: DictConfig) -> None:
    training_folder = _abspath(str(cfg.training_folder))
    mcmc_root = _abspath(str(cfg.mcmc_chain))
    save_dir = _abspath(str(cfg.save_dir))

    print(f"PWD (hydra chdir): {os.getcwd()}", flush=True)
    print(f"training_folder: {training_folder}", flush=True)
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

    ckpt_folder = os.path.join(training_folder, "checkpoints")
    pattern = re.compile(r"sampler_epoch(\d+)\.pt")
    max_file = None
    max_epoch = -1
    for fname in os.listdir(ckpt_folder):
        m = pattern.match(fname)
        if m:
            ep = int(m.group(1))
            if ep > max_epoch:
                max_epoch = ep
                max_file = fname
    if not max_file:
        raise RuntimeError(f"No checkpoints found in {ckpt_folder} matching {pattern.pattern}")
    ckpt_path = Path(os.path.join(ckpt_folder, max_file))
    print("Using latest NF model:", ckpt_path, flush=True)

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "marginals"
    img_dir.mkdir(parents=True, exist_ok=True)
    corr2d_dir = out_dir / "corr2d"
    corr2d_dir.mkdir(parents=True, exist_ok=True)

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

    num_samples = int(cfg.num_samples)
    batch_size = int(cfg.batch_size)

    print(f"Sampling {num_samples} events from NF (physical space)...", flush=True)
    t0 = time.time()
    samples_nf = sample_nf_physical(model, dataset, parameter_limits, num_samples, batch_size)
    print(f"NF sampling done: {samples_nf.shape} in {time.time()-t0:.1f}s", flush=True)

    print(f"Loading ToyNDFit MCMC from: {mcmc_root}", flush=True)
    mcmc_max_steps = cfg.mcmc_max_steps if cfg.mcmc_max_steps is not None else None
    mcmc_burnin_frac = float(cfg.mcmc_burnin_frac)
    mcmc_thin = int(cfg.mcmc_thin)

    base_dir, mcmc_names, mcmc_pts_raw, meta = load_mcmc_gundamworkspace(mcmc_root, mcmc_max_steps)
    mcmc_post, burn = apply_burnin_thin(mcmc_pts_raw, mcmc_burnin_frac, mcmc_thin)
    print(f"MCMC loaded: raw {mcmc_pts_raw.shape} -> post {mcmc_post.shape} (burn={burn}, thin={mcmc_thin})", flush=True)
    print(f"MCMC base_dir: {base_dir}  meta: {meta}", flush=True)

    d_nf = int(samples_nf.shape[1])
    d_mcmc = int(mcmc_post.shape[1])
    d_fit = int(bestfit_parameter_values.shape[0])
    d_cov = int(postfit_covariance.shape[0])
    d_names = len(nf_param_names_short)
    d = min(d_nf, d_mcmc, d_fit, d_cov, d_names)
    if d == 0:
        raise RuntimeError("Cannot compare: zero dimension after resolving NF/MCMC/postfit shapes.")

    samples_nf_c = samples_nf[:, :d]
    samples_mcmc_c = mcmc_post[:, :d]
    mu_vec = bestfit_parameter_values[:d]
    cov_mat = postfit_covariance[:d, :d]
    sig_vec = np.sqrt(np.clip(np.diag(cov_mat), 0.0, np.inf))
    labels = nf_param_names_short[:d]

    with open(out_dir / "report.txt", "w") as f:
        f.write(f"pwd: {os.getcwd()}\n")
        f.write(f"training_folder: {training_folder}\n")
        f.write(f"checkpoint: {str(ckpt_path)}\n")
        f.write(f"mcmc_root: {mcmc_root}\n")
        f.write(f"mcmc_base_dir: {base_dir}\n")
        f.write(f"mcmc_meta: {meta}\n")
        f.write(f"nf_samples: {samples_nf.shape}\n")
        f.write(f"mcmc_raw: {mcmc_pts_raw.shape}\n")
        f.write(f"mcmc_post: {mcmc_post.shape}\n")
        f.write(f"burnin_frac: {mcmc_burnin_frac}\n")
        f.write(f"thin: {mcmc_thin}\n")
        f.write(f"matching_mode: index_order_only\n")
        f.write(f"compared_dims: {d}\n")
        for i in range(d):
            f.write(f"  {i:03d} {labels[i]}\n")

    bins_n = int(cfg.bins)
    print(f"Comparing {d} parameters.", flush=True)
    print("Plotting marginals sequentially.", flush=True)

    for k in range(d):
        name = labels[k]
        x_m = samples_mcmc_c[:, k]
        x_n = samples_nf_c[:, k]
        mu = float(mu_vec[k])
        sig = float(sig_vec[k])

        xmin = float(min(np.min(x_m), np.min(x_n), mu))
        xmax = float(max(np.max(x_m), np.max(x_n), mu))
        if np.isfinite(sig) and sig > 0:
            xmin = min(xmin, mu - 5.0 * sig)
            xmax = max(xmax, mu + 5.0 * sig)

        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            xmin, xmax = 0.0, 1.0

        span = xmax - xmin
        xmin -= 0.02 * span
        xmax += 0.02 * span

        edges = np.linspace(xmin, xmax, bins_n + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        xgrid = np.linspace(xmin, xmax, 400)

        fig = plt.figure(figsize=(6, 4))
        plt.hist(x_m, bins=edges, histtype="step", density=True, label=f"MCMC (n={len(x_m)})")
        plt.hist(x_n, bins=edges, histtype="step", density=True, label=f"NF (n={len(x_n)})")

        pdf = gaussian_pdf(xgrid, mu, sig)
        if pdf.max() > 0:
            plt.plot(xgrid, pdf, label="Post frequentist fit gaussian", linewidth=1.2)

        plt.axvline(mu, linestyle="--", linewidth=1.2, label="Frequentist best fit")

        plt.xlabel(_short(name, 55))
        plt.ylabel("a.u.")
        plt.title(f"Marginal: {_short(name, 90)}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)

        out_path = img_dir / f"marginal_{k:03d}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

        if (k + 1) % 10 == 0 or (k + 1) == d:
            print(f"  Plotted {k+1}/{d}", flush=True)

    corr2d_bins = int(getattr(cfg, "corr2d_bins", 60))
    dims_cfg = getattr(cfg, "corr2d_dims", None)

    if dims_cfg is None or (isinstance(dims_cfg, (list, tuple)) and len(dims_cfg) == 0) or (isinstance(dims_cfg, str) and dims_cfg.strip() == ""):
        dims = list(range(max(0, d - 5), d))
    else:
        dims = parse_dim_list(dims_cfg, d)

    if len(dims) >= 2:
        print(f"Plotting 2D correlations for dims: {dims}", flush=True)
        for i in range(len(dims)):
            for j in range(i + 1, len(dims)):
                a = dims[i]
                b = dims[j]
                outp = corr2d_dir / f"corr2d_{a:03d}_{b:03d}.png"
                plot_2d_hist_side_by_side(
                    samples_nf_c[:, a], samples_nf_c[:, b],
                    samples_mcmc_c[:, a], samples_mcmc_c[:, b],
                    labels[a], labels[b],
                    outp,
                    bins=corr2d_bins
                )
    else:
        print("corr2d_dims has <2 dims, skipping 2D plots.", flush=True)

    print(f"Done. Outputs in: {str(out_dir)}", flush=True)


if __name__ == "__main__":
    main()
