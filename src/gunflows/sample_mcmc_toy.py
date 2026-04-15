#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Title: sample_nf_mcmc_toy.py
#  Author: Mathias El Baz (adapted version of sample_mcmc.py for the Toy OA configuration)
#  Date: 21/01/2026
#  Description:
#    Sample from a trained Normalizing Flow model and optionally compare MCMC samples.
#    This uses GUNDAM format for the MCMC output.
# =============================================================================

from __future__ import annotations

import os
import sys
import re
import time
from pathlib import Path
import multiprocessing as mp

import hydra
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

import ROOT

NF_LOCAL = os.path.join(os.path.dirname(__file__), "..", "normalizing-flows")
sys.path.append(os.path.abspath(NF_LOCAL))

from gunflows.utils.build_flow import build_base, build_flow_layers, build_model
from gunflows.likelihood_sampler import LikelihoodSampler


_REWEIGHT_LIKELIHOOD_SAMPLER = None


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


def sample_nf_physical_with_logq(
    model,
    dataset,
    parameter_limits: dict[str, tuple[float, float]],
    num_samples: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample from NF and return:
      - samples in physical/data space (N, D)
      - logq from model.sample corresponding to the accepted samples (N,)
    This mirrors the "return_probs" pathway of sample_mcmc.py.
    """
    samples_nf: list[np.ndarray] = []
    logqs_nf: list[float] = []

    need_total = int(num_samples)

    with torch.no_grad():
        while len(samples_nf) < need_total:
            need = need_total - len(samples_nf)
            b = min(int(batch_size), need)

            z_batch, lq_batch = model.sample(b)
            lq_np = lq_batch.detach().to(dtype=torch.float32, device="cpu").cpu().numpy()
            lq_np = np.asarray(lq_np).reshape(-1)

            z_batch = z_batch.detach().to(dtype=torch.float32, device="cpu")
            phys_batch = dataset.transform_eigen_space_to_data_space(z_batch)
            phys_np = phys_batch.detach().cpu().numpy().astype(np.float32)

            mask = check_parameters_array_limits(phys_np, parameter_limits)
            acc_idx = np.nonzero(mask)[0]

            for idx in acc_idx:
                samples_nf.append(phys_np[idx])
                logqs_nf.append(float(lq_np[idx]))

            print(f" NF: accepted {int(mask.sum())}/{b}  -> total {len(samples_nf)}/{need_total}", flush=True)

            if int(mask.sum()) == 0 and b <= 16:
                for _ in range(128):
                    if len(samples_nf) >= need_total:
                        break
                    z1, lq1 = model.sample(1)
                    z1 = z1.detach().to(dtype=torch.float32, device="cpu")
                    phys1 = dataset.transform_eigen_space_to_data_space(z1).detach().cpu().numpy()[0].astype(np.float32)
                    m1 = check_parameters_array_limits(phys1[None, :], parameter_limits)[0]
                    if m1:
                        samples_nf.append(phys1)
                        logqs_nf.append(float(lq1.detach().to(device="cpu").cpu().numpy().reshape(-1)[0]))

    s = np.asarray(samples_nf[:need_total], dtype=np.float32)
    lq = np.asarray(logqs_nf[:need_total], dtype=np.float32).reshape(-1)
    return s, lq


def _init_reweight_worker(llh_config, llh_overrides, data_is_asimov, threads, llh_cwd):
    global _REWEIGHT_LIKELIHOOD_SAMPLER
    _REWEIGHT_LIKELIHOOD_SAMPLER = LikelihoodSampler(
        config_file=llh_config,
        override_files=llh_overrides,
        data_is_asimov=data_is_asimov,
        threads=threads,
        llh_cwd=llh_cwd,
        light_mode=False,
    )


def _compute_single_reweight(args):
    global _REWEIGHT_LIKELIHOOD_SAMPLER
    it, nf_vector, logq = args
    logp, nll_stat, nll_syst = _REWEIGHT_LIKELIHOOD_SAMPLER.inject_params_and_compute_likelihood(
        nf_vector, extend_continue=False
    )
    return it, -float(logq) - float(logp), -float(logp), float(logq), float(logp)
class SamplingDatasetTarget:
    """Lightweight target for sampling-only workflows.

    This mirrors the attributes used by CovFlow/SystematicFlow and the
    eigen-to-physical transform used in this script, without loading
    batch*.npz files from a dataset folder.
    """

    def __init__(self, phase_space_dim, mean_vec: np.ndarray, cov_mat: np.ndarray):
        mean = torch.as_tensor(mean_vec, dtype=torch.float32).reshape(-1)
        cov = torch.as_tensor(cov_mat, dtype=torch.float32)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise RuntimeError(f"postfit covariance must be square, got shape={tuple(cov.shape)}")

        ndim = int(mean.shape[0])
        if int(cov.shape[0]) != ndim:
            raise RuntimeError(
                f"postfit mean/covariance mismatch: mean has {ndim} dims, covariance is {tuple(cov.shape)}"
            )

        phase_dims = [int(i) for i in phase_space_dim]
        phase_set = set(phase_dims)
        if any((i < 0 or i >= ndim) for i in phase_dims):
            raise RuntimeError(
                f"phase_space_dim has out-of-range indices for ndim={ndim}: {phase_dims}"
            )

        self.phase_space_dim = phase_dims
        self.list_dim_conditionnal = [i for i in range(ndim) if i not in phase_set]

        std = torch.sqrt(torch.clamp(torch.diag(cov), min=1e-12))
        dinv = torch.diag(1.0 / std)
        cov_std = dinv @ cov @ dinv
        chol_std = torch.linalg.cholesky(cov_std + 1e-6 * torch.eye(ndim, dtype=cov.dtype))

        self.mean = mean
        self.std_per_dim = std
        self.cholesky = chol_std

    def transform_eigen_space_to_data_space(self, x: torch.Tensor) -> torch.Tensor:
        std = self.std_per_dim.to(device=x.device, dtype=x.dtype)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        return x * std + mean


def build_sampling_dataset_target(cfg: DictConfig, mean_vec: np.ndarray, cov_mat: np.ndarray) -> SamplingDatasetTarget:
    phase_space_dim = list(cfg.experiment.dataset.phase_space_dim)
    return SamplingDatasetTarget(phase_space_dim, mean_vec, cov_mat)


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


def plot_2d_hist_side_by_side(
    x_nf: np.ndarray,
    y_nf: np.ndarray,
    x_mc: np.ndarray,
    y_mc: np.ndarray,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    bins: int = 60,
) -> None:
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


def plot_2d_hist_nf_only(x_nf: np.ndarray, y_nf: np.ndarray, xlabel: str, ylabel: str, outpath: Path, bins: int = 60) -> None:
    xmin = float(np.min(x_nf))
    xmax = float(np.max(x_nf))
    ymin = float(np.min(y_nf))
    ymax = float(np.max(y_nf))

    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        xmin, xmax = 0.0, 1.0
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin:
        ymin, ymax = 0.0, 1.0

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist2d(x_nf, y_nf, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    ax.set_title("NF")
    ax.set_xlabel(_short(xlabel, 45))
    ax.set_ylabel(_short(ylabel, 45))

    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


@hydra.main(config_path="/workspace/work/GuNFlows/configs", config_name="sample_mcmc_nf_toyOA", version_base=None)
def main(cfg: DictConfig) -> None:
    training_folder = _abspath(str(cfg.training_folder))
    save_dir = _abspath(str(cfg.save_dir))

    do_plot_mcmc = bool(getattr(cfg, "do_plot_mcmc", True))
    do_reweight_nf = bool(getattr(cfg, "do_reweight_nf", False))
    reweight_num_workers = int(getattr(cfg, "reweight_num_workers", 1))

    if do_plot_mcmc:
        mcmc_root = _abspath(str(cfg.mcmc_chain))
        print(f"Is MCMC file here ? {os.path.isfile(mcmc_root)}", flush=True)
    else:
        mcmc_root = None

    print(f"PWD (hydra chdir): {os.getcwd()}", flush=True)
    print(f"training_folder: {training_folder}", flush=True)
    if do_plot_mcmc:
        print(f"mcmc_chain: {mcmc_root}", flush=True)
    else:
        print("mcmc_chain: <disabled>", flush=True)
    print(f"save_dir: {save_dir}", flush=True)
    print(f"do_plot_mcmc: {do_plot_mcmc}", flush=True)
    print(f"do_reweight_nf: {do_reweight_nf}", flush=True)
    print(f"reweight_num_workers: {reweight_num_workers}", flush=True)

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

    dataset = build_sampling_dataset_target(cfg, bestfit_parameter_values, postfit_covariance)
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
    if do_reweight_nf:
        samples_nf, logq_nf = sample_nf_physical_with_logq(
            model=model,
            dataset=dataset,
            parameter_limits=parameter_limits,
            num_samples=num_samples,
            batch_size=batch_size,
        )
        print(f"NF sampling done: {samples_nf.shape} (+logq) in {time.time()-t0:.1f}s", flush=True)
    else:
        samples_nf = sample_nf_physical(model, dataset, parameter_limits, num_samples, batch_size)
        logq_nf = None
        print(f"NF sampling done: {samples_nf.shape} in {time.time()-t0:.1f}s", flush=True)

    # Optional MCMC loading (ONLY if do_plot_mcmc=True)
    if do_plot_mcmc:
        print(f"Loading ToyNDFit MCMC from: {mcmc_root}", flush=True)
        mcmc_max_steps = cfg.mcmc_max_steps if cfg.mcmc_max_steps is not None else None
        mcmc_burnin_frac = float(cfg.mcmc_burnin_frac)
        mcmc_thin = int(cfg.mcmc_thin)

        base_dir, mcmc_names, mcmc_pts_raw, meta = load_mcmc_gundamworkspace(mcmc_root, mcmc_max_steps)
        mcmc_post, burn = apply_burnin_thin(mcmc_pts_raw, mcmc_burnin_frac, mcmc_thin)
        print(f"MCMC loaded: raw {mcmc_pts_raw.shape} -> post {mcmc_post.shape} (burn={burn}, thin={mcmc_thin})", flush=True)
        print(f"MCMC base_dir: {base_dir}  meta: {meta}", flush=True)
    else:
        base_dir, mcmc_names, mcmc_pts_raw, meta = None, None, None, None
        mcmc_post, burn = None, None
        mcmc_burnin_frac, mcmc_thin = None, None

    # Resolve comparison dimension
    d_nf = int(samples_nf.shape[1])
    d_fit = int(bestfit_parameter_values.shape[0])
    d_cov = int(postfit_covariance.shape[0])
    d_names = len(nf_param_names_short)

    if do_plot_mcmc:
        d_mcmc = int(mcmc_post.shape[1])
        d = min(d_nf, d_mcmc, d_fit, d_cov, d_names)
    else:
        d = min(d_nf, d_fit, d_cov, d_names)

    if d == 0:
        raise RuntimeError("Cannot proceed: zero dimension after resolving NF/postfit shapes (and MCMC if enabled).")

    samples_nf_c = samples_nf[:, :d]
    mu_vec = bestfit_parameter_values[:d]
    cov_mat = postfit_covariance[:d, :d]
    sig_vec = np.sqrt(np.clip(np.diag(cov_mat), 0.0, np.inf))
    labels = nf_param_names_short[:d]

    if do_plot_mcmc:
        samples_mcmc_c = mcmc_post[:, :d]
    else:
        samples_mcmc_c = None

    if do_reweight_nf:
        if logq_nf is None or logq_nf.shape[0] != samples_nf.shape[0]:
            raise RuntimeError("do_reweight_nf=True but logq_nf is missing or mis-shaped.")
        logq_nf = np.asarray(logq_nf).reshape(-1)[: samples_nf.shape[0]]

    # -------------------------
    # NF -> LH reweighting (EXACTLY like sample_mcmc.py)
    # -------------------------
    reweight_nf_to_lh = None
    outlier_mask = None
    if do_reweight_nf:
        print("Computing reweighting factors from NF to LH (Toy OA)...", flush=True)
        t_rw = time.time()
        reweight_nf_to_lh_list = []
        lh_values = []

        if reweight_num_workers <= 1:
            for it, (nf_vector, logq) in enumerate(zip(samples_nf_c, logq_nf)):
                logp, nll_stat, nll_syst = likelihood_sampler.inject_params_and_compute_likelihood(
                    nf_vector, extend_continue=False
                )
                if (it % max(1, int(num_samples // 100)) == 0):
                    print(f"iter {it} NLL/2: {logp}, log_q_nf: {float(logq)}", flush=True)
                reweight_nf_to_lh_list.append(-float(logq) - float(logp))
                lh_values.append(-float(logp))
        else:
            worker_args = [
                (
                    int(it),
                    np.asarray(nf_vector, dtype=np.float64),
                    float(logq),
                )
                for it, (nf_vector, logq) in enumerate(zip(samples_nf_c, logq_nf))
            ]

            ctx = mp.get_context("spawn")
            chunksize = max(1, len(worker_args) // (reweight_num_workers * 20))

            with ctx.Pool(
                processes=reweight_num_workers,
                initializer=_init_reweight_worker,
                initargs=(
                    cfg.experiment.dataset.llh_config,
                    cfg.experiment.dataset.llh_overrides,
                    cfg.experiment.dataset.data_is_asimov,
                    cfg.experiment.sampler.threads,
                    cfg.experiment.dataset.llh_cwd,
                ),
            ) as pool:
                for it, rw_val, lh_val, logq_val, logp_val in pool.imap(_compute_single_reweight, worker_args, chunksize=chunksize):
                    if (it % max(1, int(num_samples // 100)) == 0):
                        print(f"iter {it} NLL/2: {logp_val}, log_q_nf: {float(logq_val)}", flush=True)
                    reweight_nf_to_lh_list.append(rw_val)
                    lh_values.append(lh_val)

        reweight_nf_to_lh = np.asarray(reweight_nf_to_lh_list, dtype=np.float64).reshape(-1)
        lh_values = np.asarray(lh_values, dtype=np.float64).reshape(-1)

        if reweight_nf_to_lh.size > 0:
            median_reweight = np.median(reweight_nf_to_lh)
            reweight_nf_to_lh = reweight_nf_to_lh - median_reweight

            median_lh = np.median(lh_values)
            lh_values = lh_values - median_lh

            median_logq = np.median(logq_nf)
            logq_nf = logq_nf - median_logq

        lower_bound = np.quantile(reweight_nf_to_lh, 0.001)
        upper_bound = np.quantile(reweight_nf_to_lh, 0.999)
        outlier_mask = (reweight_nf_to_lh >= lower_bound) & (reweight_nf_to_lh <= upper_bound)

        filtered_reweights = reweight_nf_to_lh[outlier_mask]
        variance_reweight = float(np.var(reweight_nf_to_lh))
        variance_filtered = float(np.var(filtered_reweights)) if filtered_reweights.size > 0 else float("nan")

        weights = np.exp(reweight_nf_to_lh)
        eff = float((np.sum(weights) ** 2) / np.sum(weights ** 2)) if np.sum(weights ** 2) > 0 else 0.0
        filt_w = np.exp(filtered_reweights)
        eff_f = float((np.sum(filt_w) ** 2) / np.sum(filt_w ** 2)) if np.sum(filt_w ** 2) > 0 else 0.0

        print(f"Effective sample size (NF to LH): {eff:.1f} / {len(reweight_nf_to_lh)}", flush=True)
        print(f"Effective sample size (NF to LH, filtered): {eff_f:.1f} / {int(outlier_mask.sum())}", flush=True)

        fig = plt.figure(figsize=(6, 4))
        plt.hist(reweight_nf_to_lh, bins=100, histtype="step", alpha=1.0, label="reweight_nf_to_lh")
        plt.hist(np.asarray(logq_nf).reshape(-1), bins=100, histtype="step", alpha=0.7, label="logq_nf (shifted)")
        plt.hist(lh_values, bins=100, histtype="step", alpha=0.7, label="lh_values (shifted)")
        plt.xlabel("Reweighting factor (logq_NF - logp_LH)")
        plt.ylabel("Entries")
        plt.title("Reweighting factors from NF to LH")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        out_path = out_dir / "LogWeights_NF_to_LH.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

        print(
            f"Reweighting stats: var={variance_reweight:.6g}, var(filtered 0.001-0.999)={variance_filtered:.6g}",
            flush=True,
        )
        print(f"Reweighting done in {time.time() - t_rw:.1f}s", flush=True)

    # -------------------------
    # Report
    # -------------------------
    with open(out_dir / "report.txt", "w") as f:
        f.write(f"pwd: {os.getcwd()}\n")
        f.write(f"training_folder: {training_folder}\n")
        f.write(f"checkpoint: {str(ckpt_path)}\n")
        f.write(f"do_plot_mcmc: {do_plot_mcmc}\n")
        if do_plot_mcmc:
            f.write(f"mcmc_root: {mcmc_root}\n")
            f.write(f"mcmc_base_dir: {base_dir}\n")
            f.write(f"mcmc_meta: {meta}\n")
            f.write(f"mcmc_raw: {mcmc_pts_raw.shape}\n")
            f.write(f"mcmc_post: {mcmc_post.shape}\n")
            f.write(f"burnin_frac: {mcmc_burnin_frac}\n")
            f.write(f"thin: {mcmc_thin}\n")
        f.write(f"nf_samples: {samples_nf.shape}\n")
        f.write(f"matching_mode: index_order_only\n")
        f.write(f"compared_dims: {d}\n")
        f.write(f"do_reweight_nf: {do_reweight_nf}\n")
        if do_reweight_nf:
            f.write(f"reweight_num_workers: {reweight_num_workers}\n")
            f.write(f"reweight_outlier_keep: {int(outlier_mask.sum())}/{len(outlier_mask)}\n")
        for i in range(d):
            f.write(f"  {i:03d} {labels[i]}\n")

    # -------------------------
    # Marginals
    # -------------------------
    bins_n = int(cfg.bins)
    if run_without_mcmc:
        print(f"Plotting {d} NF-only parameter marginals.", flush=True)
    else:
        print(f"Comparing {d} parameters.", flush=True)
    print("Plotting marginals sequentially.", flush=True)

    global_xmin = float(np.min(samples_nf_c))
    global_xmax = float(np.max(samples_nf_c))
    if samples_mcmc_c is not None:
        global_xmin = float(min(global_xmin, np.min(samples_mcmc_c)))
        global_xmax = float(max(global_xmax, np.max(samples_mcmc_c)))
    if np.isfinite(mu_vec).all():
        global_xmin = float(min(global_xmin, np.min(mu_vec)))
        global_xmax = float(max(global_xmax, np.max(mu_vec)))
    finite_sig_mask = np.isfinite(sig_vec) & (sig_vec > 0)
    if finite_sig_mask.any():
        global_xmin = float(min(global_xmin, np.min(mu_vec[finite_sig_mask] - 5.0 * sig_vec[finite_sig_mask])))
        global_xmax = float(max(global_xmax, np.max(mu_vec[finite_sig_mask] + 5.0 * sig_vec[finite_sig_mask])))

    if not np.isfinite(global_xmin) or not np.isfinite(global_xmax) or global_xmax <= global_xmin:
        global_xmin, global_xmax = 0.0, 1.0

    global_span = global_xmax - global_xmin
    global_xmin -= 0.02 * global_span
    global_xmax += 0.02 * global_span
    common_edges = np.linspace(global_xmin, global_xmax, bins_n + 1)
    common_xgrid = np.linspace(global_xmin, global_xmax, 400)

    for k in range(d):
        name = labels[k]
        x_n = samples_nf_c[:, k]
        mu = float(mu_vec[k])
        sig = float(sig_vec[k])

        if do_plot_mcmc:
            x_m = samples_mcmc_c[:, k]
            xmin = float(min(np.min(x_m), np.min(x_n), mu))
            xmax = float(max(np.max(x_m), np.max(x_n), mu))
        else:
            xmin = float(min(np.min(x_n), mu))
            xmax = float(max(np.max(x_n), mu))

        if np.isfinite(sig) and sig > 0:
            xmin = min(xmin, mu - 5.0 * sig)
            xmax = max(xmax, mu + 5.0 * sig)

        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            xmin, xmax = 0.0, 1.0

        span = xmax - xmin
        xmin -= 0.02 * span
        xmax += 0.02 * span

        edges = np.linspace(xmin, xmax, bins_n + 1)
        xgrid = np.linspace(xmin, xmax, 400)

        fig = plt.figure(figsize=(6, 4))

        if do_plot_mcmc:
            plt.hist(x_m, bins=edges, histtype="step", density=True, label=f"MCMC (n={len(x_m)})")

        plt.hist(x_n, bins=edges, histtype="step", density=True, label=f"NF (n={len(x_n)})")

        if do_reweight_nf:
            nf_filtered = x_n[outlier_mask] if outlier_mask is not None else x_n
            w = np.exp(reweight_nf_to_lh[outlier_mask]) if (outlier_mask is not None) else np.exp(reweight_nf_to_lh)
            plt.hist(
                nf_filtered,
                bins=edges,
                weights=w,
                histtype="step",
                density=True,
                label="NF (reweighted)",
                alpha=0.8,
            )

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

    if do_plot_mcmc:
        corr2d_bins = int(getattr(cfg, "corr2d_bins", 60))
        dims_cfg = getattr(cfg, "corr2d_dims", None)

        if dims_cfg is None or (isinstance(dims_cfg, (list, tuple)) and len(dims_cfg) == 0) or (isinstance(dims_cfg, str) and dims_cfg.strip() == ""):
            dims = list(range(max(0, d - 6), d))
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
    else:
        print("do_plot_mcmc=False, skipping 2D NF-vs-MCMC plots.", flush=True)

    print(f"Done. Outputs in: {str(out_dir)}", flush=True)


if __name__ == "__main__":
    main()