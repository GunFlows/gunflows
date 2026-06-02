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
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from omegaconf import DictConfig, OmegaConf
from scipy.stats import chi2

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


def _head_tail3(arr: np.ndarray) -> str:
    a = np.asarray(arr)
    if a.ndim != 1:
        a = a.reshape(-1)
    if a.size <= 6:
        return np.array2string(a, precision=6, separator=", ")
    head = np.array2string(a[:3], precision=6, separator=", ")
    tail = np.array2string(a[-3:], precision=6, separator=", ")
    return f"{head} ... {tail}"


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
    n_total = int(t_mcmc.GetEntries())
    if not t_mcmc.GetBranch("Points"):
        raise RuntimeError("MCMC tree has no 'Points' branch.")

    # Deduplicate consecutive entries based on the first Points component
    # and keep only the first entry of each repeated run.

    kept_entry_indices: list[int] = []
    prev_key = None
    for entry_idx in range(n_total):
        t_mcmc.GetEntry(entry_idx)
        v = getattr(t_mcmc, "Points")
        if int(v.size()) <= 0:
            continue
        key_val = float(v.at(0))
        if prev_key is None or key_val != prev_key:
            kept_entry_indices.append(entry_idx)
            prev_key = key_val

    if max_steps is not None:
        n = min(len(kept_entry_indices), int(max_steps))
        kept_entry_indices = kept_entry_indices[-n:]
    else:
        n = len(kept_entry_indices)

    if n == 0:
        t_mcmc.GetEntry(0)
        d0 = int(getattr(t_mcmc, "Points").size())
        return np.empty((0, d0), dtype=np.float64)

    t_mcmc.GetEntry(kept_entry_indices[0])
    d = int(getattr(t_mcmc, "Points").size())
    pts = np.empty((n, d), dtype=np.float64)
    for i, entry_idx in enumerate(kept_entry_indices):
        t_mcmc.GetEntry(entry_idx)
        v = getattr(t_mcmc, "Points")
        for j in range(d):
            pts[i, j] = float(v.at(j))
    return pts


def read_ttree_nll_from_llh_branches(t_mcmc, max_steps: int | None) -> np.ndarray:
    """Read TTree NLL proxy defined as (LLHStatistical + LLHPenalty) / 2.

    Uses the same deduplication/capping convention as read_points_vector_tree:
    deduplicate consecutive entries by Points[0], then apply max_steps on tail.
    """
    n_total = int(t_mcmc.GetEntries())
    if not t_mcmc.GetBranch("Points"):
        raise RuntimeError("MCMC tree has no 'Points' branch.")
    if not t_mcmc.GetBranch("LLHStatistical") or not t_mcmc.GetBranch("LLHPenalty"):
        raise RuntimeError("MCMC tree must contain branches 'LLHStatistical' and 'LLHPenalty'.")

    kept_entry_indices: list[int] = []
    prev_key = None
    for entry_idx in range(n_total):
        t_mcmc.GetEntry(entry_idx)
        v = getattr(t_mcmc, "Points")
        if int(v.size()) <= 0:
            continue
        key_val = float(v.at(0))
        if prev_key is None or key_val != prev_key:
            kept_entry_indices.append(entry_idx)
            prev_key = key_val

    if max_steps is not None:
        n = min(len(kept_entry_indices), int(max_steps))
        kept_entry_indices = kept_entry_indices[-n:]

    out = np.empty(len(kept_entry_indices), dtype=np.float64)
    for i, entry_idx in enumerate(kept_entry_indices):
        t_mcmc.GetEntry(entry_idx)
        llh_stat = float(getattr(t_mcmc, "LLHStatistical"))
        llh_pen = float(getattr(t_mcmc, "LLHPenalty"))
        out[i] = 0.5 * (llh_stat + llh_pen)
    return out


def load_mcmc_gundamworkspace(input_root: str, max_steps: int | None) -> tuple[str, list[str], np.ndarray, np.ndarray, dict]:
    tf = ROOT.TFile.Open(input_root, "READ")
    if not tf or tf.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {input_root}")

    base_dir = infer_base_dir(tf)
    names, pcyc = read_param_names_parameterSets(tf, base_dir)
    t_mcmc, mcyc = get_latest_obj_in_dir(tf, base_dir, "MCMC")
    pts = read_points_vector_tree(t_mcmc, max_steps)
    nll_from_tree = read_ttree_nll_from_llh_branches(t_mcmc, max_steps)

    if int(pts.shape[0]) != int(nll_from_tree.shape[0]):
        raise RuntimeError(
            f"Internal mismatch while loading MCMC tree: points={pts.shape[0]} vs nll_from_tree={nll_from_tree.shape[0]}"
        )

    meta = {"base_dir": base_dir, "parameterSets_cycle": pcyc, "MCMC_cycle": mcyc}
    return base_dir, names, pts, nll_from_tree, meta


def apply_burnin_thin(pts: np.ndarray, burnin_frac: float, thin: int) -> tuple[np.ndarray, int]:
    nsteps = int(pts.shape[0])

    # Burn-in is interpreted strictly as a FRACTION of total steps and
    # removed from the START of the chain before thinning.
    burnin_frac = float(burnin_frac)
    if not np.isfinite(burnin_frac):
        raise ValueError(f"burnin_frac must be finite, got {burnin_frac}")
    if burnin_frac < 0.0 or burnin_frac > 1.0:
        raise ValueError(
            f"burnin_frac must be in [0, 1] and is interpreted as a fraction of total steps; got {burnin_frac}"
        )

    burn = int(np.floor(burnin_frac * nsteps))
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


def eval_nf_neglogq_on_physical_points(
    model,
    dataset,
    points_physical: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Evaluate -log q_NF for points in physical/data space."""
    x_phys = np.asarray(points_physical, dtype=np.float32)
    if x_phys.ndim != 2:
        raise RuntimeError(f"points_physical must be 2D, got shape={x_phys.shape}")

    n, d = x_phys.shape
    d_model = int(dataset.mean.shape[0])
    if d != d_model:
        raise RuntimeError(f"Dimension mismatch for NF eval: points have d={d}, model expects d={d_model}")

    phase_dims = list(dataset.phase_space_dim)
    cond_dims = list(dataset.list_dim_conditionnal)
    if len(phase_dims) == 0 or len(cond_dims) == 0:
        raise RuntimeError("Invalid phase/context split in dataset target for NF evaluation.")

    out = np.empty(n, dtype=np.float64)
    dev = torch.device(device)
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            end = min(n, start + int(batch_size))
            xb = torch.as_tensor(x_phys[start:end], dtype=torch.float32, device=dev)

            # Invert the script's eigen->data map to evaluate NF in eigen space.
            mean = dataset.mean.to(device=dev, dtype=xb.dtype)
            std = dataset.std_per_dim.to(device=dev, dtype=xb.dtype)
            x_eig = (xb - mean) / std

            z = x_eig[:, phase_dims]
            c = x_eig[:, cond_dims]
            logq = model.log_prob(z, context=c)
            out[start:end] = (-logq).detach().to(device="cpu", dtype=torch.float64).numpy().reshape(-1)
    return out


def eval_nll_on_physical_points(likelihood_sampler, points_physical: np.ndarray, batch_size: int) -> np.ndarray:
    """Evaluate the LH NLL for points already expressed in physical/data space."""
    x_phys = np.asarray(points_physical, dtype=np.float64)
    if x_phys.ndim != 2:
        raise RuntimeError(f"points_physical must be 2D, got shape={x_phys.shape}")

    n = int(x_phys.shape[0])
    out = np.empty(n, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            end = min(n, start + int(batch_size))
            for i in range(start, end):
                nll_val, _, _ = likelihood_sampler.inject_params_and_compute_likelihood(
                    np.asarray(x_phys[i], dtype=np.float64), extend_continue=False
                )
                out[i] = float(nll_val)
    return out


def compute_delta_nll_cutoff(p_3sigma: float, ndim: int) -> float:
    p = float(p_3sigma)
    if not np.isfinite(p) or not (0.0 < p < 1.0):
        raise ValueError(f"p_3sigma must be in (0, 1), got {p_3sigma}")
    if int(ndim) <= 0:
        raise ValueError(f"ndim must be positive, got {ndim}")
    delta2 = float(chi2.ppf(p, df=int(ndim)))
    return 0.5 * delta2


def plot_nll_vs_neglogq(
    nll: np.ndarray,
    neglogq: np.ndarray,
    keep_mask: np.ndarray,
    delta_nll_cut: float,
    outpath: Path,
    bins: int = 80,
) -> None:
    nll = np.asarray(nll, dtype=np.float64).reshape(-1)
    neglogq = np.asarray(neglogq, dtype=np.float64).reshape(-1)
    keep_mask = np.asarray(keep_mask, dtype=bool).reshape(-1)

    # Shift -log_q by its median, same convention as logq_nf shifting.
    finite_neglogq = np.isfinite(neglogq)
    if finite_neglogq.any():
        median_neglogq = np.median(neglogq[finite_neglogq])
        neglogq = neglogq - median_neglogq

    finite_mask = np.isfinite(nll) & np.isfinite(neglogq)
    all_nll = nll[finite_mask]
    all_nq = neglogq[finite_mask]

    kept_mask_finite = keep_mask[finite_mask]
    kept_nll = all_nll[kept_mask_finite]
    kept_nq = all_nq[kept_mask_finite]

    if all_nll.size == 0:
        return

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    def _plot_hist2d_log(ax, x, y):
        h, xe, ye = np.histogram2d(x, y, bins=bins)
        h = np.ma.masked_less_equal(h, 0)
        positive = h.compressed()
        if positive.size == 0:
            ax.hist2d(x, y, bins=bins)
            return
        mesh = ax.pcolormesh(
            xe,
            ye,
            h.T,
            cmap="viridis",
            norm=LogNorm(vmin=float(positive.min()), vmax=float(positive.max())),
            shading="auto",
        )
        fig.colorbar(mesh, ax=ax)

    _plot_hist2d_log(ax1, all_nll, all_nq)
    ax1.set_title("All MCMC steps")
    ax1.set_xlabel("NLL")
    ax1.set_ylabel("-log q_NF")
    # Add y=x diagonal line
    lims = [np.min([ax1.get_xlim(), ax1.get_ylim()]), 
        np.max([ax1.get_xlim(), ax1.get_ylim()])]
    ax1.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.7)

    if kept_nll.size > 0:
        _plot_hist2d_log(ax2, kept_nll, kept_nq)
    ax2.set_title(f"Kept steps (ΔNLL <= {delta_nll_cut:.3g})")
    ax2.set_xlabel("NLL")
    ax2.set_ylabel("-log q_NF")
    # Add y=x diagonal line
    lims = [np.min([ax2.get_xlim(), ax2.get_ylim()]), 
        np.max([ax2.get_xlim(), ax2.get_ylim()])]
    ax2.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_delta_nll_overlay(
    delta_nll_nf: np.ndarray,
    delta_nll_mcmc: np.ndarray,
    outpath: Path,
    bins: int = 80,
) -> None:
    delta_nll_nf = np.asarray(delta_nll_nf, dtype=np.float64).reshape(-1)
    delta_nll_mcmc = np.asarray(delta_nll_mcmc, dtype=np.float64).reshape(-1)

    finite_nf = np.isfinite(delta_nll_nf)
    finite_mcmc = np.isfinite(delta_nll_mcmc)
    delta_nll_nf = delta_nll_nf[finite_nf]
    delta_nll_mcmc = delta_nll_mcmc[finite_mcmc]

    if delta_nll_nf.size == 0 and delta_nll_mcmc.size == 0:
        return

    xmin = float(min(np.min(delta_nll_nf) if delta_nll_nf.size else 0.0, np.min(delta_nll_mcmc) if delta_nll_mcmc.size else 0.0))
    xmax = float(max(np.max(delta_nll_nf) if delta_nll_nf.size else 0.0, np.max(delta_nll_mcmc) if delta_nll_mcmc.size else 0.0))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        xmin, xmax = -1.0, 1.0

    edges = np.linspace(xmin, xmax, bins + 1)

    fig = plt.figure(figsize=(7, 4))
    if delta_nll_mcmc.size:
        plt.hist(
            delta_nll_mcmc,
            bins=edges,
            histtype="step",
            density=True,
            linewidth=1.4,
            label=f"MCMC (n={len(delta_nll_mcmc)})",
        )
    if delta_nll_nf.size:
        plt.hist(
            delta_nll_nf,
            bins=edges,
            histtype="step",
            density=True,
            linewidth=1.4,
            label=f"NF (n={len(delta_nll_nf)})",
        )

    plt.xlabel("NLL - NLL_bestfit")
    plt.ylabel("Density")
    plt.title("Delta NLL distribution")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


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

    def _plot_hist2d_log(ax, x, y):
        h, xe, ye = np.histogram2d(x, y, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
        h = np.ma.masked_less_equal(h, 0)
        positive = h.compressed()
        if positive.size == 0:
            ax.hist2d(x, y, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
            return
        mesh = ax.pcolormesh(
            xe,
            ye,
            h.T,
            cmap="viridis",
            norm=LogNorm(vmin=float(positive.min()), vmax=float(positive.max())),
            shading="auto",
        )
        fig.colorbar(mesh, ax=ax)

    _plot_hist2d_log(ax1, x_nf, y_nf)
    ax1.set_title("NF")
    ax1.set_xlabel(_short(xlabel, 45))
    ax1.set_ylabel(_short(ylabel, 45))

    _plot_hist2d_log(ax2, x_mc, y_mc)
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
    h, xe, ye = np.histogram2d(x_nf, y_nf, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    h = np.ma.masked_less_equal(h, 0)
    positive = h.compressed()
    if positive.size == 0:
        ax.hist2d(x_nf, y_nf, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    else:
        mesh = ax.pcolormesh(
            xe,
            ye,
            h.T,
            cmap="viridis",
            norm=LogNorm(vmin=float(positive.min()), vmax=float(positive.max())),
            shading="auto",
        )
        fig.colorbar(mesh, ax=ax)
    ax.set_title("NF")
    ax.set_xlabel(_short(xlabel, 45))
    ax.set_ylabel(_short(ylabel, 45))

    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


_CLIP_QUANTILE_DEFAULT = 0.02  # fraction clipped from each tail for display


def _clip_logw(log_weights: np.ndarray, q: float = _CLIP_QUANTILE_DEFAULT) -> tuple[np.ndarray, float, float]:
    """Return (clipped_array, lo, hi) where lo/hi are the quantile boundaries used."""
    lw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    finite = lw[np.isfinite(lw)]
    if finite.size == 0:
        return lw, float("nan"), float("nan")
    lo = float(np.quantile(finite, q))
    hi = float(np.quantile(finite, 1.0 - q))
    return np.clip(lw, lo, hi), lo, hi


# ---------------------------------------------------------------------------
# Weight distribution diagnostics
# ---------------------------------------------------------------------------

def plot_weight_summary(
    log_weights: np.ndarray,
    outlier_mask: np.ndarray,
    clip_q: float,
    outpath: Path,
) -> None:
    """4-panel weight summary figure.

    Panel 1 — full log_weight histogram (marks clip bounds).
    Panel 2 — log_weight histogram clipped to [clip_q, 1-clip_q] quantile (bulk view).
    Panel 3 — Lorenz curve: x = cumulative sample fraction sorted by weight ascending,
               y = cumulative weight fraction.  ESS corresponds to the Gini area.
    Panel 4 — cumulative fraction of total weight carried by top-k% of samples.
    """
    lw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(lw)
    lw_f = lw[finite_mask]
    if lw_f.size == 0:
        return

    _, lo, hi = _clip_logw(lw_f, q=clip_q)
    lw_clip = np.clip(lw_f, lo, hi)

    w = np.exp(lw_f - np.max(lw_f))  # numerically stable
    w_norm = w / float(np.sum(w))
    w_sorted = np.sort(w_norm)
    cumw = np.cumsum(w_sorted)
    cum_frac = np.linspace(0, 1, len(w_sorted) + 1)[1:]

    w_sort_desc = np.sort(w_norm)[::-1]
    cumw_desc = np.cumsum(w_sort_desc)
    cum_frac_desc = np.linspace(0, 1, len(w_sort_desc) + 1)[1:]

    eff = float((np.sum(w) ** 2) / np.sum(w ** 2))
    eff_frac = eff / len(lw_f)
    n_out = int((~finite_mask).sum()) + int((~outlier_mask).sum())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Panel 1: full distribution
    ax = axes[0, 0]
    ax.hist(lw_f, bins=120, histtype="step", linewidth=1.2, color="steelblue", density=True)
    ax.axvline(lo, color="red", linestyle="--", linewidth=1.0, label=f"clip {clip_q:.0%}/{1-clip_q:.0%}")
    ax.axvline(hi, color="red", linestyle="--", linewidth=1.0)
    ax.axvline(float(np.median(lw_f)), color="orange", linestyle="-", linewidth=1.2, label="median")
    ax.set_xlabel("log_weight (raw)")
    ax.set_ylabel("Density")
    ax.set_title("Full log-weight distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: clipped bulk view
    ax = axes[0, 1]
    ax.hist(lw_clip, bins=100, histtype="stepfilled", linewidth=1.0, color="steelblue", alpha=0.6, density=True)
    ax.set_xlabel(f"log_weight (clipped {clip_q:.0%}–{1-clip_q:.0%})")
    ax.set_ylabel("Density")
    ax.set_title(f"Bulk log-weight distribution\n(ESS={eff:.0f}/{len(lw_f)}  = {eff_frac:.1%})")
    ax.grid(True, alpha=0.3)

    # Panel 3: Lorenz curve
    ax = axes[1, 0]
    ax.plot(cum_frac, cumw, color="steelblue", linewidth=1.5, label="Lorenz curve")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect equality")
    ax.fill_between(cum_frac, cumw, cum_frac, alpha=0.15, color="steelblue")
    ax.set_xlabel("Cumulative sample fraction (ascending weight)")
    ax.set_ylabel("Cumulative weight fraction")
    ax.set_title("Lorenz curve of weights\n(concave = weight concentrated in few samples)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: top-k% weight concentration
    ax = axes[1, 1]
    top_pct = np.array([1, 2, 5, 10, 20, 50])
    top_w_frac = np.array([
        float(np.sum(w_sort_desc[: max(1, int(p / 100.0 * len(w_sort_desc)))]))
        for p in top_pct
    ])
    ax.bar(np.arange(len(top_pct)), top_w_frac * 100, color="steelblue", alpha=0.8)
    ax.set_xticks(np.arange(len(top_pct)))
    ax.set_xticklabels([f"top {p}%" for p in top_pct], fontsize=8)
    ax.axhline(50, color="orange", linestyle="--", linewidth=1.0, label="50% of total weight")
    ax.axhline(90, color="red", linestyle="--", linewidth=1.0, label="90% of total weight")
    ax.set_ylabel("% of total weight")
    ax.set_title(f"Weight concentration\n(outliers removed: {n_out})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# log_weight vs parameter grids
# ---------------------------------------------------------------------------

def plot_logw_vs_param_grid(
    samples: np.ndarray,
    log_weights: np.ndarray,
    labels: list[str],
    dims: list[int],
    clip_q: float,
    outpath: Path,
    n_cols: int = 5,
    bins: int = 40,
    title: str = "",
) -> None:
    """Grid of 2D density histograms: x = param value, y = clipped log_weight.

    A visible slope (correlation) in any panel means the NF's density
    is systematically wrong along that parameter direction.
    """
    if not dims:
        return

    lw_all = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(lw_all)
    lw_clip, lo, hi = _clip_logw(lw_all[finite_mask], q=clip_q)

    n_dims = len(dims)
    n_rows = int(np.ceil(n_dims / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.8 * n_rows), squeeze=False)
    axes_flat = axes.reshape(-1)

    for ax_idx, k in enumerate(dims):
        ax = axes_flat[ax_idx]
        xk_all = samples[:, k].astype(np.float64)
        xk = xk_all[finite_mask]

        # Clip x too at 1%-99% to avoid extreme outliers dominating the range
        xlo = float(np.quantile(xk, 0.01))
        xhi = float(np.quantile(xk, 0.99))
        if not (np.isfinite(xlo) and np.isfinite(xhi) and xhi > xlo):
            xlo, xhi = float(xk.min()), float(xk.max())

        h, xe, ye = np.histogram2d(
            np.clip(xk, xlo, xhi), lw_clip,
            bins=bins,
            range=[[xlo, xhi], [lo, hi]],
        )
        h_ma = np.ma.masked_less_equal(h, 0)
        pos = h_ma.compressed()
        if pos.size > 0:
            mesh = ax.pcolormesh(xe, ye, h_ma.T, cmap="viridis",
                                 norm=LogNorm(vmin=float(pos.min()), vmax=float(pos.max())),
                                 shading="auto")
            fig.colorbar(mesh, ax=ax, fraction=0.04)
        else:
            ax.hist2d(np.clip(xk, xlo, xhi), lw_clip, bins=bins)

        ax.axhline(0.0, color="white", linewidth=0.6, linestyle="--", alpha=0.7)

        # Trend line (mean of y in x-bins)
        try:
            bin_means_x = 0.5 * (xe[:-1] + xe[1:])
            bin_means_y = []
            for bx in range(len(xe) - 1):
                m = (np.clip(xk, xlo, xhi) >= xe[bx]) & (np.clip(xk, xlo, xhi) < xe[bx + 1])
                bin_means_y.append(float(np.mean(lw_clip[m])) if m.any() else float("nan"))
            valid = np.isfinite(bin_means_y)
            ax.plot(bin_means_x[valid], np.array(bin_means_y)[valid],
                    "w-", linewidth=1.2, alpha=0.9)
        except Exception:
            pass

        corr = float(np.corrcoef(xk, lw_clip)[0, 1]) if xk.std() > 0 else 0.0
        short_name = str(labels[k]).split("/")[-1]
        ax.set_title(f"{short_name}\nr={corr:.3f}", fontsize=6.5)
        ax.set_xlabel("param value", fontsize=5)
        ax.set_ylabel("log_w", fontsize=5)
        ax.tick_params(labelsize=4.5)

    for ax_idx in range(n_dims, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=10)

    plt.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# NF pull distributions
# ---------------------------------------------------------------------------

def plot_nf_pull_grid(
    samples: np.ndarray,
    mu_vec: np.ndarray,
    sig_vec: np.ndarray,
    labels: list[str],
    param_groups: dict[str, list[int]],
    outpath: Path,
    bins: int = 50,
) -> None:
    """Grid of pull histograms: (x_NF - mu) / sigma per parameter.

    Should follow N(0,1) if the NF perfectly reproduced the postfit Gaussian.
    Deviations reveal the NF's intrinsic bias — independent of reweighting.
    """
    d = samples.shape[1]
    group_order = [g for g in ("physics", "detector", "nonlinear") if param_groups.get(g)]
    if not group_order:
        return

    from scipy.stats import norm as scipy_norm
    x_std = np.linspace(-5, 5, 300)
    pdf_std = scipy_norm.pdf(x_std)

    fig, all_axes = plt.subplots(
        1, len(group_order),
        figsize=(5 * len(group_order), 5),
    )
    if len(group_order) == 1:
        all_axes = [all_axes]

    for ax, group in zip(all_axes, group_order):
        idxs = param_groups[group]
        pulls = []
        for k in idxs:
            if k >= d:
                continue
            sig = float(sig_vec[k])
            if not (np.isfinite(sig) and sig > 0):
                continue
            pulls.append((samples[:, k].astype(np.float64) - float(mu_vec[k])) / sig)

        if not pulls:
            ax.set_visible(False)
            continue

        pulls_all = np.concatenate(pulls)
        finite = pulls_all[np.isfinite(pulls_all)]
        pulls_clipped = np.clip(finite, -5, 5)

        ax.hist(pulls_clipped, bins=bins, histtype="stepfilled", density=True,
                alpha=0.5, color=_GROUP_COLORS.get(group, "gray"),
                label=f"{group} (n params={len(pulls)})")
        ax.plot(x_std, pdf_std, "k--", linewidth=1.4, label="N(0,1)")
        ax.set_xlabel("(x_NF − μ) / σ")
        ax.set_ylabel("Density")
        ax.set_title(f"{group} group NF pull\n(mean={float(np.mean(finite)):.3f}, std={float(np.std(finite)):.3f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("NF pull distributions: deviation from postfit Gaussian (no reweighting)", fontsize=10)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Correlation matrix comparison
# ---------------------------------------------------------------------------

def plot_corr_matrix_comparison(
    samples: np.ndarray,
    log_weights: np.ndarray,
    outlier_mask: np.ndarray,
    postfit_cov: np.ndarray,
    labels: list[str],
    dims: list[int],
    outpath: Path,
) -> None:
    """Side-by-side correlation matrices: NF raw, NF reweighted, postfit Gaussian, NF−postfit diff.

    Focused on dims (e.g. detector + nonlinear systematics) to keep the plot readable.
    """
    if len(dims) < 2:
        return

    sub = samples[:, dims].astype(np.float64)
    sub_f = sub[outlier_mask]
    w = np.exp(log_weights[outlier_mask] - np.max(log_weights[outlier_mask]))
    w = w / float(np.sum(w))

    # Weighted correlation
    mean_w = np.sum(sub_f * w[:, None], axis=0)
    cov_w = np.zeros((len(dims), len(dims)))
    for ii in range(len(dims)):
        for jj in range(ii, len(dims)):
            c = float(np.sum(w * (sub_f[:, ii] - mean_w[ii]) * (sub_f[:, jj] - mean_w[jj])))
            cov_w[ii, jj] = c
            cov_w[jj, ii] = c
    std_w = np.sqrt(np.clip(np.diag(cov_w), 1e-12, None))
    corr_w = cov_w / (std_w[:, None] * std_w[None, :])
    np.fill_diagonal(corr_w, 1.0)

    corr_nf = np.corrcoef(sub.T)
    sub_cov = postfit_cov[np.ix_(dims, dims)]
    std_pf = np.sqrt(np.clip(np.diag(sub_cov), 1e-12, None))
    corr_pf = sub_cov / (std_pf[:, None] * std_pf[None, :])
    np.fill_diagonal(corr_pf, 1.0)

    diff = corr_nf - corr_pf
    diff_rw = corr_w - corr_pf

    short_labels = [str(labels[k]).split("/")[-1][:12] for k in dims]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    def _heatmap(ax, mat, ttl, vmin=-1, vmax=1, cmap="RdBu_r"):
        im = ax.imshow(mat, aspect="auto", interpolation="nearest",
                       vmin=vmin, vmax=vmax, cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046)
        n = len(dims)
        step = max(1, n // 15)
        ax.set_xticks(range(0, n, step))
        ax.set_yticks(range(0, n, step))
        ax.set_xticklabels(short_labels[::step], fontsize=5, rotation=90)
        ax.set_yticklabels(short_labels[::step], fontsize=5)
        ax.set_title(ttl, fontsize=9)

    _heatmap(axes[0, 0], corr_nf, "NF raw correlation")
    _heatmap(axes[0, 1], corr_w, "NF reweighted correlation")
    _heatmap(axes[1, 0], corr_pf, "Postfit Gaussian correlation")
    _heatmap(axes[1, 1], diff, "NF raw − postfit\n(blue=NF under-estimates, red=over-estimates)", vmin=-0.5, vmax=0.5)

    fig.suptitle(f"Correlation matrices for {len(dims)} selected parameters", fontsize=11)
    plt.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Conditional high-vs-low weight marginals
# ---------------------------------------------------------------------------

def plot_high_low_weight_marginals(
    samples: np.ndarray,
    log_weights: np.ndarray,
    labels: list[str],
    dims: list[int],
    mu_vec: np.ndarray,
    sig_vec: np.ndarray,
    outpath: Path,
    n_cols: int = 5,
    bins: int = 50,
    title: str = "",
) -> None:
    """Grid: for each param, overlay samples with top-25% vs bottom-25% log_weight.

    Shows WHICH parameter regions are systematically up-weighted vs down-weighted,
    giving direct intuition for why the reweighted mean shifts.
    """
    if not dims:
        return

    lw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    finite = np.isfinite(lw)
    q25 = float(np.quantile(lw[finite], 0.25))
    q75 = float(np.quantile(lw[finite], 0.75))
    mask_lo = finite & (lw <= q25)
    mask_hi = finite & (lw >= q75)

    n_dims = len(dims)
    n_rows = int(np.ceil(n_dims / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), squeeze=False)
    axes_flat = axes.reshape(-1)

    for ax_idx, k in enumerate(dims):
        ax = axes_flat[ax_idx]
        x_lo = samples[:, k][mask_lo]
        x_hi = samples[:, k][mask_hi]
        mu = float(mu_vec[k])
        sig = float(sig_vec[k]) if np.isfinite(sig_vec[k]) and sig_vec[k] > 0 else 0.0

        all_vals = np.concatenate([x_lo, x_hi])
        xmin = float(np.nanmin(all_vals))
        xmax = float(np.nanmax(all_vals))
        if sig > 0:
            xmin = min(xmin, mu - 3.5 * sig)
            xmax = max(xmax, mu + 3.5 * sig)
        span = xmax - xmin
        if span <= 0:
            xmin, xmax, span = mu - 1.0, mu + 1.0, 2.0
        edges = np.linspace(xmin - 0.02 * span, xmax + 0.02 * span, bins + 1)

        ax.hist(x_lo, bins=edges, histtype="step", density=True, color="blue",
                linewidth=1.0, label=f"low-w (≤Q25)")
        ax.hist(x_hi, bins=edges, histtype="step", density=True, color="red",
                linewidth=1.0, label=f"high-w (≥Q75)")
        ax.axvline(mu, color="green", linestyle=":", linewidth=0.8)

        mean_lo = float(np.mean(x_lo))
        mean_hi = float(np.mean(x_hi))
        ax.axvline(mean_lo, color="blue", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(mean_hi, color="red", linestyle="--", linewidth=0.8, alpha=0.7)

        shift = (mean_hi - mean_lo) / sig if sig > 0 else 0.0
        short_name = str(labels[k]).split("/")[-1]
        ax.set_title(f"{short_name}\nΔμ/σ={shift:.3f}", fontsize=6.5)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.15)

    for ax_idx in range(n_dims, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    h, l = axes_flat[0].get_legend_handles_labels()
    if h:
        axes_flat[0].legend(h, l, fontsize=5)

    if title:
        fig.suptitle(title, fontsize=9)

    plt.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sorted correlation summary (annotated)
# ---------------------------------------------------------------------------

def plot_sorted_corr_summary(
    samples: np.ndarray,
    log_weights: np.ndarray,
    labels: list[str],
    param_groups: dict[str, list[int]],
    clip_q: float,
    outpath: Path,
) -> None:
    """Horizontal bar chart of |corr(log_w, x_k)| sorted descending.

    Shows the top-N most important parameters driving the reweight.
    Each bar is annotated with the parameter name and colored by group.
    """
    lw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    finite = np.isfinite(lw)
    lw_clip, _, _ = _clip_logw(lw[finite], q=clip_q)
    lw_c = lw_clip - lw_clip.mean()
    lw_std = float(lw_clip.std())
    if lw_std < 1e-12:
        return

    n, d = samples.shape
    corrs = np.zeros(d)
    for k in range(d):
        xk = samples[:, k][finite].astype(np.float64)
        xk_std = float(xk.std())
        if xk_std < 1e-12:
            continue
        xk_c = xk - xk.mean()
        corrs[k] = float(np.mean(lw_c * xk_c)) / (lw_std * xk_std)

    abs_corrs = np.abs(corrs)
    order = np.argsort(abs_corrs)[::-1]
    top_n = min(40, d)
    top_idx = order[:top_n]

    colors = _group_color_array(d, param_groups)
    top_colors = [colors[i] for i in top_idx]
    top_corrs = corrs[top_idx]
    top_names = [str(labels[i]).split("/")[-1][:30] for i in top_idx]

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    ys = np.arange(top_n)
    bars = ax.barh(ys, top_corrs, color=top_colors, alpha=0.85)
    ax.set_yticks(ys)
    ax.set_yticklabels(top_names, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    for thr in (0.1, -0.1, 0.2, -0.2):
        ax.axvline(thr, color="red", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel("Pearson corr(log_weight_clipped, x_k)")
    ax.set_title(
        f"Top-{top_n} parameters by |corr(log_w, x_k)|\n"
        f"(log_weight clipped at [{clip_q:.0%}, {1-clip_q:.0%}])\n"
        "positive = parameter gets up-weighted by reweighting"
    )
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    handles = [Patch(facecolor=_GROUP_COLORS[g], label=g) for g in param_groups if param_groups[g]]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _classify_param_groups(labels: list[str]) -> dict[str, list[int]]:
    """Split parameter indices into physics / detector / nonlinear groups."""
    groups: dict[str, list[int]] = {"physics": [], "detector": [], "nonlinear": []}
    for i, lab in enumerate(labels):
        s = str(lab)
        if "Detector Systematics" in s:
            groups["detector"].append(i)
        elif "Non-Linear Systematics" in s:
            groups["nonlinear"].append(i)
        else:
            groups["physics"].append(i)
    return groups


_GROUP_COLORS = {"physics": "steelblue", "detector": "darkorange", "nonlinear": "forestgreen"}


def _group_color_array(n: int, groups: dict[str, list[int]]) -> list[str]:
    colors = ["gray"] * n
    for g, idxs in groups.items():
        for i in idxs:
            if i < n:
                colors[i] = _GROUP_COLORS.get(g, "gray")
    return colors


def plot_reweight_mean_shift_summary(
    samples: np.ndarray,
    log_weights: np.ndarray,
    labels: list[str],
    mu_vec: np.ndarray,
    sig_vec: np.ndarray,
    param_groups: dict[str, list[int]],
    outpath: Path,
) -> None:
    """Bar chart of (weighted_mean - raw_mean) / sigma for every parameter.

    A non-zero bar means the reweighting shifts the NF mean for that parameter,
    which is the primary symptom the user is investigating.
    """
    n, d = samples.shape
    if log_weights.shape[0] != n or d == 0:
        return

    lw = log_weights.astype(np.float64)
    # Numerically stable softmax for weights
    lw_stable = lw - np.max(lw)
    w = np.exp(lw_stable)
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        return

    raw_means = np.mean(samples, axis=0)
    weighted_means = np.sum(samples * w[:, None], axis=0) / w_sum
    sigma = np.where(np.isfinite(sig_vec) & (sig_vec > 0), sig_vec, 1.0)
    shifts = (weighted_means[:d] - raw_means[:d]) / sigma[:d]

    colors = _group_color_array(d, param_groups)
    fig, ax = plt.subplots(figsize=(max(14, d // 4), 4))
    ax.bar(np.arange(d), shifts, color=colors, alpha=0.85, width=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(0.1, color="red", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axhline(-0.1, color="red", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("Parameter index")
    ax.set_ylabel(r"$(μ_{weighted} - μ_{raw})\ /\ σ$")
    ax.set_title("Mean shift per parameter after NF→LH reweighting\n(non-zero = NF sampling is biased for this parameter)")
    ax.grid(True, alpha=0.3, axis="y")
    handles = [Patch(facecolor=_GROUP_COLORS[g], label=g) for g in param_groups if param_groups[g]]
    ax.legend(handles=handles, fontsize=9)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_reweight_logw_param_corr(
    samples: np.ndarray,
    log_weights: np.ndarray,
    labels: list[str],
    param_groups: dict[str, list[int]],
    outpath: Path,
) -> None:
    """Bar chart of Pearson corr(log_weight, x_k) per parameter.

    A large |corr| means the NF's density is systematically wrong in the
    direction of that parameter — positive = NF under-samples high x_k,
    negative = NF over-samples high x_k.
    """
    n, d = samples.shape
    if log_weights.shape[0] != n or n < 4:
        return

    lw = log_weights.astype(np.float64)
    lw_c = lw - lw.mean()
    lw_std = float(np.std(lw))
    if lw_std < 1e-12:
        return

    corrs = np.zeros(d, dtype=np.float64)
    for k in range(d):
        xk = samples[:, k].astype(np.float64)
        if not np.isfinite(xk).all():
            continue
        xk_std = float(np.std(xk))
        if xk_std < 1e-12:
            continue
        xk_c = xk - xk.mean()
        corrs[k] = float(np.mean(lw_c * xk_c)) / (lw_std * xk_std)

    colors = _group_color_array(d, param_groups)
    fig, ax = plt.subplots(figsize=(max(14, d // 4), 4))
    ax.bar(np.arange(d), corrs, color=colors, alpha=0.85, width=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8)
    for thr in (0.1, -0.1, 0.2, -0.2):
        ax.axhline(thr, color="red", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel("Parameter index")
    ax.set_ylabel("Pearson corr(log_weight, x_k)")
    ax.set_title(
        "Correlation of reweight factor with each parameter\n"
        "positive → NF under-estimates LH at high x_k (parameter gets up-weighted)"
    )
    ax.grid(True, alpha=0.3, axis="y")
    handles = [Patch(facecolor=_GROUP_COLORS[g], label=g) for g in param_groups if param_groups[g]]
    ax.legend(handles=handles, fontsize=9)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_reweight_decomposition(
    log_weights: np.ndarray,
    logq_nf: np.ndarray,
    lh_values: np.ndarray,
    outpath: Path,
    bins: int = 80,
) -> None:
    """Three 2D histograms: log_w vs logq, log_w vs LH, and LH vs logq.

    Shows whether weight variance is driven by the NF (logq) or the LH.
    """
    lw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    lq = np.asarray(logq_nf, dtype=np.float64).reshape(-1)
    lh = np.asarray(lh_values, dtype=np.float64).reshape(-1)

    finite = np.isfinite(lw) & np.isfinite(lq) & np.isfinite(lh)
    lw, lq, lh = lw[finite], lq[finite], lh[finite]
    if lw.size == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    def _hist2d_ax(ax, x, y, xl, yl, ttl):
        h, xe, ye = np.histogram2d(x, y, bins=bins)
        h_ma = np.ma.masked_less_equal(h, 0)
        pos = h_ma.compressed()
        if pos.size > 0:
            mesh = ax.pcolormesh(xe, ye, h_ma.T, cmap="viridis",
                                 norm=LogNorm(vmin=float(pos.min()), vmax=float(pos.max())),
                                 shading="auto")
            fig.colorbar(mesh, ax=ax)
        else:
            ax.hist2d(x, y, bins=bins)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(ttl)

    _hist2d_ax(axes[0], lq, lw, "logq_NF (shifted)", "log_weight (shifted)",
               "log_weight vs logq_NF\n(slope → NF density drives reweight)")
    _hist2d_ax(axes[1], lh, lw, "log LH (shifted)", "log_weight (shifted)",
               "log_weight vs LH\n(slope → LH landscape drives reweight)")
    _hist2d_ax(axes[2], lq, lh, "logq_NF (shifted)", "log LH (shifted)",
               "LH vs logq_NF\n(perfect NF would give diagonal)")

    # Diagonal reference on the NF vs LH panel
    xlim = axes[2].get_xlim()
    ylim = axes[2].get_ylim()
    lo = max(xlim[0], ylim[0])
    hi = min(xlim[1], ylim[1])
    if hi > lo:
        axes[2].plot([lo, hi], [lo, hi], "r--", linewidth=1.2, alpha=0.7, label="ideal")
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_marginals_group_grid(
    samples_nf: np.ndarray,
    log_weights: np.ndarray | None,
    outlier_mask: np.ndarray | None,
    labels: list[str],
    dims: list[int],
    mu_vec: np.ndarray,
    sig_vec: np.ndarray,
    outpath: Path,
    n_cols: int = 5,
    bins: int = 60,
    title: str = "",
    samples_mcmc: np.ndarray | None = None,
) -> None:
    """Grid page of marginals for a list of parameter indices.

    Shows NF (blue), NF reweighted (red dashed), MCMC (gray, optional),
    and the postfit Gaussian (green) on every panel.
    """
    if not dims:
        return

    n_dims = len(dims)
    n_rows = int(np.ceil(n_dims / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), squeeze=False)
    axes_flat = axes.reshape(-1)

    for ax_idx, k in enumerate(dims):
        ax = axes_flat[ax_idx]
        x_n = samples_nf[:, k]
        mu = float(mu_vec[k])
        sig = float(sig_vec[k]) if np.isfinite(sig_vec[k]) and sig_vec[k] > 0 else 0.0

        xmin_v = float(np.min(x_n))
        xmax_v = float(np.max(x_n))
        if samples_mcmc is not None:
            xmin_v = min(xmin_v, float(np.min(samples_mcmc[:, k])))
            xmax_v = max(xmax_v, float(np.max(samples_mcmc[:, k])))
        if sig > 0:
            xmin_v = min(xmin_v, mu - 4.0 * sig)
            xmax_v = max(xmax_v, mu + 4.0 * sig)
        span = xmax_v - xmin_v
        if span <= 0:
            xmin_v, xmax_v, span = mu - 1.0, mu + 1.0, 2.0
        xmin_v -= 0.05 * span
        xmax_v += 0.05 * span
        edges = np.linspace(xmin_v, xmax_v, bins + 1)
        xgrid = np.linspace(xmin_v, xmax_v, 200)

        ax.hist(x_n, bins=edges, histtype="step", density=True, linewidth=1.0,
                color="steelblue", label="NF")

        if log_weights is not None and outlier_mask is not None:
            nf_f = x_n[outlier_mask]
            w = np.exp(log_weights[outlier_mask])
            # density=True with weights: matplotlib normalises by total weight * bin_width
            ax.hist(nf_f, bins=edges, weights=w, histtype="step", density=True,
                    linewidth=1.2, linestyle="--", color="red", label="NF reweighted")

        if samples_mcmc is not None:
            ax.hist(samples_mcmc[:, k], bins=edges, histtype="stepfilled", density=True,
                    alpha=0.25, color="gray", label="MCMC")

        pdf = gaussian_pdf(xgrid, mu, sig)
        if pdf.max() > 0:
            ax.plot(xgrid, pdf, linewidth=1.0, color="green")
        ax.axvline(mu, linestyle=":", linewidth=0.8, color="green")

        short_name = str(labels[k]).split("/")[-1]
        ax.set_title(short_name, fontsize=7)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.15)

    for ax_idx in range(n_dims, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    # Single legend on first panel
    h, l = axes_flat[0].get_legend_handles_labels()
    if h:
        axes_flat[0].legend(h, l, fontsize=5, loc="upper right")

    if title:
        fig.suptitle(title, fontsize=10)

    plt.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_nf_only_corr2d(
    samples_nf: np.ndarray,
    labels: list[str],
    dims: list[int],
    outpath_dir: Path,
    bins: int = 60,
    log_weights: np.ndarray | None = None,
    outlier_mask: np.ndarray | None = None,
) -> None:
    """NF-only 2D histograms for all pairs in dims.

    When log_weights / outlier_mask are provided a second panel shows the
    reweighted distribution side-by-side.
    """
    if len(dims) < 2:
        return

    do_reweight_panel = log_weights is not None and outlier_mask is not None
    n_panels = 2 if do_reweight_panel else 1

    for ii in range(len(dims)):
        for jj in range(ii + 1, len(dims)):
            a, b = dims[ii], dims[jj]
            x_nf = samples_nf[:, a]
            y_nf = samples_nf[:, b]

            xmin = float(np.nanmin(x_nf))
            xmax = float(np.nanmax(x_nf))
            ymin = float(np.nanmin(y_nf))
            ymax = float(np.nanmax(y_nf))
            if not (np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin):
                xmin, xmax = 0.0, 1.0
            if not (np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin):
                ymin, ymax = 0.0, 1.0

            fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)

            def _fill(ax, x, y, w=None, title="NF"):
                h, xe, ye = np.histogram2d(x, y, bins=bins,
                                           range=[[xmin, xmax], [ymin, ymax]],
                                           weights=w)
                h_ma = np.ma.masked_less_equal(h, 0)
                pos = h_ma.compressed()
                if pos.size > 0:
                    mesh = ax.pcolormesh(xe, ye, h_ma.T, cmap="viridis",
                                         norm=LogNorm(vmin=float(pos.min()), vmax=float(pos.max())),
                                         shading="auto")
                    fig.colorbar(mesh, ax=ax)
                else:
                    ax.hist2d(x, y, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
                ax.set_xlabel(_short(labels[a], 40))
                ax.set_ylabel(_short(labels[b], 40))
                ax.set_title(title)

            _fill(axes[0, 0], x_nf, y_nf, title="NF (unweighted)")
            if do_reweight_panel:
                w_vals = np.exp(log_weights[outlier_mask])
                _fill(axes[0, 1], x_nf[outlier_mask], y_nf[outlier_mask],
                      w=w_vals, title="NF (reweighted)")

            plt.tight_layout()
            outp = outpath_dir / f"nf_{a:03d}_{b:03d}.png"
            fig.savefig(outp, dpi=120)
            plt.close(fig)


@hydra.main(config_path="/workspace/work/GuNFlows/configs", config_name="sample_mcmc_nf_toyOA", version_base=None)
def main(cfg: DictConfig) -> None:
    training_folder = _abspath(str(cfg.training_folder))
    save_dir = _abspath(str(cfg.save_dir))

    do_plot_mcmc = bool(getattr(cfg, "do_plot_mcmc", True))
    do_reweight_nf = bool(getattr(cfg, "do_reweight_nf", False))
    reweight_num_workers = int(getattr(cfg, "reweight_num_workers", 1))
    do_mcmc_step_lh_nf_eval = bool(getattr(cfg, "do_mcmc_step_lh_nf_eval", True))
    mcmc_delta_lh_p_3sigma = float(getattr(cfg, "mcmc_delta_lh_p_3sigma", getattr(cfg, "mcmc_lh_keep_quantile", 0.95)))
    mcmc_nf_eval_batch_size = int(getattr(cfg, "mcmc_nf_eval_batch_size", 2048))

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
    print(f"do_mcmc_step_lh_nf_eval: {do_mcmc_step_lh_nf_eval}", flush=True)
    print(f"mcmc_delta_lh_p_3sigma: {mcmc_delta_lh_p_3sigma}", flush=True)
    print(f"mcmc_nf_eval_batch_size: {mcmc_nf_eval_batch_size}", flush=True)

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

    save_samples = bool(getattr(cfg, "save_samples", True))
    diag_clip_q = float(getattr(cfg, "diag_clip_q", _CLIP_QUANTILE_DEFAULT))

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "marginals"
    img_dir.mkdir(parents=True, exist_ok=True)
    corr2d_dir = out_dir / "corr2d"
    corr2d_dir.mkdir(parents=True, exist_ok=True)

    # Diagnostics split into three subdirs for organisation
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_weights_dir = diag_dir / "weights"
    diag_weights_dir.mkdir(parents=True, exist_ok=True)
    diag_params_dir = diag_dir / "params"
    diag_params_dir.mkdir(parents=True, exist_ok=True)
    diag_corr_dir = diag_dir / "correlations"
    diag_corr_dir.mkdir(parents=True, exist_ok=True)

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

        base_dir, mcmc_names, mcmc_pts_raw, mcmc_nll_tree_raw, meta = load_mcmc_gundamworkspace(mcmc_root, mcmc_max_steps)
        mcmc_post, burn = apply_burnin_thin(mcmc_pts_raw, mcmc_burnin_frac, mcmc_thin)
        mcmc_nll_tree_post = mcmc_nll_tree_raw[burn:mcmc_nll_tree_raw.shape[0]:mcmc_thin]
        print(f"MCMC loaded: raw {mcmc_pts_raw.shape} -> post {mcmc_post.shape} (burn={burn}, thin={mcmc_thin})", flush=True)
        print(f"MCMC base_dir: {base_dir}  meta: {meta}", flush=True)
    else:
        base_dir, mcmc_names, mcmc_pts_raw, mcmc_nll_tree_raw, meta = None, None, None, None, None
        mcmc_post, burn = None, None
        mcmc_nll_tree_post = None
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
    delta_nll_cut = compute_delta_nll_cutoff(mcmc_delta_lh_p_3sigma, int(dataset.mean.shape[0]))
    print(
        f"MCMC keep cut from chi2: p_3sigma={mcmc_delta_lh_p_3sigma} -> DeltaNLL_cut={delta_nll_cut:.6g} (ndim={int(dataset.mean.shape[0])})",
        flush=True,
    )

    mcmc_nll = None
    mcmc_neglogq = None
    mcmc_keep_mask = None
    mcmc_keep_threshold = None
    mcmc_post_all = None

    if do_plot_mcmc and do_mcmc_step_lh_nf_eval:
        if mcmc_post.shape[1] != bestfit_parameter_values.shape[0]:
            print(
                "Skipping MCMC-step LH/NF diagnostics because MCMC dim does not match likelihood/NF dim: "
                f"mcmc={mcmc_post.shape[1]} vs model={bestfit_parameter_values.shape[0]}",
                flush=True,
            )
        else:
            nll_bestfit, _, _ = likelihood_sampler.inject_params_and_compute_likelihood(
                bestfit_parameter_values, extend_continue=False
            )

            if samples_nf.shape[1] != bestfit_parameter_values.shape[0]:
                print(
                    "Skipping NF debug print because NF dim does not match likelihood dim: "
                    f"nf={samples_nf.shape[1]} vs model={bestfit_parameter_values.shape[0]}",
                    flush=True,
                )
            else:
                n_nf_debug = min(20, int(samples_nf.shape[0]))
                print(f"Computing LH NLL debug for first {n_nf_debug} NF samples...", flush=True)
                nf_neglogq_debug = eval_nf_neglogq_on_physical_points(
                    model=model,
                    dataset=dataset,
                    points_physical=samples_nf[:n_nf_debug],
                    batch_size=max(1, min(mcmc_nf_eval_batch_size, n_nf_debug)),
                    device=str(cfg.device),
                )
                for it, vec in enumerate(samples_nf[:n_nf_debug]):
                    nll_val, _, _ = likelihood_sampler.inject_params_and_compute_likelihood(vec, extend_continue=False)
                    neglogq_val = float(nf_neglogq_debug[it])
                    delta_nll_val = float(nll_val) - float(nll_bestfit)
                    accepted = bool(np.isfinite(nll_val) and (nll_val <= (nll_bestfit + delta_nll_cut)))
                    print(
                        f"  debug NF[{it:03d}] NLL={float(nll_val):.6g} -log_q={neglogq_val:.6g} best_fit={float(nll_bestfit):.6g} "
                        f"Delta_NLL={delta_nll_val:.6g} {'accepted' if accepted else 'rejected'}",
                        flush=True,
                    )
                    print(f"    params(head/tail): {_head_tail3(vec[:d])}", flush=True)

            print("Computing LH NLL at each MCMC step...", flush=True)
            t_lh = time.time()
            if mcmc_nll_tree_post is None or int(mcmc_nll_tree_post.shape[0]) != int(mcmc_post.shape[0]):
                raise RuntimeError(
                    "MCMC TTree-derived NLL is missing or misaligned with post chain after burn/thin."
                )
            n_mcmc_debug = min(100, int(mcmc_post.shape[0]))
            mcmc_neglogq_debug = eval_nf_neglogq_on_physical_points(
                model=model,
                dataset=dataset,
                points_physical=mcmc_post[:n_mcmc_debug],
                batch_size=max(1, min(mcmc_nf_eval_batch_size, n_mcmc_debug)),
                device=str(cfg.device),
            )
            mcmc_nll = np.empty(mcmc_post.shape[0], dtype=np.float64)
            for it, vec in enumerate(mcmc_post):
                nll_val, _, _ = likelihood_sampler.inject_params_and_compute_likelihood(vec, extend_continue=False)
                mcmc_nll[it] = float(nll_val)
                delta_nll_val = float(nll_val) - float(nll_bestfit)
                accepted = bool(np.isfinite(nll_val) and (nll_val <= (nll_bestfit + delta_nll_cut)))
                if it < 100:
                    neglogq_val = float(mcmc_neglogq_debug[it])
                    nll_tree_val = float(mcmc_nll_tree_post[it])
                    nll_diff_val = float(nll_val) - nll_tree_val
                    print(
                        f"  debug MCMC[{it:03d}] NLL={float(nll_val):.6g} -log_q={neglogq_val:.6g} best_fit={float(nll_bestfit):.6g} "
                        f"Delta_NLL={delta_nll_val:.6g} {'accepted' if accepted else 'rejected'}",
                        flush=True,
                    )
                    print(
                        f"    NLL_tree((LLHStatistical+LLHPenalty)/2)={nll_tree_val:.6g} "
                        f"NLL_computed={float(nll_val):.6g} diff={nll_diff_val:.6g}",
                        flush=True,
                    )
                    print(f"    params(head/tail): {_head_tail3(vec[:d])}", flush=True)
                if (it % max(1, int(mcmc_post.shape[0] // 100)) == 0):
                    print(f"  LH eval {it}/{mcmc_post.shape[0]}", flush=True)
            print(f"LH evaluation done in {time.time()-t_lh:.1f}s", flush=True)

            print("Computing NF -log q at each MCMC step...", flush=True)
            t_nf = time.time()
            mcmc_neglogq = eval_nf_neglogq_on_physical_points(
                model=model,
                dataset=dataset,
                points_physical=mcmc_post,
                batch_size=mcmc_nf_eval_batch_size,
                device=str(cfg.device),
            )
            print(f"NF evaluation done in {time.time()-t_nf:.1f}s", flush=True)

            finite_lh = np.isfinite(mcmc_nll)
            if not finite_lh.any():
                raise RuntimeError("All MCMC likelihood evaluations are non-finite.")

            mcmc_keep_threshold = float(nll_bestfit + delta_nll_cut)
            mcmc_keep_mask = finite_lh & (mcmc_nll <= mcmc_keep_threshold)
            n_keep = int(mcmc_keep_mask.sum())
            print(
                f"Keeping {n_keep}/{len(mcmc_keep_mask)} MCMC steps with NLL <= bestfit + {delta_nll_cut:.6g} ({mcmc_keep_threshold:.6g})",
                flush=True,
            )

            mcmc_nll_all = mcmc_nll.copy()
            mcmc_neglogq_all = mcmc_neglogq.copy()
            mcmc_post_all = mcmc_post.copy()

            # Keep only selected MCMC throws for subsequent comparisons.
            mcmc_post = mcmc_post[mcmc_keep_mask]
            mcmc_nll = mcmc_nll[mcmc_keep_mask]
            mcmc_neglogq = mcmc_neglogq[mcmc_keep_mask]

            plot_nll_vs_neglogq(
                nll=mcmc_nll_all,
                neglogq=mcmc_neglogq_all,
                keep_mask=mcmc_keep_mask,
                delta_nll_cut=delta_nll_cut,
                outpath=out_dir / "mcmc_nll_vs_neglogq_kept.png",
                bins=int(getattr(cfg, "mcmc_diag_bins", 80)),
            )

            np.savez(
                out_dir / "mcmc_lh_nf_eval_kept.npz",
                nll=mcmc_nll,
                neglogq=mcmc_neglogq,
                nll_all=mcmc_nll_all,
                nll_tree_all=mcmc_nll_tree_post,
                nll_tree_kept=mcmc_nll_tree_post[mcmc_keep_mask],
                neglogq_all=mcmc_neglogq_all,
                keep_mask=mcmc_keep_mask,
                p_3sigma=float(mcmc_delta_lh_p_3sigma),
                delta_nll_cut=float(delta_nll_cut),
                keep_threshold=float(mcmc_keep_threshold),
            )

    if do_plot_mcmc and mcmc_keep_mask is not None:
        samples_mcmc_c = mcmc_post[:, :d]
    else:
        samples_mcmc_c = None

    if do_plot_mcmc and mcmc_post_all is not None:
        samples_mcmc_all_c = mcmc_post_all[:, :d]
    else:
        samples_mcmc_all_c = None

    delta_nll_nf = None
    delta_nll_mcmc = None
    nll_bestfit = None
    if do_plot_mcmc and mcmc_nll is not None:
        print("Computing best-fit NLL and delta-NLL distributions...", flush=True)
        nll_bestfit, _, _ = likelihood_sampler.inject_params_and_compute_likelihood(
            bestfit_parameter_values, extend_continue=False
        )
        nf_nll = eval_nll_on_physical_points(likelihood_sampler, samples_nf, batch_size=max(1, batch_size))
        delta_nll_nf = np.asarray(nf_nll, dtype=np.float64) - float(nll_bestfit)
        delta_nll_mcmc = np.asarray(mcmc_nll, dtype=np.float64) - float(nll_bestfit)

        plot_delta_nll_overlay(
            delta_nll_nf=delta_nll_nf,
            delta_nll_mcmc=delta_nll_mcmc,
            outpath=out_dir / "delta_nll_nf_vs_mcmc.png",
            bins=int(getattr(cfg, "delta_nll_bins", 80)),
        )

        np.savez(
            out_dir / "delta_nll_nf_vs_mcmc.npz",
            delta_nll_nf=delta_nll_nf,
            delta_nll_mcmc=delta_nll_mcmc,
            nll_bestfit=float(nll_bestfit),
        )

    if do_reweight_nf:
        if logq_nf is None or logq_nf.shape[0] != samples_nf.shape[0]:
            raise RuntimeError("do_reweight_nf=True but logq_nf is missing or mis-shaped.")
        logq_nf = np.asarray(logq_nf).reshape(-1)[: samples_nf.shape[0]]

    # -------------------------
    # NF -> LH reweighting (EXACTLY like sample_mcmc.py)
    # -------------------------
    reweight_nf_to_lh = None
    outlier_mask = None
    rw_lh_values = None  # log-LH values (median-centered), saved for diagnostics
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
            rw_lh_values = lh_values.copy()  # save for diagnostics

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
        # Use shared bin edges so all overlaid histograms have identical bin widths.
        h_rew = np.asarray(reweight_nf_to_lh, dtype=np.float64).reshape(-1)
        h_logq = np.asarray(logq_nf, dtype=np.float64).reshape(-1)
        h_lh = np.asarray(lh_values, dtype=np.float64).reshape(-1)
        h_all = np.concatenate([h_rew, h_logq, h_lh])
        h_all = h_all[np.isfinite(h_all)]
        if h_all.size == 0:
            bin_edges = np.linspace(-1.0, 1.0, 101)
        else:
            x_min = float(np.min(h_all))
            x_max = float(np.max(h_all))
            if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
                x_min, x_max = x_min - 0.5, x_min + 0.5
            bin_edges = np.linspace(x_min, x_max, 101)

        plt.hist(h_rew, bins=bin_edges, histtype="step", alpha=1.0, label="reweight_nf_to_lh")
        plt.hist(h_logq, bins=bin_edges, histtype="step", alpha=0.7, label="logq_nf (shifted)")
        plt.hist(h_lh, bins=bin_edges, histtype="step", alpha=0.7, label="lh_values (shifted)")
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

        # ---- Save samples + weights as npz ----
        if save_samples:
            npz_path = out_dir / "nf_samples_with_weights.npz"
            print(f"Saving samples+weights to {npz_path} ...", flush=True)
            np.savez(
                npz_path,
                samples_nf=samples_nf,
                logq_nf=logq_nf,
                log_weights=reweight_nf_to_lh,
                weights=np.exp(reweight_nf_to_lh),
                outlier_mask=outlier_mask,
                param_names=np.array(labels, dtype=object),
                bestfit=mu_vec,
                sigma=sig_vec,
            )
            print("Saved.", flush=True)

        # ---- Reweighting diagnostics ----
        param_groups = _classify_param_groups(labels)
        print(
            f"Parameter groups: physics={len(param_groups['physics'])}, "
            f"detector={len(param_groups['detector'])}, "
            f"nonlinear={len(param_groups['nonlinear'])}",
            flush=True,
        )

        # --- weights/ subdir ---
        print("Plotting weight summary (distribution + Lorenz curve + concentration)...", flush=True)
        plot_weight_summary(
            log_weights=reweight_nf_to_lh,
            outlier_mask=outlier_mask,
            clip_q=diag_clip_q,
            outpath=diag_weights_dir / "weight_summary.png",
        )

        print("Plotting sorted parameter-weight correlation (top-40)...", flush=True)
        plot_sorted_corr_summary(
            samples=samples_nf_c,
            log_weights=reweight_nf_to_lh,
            labels=labels,
            param_groups=param_groups,
            clip_q=diag_clip_q,
            outpath=diag_weights_dir / "sorted_corr_summary.png",
        )

        print("Plotting reweight mean-shift summary bar chart...", flush=True)
        plot_reweight_mean_shift_summary(
            samples=samples_nf_c,
            log_weights=reweight_nf_to_lh,
            labels=labels,
            mu_vec=mu_vec,
            sig_vec=sig_vec,
            param_groups=param_groups,
            outpath=diag_weights_dir / "reweight_mean_shift.png",
        )

        print("Plotting Pearson corr(log_w, x_k) per parameter...", flush=True)
        plot_reweight_logw_param_corr(
            samples=samples_nf_c,
            log_weights=reweight_nf_to_lh,
            labels=labels,
            param_groups=param_groups,
            outpath=diag_weights_dir / "reweight_logw_param_corr.png",
        )

        if rw_lh_values is not None:
            print("Plotting reweight decomposition (logq vs LH contribution)...", flush=True)
            plot_reweight_decomposition(
                log_weights=reweight_nf_to_lh,
                logq_nf=logq_nf,
                lh_values=rw_lh_values,
                outpath=diag_weights_dir / "reweight_decomposition.png",
                bins=int(getattr(cfg, "mcmc_diag_bins", 80)),
            )

        # --- params/ subdir ---
        print("Plotting NF pull distributions...", flush=True)
        plot_nf_pull_grid(
            samples=samples_nf_c,
            mu_vec=mu_vec,
            sig_vec=sig_vec,
            labels=labels,
            param_groups=param_groups,
            outpath=diag_params_dir / "nf_pull_distributions.png",
            bins=60,
        )

        for _group_name, _n_cols in [("detector", 5), ("nonlinear", 5), ("physics", 10)]:
            _gdims = param_groups.get(_group_name, [])
            if not _gdims:
                continue
            _title_base = f"{_group_name} systematics" if _group_name != "physics" else "physics parameters"

            print(f"Plotting {_group_name} marginals grid ({len(_gdims)} params)...", flush=True)
            plot_marginals_group_grid(
                samples_nf=samples_nf_c,
                log_weights=reweight_nf_to_lh,
                outlier_mask=outlier_mask,
                labels=labels,
                dims=_gdims,
                mu_vec=mu_vec,
                sig_vec=sig_vec,
                outpath=diag_params_dir / f"{_group_name}_marginals_grid.png",
                n_cols=_n_cols,
                bins=60,
                title=f"{_title_base}: NF (blue) vs NF-reweighted (red dashed) vs postfit Gaussian (green)",
                samples_mcmc=samples_mcmc_c,
            )

            print(f"Plotting log_w vs {_group_name} params 2D grid...", flush=True)
            plot_logw_vs_param_grid(
                samples=samples_nf_c,
                log_weights=reweight_nf_to_lh,
                labels=labels,
                dims=_gdims,
                clip_q=diag_clip_q,
                outpath=diag_params_dir / f"{_group_name}_logw_vs_param_grid.png",
                n_cols=_n_cols,
                bins=40,
                title=(
                    f"log_weight vs {_title_base}\n"
                    f"(white line = conditional mean, r = Pearson corr  |  "
                    f"log_w clipped at [{diag_clip_q:.0%}, {1-diag_clip_q:.0%}])"
                ),
            )

            print(f"Plotting high vs low weight marginals for {_group_name}...", flush=True)
            plot_high_low_weight_marginals(
                samples=samples_nf_c,
                log_weights=reweight_nf_to_lh,
                labels=labels,
                dims=_gdims,
                mu_vec=mu_vec,
                sig_vec=sig_vec,
                outpath=diag_params_dir / f"{_group_name}_high_vs_low_weight.png",
                n_cols=_n_cols,
                bins=50,
                title=(
                    f"{_title_base}: samples with bottom-25% weight (blue) vs top-25% weight (red)\n"
                    "Δμ/σ annotated per panel — shows WHICH parameter regions get up/down-weighted"
                ),
            )

        # --- correlations/ subdir ---
        # Correlation matrix for the detector+nonlinear block (most relevant)
        _corr_dims = param_groups.get("detector", []) + param_groups.get("nonlinear", [])
        if len(_corr_dims) >= 2:
            print(f"Plotting correlation matrix comparison ({len(_corr_dims)} dims)...", flush=True)
            plot_corr_matrix_comparison(
                samples=samples_nf_c,
                log_weights=reweight_nf_to_lh,
                outlier_mask=outlier_mask,
                postfit_cov=cov_mat,
                labels=labels,
                dims=_corr_dims,
                outpath=diag_corr_dir / "corr_matrix_detector_nonlinear.png",
            )

        # Also do physics block if it fits
        _phys_dims = param_groups.get("physics", [])
        if len(_phys_dims) >= 2:
            print(f"Plotting correlation matrix for physics params ({len(_phys_dims)} dims)...", flush=True)
            plot_corr_matrix_comparison(
                samples=samples_nf_c,
                log_weights=reweight_nf_to_lh,
                outlier_mask=outlier_mask,
                postfit_cov=cov_mat,
                labels=labels,
                dims=_phys_dims,
                outpath=diag_corr_dir / "corr_matrix_physics.png",
            )

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
            if mcmc_nll_tree_raw is not None:
                f.write(f"mcmc_nll_tree_raw: {mcmc_nll_tree_raw.shape}\n")
            if mcmc_nll_tree_post is not None:
                f.write(f"mcmc_nll_tree_post: {mcmc_nll_tree_post.shape}\n")
                f.write("mcmc_nll_tree_definition: (LLHStatistical+LLHPenalty)/2\n")
            f.write(f"burnin_frac: {mcmc_burnin_frac}\n")
            f.write(f"thin: {mcmc_thin}\n")
            f.write(f"do_mcmc_step_lh_nf_eval: {do_mcmc_step_lh_nf_eval}\n")
            if mcmc_keep_mask is not None:
                f.write(f"mcmc_delta_lh_p_3sigma: {mcmc_delta_lh_p_3sigma}\n")
                f.write(f"mcmc_lh_keep_threshold: {mcmc_keep_threshold}\n")
                f.write(f"mcmc_lh_keep_count: {int(len(mcmc_nll))}\n")
                f.write("mcmc_nll_neglogq_plot: mcmc_nll_vs_neglogq_kept.png\n")
            if delta_nll_nf is not None:
                f.write("delta_nll_plot: delta_nll_nf_vs_mcmc.png\n")
                f.write(f"nll_bestfit: {float(nll_bestfit)}\n")
        f.write(f"nf_samples: {samples_nf.shape}\n")
        f.write(f"matching_mode: index_order_only\n")
        f.write(f"compared_dims: {d}\n")
        f.write(f"do_reweight_nf: {do_reweight_nf}\n")
        if do_reweight_nf:
            f.write(f"reweight_num_workers: {reweight_num_workers}\n")
            f.write(f"reweight_outlier_keep: {int(outlier_mask.sum())}/{len(outlier_mask)}\n")
            if save_samples:
                f.write("nf_samples_with_weights.npz: NF samples, log_weights, weights, param_names, bestfit, sigma\n")
            f.write(f"diag_clip_q: {diag_clip_q}\n")
            f.write("--- diagnostics/weights/ ---\n")
            f.write("  weight_summary.png: full + clipped log_w dist, Lorenz curve, top-k concentration\n")
            f.write("  sorted_corr_summary.png: top-40 params by |corr(log_w, x_k)|\n")
            f.write("  reweight_mean_shift.png: (weighted_mean - raw_mean)/sigma bar chart\n")
            f.write("  reweight_logw_param_corr.png: Pearson corr(log_w, x_k) all params\n")
            f.write("  reweight_decomposition.png: log_w vs logq_NF, log_w vs LH, LH vs logq_NF\n")
            f.write("--- diagnostics/params/ ---\n")
            f.write("  nf_pull_distributions.png: (x_NF - mu)/sigma vs N(0,1) per group\n")
            f.write("  *_marginals_grid.png: NF vs NF-reweighted per parameter group\n")
            f.write("  *_logw_vs_param_grid.png: 2D density log_w vs each param (slope = NF bias)\n")
            f.write("  *_high_vs_low_weight.png: top-25% vs bottom-25% weight samples per param\n")
            f.write("--- diagnostics/correlations/ ---\n")
            f.write("  corr_matrix_detector_nonlinear.png: NF / NF-reweighted / postfit corr matrices\n")
            f.write("  corr_matrix_physics.png: same for physics parameter block\n")
        f.write("corr2d/nf_*.png: NF-only 2D histograms (unweighted + reweighted if applicable)\n")
        for i in range(d):
            f.write(f"  {i:03d} {labels[i]}\n")

    # -------------------------
    # Marginals
    # -------------------------
    bins_n = int(cfg.bins)
    if do_plot_mcmc and samples_mcmc_c is not None:
        print(f"Comparing {d} parameters.", flush=True)
    else:
        print(f"Plotting {d} NF-only parameter marginals.", flush=True)
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

        if do_plot_mcmc and samples_mcmc_c is not None:
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

        if do_plot_mcmc and samples_mcmc_c is not None:
            plt.hist(x_m, bins=edges, histtype="step", density=True, label=f"MCMC kept (n={len(x_m)})")
            if samples_mcmc_all_c is not None:
                x_m_all = samples_mcmc_all_c[:, k]
                plt.hist(
                    x_m_all,
                    bins=edges,
                    histtype="stepfilled",
                    density=True,
                    alpha=0.18,
                    label=f"MCMC all (n={len(x_m_all)})",
                )

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

    # -------------------------
    # 2D correlation plots
    # -------------------------
    corr2d_bins = int(getattr(cfg, "corr2d_bins", 60))
    dims_cfg = getattr(cfg, "corr2d_dims", None)

    if dims_cfg is None or (isinstance(dims_cfg, (list, tuple)) and len(dims_cfg) == 0) or (isinstance(dims_cfg, str) and str(dims_cfg).strip() == ""):
        # Default: sample a cross-section across parameter groups
        _pg = _classify_param_groups(labels) if not do_reweight_nf else param_groups
        _default_dims = []
        for _g in ("physics", "detector", "nonlinear"):
            _gd = _pg.get(_g, [])
            _default_dims += _gd[:5]
        dims = _default_dims if len(_default_dims) >= 2 else list(range(max(0, d - 6), d))
    else:
        dims = parse_dim_list(dims_cfg, d)

    if len(dims) >= 2:
        # NF-vs-MCMC side-by-side (only when MCMC is available)
        if do_plot_mcmc and samples_mcmc_c is not None:
            print(f"Plotting NF-vs-MCMC 2D correlations for dims: {dims}", flush=True)
            for i in range(len(dims)):
                for j in range(i + 1, len(dims)):
                    a = dims[i]
                    b = dims[j]
                    outp = corr2d_dir / f"nfvsmcmc_{a:03d}_{b:03d}.png"
                    plot_2d_hist_side_by_side(
                        samples_nf_c[:, a], samples_nf_c[:, b],
                        samples_mcmc_c[:, a], samples_mcmc_c[:, b],
                        labels[a], labels[b],
                        outp,
                        bins=corr2d_bins,
                    )
        else:
            print("do_plot_mcmc=False or no filtered MCMC samples, skipping NF-vs-MCMC 2D plots.", flush=True)

        # NF-only 2D correlations (always generated)
        print(f"Plotting NF-only 2D correlations for {len(dims)} dims ({len(dims)*(len(dims)-1)//2} pairs)...", flush=True)
        plot_nf_only_corr2d(
            samples_nf=samples_nf_c,
            labels=labels,
            dims=dims,
            outpath_dir=corr2d_dir,
            bins=corr2d_bins,
            log_weights=reweight_nf_to_lh if do_reweight_nf else None,
            outlier_mask=outlier_mask if do_reweight_nf else None,
        )
    else:
        print("corr2d_dims resolved to <2 dims, skipping all 2D correlation plots.", flush=True)

    print(f"Done. Outputs in: {str(out_dir)}", flush=True)


if __name__ == "__main__":
    main()