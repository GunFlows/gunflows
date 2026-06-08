#!/usr/bin/env python3
"""
make_paper_plots.py — completely independent end-to-end script.

First run  : loads GUNDAM + NF model, draws throws, computes everything,
             saves all arrays to <save_dir>/cache/.
Later runs : detects the cache, skips GUNDAM/NF entirely, goes straight
             to plotting.  Set force_recompute: true in config to redo.

Output tree
-----------
<save_dir>/
  cache/
    bin_edges_<var>.npy
    histograms_<src>_<var>_<stream>.npy   (n_throws, n_bins)
    nll_data.npz   {nll_gundam, log_nf, log_g, samples, bestfit, cov, par_names}
  plots/
    kinematic/<VAR>/<STREAM>/
      spectrum.pdf          4-panel: yield + 1σ hw + Δ⟨N⟩% + Δσ%
      violin.pdf            split violin: MCMC left | NF right vs Gaussian
      correlation_<SRC>.pdf bin-yield Pearson-r matrix
      corner_<SRC>.pdf      bin-yield corner (all bins)
    nll/
      hist2d_nf.pdf         ΔNLL vs −log q_NF  (2-D histogram)
      hist2d_gauss.pdf      ΔNLL vs −log g      (2-D histogram)
      logweights.pdf        IS log-weight distributions
    parameters/
      corner_weighted.pdf   IS-weighted parameter corner (KS-ranked)
      corner_unweighted.pdf same, uniform weights
      pulls.pdf             per-parameter pull histograms vs N(0,1)
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from omegaconf import DictConfig, OmegaConf
from scipy.stats import gaussian_kde
from scipy.special import erf
from matplotlib.ticker import MaxNLocator

try:
    import mplhep as hep
    _HEP = True
except ImportError:
    _HEP = False
    warnings.warn("mplhep not found — using plain matplotlib style.")

NF_LOCAL = os.path.join(os.path.dirname(__file__), "normalizing-flows")
sys.path.append(os.path.abspath(NF_LOCAL))


# ─────────────────────────────────────────────────────────────────────────────
# Global style constants
# ─────────────────────────────────────────────────────────────────────────────

COLORS       = {"NF": "#1f77b4", "Gaussian": "#d62728", "MCMC": "#2ca02c"}
SOURCE_ORDER = ("Gaussian", "NF", "MCMC")
STREAMS      = ("FHC", "RHC")

_VAR_LABELS = {
    "Enu":        r"$E_\nu$ [GeV]",
    "Pmu":        r"$p_\mu$ [MeV/$c$]",
    "CosThetamu": r"$\cos\theta_\mu$",
}
_DEFAULT_CI = (0.6827, 0.9545, 0.9973)


# ─────────────────────────────────────────────────────────────────────────────
# Style helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_style() -> None:
    # Do NOT apply mplhep style — it resets font sizes and fights our settings.
    # Set everything explicitly here and in each plot function.

    _usetex = False
    try:
        import subprocess
        subprocess.run(["latex", "--version"], capture_output=True, check=True)
        _usetex = True
    except Exception:
        pass

    plt.rcParams.update({
        # --- text ---
        "text.usetex":          _usetex,
        "mathtext.fontset":     "cm",        # fallback: Computer Modern math
        "font.family":          "serif",
        "font.size":            16,          # default; each figure overrides explicitly
        "axes.labelsize":       16,
        "xtick.labelsize":      14,
        "ytick.labelsize":      14,
        "legend.fontsize":      14,
        "legend.frameon":       False,
        **({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}"}
           if _usetex else {}),
        # --- ticks ---
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.top":            False,
        "ytick.right":          False,
        "xtick.minor.visible":  False,
        "ytick.minor.visible":  False,
        "xtick.major.size":     7,
        "ytick.major.size":     7,
        "xtick.major.width":    1.4,
        "ytick.major.width":    1.4,
        # --- figure ---
        "axes.linewidth":       1.4,
        "axes.labelpad":        10,
        "lines.linewidth":      2.0,
        "figure.dpi":           150,
        "savefig.dpi":          200,        # moderate DPI keeps file size sane
        "savefig.bbox":         "tight",
    })
    print(f"  Text backend: {'LaTeX' if _usetex else 'CM mathtext'}", flush=True)


# Helper: apply explicit font sizes to an axis after the fact
def _ax_fontsize(ax, label_fs: int, tick_fs: int | None = None,
                 legend_fs: int | None = None) -> None:
    """Force explicit font sizes on a single axes, overriding any style.

    label_fs  : axis-label size.
    tick_fs   : tick-value size  (default label_fs - 2). Decoupled so axis
                labels can grow while tick values stay fixed.
    legend_fs : legend text size (default label_fs - 2).
    """
    if tick_fs is None:
        tick_fs = label_fs - 2
    if legend_fs is None:
        legend_fs = label_fs - 2
    ax.xaxis.label.set_size(label_fs)
    ax.yaxis.label.set_size(label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(tick_fs)
    if ax.get_legend():
        for t in ax.get_legend().get_texts():
            t.set_fontsize(legend_fs)


def _savefig(fig: plt.Figure, path: Path, fmt: str = "pdf") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(f".{fmt}"))
    plt.close(fig)
    print(f"  Saved: {path.with_suffix('.' + fmt)}", flush=True)


def _hep_label(ax: plt.Axes, label: str = "") -> None:
    """Optional text label in the top-left corner (e.g. experiment name)."""
    if not label:
        return
    if _HEP:
        try:
            hep.label.exp_label(exp=label, ax=ax)
            return
        except Exception:
            pass
    ax.text(0.04, 0.97, label, transform=ax.transAxes,
            va="top", ha="left", fontsize=12, fontweight="bold", style="italic")


def _step_xy(edges: np.ndarray, vals: np.ndarray):
    x = np.concatenate([[edges[0]], np.repeat(edges[1:-1], 2), [edges[-1]]])
    return x, np.repeat(vals, 2)


def _percentile_ranges(arrays: list[np.ndarray], q: float = 0.999,
                       pad: float = 0.0) -> list[tuple]:
    """Common per-column (lo,hi) ranges over one or more (N, D) arrays, using
    the (1-q, q) percentiles so all sources share identical corner axes.
    pad: fraction of the range added on EACH side (e.g. 0.15 → +15% both ends)."""
    stacked = np.concatenate([np.asarray(a, dtype=np.float64) for a in arrays
                              if a is not None and len(a)], axis=0)
    lo = np.nanpercentile(stacked, 100 * (1 - q), axis=0)
    hi = np.nanpercentile(stacked, 100 * q,       axis=0)
    # guard against degenerate (lo == hi) columns
    eq = hi <= lo
    hi = np.where(eq, lo + 1e-9, hi)
    if pad:
        w  = hi - lo
        lo = lo - pad * w
        hi = hi + pad * w
    return list(zip(lo.tolist(), hi.tolist()))


def _percentile_bands(
    arr: np.ndarray,
    levels: tuple = _DEFAULT_CI,
) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (np.percentile(arr, 100 * (1 - l) / 2, axis=0),
         np.percentile(arr, 100 * (1 + l) / 2, axis=0))
        for l in levels
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Computation: NF model loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_nf(cfg, dataset, ckpt_path: str):
    from gunflows.utils.build_flow import build_base, build_flow_layers, build_model
    m = cfg.experiment.model
    device = cfg.get("device", cfg.experiment.get("device", "cpu"))

    base   = build_base(int(m.total_dim))
    n_spline = len(dataset.phase_space_dim)
    tail   = torch.ones(n_spline) * float(m.tail_bound)
    flows  = build_flow_layers(
        int(m.nflows), n_spline, int(m.hidden), int(m.nlayers), int(m.nbins),
        tail, n_context=int(m.total_dim) - n_spline,
    )
    kw = {}
    for k in ("n_context_flows", "hidden_dim", "n_hidden_layers"):
        if hasattr(m, k):
            kw[k] = int(getattr(m, k))

    freeze = bool(m.freeze_covflow) if hasattr(m, "freeze_covflow") else False
    model  = build_model(base, flows, dataset, m.context_transform, freeze, **kw)
    state  = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Computation: sampling dataset (eigen → physical space transform only)
# ─────────────────────────────────────────────────────────────────────────────

class _SamplingDataset:
    def __init__(self, phase_space_dim, mean: np.ndarray, cov: np.ndarray):
        self.phase_space_dim      = list(phase_space_dim)
        self.list_dim_conditionnal = [i for i in range(len(mean))
                                      if i not in set(self.phase_space_dim)]
        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        cov_t  = torch.as_tensor(cov,  dtype=torch.float32)
        std    = torch.sqrt(torch.clamp(torch.diag(cov_t), min=1e-12))
        dinv   = torch.diag(1.0 / std)
        cov_std = dinv @ cov_t @ dinv
        self.mean        = mean_t
        self.std_per_dim = std
        self.cholesky    = torch.linalg.cholesky(cov_std + 1e-6 * torch.eye(cov_std.shape[0]))

    def transform_eigen_space_to_data_space(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std_per_dim.to(x) + self.mean.to(x)


# ─────────────────────────────────────────────────────────────────────────────
# Computation: GUNDAM event cache (single pass)
# ─────────────────────────────────────────────────────────────────────────────

def _build_event_caches(
    sampler,
    var_bin_edges: dict[str, np.ndarray],
    stream_var: str = "isRHC",
) -> dict[str, dict[str, tuple[list, np.ndarray]]]:
    n_bins_map = {v: len(e) - 1 for v, e in var_bin_edges.items()}
    buckets    = {v: {s: ([], []) for s in STREAMS} for v in var_bin_edges}

    for sp in sampler.likelihood_interface.getSamplePairList():
        mc = sp.model
        if not mc.isEnabled():
            continue
        for event in mc.getEventList():
            leaves = event.getVariables()
            is_rhc = int(leaves.fetchVariable(stream_var).getVarAsDouble())
            stream = "RHC" if is_rhc else "FHC"
            for var, edges in var_bin_edges.items():
                val = leaves.fetchVariable(var).getVarAsDouble()
                idx = int(np.digitize(val, edges)) - 1
                if 0 <= idx < n_bins_map[var]:
                    buckets[var][stream][0].append(event)
                    buckets[var][stream][1].append(idx)

    out = {}
    for var in var_bin_edges:
        out[var] = {}
        for stream in STREAMS:
            ev, ix = buckets[var][stream]
            out[var][stream] = (ev, np.array(ix, dtype=np.int32))
            print(f"  Cache [{var}][{stream}]: {len(ev)} events", flush=True)
    return out


def _fill_histogram(events: list, indices: np.ndarray, n_bins: int) -> np.ndarray:
    w = np.fromiter((e.getEventWeight() for e in events), dtype=np.float64, count=len(events))
    return np.bincount(indices, weights=w, minlength=n_bins)


# ─────────────────────────────────────────────────────────────────────────────
# Computation: histogram sampling loop (NF / Gaussian / MCMC)
# ─────────────────────────────────────────────────────────────────────────────

def _inject_and_fill(
    sampler,
    params_batch: np.ndarray,
    var_caches: dict[str, dict[str, tuple[list, np.ndarray]]],
    n_bins_map: dict[str, int],
    label: str,
    verbose: bool = True,
) -> tuple[dict[str, dict[str, list[np.ndarray]]], list[float], list[int]]:
    hists: dict[str, dict[str, list]] = {v: {s: [] for s in STREAMS} for v in var_caches}
    nll_throws: list[float] = []
    acc_idx: list[int] = []           # indices of accepted throws (nll != -1)
    for i, theta in enumerate(params_batch):
        nll, _, _ = sampler.inject_params_and_compute_likelihood(
            theta.tolist(), extend_continue=False)
        if nll == -1:
            continue
        for var, streams in var_caches.items():
            for stream, (ev, ix) in streams.items():
                if ev:
                    hists[var][stream].append(
                        _fill_histogram(ev, ix, n_bins_map[var]))
        nll_throws.append(float(nll))   # per-throw NLL, aligned with histograms
        acc_idx.append(i)
        if verbose:
            n_ok = max(max(len(hists[v][s]) for s in STREAMS) for v in hists)
            print(f"  [{label} {n_ok:5d}] NLL={nll:.4f}", flush=True, end="\r")
    if verbose:
        print()
    return hists, nll_throws, acc_idx


# ── Parallel histogram filling: one full-data sampler + event cache per worker ──
_HIST_SAMPLER = None
_HIST_CACHE = None
_HIST_NBINS = None


def _init_hist_worker(llh_kw: dict, var_bin_edges_lists: dict) -> None:
    global _HIST_SAMPLER, _HIST_CACHE, _HIST_NBINS
    from gunflows.likelihood_sampler import LikelihoodSampler
    _HIST_SAMPLER = LikelihoodSampler(**llh_kw)          # light_mode=False → full data
    vbe = {k: np.asarray(v, dtype=np.float64) for k, v in var_bin_edges_lists.items()}
    _HIST_CACHE = _build_event_caches(_HIST_SAMPLER, vbe)
    _HIST_NBINS = {k: len(v) - 1 for k, v in vbe.items()}


def _fill_chunk(params_chunk):
    """Worker: fill histograms for a chunk of throws. Returns only numpy/scalars
    (no GUNDAM objects) so it pickles cleanly across the process boundary."""
    new_h, nll_list, acc = _inject_and_fill(
        _HIST_SAMPLER, np.asarray(params_chunk, dtype=np.float64),
        _HIST_CACHE, _HIST_NBINS, "worker", verbose=False)
    out = {v: {s: np.asarray(new_h[v][s], dtype=np.float64) for s in STREAMS}
           for v in new_h}
    return nll_list, acc, out


def _fill_source_parallel(pool, n_workers, params):
    """Map a source's throws over the worker pool; combine results in order.
    Returns (hists_stacked[var][stream]=(m,nbins), nll_array, accepted_global_idx)."""
    n_chunks = max(1, n_workers * 4)
    chunks = [c for c in np.array_split(np.asarray(params), n_chunks) if len(c)]
    results = pool.map(_fill_chunk, chunks)

    hists = None
    nll_all: list[float] = []
    acc_global: list[int] = []
    offset = 0
    for (nll_list, acc, out), chunk in zip(results, chunks):
        if hists is None:
            hists = {v: {s: [] for s in STREAMS} for v in out}
        for v in out:
            for s in STREAMS:
                if out[v][s].size:
                    hists[v][s].append(out[v][s])
        nll_all.extend(nll_list)
        acc_global.extend(offset + int(a) for a in acc)
        offset += len(chunk)

    hists_stacked = {
        v: {s: (np.concatenate(hists[v][s], axis=0) if hists[v][s] else None)
            for s in STREAMS}
        for v in (hists or {})
    }
    return hists_stacked, np.asarray(nll_all, dtype=np.float64), np.asarray(acc_global, dtype=int)


def _run_histogram_loop(
    sampler,
    nf_model,
    dataset: _SamplingDataset,
    mcmc_throws: np.ndarray | None,
    var_caches: dict,
    n_bins_map: dict,
    num_samples: int,
    batch_size: int,
    use_nf: bool,
    use_gaussian: bool,
    use_mcmc: bool,
    bestfit: np.ndarray,
    cov: np.ndarray,
    rng: np.random.Generator,
    cache_dir: Path,
    save_every: int,
    var_bin_edges: dict,
    preloaded: dict | None = None,
    hist_num_workers: int = 1,
    llh_kw: dict | None = None,
    force: bool = False,
) -> dict[str, dict[str, dict[str, list[np.ndarray]]]]:
    """Returns all_hists[source][var][stream] = [array(n_bins), ...]
    preloaded: optionally pre-populate from cache so only missing sources run.
    hist_num_workers>1: fill histograms in parallel (each worker = own full-data
    sampler + own event cache); llh_kw must then provide the sampler config.
    """
    def _empty():
        return {v: {s: [] for s in STREAMS} for v in var_caches}

    if preloaded is not None:
        all_hists = preloaded
    else:
        all_hists = {
            "NF":       _empty(),
            "Gaussian": _empty(),
            "MCMC":     _empty(),
        }

    log_norm = float(getattr(nf_model, "log_norm", torch.tensor(0.0))) if nf_model else 0.0

    def _load(name):
        # Never read stale per-throw arrays on a forced recompute, otherwise
        # they get prepended to the fresh ones and corrupt the alignment.
        p = cache_dir / name
        return list(np.load(p)) if (p.exists() and not force) else []

    # Per-throw records, aligned with the histogram throws. Reused for the NLL
    # comparison + rate-vs-NLL plots (no separate resampling / parallel pool).
    all_nll  = {"NF": _load("nll_throws_nf.npy"),  "Gaussian": _load("nll_throws_gaussian.npy")}
    all_logq = {"NF": _load("logq_nf.npy")}                       # NF log q (normalised)
    all_samp = {"NF": _load("samp_nf.npy"),        "Gaussian": _load("samp_gaussian.npy")}

    def _count(h):
        fv = next(iter(h))
        return max(len(h[fv][s]) for s in STREAMS)

    def _extend(dst, src):
        for v in src:
            for s in STREAMS:
                dst[v][s].extend(src[v].get(s, []))

    def _done(label):
        flags = {"NF": use_nf, "Gaussian": use_gaussian, "MCMC": use_mcmc}
        return (not flags[label]) or _count(all_hists[label]) >= num_samples

    def _checkpoint():
        for label in SOURCE_ORDER:
            for var, edges in var_bin_edges.items():
                for stream in STREAMS:
                    hlist = all_hists[label][var][stream]
                    if hlist:
                        arr = np.array(hlist, dtype=np.float64)
                        tag = label.lower()
                        np.save(cache_dir / f"histograms_{tag}_{var.lower()}_{stream.lower()}.npy", arr)
        if all_nll["NF"]:
            np.save(cache_dir / "nll_throws_nf.npy",  np.array(all_nll["NF"],  dtype=np.float64))
            np.save(cache_dir / "logq_nf.npy",        np.array(all_logq["NF"], dtype=np.float64))
            np.save(cache_dir / "samp_nf.npy",        np.array(all_samp["NF"], dtype=np.float32))
        if all_nll["Gaussian"]:
            np.save(cache_dir / "nll_throws_gaussian.npy", np.array(all_nll["Gaussian"], dtype=np.float64))
            np.save(cache_dir / "samp_gaussian.npy",       np.array(all_samp["Gaussian"], dtype=np.float32))
        print("  [checkpoint saved]", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # Parallel path: each worker owns a full-data sampler + event cache.
    # ════════════════════════════════════════════════════════════════════════
    if hist_num_workers and hist_num_workers > 1:
        import multiprocessing as mp
        if llh_kw is None:
            raise ValueError("hist_num_workers>1 requires llh_kw for worker samplers")
        ctx = mp.get_context("spawn")
        vbe_lists = {k: np.asarray(v).tolist() for k, v in var_bin_edges.items()}
        print(f"  Parallel histogram fill: {hist_num_workers} workers", flush=True)

        def _absorb(label, hs, nll, acc, x_params=None, lq=None):
            for v in hs:
                for s in STREAMS:
                    if hs[v][s] is not None:
                        all_hists[label][v][s].extend(list(hs[v][s]))
            if label in all_nll:
                all_nll[label].extend(nll.tolist())
            if x_params is not None and label in all_samp:
                all_samp[label].extend(x_params[acc])
            if lq is not None and label == "NF":
                all_logq["NF"].extend((lq[acc] + log_norm).tolist())

        with ctx.Pool(hist_num_workers, initializer=_init_hist_worker,
                      initargs=(llh_kw, vbe_lists)) as pool:
            with torch.no_grad():
                # NF
                if not _done("NF"):
                    z, lq = nf_model.sample(num_samples)
                    lq    = lq.cpu().numpy().reshape(-1)
                    x_nf  = dataset.transform_eigen_space_to_data_space(z).cpu().numpy()
                    hs, nll, acc = _fill_source_parallel(pool, hist_num_workers, x_nf)
                    _absorb("NF", hs, nll, acc, x_params=x_nf, lq=lq)
                    print(f"  NF: {_count(all_hists['NF'])}/{num_samples}", flush=True)
                    _checkpoint()
                # Gaussian
                if not _done("Gaussian"):
                    x_g = rng.multivariate_normal(bestfit, cov, size=num_samples)
                    hs, nll, acc = _fill_source_parallel(pool, hist_num_workers, x_g)
                    _absorb("Gaussian", hs, nll, acc, x_params=x_g)
                    print(f"  Gaussian: {_count(all_hists['Gaussian'])}/{num_samples}", flush=True)
                    _checkpoint()
                # MCMC
                if not _done("MCMC") and mcmc_throws is not None:
                    hs, nll, acc = _fill_source_parallel(
                        pool, hist_num_workers, mcmc_throws[:num_samples])
                    _absorb("MCMC", hs, nll, acc)   # NLL not stored for MCMC
                    print(f"  MCMC: {_count(all_hists['MCMC'])}/{num_samples}", flush=True)
                    _checkpoint()
        _checkpoint()
        return all_hists

    with torch.no_grad():
        while not (_done("NF") and _done("Gaussian") and _done("MCMC")):
            # ── NF ─────────────────────────────────────────────────────────
            if not _done("NF"):
                prev = _count(all_hists["NF"])
                need = min(batch_size, num_samples - prev)
                z, lq = nf_model.sample(need)
                lq    = lq.cpu().numpy().reshape(-1)
                x_nf  = dataset.transform_eigen_space_to_data_space(z).cpu().numpy()
                new_h, new_nll, acc = _inject_and_fill(sampler, x_nf, var_caches, n_bins_map, "NF")
                _extend(all_hists["NF"], new_h)
                all_nll["NF"].extend(new_nll)
                all_logq["NF"].extend((lq[acc] + log_norm).tolist())   # normalised NF log q
                all_samp["NF"].extend(x_nf[acc])
                cur = _count(all_hists["NF"])
                print(f"  NF: {cur}/{num_samples}", flush=True)
                if save_every > 0 and (cur // save_every) > (prev // save_every):
                    _checkpoint()

            # ── Gaussian ────────────────────────────────────────────────────
            if not _done("Gaussian"):
                prev = _count(all_hists["Gaussian"])
                need = min(batch_size, num_samples - prev)
                x_g  = rng.multivariate_normal(bestfit, cov, size=need)
                new_h, new_nll, acc = _inject_and_fill(sampler, x_g, var_caches, n_bins_map, "Gauss")
                _extend(all_hists["Gaussian"], new_h)
                all_nll["Gaussian"].extend(new_nll)
                all_samp["Gaussian"].extend(x_g[acc])
                cur = _count(all_hists["Gaussian"])
                print(f"  Gaussian: {cur}/{num_samples}", flush=True)
                if save_every > 0 and (cur // save_every) > (prev // save_every):
                    _checkpoint()

            # ── MCMC ────────────────────────────────────────────────────────
            if not _done("MCMC") and mcmc_throws is not None:
                prev = _count(all_hists["MCMC"])
                need = min(batch_size, num_samples - prev)
                x_mc = mcmc_throws[prev: prev + need]
                new_h, _, _ = _inject_and_fill(sampler, x_mc, var_caches, n_bins_map, "MCMC")
                _extend(all_hists["MCMC"], new_h)
                # MCMC per-throw NLL is not used anywhere → not stored
                cur = _count(all_hists["MCMC"])
                print(f"  MCMC: {cur}/{num_samples}", flush=True)
                if save_every > 0 and (cur // save_every) > (prev // save_every):
                    _checkpoint()

    _checkpoint()
    return all_hists


# ─────────────────────────────────────────────────────────────────────────────
# Computation: NLL comparison arrays (log_p, log_NF, log_g)
# ─────────────────────────────────────────────────────────────────────────────

def _log_gaussian_batch(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    D   = mean.shape[0]
    L   = np.linalg.cholesky(cov + 1e-10 * np.eye(D))
    diff = X - mean[None, :]
    y   = np.linalg.solve(L, diff.T).T
    return -0.5 * (np.sum(y ** 2, axis=1)
                   + 2 * np.sum(np.log(np.diag(L)))
                   + D * np.log(2 * np.pi))


# ── Parallel GUNDAM NLL evaluation (one LikelihoodSampler per worker) ────────
# Mirrors the reweight pool in sample_mcmc_toy_mathias.py.

_REWEIGHT_SAMPLER = None


def _init_reweight_worker(llh_kw: dict) -> None:
    global _REWEIGHT_SAMPLER
    from gunflows.likelihood_sampler import LikelihoodSampler
    _REWEIGHT_SAMPLER = LikelihoodSampler(**llh_kw)


def _reweight_one(theta):
    nll, _, _ = _REWEIGHT_SAMPLER.inject_params_and_compute_likelihood(
        list(theta), extend_continue=False)
    return float(nll)


def _nll_batch(
    params: np.ndarray,
    llh_kw: dict,
    n_workers: int,
    fallback_sampler,
    label: str = "",
) -> np.ndarray:
    """GUNDAM NLL for each row of `params`. Parallel over n_workers processes
    (each with its own LikelihoodSampler) when n_workers > 1, else serial."""
    rows = [r.tolist() for r in np.asarray(params, dtype=np.float64)]
    if n_workers and n_workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")   # safe alongside CUDA / ROOT in the parent
        chunk = max(1, len(rows) // (n_workers * 8))
        print(f"  [{label}] parallel NLL: {len(rows)} evals on {n_workers} workers",
              flush=True)
        with ctx.Pool(n_workers, initializer=_init_reweight_worker,
                      initargs=(llh_kw,)) as pool:
            out = pool.map(_reweight_one, rows, chunksize=chunk)
        return np.array(out, dtype=np.float64)
    # serial fallback
    out = []
    for i, r in enumerate(rows):
        nll, _, _ = fallback_sampler.inject_params_and_compute_likelihood(
            r, extend_continue=False)
        out.append(float(nll))
        if (i + 1) % 500 == 0:
            print(f"  [{label}] serial NLL: {i+1}/{len(rows)}", flush=True, end="\r")
    print()
    return np.array(out, dtype=np.float64)


def _compute_nll_data(
    sampler,
    nf_model,
    dataset: _SamplingDataset,
    bestfit: np.ndarray,
    cov: np.ndarray,
    n_samples: int,
    batch_size: int,
    llh_kw: dict,
    n_workers: int = 1,
    seed: int = 12345,
) -> dict:
    """
    Two independent sample sets evaluated through GUNDAM:

      NF draws        → ΔNLL, log q_NF, log g, samples   (for the NF hist2d,
                        log(p/q_NF) weights, and the NF parameter corners)
      Gaussian draws  → ΔNLL, log g, samples             (for the Gaussian
                        hist2d and the log(p/g) weights)

    GUNDAM NLL evaluations (the reweighting step) are parallelised over
    n_workers processes, each with its own LikelihoodSampler.
    ΔNLL is centred at the best fit (= NLL − NLL_best).
    """
    log_norm = float(getattr(nf_model, "log_norm", torch.tensor(0.0)))

    bf_nll, _, _ = sampler.inject_params_and_compute_likelihood(
        bestfit.tolist(), extend_continue=False)
    print(f"  Best-fit NLL = {bf_nll:.4f}", flush=True)

    # ── 1. Generate parameter vectors (GPU/NF in the main process) ───────────
    x_nf_list, lq_list = [], []
    with torch.no_grad():
        while len(x_nf_list) < n_samples:
            n_need = min(batch_size, n_samples - len(x_nf_list))
            z, lq_z = nf_model.sample(n_need)
            x = dataset.transform_eigen_space_to_data_space(z).cpu().numpy()
            lq = lq_z.cpu().numpy().reshape(-1)
            x_nf_list.extend(list(x))
            lq_list.extend(list(lq))
    x_nf  = np.asarray(x_nf_list[:n_samples], dtype=np.float64)
    lq_nf = np.asarray(lq_list[:n_samples],  dtype=np.float64) + log_norm

    rng = np.random.default_rng(seed)
    x_g = rng.multivariate_normal(bestfit, cov, size=n_samples)

    # ── 2. Evaluate NLL in parallel for both sample sets ─────────────────────
    nll_x_nf = _nll_batch(x_nf, llh_kw, n_workers, sampler, label="NF draws")
    nll_x_g  = _nll_batch(x_g,  llh_kw, n_workers, sampler, label="Gaussian draws")

    # ── 3. Filter invalid (out-of-domain) draws and assemble ─────────────────
    m_nf = nll_x_nf != -1
    samp_nf  = x_nf[m_nf].astype(np.float32)
    nll_nf   = nll_x_nf[m_nf] - float(bf_nll)
    lnf      = lq_nf[m_nf]
    lg_at_nf = _log_gaussian_batch(x_nf[m_nf], bestfit, cov)

    m_g    = nll_x_g != -1
    samp_g = x_g[m_g].astype(np.float32)
    nll_g  = nll_x_g[m_g] - float(bf_nll)
    lg_g   = _log_gaussian_batch(x_g[m_g], bestfit, cov)

    print(f"  Accepted: NF {m_nf.sum()}/{len(m_nf)}, "
          f"Gaussian {m_g.sum()}/{len(m_g)}", flush=True)

    return {
        # NF draws
        "nll_nf":      nll_nf,
        "log_nf":      lnf,
        "log_g_nf":    lg_at_nf,
        "samples_nf":  samp_nf,
        # Gaussian draws
        "nll_g":       nll_g,
        "log_g_g":     lg_g,
        "samples_g":   samp_g,
        "bestfit_nll": float(bf_nll),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Computation: MCMC chain reader
# ─────────────────────────────────────────────────────────────────────────────

def _load_mcmc(
    mcmc_chain: str,
    n_samples: int,
    burnin_frac: float = 0.0,
    max_steps: int | None = None,
    thin: int | None = None,
) -> np.ndarray:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    f = ROOT.TFile(mcmc_chain, "READ")
    if not f or f.IsZombie():
        raise FileNotFoundError(f"Cannot open MCMC file: {mcmc_chain}")
    tree = f.Get("FitterEngine/fit/MCMC")
    if not tree:
        raise RuntimeError("TTree 'FitterEngine/fit/MCMC' not found")
    n_total    = int(tree.GetEntries())
    start      = int(n_total * burnin_frac)
    stop       = min(n_total, start + max_steps) if max_steps else n_total
    n_avail    = stop - start
    m          = thin if thin else max(1, n_avail // n_samples)
    indices    = list(range(start, stop, m))[:n_samples]
    rows = []
    for i in indices:
        tree.GetEntry(i)
        rows.append(np.array(list(tree.Points), dtype=np.float64))
    f.Close()
    print(f"  MCMC: {len(rows)} throws loaded (thin={m})", flush=True)
    return np.stack(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Computation: parameter names from GUNDAM
# ─────────────────────────────────────────────────────────────────────────────

def _get_par_names(sampler) -> list[str]:
    names = []
    try:
        pm = sampler.likelihood_interface.getModelPropagator().getParametersManager()
        for pset in pm.getParameterSetsList():
            if not pset.isEnabled():
                continue
            for p in pset.getParameterList():
                if p.isEnabled():
                    names.append(str(p.getName()))
    except Exception:
        pass
    return names


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — kinematic spectrum (4-panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectrum(
    results: dict[str, np.ndarray],
    bin_edges: np.ndarray,
    save_path: Path,
    xlabel: str = "variable",
    ci_levels: tuple = _DEFAULT_CI,
    legend_loc: str = "upper right",
    width_scale: float = 1.0,
    font_scale: float = 1.0,
    fmt: str = "pdf",
) -> None:
    """Main spectrum: log-yield panel + optional NF−Gaussian diff panels (no ratio panel)."""
    if not results:
        return
    centers   = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths    = bin_edges[1:] - bin_edges[:-1]
    has_nf    = "NF"       in results
    has_gauss = "Gaussian" in results
    show_diff = has_nf and has_gauss

    # 1 panel (main) or 3 panels (main + Δmean + Δσ)
    n_panels      = 3 if show_diff else 1
    height_ratios = [4, 1, 1][:n_panels]
    alphas        = np.linspace(0.45, 0.12, len(ci_levels))

    FS  = int(32 * font_scale)
    fig = plt.figure(figsize=(16 * width_scale, 5 + 4 * n_panels))
    gs  = gridspec.GridSpec(n_panels, 1, height_ratios=height_ratios, hspace=0.08)
    ax  = fig.add_subplot(gs[0])
    ax_dm = fig.add_subplot(gs[1], sharex=ax) if show_diff else None
    ax_ds = fig.add_subplot(gs[2], sharex=ax) if show_diff else None

    gauss_raw = results["Gaussian"].mean(axis=0) if has_gauss else None
    gauss_std = results["Gaussian"].std(axis=0)  if has_gauss else None
    means_raw, stds_raw = {}, {}

    for label in SOURCE_ORDER:
        if label not in results:
            continue
        color = COLORS[label]
        arr   = results[label]
        means_raw[label] = arr.mean(axis=0)
        stds_raw[label]  = arr.std(axis=0)
        mean_d = means_raw[label] / widths
        bands  = _percentile_bands(arr, ci_levels)

        for (lo_r, hi_r), alpha in zip(bands, alphas):
            sx, sy_lo = _step_xy(bin_edges, np.maximum(lo_r, 1e-10) / widths)
            _,  sy_hi = _step_xy(bin_edges, hi_r / widths)
            ax.fill_between(sx, sy_lo, sy_hi, facecolor=color, alpha=alpha, linewidth=0)
        sx, sy = _step_xy(bin_edges, mean_d)
        ax.plot(sx, sy, color=color, linewidth=2.2, label=label)

    ax.set_ylabel("Event yield / bin width")
    ax.legend(loc=legend_loc)
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.xaxis.set_major_locator(MaxNLocator(6, prune="both"))
    plt.setp(ax.get_xticklabels(), visible=not show_diff)
    if not show_diff:
        ax.set_xlabel(xlabel)
    _ax_fontsize(ax, FS)

    if show_diff:
        bar_kw  = dict(align="center", edgecolor="none", alpha=0.82)
        pct_fmt = plt.FuncFormatter(lambda v, _: f"{v:.2g}%")   # 2 sig-figs, no trailing zeros
        mn, mg  = means_raw["NF"], gauss_raw
        sn, sg  = stds_raw["NF"],  gauss_std

        d_m   = 0.5 * (mn + mg)
        rel_m = np.where(d_m > 0, (mn - mg) / d_m * 100.0, 0.0)
        ax_dm.bar(centers, rel_m, width=widths, color=COLORS["NF"], **bar_kw)
        ax_dm.axhline(0, color="k", linewidth=1.0)
        ax_dm.set_ylabel(r"$\Delta\langle N \rangle$")
        ax_dm.yaxis.set_major_formatter(pct_fmt)
        ax_dm.yaxis.set_major_locator(MaxNLocator(4, prune="both"))
        ax_dm.grid(True, alpha=0.2, axis="y")
        ax_dm.set_xlim(bin_edges[0], bin_edges[-1])
        plt.setp(ax_dm.get_xticklabels(), visible=False)
        _ax_fontsize(ax_dm, FS)

        d_s   = 0.5 * (sn + sg)
        rel_s = np.where(d_s > 0, (sn - sg) / d_s * 100.0, 0.0)
        ax_ds.bar(centers, rel_s, width=widths, color=COLORS["Gaussian"], **bar_kw)
        ax_ds.axhline(0, color="k", linewidth=1.0)
        ax_ds.set_ylabel(r"$\Delta\sigma$")
        ax_ds.set_xlabel(xlabel)
        ax_ds.yaxis.set_major_formatter(pct_fmt)
        ax_ds.yaxis.set_major_locator(MaxNLocator(4, prune="both"))
        ax_ds.grid(True, alpha=0.2, axis="y")
        ax_ds.set_xlim(bin_edges[0], bin_edges[-1])
        ax_ds.xaxis.set_major_locator(MaxNLocator(6, prune="both"))
        _ax_fontsize(ax_ds, FS)

    fig.tight_layout(pad=1.5)
    # Apply font sizes AFTER tight_layout so they aren't overridden.
    # Axis labels scale with FS; tick values stay fixed at 22.
    for _ax in fig.axes:
        _ax_fontsize(_ax, FS, tick_fs=22)
    _savefig(fig, save_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — split violin (MCMC left | NF right, relative to Gaussian)
# ─────────────────────────────────────────────────────────────────────────────

def plot_violin(
    results: dict[str, np.ndarray],
    bin_edges: np.ndarray,
    save_path: Path,
    xlabel: str = "variable",
    legend_loc: str = "upper right",
    font_scale: float = 1.0,
    tick_scale: float = 1.0,
    height_scale: float = 1.0,
    fmt: str = "pdf",
) -> None:
    has_gauss = "Gaussian" in results
    sides = [("MCMC", -1), ("NF", +1)]
    sides = [(l, sg) for l, sg in sides if l in results]
    if not sides or not has_gauss:
        warnings.warn("plot_violin: need NF or MCMC plus Gaussian — skipping.")
        return

    n_bins  = next(iter(results.values())).shape[1]
    mu_g    = results["Gaussian"].mean(axis=0)
    # Equal width for every bin: violins sit at integer positions 0..n-1
    bin_labs = [_bin_label(bin_edges[i], bin_edges[i + 1]) for i in range(n_bins)]

    fig, ax = plt.subplots(figsize=(max(14, n_bins * 1.4), 10 * height_scale))
    hw = 0.40   # fixed half-width per bin (slot is [i-0.5, i+0.5])

    legend_done = set()
    for i in range(n_bins):
        mu = mu_g[i]
        if mu <= 0:
            continue
        xc = i

        for label, sign in sides:
            color = COLORS[label]
            rel   = (results[label][:, i] - mu) / mu
            rel   = rel[np.isfinite(rel)]
            if len(rel) < 5:
                continue
            lo_q, hi_q = np.quantile(rel, [0.01, 0.99])
            if hi_q <= lo_q:
                continue
            ys = np.linspace(lo_q, hi_q, 200)
            try:
                kde     = gaussian_kde(rel, bw_method="scott")
                pdf     = kde(ys)
                pdf_max = pdf.max()
            except Exception:
                continue
            if pdf_max <= 0:
                continue
            pn = pdf / pdf_max
            ax.fill_betweenx(ys, xc, xc + sign * hw * pn,
                             facecolor=color, alpha=0.60, linewidth=0)
            ax.plot(xc + sign * hw * pn, ys, color=color, linewidth=0.9)
            med = float(np.median(rel))
            m_p = float(np.asarray(kde([med])).flat[0]) / pdf_max
            ax.plot([xc, xc + sign * hw * m_p], [med, med],
                    color=color, linewidth=1.8, solid_capstyle="round")
            if label not in legend_done:
                ax.fill_between([], [], [], color=color, alpha=0.60, label=label)
                legend_done.add(label)

    # Fonts scale with font_scale AND the figure height, so a taller figure
    # keeps proportionally-sized text.
    VFS  = int(28 * font_scale * height_scale)
    TFS  = int(22 * tick_scale * height_scale)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$(N - \mu_{\mathrm{Gauss}}) / \mu_{\mathrm{Gauss}}$")
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_labs, rotation=45, ha="right")
    ax.legend(loc=legend_loc)
    fig.tight_layout(pad=1.2)
    _ax_fontsize(ax, VFS, tick_fs=TFS, legend_fs=VFS - 4)
    _savefig(fig, save_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — bin-yield correlation matrix
# ─────────────────────────────────────────────────────────────────────────────

def _bin_label(lo: float, hi: float) -> str:
    """Short, round-number bin label."""
    def _fmt(v):
        if v == int(v):
            return str(int(v))
        return f"{v:.3g}"
    return f"[{_fmt(lo)},{_fmt(hi)})"


def plot_correlation(
    hists_arr: np.ndarray,
    bin_edges: np.ndarray,
    label: str,
    save_path: Path,
    fmt: str = "pdf",
) -> None:
    corr = np.corrcoef(hists_arr.T)
    labs = [_bin_label(lo, hi) for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
    n    = len(labs)
    lbl_fs  = max(10, 20 - n // 2)   # scale label fontsize with number of bins
    cell    = max(0.8, 9.0 / n)
    fig, ax = plt.subplots(figsize=(n * cell + 2.0, n * cell))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(r"Pearson $r$", fontsize=lbl_fs + 4)
    cb.ax.tick_params(labelsize=lbl_fs + 2)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=lbl_fs)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labs, fontsize=lbl_fs)
    ax.tick_params(top=False, right=False)
    fig.tight_layout()
    _savefig(fig, save_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — bin-yield corner (all bins)
# ─────────────────────────────────────────────────────────────────────────────

def _corner_ticks(
    axes: np.ndarray,
    n: int,
    labs: list[str],
    label_size: int = 14,
    x_only: bool = False,
    rotate_x: bool = False,
) -> None:
    """Corner-plot tick style.
    x_only=False (default): x labels on bottom row, y labels on left column.
    x_only=True:            x labels on bottom row only; no y-axis labels.
    rotate_x=True:          rotate x labels (for long parameter names that
                            would otherwise overlap and hide one another).
    """
    xlbl_kw = dict(fontsize=label_size + 3, labelpad=8)
    if rotate_x:
        xlbl_kw.update(rotation=35, ha="right", rotation_mode="anchor")

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if col > row or ax is None:
                continue
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ax.tick_params(which="both", top=False, right=False,
                           length=5, width=1.2)

            # ── x labels ──────────────────────────────────────────────────
            if x_only:
                # Only bottom row gets x tick labels and axis labels
                if row == n - 1:
                    ax.tick_params(labelbottom=True, labelsize=label_size)
                    ax.set_xlabel(labs[col], **xlbl_kw)
                else:
                    ax.tick_params(labelbottom=False, bottom=True)
            else:
                if row < n - 1:
                    ax.tick_params(labelbottom=False, bottom=True)
                else:
                    ax.tick_params(labelbottom=True, labelsize=label_size)
                    ax.set_xlabel(labs[col], **xlbl_kw)

            # ── y labels ──────────────────────────────────────────────────
            if x_only:
                ax.tick_params(labelleft=False, left=True)
            else:
                if col == 0 and row > 0:
                    ax.tick_params(labelleft=True, labelsize=label_size)
                    ax.set_ylabel(labs[row], fontsize=label_size + 3, labelpad=8)
                else:
                    ax.tick_params(labelleft=False, left=True)


def plot_bin_corner(
    hists_arr: np.ndarray,
    bin_edges: np.ndarray,
    label: str,
    save_path: Path,
    bins: int = 50,
    n_show: int | None = None,
    first_n: bool = False,
    clean: bool = False,
    ranges_all: list | None = None,   # per-bin (lo,hi) shared across sources
    log_scale: bool = True,           # hist2d colour scale (log vs linear)
    fmt: str = "pdf",
) -> None:
    """Bin-yield corner.
    n_show=None → all bins.
    n_show=k, first_n=True  → first k bins by index (default for corner5).
    n_show=k, first_n=False → k most non-Gaussian bins.
    ranges_all → per-bin (lo,hi) axis ranges (length n_total) shared across
                 NF/Gaussian/MCMC so the corners are directly comparable.
    log_scale  → True: LogNorm colour scale; False: linear.
    """
    norm = LogNorm() if log_scale else None
    n_total  = hists_arr.shape[1]
    labs_all = [_bin_label(lo, hi) for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]

    sel = np.arange(n_total)
    if n_show is not None and n_show < n_total:
        if first_n:
            sel = np.arange(n_show)
        else:
            cv  = hists_arr.std(axis=0) / np.maximum(hists_arr.mean(axis=0), 1e-12)
            sel = np.argsort(-cv)[:n_show]
    hists_arr = hists_arr[:, sel]
    labs_all  = [labs_all[i] for i in sel]
    ranges = [ranges_all[i] for i in sel] if ranges_all is not None else None

    n    = hists_arr.shape[1]
    labs = labs_all
    color = COLORS.get(label, "#1f77b4")
    cell  = 2.8 * 1.1      # base cell × 1.1 as requested

    fig, axes = plt.subplots(n, n, figsize=(n * cell, n * cell))
    if n == 1:
        axes = np.array([[axes]])

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if col > row:
                ax.axis("off")
            elif row == col:
                rng = ranges[row] if ranges else None
                ax.hist(hists_arr[:, row], bins=bins, range=rng, density=True,
                        histtype="stepfilled", alpha=0.45, color=color)
                ax.hist(hists_arr[:, row], bins=bins, range=rng, density=True,
                        histtype="step", linewidth=1.2, color=color)
                if ranges:
                    ax.set_xlim(*ranges[row])
            else:
                rng2 = [ranges[col], ranges[row]] if ranges else None
                ax.hist2d(hists_arr[:, col], hists_arr[:, row],
                          bins=bins, range=rng2, norm=norm, cmap="viridis")
                if ranges:
                    ax.set_xlim(*ranges[col]); ax.set_ylim(*ranges[row])

    if clean:
        for row in range(n):
            for col in range(n):
                ax = axes[row, col]
                if col > row:
                    continue
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
                ax.tick_params(labelbottom=False, labelleft=False,
                               top=False, right=False, length=4, width=1.0)
    else:
        _corner_ticks(axes, n, labs, label_size=16)
    fig.tight_layout(pad=0.5)
    _savefig(fig, save_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — NLL 2-D histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_nll_hist2d(
    nll_delta: np.ndarray,       # ΔNLL (GUNDAM)  → goes on the X axis
    nll_other: np.ndarray,       # -log q_NF or -log g → goes on the Y axis
    other_label_tex: str,
    save_path: Path,
    n_bins: int = 60,
    q_lo: float = 0.001,         # lower-quantile cut (both axes)
    q_hi: float = 0.999,         # upper-quantile cut (both axes)
    fmt: str = "pdf",
) -> None:
    # Cut each axis by its OWN quantiles — ΔNLL and -log q/g are clipped
    # independently (and q_NF vs g get their own cuts, one per plot).
    mask = (np.isfinite(nll_delta) & np.isfinite(nll_other)
            & (nll_delta < np.quantile(nll_delta, q_hi))
            & (nll_delta > np.quantile(nll_delta, q_lo))
            & (nll_other < np.quantile(nll_other, q_hi))
            & (nll_other > np.quantile(nll_other, q_lo)))
    delta = nll_delta[mask]
    other = nll_other[mask]
    # Align medians so the distribution sits on the diagonal
    delta = delta + (np.median(other) - np.median(delta))
    # Shift BOTH by the same constant so the smaller of the two starts at 0
    # (preserves the y = x relationship; at least one NLL now begins at 0).
    offset = min(other.min(), delta.min())
    delta -= offset
    other -= offset
    lo, hi = 0.0, max(other.max(), delta.max())

    HFS = 18
    fig, ax = plt.subplots(figsize=(7 * 1.1, 6))
    # X = ΔNLL  ;  Y = -log q/g
    h = ax.hist2d(delta, other, bins=n_bins, range=[[lo, hi], [lo, hi]],
                  norm=LogNorm(), cmap="viridis")
    cb = plt.colorbar(h[3], ax=ax)
    cb.set_label("")          # no "Counts" text
    # Plain round numbers on colorbar instead of 10^x notation
    cb.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.3g}"))
    cb.ax.tick_params(labelsize=HFS - 2)
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1.2,
            linestyle="--", label="$y = x$")
    ax.set_xlabel(r"$\Delta$NLL (GUNDAM)")
    ax.set_ylabel(other_label_tex)
    ax.legend()
    _ax_fontsize(ax, HFS)
    fig.tight_layout(pad=1.2)
    _savefig(fig, save_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — log-weight distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_logweights(
    logw_g: np.ndarray,        # log(p/g) from GAUSSIAN draws
    logw_nf: np.ndarray,       # log(p/q_NF) from NF draws
    save_path: Path,
    n_bins: int = 60,
    q_clip: float = 0.99,      # clip at 1st / 99th percentile (both curves)
    log_y: bool = False,       # log-scale y axis
    fmt: str = "pdf",
) -> None:
    def _shift_logw(arr: np.ndarray) -> np.ndarray:
        """Median-centre the log-weight distribution at 0 — the same median
        alignment used by the hist2d plots (no extra mean-weight step, which
        would over-shift the heavy-tailed Gaussian weights)."""
        arr = arr.copy()
        arr -= np.median(arr)
        return arr

    entries = [
        (logw_g,  COLORS["Gaussian"], r"$\log(p/g)$"),
        (logw_nf, COLORS["NF"],       r"$\log(p/q_{\rm NF})$"),
    ]
    # log-y version widens the left tail to -15; linear stays at -10
    XLIM = (-15.0, 8.0) if log_y else (-10.0, 8.0)
    fig, ax = plt.subplots(figsize=(8, 6))
    for arr, color, lbl in entries:
        ok  = np.isfinite(arr)
        arr = _shift_logw(arr[ok])     # centre at 0
        arr = arr[(arr >= XLIM[0]) & (arr <= XLIM[1])]
        if arr.size == 0:
            continue
        ax.hist(arr, bins=n_bins, range=XLIM, density=True, histtype="stepfilled",
                alpha=0.40, color=color, label=lbl)
        ax.hist(arr, bins=n_bins, range=XLIM, density=True, histtype="step",
                linewidth=1.3, color=color)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(*XLIM)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Log importance weight")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    _ax_fontsize(ax, 16, legend_fs=22)   # larger legend
    fig.tight_layout(pad=1.2)
    _savefig(fig, save_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6b — total event rate vs ΔNLL (GUNDAM)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rate_vs_nll(
    rate: np.ndarray,            # total event rate per throw (Σ bins)
    nll: np.ndarray,             # GUNDAM NLL per throw
    save_path: Path,
    n_bins: int = 60,
    q_clip: float = 0.999,
    fmt: str = "pdf",
) -> None:
    """2-D histogram of total event rate (x) vs GUNDAM NLL (y)."""
    mask = (np.isfinite(rate) & np.isfinite(nll)
            & (rate < np.quantile(rate, q_clip))
            & (rate > np.quantile(rate, 1 - q_clip))
            & (nll  < np.quantile(nll,  q_clip)))
    x, y = rate[mask], nll[mask]
    if x.size == 0:
        return

    HFS = 18
    fig, ax = plt.subplots(figsize=(7 * 1.1, 6))
    h = ax.hist2d(x, y, bins=n_bins, norm=LogNorm(), cmap="viridis")
    cb = plt.colorbar(h[3], ax=ax)
    cb.set_label("")
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3g}"))
    cb.ax.tick_params(labelsize=HFS - 2)
    ax.set_xlabel("Total event rate")
    ax.set_ylabel("NLL (GUNDAM)")
    ax.xaxis.set_major_locator(MaxNLocator(4))   # 4 ticks on the rate axis
    _ax_fontsize(ax, HFS)
    fig.tight_layout(pad=1.2)
    _savefig(fig, save_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7 — parameter corner (KS-ranked, IS-weighted or uniform)
# ─────────────────────────────────────────────────────────────────────────────

def _short_par_name(s: str, maxlen: int = 40) -> str:
    """Strip common GUNDAM prefixes and shorten so the name fits as an
    un-rotated x-axis label without overlapping its neighbours.
    When too long, keep the END of the name (no ellipsis)."""
    s = str(s).strip()
    for pre in ("Linear Systematics/", "Systematics/", "Parameters/"):
        if s.startswith(pre):
            s = s[len(pre):]
    s = s.replace("_", " ")          # underscores → spaces (avoid mathtext issues, wrap nicer)
    if len(s) > maxlen:
        s = s[-maxlen:]              # keep the tail
    return s


def _ks_gauss(z: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(z) & np.isfinite(w)
    if not ok.any():
        return 0.0
    z_ok, w_ok = z[ok], w[ok]
    w_ok /= w_ok.sum() + 1e-40
    order    = np.argsort(z_ok)
    cdf_emp  = np.cumsum(w_ok[order])
    cdf_norm = 0.5 * (1.0 + erf(z_ok[order] / np.sqrt(2.0)))
    return float(np.max(np.abs(cdf_emp - cdf_norm)))


def plot_param_corner(
    samples: np.ndarray,
    weights: np.ndarray | None,
    par_names: list[str],
    mean: np.ndarray,
    cov: np.ndarray,
    save_path: Path,
    n_params: int = 10,
    bins: int = 50,
    clean: bool = False,
    sel: np.ndarray | None = None,
    ranges_all: list | None = None,   # per-parameter (lo,hi) shared across sources
    log_scale: bool = True,           # hist2d colour scale (log vs linear)
    fmt: str = "pdf",
) -> None:
    """Parameter corner: n_params most non-Gaussian by weighted KS test.
    clean=True → no axis labels, no tick values (for the large full corner).
    clean=False → x-axis labels on bottom row only (for corner5).
    sel        → if given, use these exact parameter indices (in order) instead
                 of ranking; lets weighted/unweighted share the same selection.
    ranges_all → per-parameter (lo,hi) axis ranges (length D) shared across
                 NF/MCMC so the corners are directly comparable.
    log_scale  → True: LogNorm colour scale; False: linear.
    """
    norm = LogNorm() if log_scale else None
    N, D   = samples.shape
    std    = np.sqrt(np.clip(np.diag(cov), 1e-14, None))
    z      = (samples - mean) / std
    w_norm = weights / (weights.sum() + 1e-40) if weights is not None else np.ones(N) / N

    if sel is None:
        ks  = np.array([_ks_gauss(z[:, i], w_norm) for i in range(D)])
        sel = np.argsort(-ks)[:n_params]
    sel = np.asarray(sel)
    n   = len(sel)

    ss  = samples[:, sel]
    ms  = mean[sel]
    sts = std[sel]
    ns  = [_short_par_name(par_names[i]) if i < len(par_names) else f"par {i}"
           for i in sel]
    ranges = [ranges_all[i] for i in sel] if ranges_all is not None else None
    cell = 2.8

    fig, axes = plt.subplots(n, n, figsize=(n * cell, n * cell))
    if n == 1:
        axes = np.array([[axes]])

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if col > row:
                ax.axis("off")
                continue
            if row == col:
                x  = ss[:, row]
                ok = np.isfinite(x)
                w  = w_norm[ok] if weights is not None else None
                rng = ranges[row] if ranges else None
                ax.hist(x[ok], bins=bins, weights=w, range=rng, density=True,
                        histtype="stepfilled", alpha=0.45, color=COLORS["NF"])
                ax.hist(x[ok], bins=bins, weights=w, range=rng, density=True,
                        histtype="step", linewidth=1.2, color=COLORS["NF"])
                xs = np.linspace(ms[row] - 3.5 * sts[row], ms[row] + 3.5 * sts[row], 200)
                ax.plot(xs,
                        np.exp(-0.5 * ((xs - ms[row]) / sts[row]) ** 2)
                        / (sts[row] * np.sqrt(2 * np.pi)),
                        color=COLORS["Gaussian"], linewidth=1.2, linestyle="--")
                if ranges:
                    ax.set_xlim(*ranges[row])
            else:
                xd, yd = ss[:, col], ss[:, row]
                ok = np.isfinite(xd) & np.isfinite(yd)
                w  = w_norm[ok] if weights is not None else None
                rng2 = [ranges[col], ranges[row]] if ranges else None
                ax.hist2d(xd[ok], yd[ok], weights=w, bins=bins, range=rng2,
                          norm=norm, cmap="viridis")
                if ranges:
                    ax.set_xlim(*ranges[col]); ax.set_ylim(*ranges[row])

    if clean:
        # Full corner: completely clean — no labels, no tick values, just distributions
        for row in range(n):
            for col in range(n):
                ax = axes[row, col]
                if col > row:
                    continue
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
                ax.tick_params(labelbottom=False, labelleft=False,
                               top=False, right=False, length=4, width=1.0)
    else:
        _corner_ticks(axes, n, ns, label_size=16, x_only=True)

    fig.tight_layout(pad=0.4)
    _savefig(fig, save_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 8 — parameter pull distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_pulls(
    samples: np.ndarray,
    weights: np.ndarray | None,
    par_names: list[str],
    mean: np.ndarray,
    cov: np.ndarray,
    save_path: Path,
    max_params: int = 30,
    fmt: str = "pdf",
) -> None:
    D      = samples.shape[1]
    std    = np.sqrt(np.clip(np.diag(cov), 1e-14, None))
    n_plt  = min(D, max_params)
    ncols  = 5
    nrows  = (n_plt + ncols - 1) // ncols
    w_norm = weights / (weights.sum() + 1e-40) if weights is not None else None

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes  = np.asarray(axes).flatten()
    xs_g  = np.linspace(-4, 4, 300)
    g_pdf = np.exp(-0.5 * xs_g ** 2) / np.sqrt(2 * np.pi)

    for i in range(n_plt):
        ax = axes[i]
        z  = (samples[:, i] - mean[i]) / std[i]
        ok = np.isfinite(z)
        w  = w_norm[ok] if w_norm is not None else None
        ax.hist(z[ok], bins=40, weights=w, density=True,
                histtype="stepfilled", alpha=0.40, color=COLORS["NF"])
        ax.hist(z[ok], bins=40, weights=w, density=True,
                histtype="step", linewidth=1.2, color=COLORS["NF"])
        ax.plot(xs_g, g_pdf, color=COLORS["Gaussian"], linewidth=1.2, linestyle="--")
        nm = par_names[i] if i < len(par_names) else f"par_{i}"
        ax.set_xlabel(nm, fontsize=9, labelpad=3)
        ax.set_xlim(-4, 4)
        ax.xaxis.set_major_locator(MaxNLocator(3, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
        ax.tick_params(labelsize=9, top=False, right=False)

    for j in range(n_plt, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    _savefig(fig, save_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../../configs", config_name="make_paper_plots", version_base=None)
def main(cfg: DictConfig) -> None:
    from gunflows.likelihood_sampler import LikelihoodSampler

    _apply_style()

    save_dir  = Path(cfg.save_dir).expanduser().resolve()
    cache_dir = save_dir / "cache"
    plots_dir = save_dir / "plots"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # hep_label removed
    
    fmt      = str(cfg.get("fig_format",   "pdf"))
    ci_levels = tuple(float(x) for x in cfg.get("ci_levels", list(_DEFAULT_CI)))
    n_corner  = int(cfg.get("n_corner_params", 10))
    do_nll    = bool(cfg.get("do_nll_plots", True))

    use_nf       = bool(cfg.get("use_nf",       True))
    use_gaussian = bool(cfg.get("use_gaussian",  True))
    use_mcmc     = bool(cfg.get("use_mcmc",      False))

    # ── Build variable map ──────────────────────────────────────────────────
    enu_var = str(cfg.get("enu_var", "Enu"))
    if "bin_edges_list" in cfg and cfg.bin_edges_list is not None:
        primary_edges = np.array(list(cfg.bin_edges_list), dtype=np.float64)
    else:
        primary_edges = np.array([0.0, 0.2, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
                                   0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 2.0, 5.0])
    var_bin_edges: dict[str, np.ndarray] = {enu_var: primary_edges}
    for extra in cfg.get("extra_vars", []) or []:
        var_bin_edges[str(extra["name"])] = np.array(list(extra["bin_edges"]))
    n_bins_map = {v: len(e) - 1 for v, e in var_bin_edges.items()}

    # ── Check which cache files exist ───────────────────────────────────────
    force    = bool(cfg.get("force_recompute", False))
    src_need = [l for l, u in [("nf", use_nf), ("gaussian", use_gaussian), ("mcmc", use_mcmc)] if u]

    def _hist_cached(src, var, stream):
        return (cache_dir / f"histograms_{src}_{var.lower()}_{stream.lower()}.npy").exists()

    hists_complete = all(
        _hist_cached(src, var, stream)
        for src in src_need
        for var in var_bin_edges
        for stream in STREAMS
    )

    # Invalidate cache if bin edges changed
    if hists_complete and not force:
        for var, cfg_edges in var_bin_edges.items():
            p = cache_dir / f"bin_edges_{var.lower()}.npy"
            if p.exists():
                cached_edges = np.load(p)
                if not np.allclose(cached_edges, cfg_edges):
                    print(f"  [CACHE] bin edges changed for {var} — forcing recompute",
                          flush=True)
                    hists_complete = False
                    break
    nll_complete = (cache_dir / "nll_data.npz").exists() or not do_nll

    need_compute = force or not hists_complete or not nll_complete

    # ════════════════════════════════════════════════════════════════════════
    # COMPUTE (first run or force_recompute)
    # ════════════════════════════════════════════════════════════════════════
    if need_compute:
        print("\n=== COMPUTE phase (GUNDAM + NF) ===", flush=True)

        # Save bin edges
        for v, edges in var_bin_edges.items():
            np.save(cache_dir / f"bin_edges_{v.lower()}.npy", edges)

        # ── Training config ─────────────────────────────────────────────────
        training_folder = Path(cfg.training_folder).expanduser().resolve()
        hydra_cfg_path  = training_folder / ".hydra" / "config.yaml"
        if not hydra_cfg_path.exists():
            raise FileNotFoundError(f"Hydra config not found: {hydra_cfg_path}")
        train_cfg = OmegaConf.merge(OmegaConf.load(hydra_cfg_path), cfg)

        # ── GUNDAM ─────────────────────────────────────────────────────────
        if "llh_config" in train_cfg:
            llh_config    = str(train_cfg.llh_config)
            llh_overrides = list(train_cfg.get("llh_overrides", []))
            asimov        = bool(train_cfg.get("data_is_asimov", True))
            llh_cwd       = str(train_cfg.get("llh_cwd", ".")) or None
            threads       = int(train_cfg.get("threads", 1))
        else:
            llh_config    = str(train_cfg.experiment.dataset.llh_config)
            llh_overrides = list(train_cfg.experiment.dataset.llh_overrides)
            asimov        = bool(train_cfg.experiment.dataset.data_is_asimov)
            llh_cwd       = str(train_cfg.experiment.dataset.llh_cwd)
            threads       = int(train_cfg.experiment.sampler.threads) \
                            if hasattr(train_cfg.experiment, "sampler") else 1
        # Top-level `threads` (from make_paper_plots.yaml / CLI) takes priority
        threads = int(cfg.get("threads", threads))

        llh_overrides += list(cfg.get("llh_extra_overrides", []))

        print("Initializing GUNDAM LikelihoodSampler...", flush=True)
        sampler = LikelihoodSampler(
            config_file=llh_config,
            override_files=llh_overrides,
            data_is_asimov=asimov,
            threads=threads,
            llh_cwd=llh_cwd,
            light_mode=False,
        )
        bestfit   = np.asarray(sampler.postfit_parameter_values, dtype=np.float64)
        cov       = np.asarray(sampler.postfit_covariance_matrix, dtype=np.float64)
        par_names = _get_par_names(sampler)
        print(f"  Parameters: {len(bestfit)}, names found: {len(par_names)}", flush=True)

        # ── NF model ────────────────────────────────────────────────────────
        ckpt_dir   = training_folder / "checkpoints"
        ckpts      = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() \
                     else sorted(training_folder.glob("*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints in {training_folder}")
        ckpt = str(ckpts[-1])
        print(f"  Checkpoint: {ckpt}", flush=True)

        dataset  = _SamplingDataset(
            train_cfg.experiment.dataset.phase_space_dim, bestfit, cov)
        nf_model = _load_nf(train_cfg, dataset, ckpt)
        print("  NF model loaded.", flush=True)

        # ── MCMC ────────────────────────────────────────────────────────────
        mcmc_throws = None
        if use_mcmc:
            mcmc_throws = _load_mcmc(
                str(cfg.mcmc_chain),
                n_samples   = int(cfg.num_samples),
                burnin_frac = float(cfg.get("mcmc_burnin_frac", 0.0)),
                max_steps   = int(cfg.mcmc_max_steps) if "mcmc_max_steps" in cfg else None,
                thin        = int(cfg.mcmc_thin)      if "mcmc_thin"      in cfg else None,
            )
            # Save a subset of MCMC parameter throws for the parameter corner plots
            n_save = min(len(mcmc_throws), int(cfg.get("n_nll_samples", 20000)))
            np.save(cache_dir / "mcmc_samples.npy",
                    mcmc_throws[:n_save].astype(np.float32))
            print(f"  Saved {n_save} MCMC parameter throws for corner plots", flush=True)

        # ── Event caches ────────────────────────────────────────────────────
        print("Building event caches...", flush=True)
        var_caches = _build_event_caches(sampler, var_bin_edges)

        # ── Histogram sampling loop ─────────────────────────────────────────
        if not hists_complete or force:
            # Pre-load any sources already cached so they are skipped in the loop
            def _empty_hists():
                return {v: {s: [] for s in STREAMS} for v in var_caches}
            preloaded: dict = {
                "NF": _empty_hists(), "Gaussian": _empty_hists(), "MCMC": _empty_hists()
            }
            for lbl in SOURCE_ORDER:
                for var in var_bin_edges:
                    for stream in STREAMS:
                        p = cache_dir / f"histograms_{lbl.lower()}_{var.lower()}_{stream.lower()}.npy"
                        if p.exists() and not force:
                            arr = np.load(p)
                            preloaded[lbl][var][stream] = list(arr)
                            print(f"  Pre-loaded {len(arr)} throws: {lbl}/{var}/{stream}",
                                  flush=True)

            rng = np.random.default_rng(int(cfg.get("seed", 0)))
            # Worker sampler config for the parallel histogram fill (full data).
            hist_workers = int(cfg.get("hist_num_workers", 1))
            hist_llh_kw = dict(
                config_file=llh_config, override_files=llh_overrides,
                data_is_asimov=asimov, threads=1, llh_cwd=llh_cwd, light_mode=False,
            )
            print(f"\nSampling loop: {cfg.num_samples} throws "
                  f"(hist_num_workers={hist_workers})...", flush=True)
            _run_histogram_loop(
                sampler=sampler, nf_model=nf_model, dataset=dataset,
                mcmc_throws=mcmc_throws, var_caches=var_caches,
                n_bins_map=n_bins_map,
                num_samples=int(cfg.num_samples),
                batch_size=int(cfg.batch_size),
                use_nf=use_nf, use_gaussian=use_gaussian, use_mcmc=use_mcmc,
                bestfit=bestfit, cov=cov, rng=rng,
                cache_dir=cache_dir,
                save_every=int(cfg.get("save_every", 5000)),
                var_bin_edges=var_bin_edges,
                preloaded=preloaded,
                hist_num_workers=hist_workers,
                llh_kw=hist_llh_kw,
                force=force,
            )

        # ── NLL comparison arrays — assembled from the SAME histogram-loop
        #    throws (no resampling, no parallel pool). ──────────────────────
        if do_nll:
            print("\nAssembling NLL comparison arrays from loop throws...", flush=True)
            bf_nll, _, _ = sampler.inject_params_and_compute_likelihood(
                bestfit.tolist(), extend_continue=False)
            print(f"  Best-fit NLL = {bf_nll:.4f}", flush=True)

            nll_nf  = np.load(cache_dir / "nll_throws_nf.npy")
            log_nf  = np.load(cache_dir / "logq_nf.npy")
            samp_nf = np.load(cache_dir / "samp_nf.npy").astype(np.float64)
            nll_g   = np.load(cache_dir / "nll_throws_gaussian.npy")
            samp_g  = np.load(cache_dir / "samp_gaussian.npy").astype(np.float64)

            # Defensive: keep the NF arrays aligned to a common length
            m_nf = min(len(nll_nf), len(log_nf), len(samp_nf))
            nll_nf, log_nf, samp_nf = nll_nf[:m_nf], log_nf[:m_nf], samp_nf[:m_nf]
            m_g = min(len(nll_g), len(samp_g))
            nll_g, samp_g = nll_g[:m_g], samp_g[:m_g]

            nll_data = {
                "nll_nf":      nll_nf - float(bf_nll),
                "log_nf":      log_nf,
                "log_g_nf":    _log_gaussian_batch(samp_nf, bestfit, cov),
                "samples_nf":  samp_nf.astype(np.float32),
                "nll_g":       nll_g - float(bf_nll),
                "log_g_g":     _log_gaussian_batch(samp_g, bestfit, cov),
                "samples_g":   samp_g.astype(np.float32),
                "bestfit_nll": float(bf_nll),
            }
            np.savez(
                cache_dir / "nll_data.npz",
                bestfit=bestfit, cov=cov,
                par_names=np.array(par_names, dtype=object),
                **nll_data,
            )
            print(f"  NLL data saved: {len(nll_nf)} NF + {len(nll_g)} Gaussian "
                  f"(reused from histogram loop)", flush=True)

        print("\n=== COMPUTE phase done ===\n", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # LOAD from cache
    # ════════════════════════════════════════════════════════════════════════
    print("=== Loading from cache ===", flush=True)

    # Load bin edges (prefer cache, fall back to config)
    loaded_edges: dict[str, np.ndarray] = {}
    for v, cfg_edges in var_bin_edges.items():
        p = cache_dir / f"bin_edges_{v.lower()}.npy"
        loaded_edges[v] = np.load(p) if p.exists() else cfg_edges

    # Load histogram arrays
    all_hists: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for label in SOURCE_ORDER:
        all_hists[label] = {}
        for var in loaded_edges:
            all_hists[label][var] = {}
            for stream in STREAMS:
                p = cache_dir / f"histograms_{label.lower()}_{var.lower()}_{stream.lower()}.npy"
                if p.exists():
                    all_hists[label][var][stream] = np.load(p)

    # Load NLL data
    nll_data = None
    par_names_cached: list[str] = []
    bestfit_cached = cov_cached = None
    if do_nll and (cache_dir / "nll_data.npz").exists():
        cd = np.load(cache_dir / "nll_data.npz", allow_pickle=True)
        nll_data = {k: cd[k] for k in cd.files if k not in ("bestfit", "cov", "par_names")}
        bestfit_cached  = cd["bestfit"]
        cov_cached      = cd["cov"]
        par_names_cached = list(cd["par_names"]) if "par_names" in cd else []

        # Normalize to unified keys (back-compat with old single-set caches that
        # only had nll_gundam / log_nf / log_g / samples).
        if "nll_nf" not in nll_data:
            nll_data = {
                "nll_nf":     nll_data["nll_gundam"], "log_nf":   nll_data["log_nf"],
                "log_g_nf":   nll_data["log_g"],      "samples_nf": nll_data["samples"],
                "nll_g":      nll_data["nll_gundam"], "log_g_g":  nll_data["log_g"],
                "samples_g":  nll_data["samples"],
                "bestfit_nll": nll_data.get("bestfit_nll", 0.0),
            }
        print(f"  NLL data: {len(nll_data['nll_nf'])} NF + "
              f"{len(nll_data['nll_g'])} Gaussian samples", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # PLOT
    # ════════════════════════════════════════════════════════════════════════
    print("\n=== Plotting ===", flush=True)

    # ── Kinematic plots ──────────────────────────────────────────────────────
    var_cfgs   = list(cfg.get("variables", []))
    # Build per-variable settings lookup
    var_settings: dict[str, dict] = {}
    for vc in var_cfgs:
        nm = str(vc["name"])
        var_settings[nm] = {
            "xlabel":           str(vc.get("xlabel", nm)),
            "legend_loc_spec":  str(vc.get("legend_loc_spectrum", "upper right")),
            "legend_loc_vio":   str(vc.get("legend_loc_violin",   "upper right")),
            "width_scale":      float(vc.get("spectrum_width_scale", 1.0)),
            "font_scale_spec":  float(vc.get("spectrum_font_scale", 1.0)),
            "font_scale_vio":   float(vc.get("violin_font_scale",   1.0)),
            "tick_scale_vio":   float(vc.get("violin_tick_scale",   1.0)),
            "height_scale_vio": float(vc.get("violin_height_scale", 1.0)),
        }

    for var, edges in loaded_edges.items():
        vs     = var_settings.get(var, {})
        xlabel = vs.get("xlabel", _VAR_LABELS.get(var, var))

        for stream in STREAMS:
            results: dict[str, np.ndarray] = {}
            for label in SOURCE_ORDER:
                arr = all_hists.get(label, {}).get(var, {}).get(stream)
                if arr is not None and len(arr) > 0:
                    results[label] = arr

            if not results:
                print(f"  [SKIP] No data for {var}/{stream}", flush=True)
                continue

            print(f"  {var}/{stream}: {list(results.keys())}", flush=True)
            base = plots_dir / "kinematic" / var / stream

            plot_spectrum(results, edges,
                          save_path=base / "spectrum",
                          xlabel=xlabel, ci_levels=ci_levels,
                          legend_loc=vs.get("legend_loc_spec", "upper right"),
                          width_scale=vs.get("width_scale", 1.0),
                          font_scale=vs.get("font_scale_spec", 1.0),
                          fmt=fmt)

            plot_violin(results, edges,
                        save_path=base / "violin",
                        xlabel=xlabel,
                        legend_loc=vs.get("legend_loc_vio", "upper right"),
                        font_scale=vs.get("font_scale_vio", 1.0),
                        tick_scale=vs.get("tick_scale_vio", 1.0),
                        height_scale=vs.get("height_scale_vio", 1.0),
                        fmt=fmt)

            # Shared per-bin ranges across all sources for comparable corners
            bin_ranges     = _percentile_ranges(list(results.values()))
            bin_ranges_pad = _percentile_ranges(list(results.values()), pad=0.15)

            for label, arr in results.items():
                plot_correlation(arr, edges, label,
                                 save_path=base / f"correlation_{label}", fmt=fmt)
                # log-scale (range padded +15%) + linear duplicate (tight range)
                for logsc, sfx in [(True, ""), (False, "_linear")]:
                    rng = bin_ranges_pad if logsc else bin_ranges
                    plot_bin_corner(arr, edges, label,         # full: clean
                                    save_path=base / f"corner_{label}{sfx}",
                                    clean=True, ranges_all=rng,
                                    log_scale=logsc, fmt=fmt)
                    if arr.shape[1] > 5:
                        plot_bin_corner(arr, edges, label,     # corner5: with labels
                                        save_path=base / f"corner5_{label}{sfx}",
                                        n_show=5, first_n=True, clean=False,
                                        ranges_all=rng, log_scale=logsc, fmt=fmt)

    # ── Total event rate vs ΔNLL (per-throw, primary variable) ───────────────
    # Only NF and Gaussian — MCMC NLL is not a meaningful comparison here.
    print("\n  Rate-vs-NLL plots...", flush=True)
    primary_var = next(iter(loaded_edges))
    rate_dir = plots_dir / "rate_vs_nll"
    for label in ("NF", "Gaussian"):
        nll_path = cache_dir / f"nll_throws_{label.lower()}.npy"
        if not nll_path.exists():
            continue
        nll_thr = np.load(nll_path)
        # total event rate per throw = Σ bins over both streams of the primary var
        fhc = all_hists.get(label, {}).get(primary_var, {}).get("FHC")
        rhc = all_hists.get(label, {}).get(primary_var, {}).get("RHC")
        rate = np.zeros(len(nll_thr))
        if fhc is not None:
            rate[: len(fhc)] += fhc.sum(axis=1)
        if rhc is not None:
            rate[: len(rhc)] += rhc.sum(axis=1)
        n = min(len(rate), len(nll_thr))
        if n == 0:
            continue
        plot_rate_vs_nll(
            rate[:n], nll_thr[:n],
            save_path=rate_dir / f"rate_vs_nll_{label.lower()}",
            fmt=fmt,
        )

    # ── NLL / density plots ──────────────────────────────────────────────────
    if nll_data is not None:
        print("\n  NLL comparison plots...", flush=True)
        nll_nf    = nll_data["nll_nf"]        # ΔNLL at NF draws
        log_nf    = nll_data["log_nf"]        # log q_NF at NF draws
        nll_gd    = nll_data["nll_g"]         # ΔNLL at Gaussian draws
        log_g_gd  = nll_data["log_g_g"]       # log g  at Gaussian draws
        nll_dir   = plots_dir / "nll"

        # hist2d_nf:    NF draws       → ΔNLL vs −log q_NF
        plot_nll_hist2d(
            nll_nf, -log_nf,
            other_label_tex=r"$-\log\,q_{\rm NF}$",
            save_path=nll_dir / "hist2d_nf",
            fmt=fmt,
        )
        # hist2d_gauss: GAUSSIAN draws → ΔNLL vs −log g
        plot_nll_hist2d(
            nll_gd, -log_g_gd,
            other_label_tex=r"$-\log\,g$",
            save_path=nll_dir / "hist2d_gauss",
            fmt=fmt,
        )
        # log(p/g) from Gaussian draws ; log(p/q_NF) from NF draws
        # linear-y and duplicated log-y versions
        for logy, sfx in [(False, ""), (True, "_logy")]:
            plot_logweights(
                logw_g  = (-nll_gd) - log_g_gd,
                logw_nf = (-nll_nf) - log_nf,
                save_path=nll_dir / f"logweights{sfx}",
                log_y=logy,
                fmt=fmt,
            )

        # ── Parameter plots ─────────────────────────────────────────────────
        if "samples_nf" in nll_data and bestfit_cached is not None and cov_cached is not None:
            samples    = nll_data["samples_nf"].astype(np.float64)
            par_dir    = plots_dir / "parameters"

            # IS weights: reweight NF samples to true posterior  (w ∝ p / q_NF)
            log_w = (-nll_nf) - log_nf
            log_w -= log_w.max()
            weights_is = np.exp(log_w)
            weights_is /= weights_is.sum()

            # ── Shared parameter selections (from UNWEIGHTED NF samples) ──
            # sel_full : full corner — n_corner least-Gaussian over all params.
            # sel5     : corner5 — among spline systematics (last N params),
            #            5 least-Gaussian.
            # Both are computed once on the unweighted NF samples and reused for
            # the weighted NF and the MCMC corners, so all are directly comparable.
            D = samples.shape[1]
            n_spline = int(cfg.get("corner5_last_n", 10))
            std_all = np.sqrt(np.clip(np.diag(cov_cached), 1e-14, None))
            z_all   = (samples - bestfit_cached) / std_all
            w_uni   = np.ones(len(samples)) / len(samples)

            ks_all   = np.array([_ks_gauss(z_all[:, i], w_uni) for i in range(D)])
            sel_full = np.argsort(-ks_all)[:n_corner]

            cand    = np.arange(max(0, D - n_spline), D)
            sel5    = cand[np.argsort(-ks_all[cand])[:5]]
            print(f"  corner5 params (spline, 5 least-Gaussian, unweighted): "
                  f"{[par_names_cached[i] if i < len(par_names_cached) else i for i in sel5]}",
                  flush=True)

            # MCMC parameter samples (if available)
            mcmc_samples = None
            mcmc_path = cache_dir / "mcmc_samples.npy"
            if mcmc_path.exists():
                mcmc_samples = np.load(mcmc_path).astype(np.float64)
                print(f"  MCMC samples for corners: {mcmc_samples.shape}", flush=True)

            # Shared per-parameter ranges across NF + MCMC so all corners match.
            # Log corners use a +15% padded range; linear corners keep it tight.
            _par_src = [samples] + ([mcmc_samples] if mcmc_samples is not None else [])
            par_ranges     = _percentile_ranges(_par_src)
            par_ranges_pad = _percentile_ranges(_par_src, pad=0.15)

            print("\n  Parameter corner + pulls...", flush=True)
            # Each corner is produced twice: log-scale (padded +15% range) and a
            # linear-scale duplicate (tight range).
            for logsc, sfx in [(True, ""), (False, "_linear")]:
                rng = par_ranges_pad if logsc else par_ranges
                # NF corners (weighted + unweighted), shared selections + ranges
                for w_is, tag in [(weights_is, "weighted"), (None, "unweighted")]:
                    plot_param_corner(
                        samples, w_is, par_names_cached, bestfit_cached, cov_cached,
                        save_path=par_dir / f"corner_{tag}{sfx}",
                        sel=sel_full, clean=True, ranges_all=rng,
                        log_scale=logsc, fmt=fmt,
                    )
                    plot_param_corner(
                        samples, w_is, par_names_cached, bestfit_cached, cov_cached,
                        save_path=par_dir / f"corner5_{tag}{sfx}",
                        sel=sel5, ranges_all=rng, log_scale=logsc, fmt=fmt,
                    )
                # MCMC corners — same parameters/order AND ranges as the NF corners
                if mcmc_samples is not None:
                    plot_param_corner(
                        mcmc_samples, None, par_names_cached, bestfit_cached, cov_cached,
                        save_path=par_dir / f"corner_mcmc{sfx}",
                        sel=sel_full, clean=True, ranges_all=rng,
                        log_scale=logsc, fmt=fmt,
                    )
                    plot_param_corner(
                        mcmc_samples, None, par_names_cached, bestfit_cached, cov_cached,
                        save_path=par_dir / f"corner5_mcmc{sfx}",
                        sel=sel5, ranges_all=rng, log_scale=logsc, fmt=fmt,
                    )

            plot_pulls(
                samples, weights_is, par_names_cached,
                bestfit_cached, cov_cached,
                save_path=par_dir / "pulls",
                fmt=fmt,
            )

    print(f"\nAll plots saved to {plots_dir}", flush=True)


if __name__ == "__main__":
    main()
