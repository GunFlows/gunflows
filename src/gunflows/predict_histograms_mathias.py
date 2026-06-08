# python
#!/usr/bin/env python3
# predict_histograms.py
#
# 1. Load likelihood interface and NF model via hydra config.
# 2. Sample parameter sets from NF, Gaussian (post-fit covariance), and/or MCMC.
# 3. For each sampled parameter set, propagate through GUNDAM and fill histograms
#    in one or more kinematic variables (Enu, Pmu, CosThetamu, …), split by
#    beam mode (FHC / RHC).
# 4. After the loop, report mean ± std-dev per bin and save results to disk.

from __future__ import annotations
import os, sys, math
from pathlib import Path
from contextlib import contextmanager

import hydra
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from omegaconf import DictConfig, OmegaConf


NF_LOCAL = os.path.join(os.path.dirname(__file__), "normalizing-flows")
sys.path.append(os.path.abspath(NF_LOCAL))


@contextmanager
def pushd(path: str | None):
    prev = os.getcwd()
    if path:
        os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


# ---------------------------------------------------------------------------
# NF model loader
# ---------------------------------------------------------------------------

def _maybe_build_nf(cfg, dataset, ckpt_path: str):
    from gunflows.utils.build_flow import build_base, build_flow_layers, build_model
    m = cfg.experiment.model
    device = cfg.device if "device" in cfg else (
        cfg.experiment.device if "device" in cfg.experiment else "cpu"
    )

    base = build_base(int(m.total_dim))
    dim_spline = len(dataset.phase_space_dim)
    tail_bounds = torch.ones(dim_spline) * float(m.tail_bound)
    flows = build_flow_layers(
        int(m.nflows), dim_spline, int(m.hidden), int(m.nlayers), int(m.nbins),
        tail_bounds, n_context=int(m.total_dim) - dim_spline,
    )
    freeze_covflow = bool(m.freeze_covflow) if hasattr(m, "freeze_covflow") else False
    kw = {}
    if hasattr(m, "n_context_flows"):
        kw["n_context_flows"] = int(m.n_context_flows)
    if hasattr(m, "hidden_dim"):
        kw["hidden_dim"] = int(m.hidden_dim)
    if hasattr(m, "n_hidden_layers"):
        kw["n_hidden_layers"] = int(m.n_hidden_layers)

    model = build_model(base, flows, dataset, m.context_transform, freeze_covflow, **kw)

    state = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STREAMS = ("FHC", "RHC")

_DEFAULT_CI_LEVELS = (0.6827, 0.9545, 0.9973)

_DEFAULT_BIN_EDGES = [
    0.0, 0.2, 0.4,
    0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
    2.0, 5.0,
]


# ---------------------------------------------------------------------------
# Event cache — single pass over all MC events for all variables
# ---------------------------------------------------------------------------

def build_var_caches(
    sampler,
    var_bin_edges: dict[str, np.ndarray],
    stream_var: str = "isRHC",
) -> dict[str, dict[str, tuple[list, np.ndarray]]]:
    """
    Single-pass cache for one or more kinematic variables, split by FHC/RHC.
    Returns {var_name: {stream: (events_list, bin_indices_array)}}.

    Events outside a variable's bin range are excluded from that variable's list
    only.  All variables AND stream_var must be in GUNDAM's additionalLeavesStorage
    (see override/mcKinStorage.yaml).
    """
    n_bins_map = {v: len(e) - 1 for v, e in var_bin_edges.items()}
    buckets: dict[str, dict[str, tuple[list, list]]] = {
        v: {s: ([], []) for s in STREAMS} for v in var_bin_edges
    }

    for sp in sampler.likelihood_interface.getSamplePairList():
        model_sample = sp.model
        if not model_sample.isEnabled():
            continue
        for event in model_sample.getEventList():
            leaves = event.getVariables()
            is_rhc = int(leaves.fetchVariable(stream_var).getVarAsDouble())
            stream = "RHC" if is_rhc else "FHC"
            for var_name, bin_edges in var_bin_edges.items():
                val = leaves.fetchVariable(var_name).getVarAsDouble()
                idx = int(np.digitize(val, bin_edges)) - 1
                if not (0 <= idx < n_bins_map[var_name]):
                    continue
                buckets[var_name][stream][0].append(event)
                buckets[var_name][stream][1].append(idx)

    out: dict[str, dict[str, tuple[list, np.ndarray]]] = {}
    for var_name in var_bin_edges:
        out[var_name] = {}
        for stream in STREAMS:
            events, indices = buckets[var_name][stream]
            out[var_name][stream] = (events, np.array(indices, dtype=np.int32))
            print(f"  Cache [{var_name}][{stream}]: {len(events)} in-range MC events",
                  flush=True)
    return out


def fill_histogram(
    events: list,
    bin_indices: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Fast weighted histogram fill using pre-computed bin indices."""
    weights = np.fromiter((e.getEventWeight() for e in events),
                          dtype=np.float64, count=len(events))
    return np.bincount(bin_indices, weights=weights, minlength=n_bins)


# ---------------------------------------------------------------------------
# MCMC chain reader
# ---------------------------------------------------------------------------

def load_mcmc_throws(
    mcmc_chain: str,
    n_samples: int,
    burnin_frac: float = 0.0,
    max_steps: int | None = None,
    thin: int | None = None,
) -> np.ndarray:
    """
    Read parameter throws from a GUNDAM MCMC ROOT file.
    Returns array of shape (n_samples, n_params).
    """
    import ROOT
    ROOT.gROOT.SetBatch(True)

    f = ROOT.TFile(mcmc_chain, "READ")
    if not f or f.IsZombie():
        raise FileNotFoundError(f"Cannot open MCMC file: {mcmc_chain}")
    tree = f.Get("FitterEngine/fit/MCMC")
    if not tree:
        raise RuntimeError("TTree 'FitterEngine/fit/MCMC' not found in file")

    n_total = int(tree.GetEntries())
    start = int(n_total * burnin_frac)
    stop  = min(n_total, start + max_steps) if max_steps is not None else n_total
    n_available = stop - start

    m = thin if thin is not None else max(1, n_available // n_samples)
    indices = list(range(start, stop, m))[:n_samples]
    print(f"  MCMC: {n_total} total steps, using [{start}:{stop}], "
          f"thin={m} → {len(indices)} throws", flush=True)

    rows = []
    for i in indices:
        tree.GetEntry(i)
        rows.append(np.array(list(tree.Points), dtype=np.float64))

    f.Close()
    return np.stack(rows)


# ---------------------------------------------------------------------------
# Per-throw inject + histogram (all variables in one GUNDAM propagation)
# ---------------------------------------------------------------------------

def _histograms_from_params(
    likelihood_sampler,
    params_array: np.ndarray,
    var_caches: dict[str, dict[str, tuple[list, np.ndarray]]],
    n_bins_map: dict[str, int],
    label: str,
) -> dict[str, dict[str, list[np.ndarray]]]:
    """
    Inject each row of params_array into GUNDAM, propagate once, and fill
    histograms for every (variable, stream) combination.
    Returns {var_name: {stream: [hist, ...]}} for all accepted throws.
    """
    histograms: dict[str, dict[str, list]] = {
        v: {s: [] for s in STREAMS} for v in var_caches
    }

    for i, params in enumerate(params_array):
        nll, _, _ = likelihood_sampler.inject_params_and_compute_likelihood(
            params.tolist(), extend_continue=False
        )
        if nll == -1:
            print(f"  [{label} {i}] out of domain — skipped", flush=True)
            continue

        for var_name, streams in var_caches.items():
            for stream, (events, indices) in streams.items():
                if not events:
                    continue
                hist = fill_histogram(events, indices, n_bins_map[var_name])
                histograms[var_name][stream].append(hist)

        n_accepted = max(
            max((len(histograms[v][s]) for s in STREAMS), default=0)
            for v in histograms
        )
        summary_parts = []
        for v in var_caches:
            fhc = histograms[v]["FHC"]
            rhc = histograms[v]["RHC"]
            fhc_sum = f"{fhc[-1].sum():.0f}" if fhc else "0"
            rhc_sum = f"{rhc[-1].sum():.0f}" if rhc else "0"
            summary_parts.append(f"{v}={fhc_sum}+{rhc_sum}")
        print(f"  [{label} {n_accepted:4d}] NLL={nll:.4f}  {'  '.join(summary_parts)}",
              flush=True)

    return histograms


def _checkpoint(
    save_dir: Path,
    var_bin_edges: dict[str, np.ndarray],
    all_hists: dict[str, dict[str, dict[str, list]]],  # {label: {var: {stream: [hist]}}}
    use_flags: dict[str, bool],
    ci_method: str = "percentile",
    ci_levels: tuple[float, ...] = _DEFAULT_CI_LEVELS,
    smooth: bool = False,
) -> None:
    """Save intermediate npy arrays and regenerate combined plots per variable per stream."""
    summary: dict[str, dict[str, int]] = {}

    for var_name, bin_edges in var_bin_edges.items():
        for stream in STREAMS:
            results:    dict[str, np.ndarray] = {}
            combined_n: dict[str, int] = {}

            for label, use in use_flags.items():
                if not use:
                    continue
                hists = all_hists[label][var_name][stream]
                if not hists:
                    continue
                hists_arr = np.array(hists, dtype=np.float64)
                tag = label.lower()
                vt  = var_name.lower()
                ss  = stream.lower()
                np.save(save_dir / f"histograms_{tag}_{vt}_{ss}.npy", hists_arr)
                np.save(save_dir / f"mean_{tag}_{vt}_{ss}.npy",       hists_arr.mean(axis=0))
                np.save(save_dir / f"std_{tag}_{vt}_{ss}.npy",        hists_arr.std(axis=0))
                results[label]    = hists_arr
                combined_n[label] = len(hists)

            if results:
                plot_combined(bin_edges, results, combined_n, save_dir,
                              var_name=var_name, stream=stream,
                              ci_method=ci_method, ci_levels=ci_levels, smooth=smooth)
                plot_violin(results, bin_edges, save_dir,
                            var_name=var_name, stream=stream)
            summary[f"{var_name}/{stream}"] = combined_n

    print(f"  [checkpoint] saved {summary}", flush=True)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _bin_labels(bin_edges: np.ndarray) -> list[str]:
    return [f"[{lo:.2f},{hi:.2f})" for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]


def _hdi_one(samples: np.ndarray, level: float) -> tuple[float, float]:
    """Sample-based highest-density interval."""
    s = np.sort(samples)
    n = len(s)
    n_in = max(1, int(np.ceil(level * n)))
    if n_in >= n:
        return float(s[0]), float(s[-1])
    widths = s[n_in:] - s[: n - n_in]
    i = int(np.argmin(widths))
    return float(s[i]), float(s[i + n_in])


def _compute_bands(
    hists_arr: np.ndarray,
    method: str = "percentile",
    levels: tuple[float, ...] = _DEFAULT_CI_LEVELS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_bins = hists_arr.shape[1]
    bands: list[tuple[np.ndarray, np.ndarray]] = []
    if method == "percentile":
        for lvl in levels:
            q_lo = 100.0 * (1.0 - lvl) / 2.0
            q_hi = 100.0 * (1.0 + lvl) / 2.0
            bands.append((
                np.percentile(hists_arr, q_lo, axis=0),
                np.percentile(hists_arr, q_hi, axis=0),
            ))
    elif method == "hdi":
        for lvl in levels:
            lo_arr = np.empty(n_bins)
            hi_arr = np.empty(n_bins)
            for b in range(n_bins):
                lo_arr[b], hi_arr[b] = _hdi_one(hists_arr[:, b], lvl)
            bands.append((lo_arr, hi_arr))
    else:
        raise ValueError(f"Unknown ci_method {method!r}; use 'percentile' or 'hdi'.")
    return bands


def _smooth_curve(centers: np.ndarray, vals: np.ndarray, n_points: int = 400):
    from scipy.interpolate import PchipInterpolator
    x_dense = np.linspace(centers[0], centers[-1], n_points)
    return x_dense, PchipInterpolator(centers, vals)(x_dense)


def _step_xy(edges, vals):
    x = np.concatenate([[edges[0]], np.repeat(edges[1:-1], 2), [edges[-1]]])
    y = np.repeat(vals, 2)
    return x, y


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_combined_one_level(
    bin_edges: np.ndarray,
    results: dict[str, np.ndarray],
    n_throws: dict[str, int],
    save_dir: Path,
    ci_method: str,
    level: float,
    smooth: bool,
    var_name: str = "",
    stream: str = "",
) -> None:
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths  = bin_edges[1:] - bin_edges[:-1]

    fig, (ax, ax_mean, ax_sig) = plt.subplots(
        3, 1, figsize=(10, 10),
        gridspec_kw={"height_ratios": [3, 1, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.05)

    COLORS = {"Gaussian": "#d62728", "NF": "#1f77b4", "MCMC": "#2ca02c"}
    BAND_ALPHA = 0.32

    # Collect per-label mean and std of bin counts (used for bottom panels)
    means_raw: dict[str, np.ndarray] = {}
    stds_raw:  dict[str, np.ndarray] = {}

    for label in ("Gaussian", "NF", "MCMC"):
        if label not in results:
            continue
        color = COLORS[label]
        hists_arr = results[label]

        mean_raw = hists_arr.mean(axis=0)
        means_raw[label] = mean_raw
        stds_raw[label]  = hists_arr.std(axis=0)

        mean = mean_raw / widths
        lo_raw, hi_raw = _compute_bands(hists_arr, method=ci_method, levels=(level,))[0]
        lo = lo_raw / widths
        hi = hi_raw / widths

        if smooth:
            x_d, lo_d = _smooth_curve(centers, np.maximum(0.0, lo))
            _,   hi_d = _smooth_curve(centers, hi)
            ax.fill_between(x_d, lo_d, hi_d, facecolor=color, alpha=BAND_ALPHA, linewidth=0)
        else:
            sx, sy_lo = _step_xy(bin_edges, np.maximum(0.0, lo))
            _,  sy_hi = _step_xy(bin_edges, hi)
            ax.fill_between(sx, sy_lo, sy_hi, facecolor=color, alpha=BAND_ALPHA, linewidth=0)

        n = n_throws.get(label, "?")
        legend_label = f"{label} ({n} throws)"
        if smooth:
            x_d, m_d = _smooth_curve(centers, mean)
            ax.plot(x_d, m_d, color=color, linewidth=1.6, label=legend_label)
        else:
            sx, sy = _step_xy(bin_edges, mean)
            ax.plot(sx, sy, color=color, linewidth=1.4, label=legend_label)

    pct = 100.0 * level
    stream_str = f" — {stream}"   if stream   else ""
    vname_str  = f" ({var_name})" if var_name else ""
    title = (f"CI band: {pct:.2f}%{stream_str}{vname_str}  "
             f"(method: {ci_method}" + (", smoothed)" if smooth else ")"))
    ax.set_title(title, fontsize=10, loc="right", color="gray")
    ax.set_ylabel("Event yield / bin width", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11)
    ax.tick_params(axis="both", labelsize=11)

    # ── Bottom panels: NF vs Gaussian relative differences ──────────────────
    _nf_present = "NF" in means_raw
    _ga_present = "Gaussian" in means_raw

    if _nf_present and _ga_present:
        mn = means_raw["NF"];   mg = means_raw["Gaussian"]
        sn = stds_raw["NF"];    sg = stds_raw["Gaussian"]

        denom_m = 0.5 * (mn + mg)
        rel_mean_pct = np.where(denom_m != 0.0, (mn - mg) / denom_m * 100.0, 0.0)

        denom_s = 0.5 * (sn + sg)
        rel_sig_pct  = np.where(denom_s != 0.0, (sn - sg) / denom_s * 100.0, 0.0)
    else:
        rel_mean_pct = np.zeros(len(centers))
        rel_sig_pct  = np.zeros(len(centers))

    bar_kw = dict(align="center", edgecolor="none", alpha=0.8)

    ax_mean.bar(centers, rel_mean_pct, width=widths, color="#1f77b4", **bar_kw)
    ax_mean.axhline(0, color="k", lw=0.8)
    ax_mean.set_ylabel("$\\Delta$ mean [%]", fontsize=10)
    ax_mean.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax_mean.grid(True, alpha=0.25, axis="y")
    ax_mean.tick_params(axis="both", labelsize=9)

    ax_sig.bar(centers, rel_sig_pct, width=widths, color="#d62728", **bar_kw)
    ax_sig.axhline(0, color="k", lw=0.8)
    ax_sig.set_ylabel("$\\Delta\\sigma$ [%]", fontsize=10)
    ax_sig.set_xlabel(var_name if var_name else "variable", fontsize=13)
    ax_sig.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax_sig.grid(True, alpha=0.25, axis="y")
    ax_sig.tick_params(axis="both", labelsize=9)
    ax_sig.set_xlim(bin_edges[0], bin_edges[-1])

    if _nf_present and _ga_present:
        ax_mean.set_title("(NF − Gaussian) / mean(NF, Gaussian)", fontsize=8,
                          loc="right", color="gray")

    tag = f"{pct:.2f}".rstrip("0").rstrip(".").replace(".", "p")
    stream_suffix = f"_{stream.lower()}" if stream else ""
    vname_suffix  = f"_{var_name.lower()}" if var_name else ""
    out = save_dir / f"combined_ci{tag}{vname_suffix}{stream_suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined(
    bin_edges: np.ndarray,
    results: dict[str, np.ndarray],
    n_throws: dict[str, int],
    save_dir: Path,
    var_name: str = "",
    stream: str = "",
    ci_method: str = "percentile",
    ci_levels: tuple[float, ...] = _DEFAULT_CI_LEVELS,
    smooth: bool = False,
) -> None:
    """One combined CI plot per coverage level."""
    for level in ci_levels:
        _plot_combined_one_level(
            bin_edges, results, n_throws, save_dir,
            ci_method=ci_method, level=float(level), smooth=smooth,
            var_name=var_name, stream=stream,
        )


def plot_correlation_matrix(
    hists_arr: np.ndarray,
    bin_edges: np.ndarray,
    label: str,
    save_dir: Path,
    var_name: str = "",
    stream: str = "",
) -> None:
    corr = np.corrcoef(hists_arr.T)
    labels = _bin_labels(bin_edges)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    stream_str = f" — {stream}" if stream else ""
    vname_str  = f" ({var_name})" if var_name else ""
    ax.set_title(f"Bin correlation — {label}{stream_str}{vname_str}")
    fig.tight_layout()
    stream_suffix = f"_{stream.lower()}" if stream else ""
    vname_suffix  = f"_{var_name.lower()}" if var_name else ""
    fig.savefig(save_dir / f"correlation_matrix_{label.lower()}{vname_suffix}{stream_suffix}.png",
                dpi=150)
    plt.close(fig)


def plot_corner(
    hists_arr: np.ndarray,
    bin_edges: np.ndarray,
    label: str,
    save_dir: Path,
    var_name: str = "",
    stream: str = "",
    bins: int = 60,
) -> None:
    """Match training-time corner style: step diagonal, viridis hist2d w/ LogNorm."""
    n_bins = hists_arr.shape[1]
    bin_labels = _bin_labels(bin_edges)

    fig, axes = plt.subplots(n_bins, n_bins, figsize=(2.2 * n_bins, 2.2 * n_bins))
    stream_str = f" — {stream}" if stream else ""
    vname_str  = f" ({var_name})" if var_name else ""
    fig.suptitle(f"Corner plot — {label}{stream_str}{vname_str}", y=1.01)

    for row in range(n_bins):
        for col in range(n_bins):
            ax = axes[row, col]
            if col > row:
                ax.axis("off")
            elif row == col:
                ax.hist(hists_arr[:, row], bins=bins, density=True, histtype="step")
                ax.set_xlabel(bin_labels[row], fontsize=6)
            else:
                ax.hist2d(hists_arr[:, col], hists_arr[:, row],
                          bins=bins, norm=LogNorm(), cmap="viridis")
                ax.set_xlabel(bin_labels[col], fontsize=6)
                ax.set_ylabel(bin_labels[row], fontsize=6)
            ax.tick_params(labelsize=5)

    fig.tight_layout()
    stream_suffix = f"_{stream.lower()}" if stream else ""
    vname_suffix  = f"_{var_name.lower()}" if var_name else ""
    fig.savefig(save_dir / f"corner_{label.lower()}{vname_suffix}{stream_suffix}.png", dpi=150)
    plt.close(fig)


def plot_violin(
    results: dict[str, np.ndarray],  # {label: hists_arr (n_throws, n_bins)}
    bin_edges: np.ndarray,
    save_dir: Path,
    var_name: str = "",
    stream: str = "",
) -> None:
    """
    Comparison violin plot: one violin per source (Gaussian / NF / MCMC) side
    by side within each bin.  Violins are offset proportionally to each bin's
    width so non-uniform binning is correctly represented.  The median is marked
    inside each violin.  Output: violin_<var>_<stream>.png.
    """
    if not results:
        return

    n_bins  = next(iter(results.values())).shape[1]
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths  = bin_edges[1:] - bin_edges[:-1]

    SOURCE_ORDER = [l for l in ("Gaussian", "NF", "MCMC") if l in results]
    COLORS = {"Gaussian": "#d62728", "NF": "#1f77b4", "MCMC": "#2ca02c"}
    n_src = len(SOURCE_ORDER)

    # Each violin is offset by a fraction of its bin width.
    # Total spread covers ±spread of bin width; individual width uses the rest.
    spread   = 0.28
    offsets  = np.linspace(-spread, spread, n_src) if n_src > 1 else np.array([0.0])
    vw_frac  = (2 * spread / max(n_src, 1)) * 0.80

    fig, ax = plt.subplots(figsize=(max(8, n_bins), 5))

    for j, label in enumerate(SOURCE_ORDER):
        hists_arr = results[label]
        density   = hists_arr / widths[np.newaxis, :]
        positions = centers + offsets[j] * widths
        vwidths   = widths * vw_frac

        vp = ax.violinplot(
            [density[:, i] for i in range(n_bins)],
            positions=positions,
            widths=vwidths,
            showmedians=True,
            showextrema=False,
        )
        color = COLORS.get(label, f"C{j}")
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.65)
        vp["cmedians"].set_color(color)
        vp["cmedians"].set_linewidth(1.5)
        # invisible patch for the legend
        ax.fill_between([], [], [], color=color, alpha=0.65, label=label)

    stream_str = f" — {stream}" if stream else ""
    vname_str  = f" ({var_name})" if var_name else ""
    ax.set_title(f"Per-bin yield distribution{stream_str}{vname_str}")
    ax.set_xlabel(var_name if var_name else "variable", fontsize=12)
    ax.set_ylabel("Event yield / bin width", fontsize=12)
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    fig.tight_layout()

    stream_suffix = f"_{stream.lower()}" if stream else ""
    vname_suffix  = f"_{var_name.lower()}" if var_name else ""
    fig.savefig(save_dir / f"violin{vname_suffix}{stream_suffix}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main (hydra entry point)
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="predict_histograms", version_base=None)
def main(cfg: DictConfig) -> None:
    from gunflows.likelihood_sampler import LikelihoodSampler
    from gunflows.sample_mcmc_toy import build_sampling_dataset_target

    use_nf       = bool(cfg.get("use_nf", True))
    use_gaussian = bool(cfg.get("use_gaussian", True))
    use_mcmc     = bool(cfg.get("use_mcmc", False))
    if not use_nf and not use_gaussian and not use_mcmc:
        raise ValueError("At least one of use_nf, use_gaussian, use_mcmc must be True.")

    # ------------------------------------------------------------------
    # 1a. Load checkpoint + training config (only needed when use_nf=True)
    # ------------------------------------------------------------------
    best_ckpt = None
    if use_nf:
        training_folder = Path(cfg.training_folder).expanduser().resolve()
        print(f"Loading NF model from: {training_folder}", flush=True)
        if not training_folder.exists():
            raise FileNotFoundError(f"Training folder not found: {training_folder}")

        checkpoint_dir = training_folder / "checkpoints"
        checkpoints = sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() \
            else sorted(training_folder.glob("*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {training_folder}")
        best_ckpt = str(checkpoints[-1])
        print(f"Using checkpoint: {best_ckpt}", flush=True)

        hydra_config_path = training_folder / ".hydra" / "config.yaml"
        if not hydra_config_path.exists():
            raise FileNotFoundError(f"Hydra config not found at {hydra_config_path}")
        train_cfg = OmegaConf.load(hydra_config_path)
        cfg = OmegaConf.merge(train_cfg, cfg)
        print(f"Loaded training config from: {hydra_config_path}", flush=True)

    # ------------------------------------------------------------------
    # 1b. Initialise LikelihoodSampler (GUNDAM interface)
    # ------------------------------------------------------------------
    if "llh_config" in cfg:
        llh_config    = str(cfg.llh_config)
        llh_overrides = list(cfg.get("llh_overrides", []))
        data_is_asimov = bool(cfg.get("data_is_asimov", True))
        llh_cwd       = str(cfg.get("llh_cwd", ".")) if cfg.get("llh_cwd") else None
        threads       = int(cfg.get("threads", 1))
    else:
        llh_config    = str(cfg.experiment.dataset.llh_config)
        llh_overrides = list(cfg.experiment.dataset.llh_overrides)
        data_is_asimov = bool(cfg.experiment.dataset.data_is_asimov)
        llh_cwd       = str(cfg.experiment.dataset.llh_cwd)
        threads       = int(cfg.experiment.sampler.threads) if hasattr(cfg.experiment, "sampler") else 1

    llh_overrides += list(cfg.get("llh_extra_overrides", []))

    print("Initializing LikelihoodSampler...", flush=True)
    likelihood_sampler = LikelihoodSampler(
        config_file=llh_config,
        override_files=llh_overrides,
        data_is_asimov=data_is_asimov,
        threads=threads,
        llh_cwd=llh_cwd,
        light_mode=False,
    )

    bestfit = np.asarray(likelihood_sampler.postfit_parameter_values, dtype=np.float64)
    cov     = np.asarray(likelihood_sampler.postfit_covariance_matrix,  dtype=np.float64)
    print(f"Covariance matrix shape: {cov.shape}", flush=True)

    # ------------------------------------------------------------------
    # 1c. Build NF model (only when use_nf=True)
    # ------------------------------------------------------------------
    nf_model = None
    dataset  = None
    if use_nf:
        dataset  = build_sampling_dataset_target(cfg, bestfit, cov)
        nf_model = _maybe_build_nf(cfg, dataset, best_ckpt)
        print("NF model loaded.", flush=True)

    # ------------------------------------------------------------------
    # Build variable → bin-edges map
    #
    # Primary variable: enu_var (default "Enu")
    #   Priority: bin_edges_list > n_bins/enu_min/enu_max > hardcoded default
    #
    # Extra variables: cfg.extra_vars (list of {name: str, bin_edges: list})
    #   Each extra variable must also be present in additionalLeavesStorage.
    #   Use override/mcKinStorage.yaml instead of mcEnuStorage.yaml when adding
    #   Pmu / CosThetamu.
    # ------------------------------------------------------------------
    enu_var = str(cfg.get("enu_var", "Enu"))
    if "bin_edges_list" in cfg and cfg.bin_edges_list is not None:
        primary_edges = np.array(list(cfg.bin_edges_list), dtype=np.float64)
    elif "n_bins" in cfg:
        n_bins_  = int(cfg.n_bins)
        enu_min  = float(cfg.get("enu_min", 0.0))
        enu_max  = float(cfg.get("enu_max", 5.0))
        primary_edges = np.linspace(enu_min, enu_max, n_bins_ + 1)
    else:
        primary_edges = np.array(_DEFAULT_BIN_EDGES, dtype=np.float64)

    var_bin_edges: dict[str, np.ndarray] = {enu_var: primary_edges}

    for extra in cfg.get("extra_vars", []) or []:
        vname  = str(extra["name"])
        vedges = np.array(list(extra["bin_edges"]), dtype=np.float64)
        var_bin_edges[vname] = vedges
        print(f"Extra variable: {vname}  ({len(vedges)-1} bins)", flush=True)

    n_bins_map = {v: len(e) - 1 for v, e in var_bin_edges.items()}
    print(f"Variables: { {v: n for v, n in n_bins_map.items()} } bins", flush=True)

    save_dir = Path(cfg.save_dir).expanduser() if "save_dir" in cfg else Path(".")
    save_dir.mkdir(parents=True, exist_ok=True)
    for v, edges in var_bin_edges.items():
        np.save(save_dir / f"bin_edges_{v.lower()}.npy", edges)

    # ------------------------------------------------------------------
    # Build event caches — single pass over all MC events
    # ------------------------------------------------------------------
    print("Building variable caches (single pass, split by isRHC)...", flush=True)
    var_caches = build_var_caches(likelihood_sampler, var_bin_edges)

    num_samples = int(cfg.num_samples)
    batch_size  = int(cfg.batch_size)
    save_every  = int(cfg.get("save_every", 1000))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    ci_method = str(cfg.get("ci_method", "percentile"))
    ci_levels = tuple(float(x) for x in cfg.get("ci_levels", list(_DEFAULT_CI_LEVELS)))
    smooth    = bool(cfg.get("smooth", False))
    print(f"CI: method={ci_method}, levels={ci_levels}, smooth={smooth}", flush=True)

    # ------------------------------------------------------------------
    # Load MCMC throws up-front
    # ------------------------------------------------------------------
    mcmc_throws: np.ndarray | None = None
    if use_mcmc:
        mcmc_chain = str(cfg.mcmc_chain)
        mcmc_throws = load_mcmc_throws(
            mcmc_chain,
            n_samples    = num_samples,
            burnin_frac  = float(cfg.get("mcmc_burnin_frac", 0.0)),
            max_steps    = int(cfg.mcmc_max_steps) if "mcmc_max_steps" in cfg else None,
            thin         = int(cfg.mcmc_thin)      if "mcmc_thin"      in cfg else None,
        )
        print(f"MCMC throws loaded: {mcmc_throws.shape}", flush=True)

    # ------------------------------------------------------------------
    # Sampling loop
    #
    # Accumulation:  all_hists[label][var_name][stream] = [hist, ...]
    # ------------------------------------------------------------------
    def _empty():
        return {v: {s: [] for s in STREAMS} for v in var_caches}

    nf_hists       = _empty()
    gaussian_hists = _empty()
    mcmc_hists     = _empty()

    use_flags = {"NF": use_nf, "Gaussian": use_gaussian, "MCMC": use_mcmc}
    all_hists = {"NF": nf_hists, "Gaussian": gaussian_hists, "MCMC": mcmc_hists}

    def _count(h):
        first = next(iter(h))
        return max((len(h[first][s]) for s in STREAMS), default=0)

    def _nf_done():    return (not use_nf)      or _count(nf_hists)       >= num_samples
    def _gauss_done(): return (not use_gaussian) or _count(gaussian_hists) >= num_samples
    def _mcmc_done():  return (not use_mcmc)     or _count(mcmc_hists)     >= num_samples

    def _should_checkpoint(old: int, new: int) -> bool:
        if save_every <= 0:
            return False
        return (new // save_every) > (old // save_every)

    def _extend(target, new):
        for v in new:
            for s in STREAMS:
                target[v][s].extend(new[v].get(s, []))

    def _do_checkpoint():
        _checkpoint(save_dir, var_bin_edges, all_hists, use_flags,
                    ci_method=ci_method, ci_levels=ci_levels, smooth=smooth)

    active = (["NF"] if use_nf else []) + (["Gaussian"] if use_gaussian else []) + (["MCMC"] if use_mcmc else [])
    print(f"\nStarting sampling loop: {num_samples} throws  [{' + '.join(active)}]", flush=True)

    with torch.no_grad():
        while not _nf_done() or not _gauss_done() or not _mcmc_done():

            # --- NF batch ---
            if not _nf_done():
                prev = _count(nf_hists)
                need = min(batch_size, num_samples - prev)
                z_nf, _ = nf_model.sample(need)
                x_nf = dataset.transform_eigen_space_to_data_space(z_nf).cpu().numpy()
                new_h = _histograms_from_params(
                    likelihood_sampler, x_nf, var_caches, n_bins_map,
                    f"NF {prev+1}–{prev+len(x_nf)}",
                )
                _extend(nf_hists, new_h)
                cur = _count(nf_hists)
                print(f"NF:    {cur}/{num_samples} valid throws", flush=True)
                if _should_checkpoint(prev, cur):
                    _do_checkpoint()

            # --- Gaussian batch ---
            if not _gauss_done():
                prev = _count(gaussian_hists)
                need = min(batch_size, num_samples - prev)
                x_g  = rng.multivariate_normal(bestfit, cov, size=need)
                new_h = _histograms_from_params(
                    likelihood_sampler, x_g, var_caches, n_bins_map,
                    f"Gauss {prev+1}–{prev+len(x_g)}",
                )
                _extend(gaussian_hists, new_h)
                cur = _count(gaussian_hists)
                print(f"Gauss: {cur}/{num_samples} valid throws", flush=True)
                if _should_checkpoint(prev, cur):
                    _do_checkpoint()

            # --- MCMC batch ---
            if not _mcmc_done():
                prev = _count(mcmc_hists)
                need = min(batch_size, num_samples - prev)
                x_mc = mcmc_throws[prev : prev + need]
                new_h = _histograms_from_params(
                    likelihood_sampler, x_mc, var_caches, n_bins_map,
                    f"MCMC {prev+1}–{prev+len(x_mc)}",
                )
                _extend(mcmc_hists, new_h)
                cur = _count(mcmc_hists)
                print(f"MCMC:  {cur}/{num_samples} valid throws", flush=True)
                if _should_checkpoint(prev, cur):
                    _do_checkpoint()

    # ------------------------------------------------------------------
    # Final summary + plots  (all variables × all streams)
    # ------------------------------------------------------------------
    for var_name, bin_edges in var_bin_edges.items():
        for stream in STREAMS:
            results:    dict[str, np.ndarray] = {}
            combined_n: dict[str, int] = {}

            for label, use in use_flags.items():
                if not use:
                    continue
                hists = all_hists[label][var_name][stream][:num_samples]
                if not hists:
                    continue
                hists_arr = np.array(hists, dtype=np.float64)
                mean_hist = hists_arr.mean(axis=0)
                std_hist  = hists_arr.std(axis=0)

                print(f"\n{'='*60}")
                print(f"  {label} [{var_name}][{stream}]  ({len(hists)} throws)")
                print(f"{'='*60}")
                for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                    print(f"  [{lo:.3g}, {hi:.3g}):  {mean_hist[i]:.4f} ± {std_hist[i]:.4f}")

                tag = label.lower()
                vt  = var_name.lower()
                ss  = stream.lower()
                np.save(save_dir / f"histograms_{tag}_{vt}_{ss}.npy", hists_arr)
                np.save(save_dir / f"mean_{tag}_{vt}_{ss}.npy",       mean_hist)
                np.save(save_dir / f"std_{tag}_{vt}_{ss}.npy",        std_hist)

                plot_correlation_matrix(hists_arr, bin_edges, label, save_dir,
                                        var_name=var_name, stream=stream)
                plot_corner(hists_arr, bin_edges, label, save_dir,
                            var_name=var_name, stream=stream)
                print(f"  Plots saved for {label} [{var_name}][{stream}]", flush=True)

                results[label]    = hists_arr
                combined_n[label] = len(hists)

            if results:
                plot_combined(bin_edges, results, combined_n, save_dir,
                              var_name=var_name, stream=stream,
                              ci_method=ci_method, ci_levels=ci_levels, smooth=smooth)
                plot_violin(results, bin_edges, save_dir,
                            var_name=var_name, stream=stream)
                print(f"  Combined + violin plots saved for {var_name} [{stream}].", flush=True)

    print(f"\nResults saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
