# python
#!/usr/bin/env python3
# predict_histograms.py
#
# 1. Load likelihood interface and NF model via hydra config.
# 2. Sample parameter sets from the NF model and a Gaussian (post-fit covariance).
# 3. For each sampled parameter set, propagate through GUNDAM and fill an E_nu
#    histogram by iterating over all MC model events.
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
# NF model loader (unchanged from remote)
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
# E_nu histogram filler — with one-time cache to avoid per-throw Python overhead
# ---------------------------------------------------------------------------

STREAMS = ("FHC", "RHC")

# 1σ, 2σ, 3σ Gaussian-equivalent coverage levels for CI bands
_DEFAULT_CI_LEVELS = (0.6827, 0.9545, 0.9973)


def build_enu_cache(
    sampler,
    bin_edges: np.ndarray,
    enu_var: str = "Enu",
) -> dict[str, tuple[list, np.ndarray]]:
    """
    Called once before the sampling loop. Splits MC events into FHC / RHC
    streams using the integer leaf 'isRHC' (0 → FHC, 1 → RHC).
    Returns {"FHC": (events, bin_indices), "RHC": (events, bin_indices)}.

    Both 'Enu' and 'isRHC' must be in GUNDAM's additionalLeavesStorage
    (see override yaml mcEnuStorage.yaml).
    """
    n_bins = len(bin_edges) - 1
    buckets: dict[str, tuple[list, list[int]]] = {s: ([], []) for s in STREAMS}

    for sp in sampler.likelihood_interface.getSamplePairList():
        model_sample = sp.model
        if not model_sample.isEnabled():
            continue
        for event in model_sample.getEventList():
            vars_ = event.getVariables()
            enu = vars_.fetchVariable(enu_var).getVarAsDouble()
            idx = int(np.digitize(enu, bin_edges)) - 1
            if not (0 <= idx < n_bins):
                continue
            is_rhc = int(vars_.fetchVariable("isRHC").getVarAsDouble())
            stream = "RHC" if is_rhc else "FHC"
            buckets[stream][0].append(event)
            buckets[stream][1].append(idx)

    out: dict[str, tuple[list, np.ndarray]] = {}
    for stream in STREAMS:
        events, indices = buckets[stream]
        out[stream] = (events, np.array(indices, dtype=np.int32))
        print(f"  Enu cache [{stream}]: {len(events)} in-range MC events", flush=True)
    return out


def fill_enu_histogram(
    events: list,
    bin_indices: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """
    Per-throw histogram fill. Only getEventWeight() is called here;
    bin indices are pre-computed. Uses np.bincount for fast accumulation.
    """
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

    Thinning: every m-th entry is kept, where m = n_available // n_samples
    (or the user-supplied thin override). This gives approximately independent
    draws spanning the full post-burnin chain.
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
# Per-sample inject + histogram
# ---------------------------------------------------------------------------

def _histograms_from_params(
    likelihood_sampler,
    params_array: np.ndarray,
    enu_cache: dict[str, tuple[list, np.ndarray]],
    n_bins: int,
    label: str,
) -> dict[str, list[np.ndarray]]:
    """
    Inject each row of params_array into GUNDAM, propagate, and collect
    one E_nu histogram per stream (FHC, RHC) per accepted throw.
    Returns {"FHC": [hist, ...], "RHC": [hist, ...]} aligned across streams.
    """
    histograms: dict[str, list[np.ndarray]] = {s: [] for s in enu_cache}
    for i, params in enumerate(params_array):
        nll, _, _ = likelihood_sampler.inject_params_and_compute_likelihood(
            params.tolist(), extend_continue=False
        )
        if nll == -1:
            print(f"  [{label} {i}] out of domain — skipped", flush=True)
            continue

        per_stream_sum = []
        for stream, (events, indices) in enu_cache.items():
            if len(events) == 0:
                continue
            hist = fill_enu_histogram(events, indices, n_bins)
            histograms[stream].append(hist)
            per_stream_sum.append(f"{stream}={hist.sum():.0f}")

        count = max((len(h) for h in histograms.values()), default=0)
        print(f"  [{label} {count:4d}] NLL={nll:.4f}  {' '.join(per_stream_sum)}", flush=True)

    return histograms


def _checkpoint(
    save_dir: Path,
    bin_edges: np.ndarray,
    nf_per_stream: dict[str, list],
    gaussian_per_stream: dict[str, list],
    mcmc_per_stream: dict[str, list],
    use_nf: bool,
    use_gaussian: bool,
    use_mcmc: bool,
    ci_method: str = "percentile",
    ci_levels: tuple[float, ...] = _DEFAULT_CI_LEVELS,
    smooth: bool = False,
) -> None:
    """Save intermediate npy arrays and regenerate combined plots per stream."""
    sources = (
        ("NF",       nf_per_stream,       use_nf),
        ("Gaussian", gaussian_per_stream, use_gaussian),
        ("MCMC",     mcmc_per_stream,     use_mcmc),
    )
    summary: dict[str, dict[str, int]] = {}
    for stream in STREAMS:
        results: dict[str, np.ndarray] = {}
        combined_n: dict[str, int] = {}
        for label, per_stream, use in sources:
            if not use:
                continue
            hists = per_stream.get(stream, [])
            if not hists:
                continue
            hists_arr = np.array(hists, dtype=np.float64)
            tag = label.lower()
            ss  = stream.lower()
            np.save(save_dir / f"enu_histograms_{tag}_{ss}.npy", hists_arr)
            np.save(save_dir / f"enu_mean_{tag}_{ss}.npy",       hists_arr.mean(axis=0))
            np.save(save_dir / f"enu_std_{tag}_{ss}.npy",        hists_arr.std(axis=0))
            results[label] = hists_arr
            combined_n[label] = len(hists)
        if results:
            plot_enu_combined(bin_edges, results, combined_n, save_dir,
                              stream=stream,
                              ci_method=ci_method, ci_levels=ci_levels, smooth=smooth)
        summary[stream] = combined_n
    print(f"  [checkpoint] saved {summary}", flush=True)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_DEFAULT_BIN_EDGES = [
    0.0, 0.2, 0.4,
    0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
    2.0, 5.0,
]

def _bin_labels(bin_edges: np.ndarray) -> list[str]:
    return [f"[{lo:.2f},{hi:.2f})" for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]


# ---------------------------------------------------------------------------
# Confidence-interval helpers
# ---------------------------------------------------------------------------


def _hdi_one(samples: np.ndarray, level: float) -> tuple[float, float]:
    """
    Sample-based highest-density interval at given coverage level.
    Sort samples, find smallest window containing ceil(level * N).
    For unimodal smooth distributions this matches a KDE-based HDI very well.
    """
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
    """
    Return list of (lo, hi) bin-content arrays, one per confidence level.
    method="percentile": central quantile intervals (default).
    method="hdi"       : sample-based highest-density intervals.
    """
    n_bins = hists_arr.shape[1]
    bands: list[tuple[np.ndarray, np.ndarray]] = []
    if method == "percentile":
        for lvl in levels:
            q_lo = 100.0 * (1.0 - lvl) / 2.0
            q_hi = 100.0 * (1.0 + lvl) / 2.0
            lo = np.percentile(hists_arr, q_lo, axis=0)
            hi = np.percentile(hists_arr, q_hi, axis=0)
            bands.append((lo, hi))
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
    """Monotonic-piecewise-cubic interpolation through bin centers."""
    from scipy.interpolate import PchipInterpolator
    x_dense = np.linspace(centers[0], centers[-1], n_points)
    return x_dense, PchipInterpolator(centers, vals)(x_dense)


# ---------------------------------------------------------------------------
# Combined E_nu plot
# ---------------------------------------------------------------------------

def _step_xy(edges, vals):
    x = np.concatenate([[edges[0]], np.repeat(edges[1:-1], 2), [edges[-1]]])
    y = np.repeat(vals, 2)
    return x, y


def _plot_enu_one_level(
    bin_edges: np.ndarray,
    results: dict[str, np.ndarray],
    n_throws: dict[str, int],
    save_dir: Path,
    ci_method: str,
    level: float,
    smooth: bool,
    stream: str = "",
) -> None:
    """Produce one combined plot for a single CI coverage level."""
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
        bands_raw = _compute_bands(hists_arr, method=ci_method, levels=(level,))
        lo_raw, hi_raw = bands_raw[0]
        lo = lo_raw / widths
        hi = hi_raw / widths

        # Single CI band
        if smooth:
            x_d, lo_d = _smooth_curve(centers, np.maximum(0.0, lo))
            _,   hi_d = _smooth_curve(centers, hi)
            ax.fill_between(x_d, lo_d, hi_d,
                            facecolor=color, alpha=BAND_ALPHA, linewidth=0)
        else:
            sx, sy_lo = _step_xy(bin_edges, np.maximum(0.0, lo))
            _,  sy_hi = _step_xy(bin_edges, hi)
            ax.fill_between(sx, sy_lo, sy_hi,
                            facecolor=color, alpha=BAND_ALPHA, linewidth=0)

        # Mean line (legend entry)
        n = n_throws.get(label, "?")
        legend_label = f"{label} ({n} throws)"
        if smooth:
            x_d, m_d = _smooth_curve(centers, mean)
            ax.plot(x_d, m_d, color=color, linewidth=1.6, label=legend_label)
        else:
            sx, sy = _step_xy(bin_edges, mean)
            ax.plot(sx, sy, color=color, linewidth=1.4, label=legend_label)

    pct = 100.0 * level
    stream_str = f" — {stream}" if stream else ""
    title = f"CI band: {pct:.2f}%{stream_str}  (method: {ci_method}" + (", smoothed)" if smooth else ")")
    ax.set_title(title, fontsize=10, loc="right", color="gray")
    ax.set_ylabel(r"Event yield / bin width  [GeV$^{-1}$]", fontsize=13)
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
    ax_sig.set_xlabel(r"$E_\nu^{\mathrm{rec}}$ [GeV]", fontsize=13)
    ax_sig.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax_sig.grid(True, alpha=0.25, axis="y")
    ax_sig.tick_params(axis="both", labelsize=9)
    ax_sig.set_xlim(bin_edges[0], bin_edges[-1])

    if _nf_present and _ga_present:
        ax_mean.set_title("(NF − Gaussian) / mean(NF, Gaussian)", fontsize=8,
                          loc="right", color="gray")

    # File-safe tag from coverage percentage
    tag = f"{pct:.2f}".rstrip("0").rstrip(".").replace(".", "p")
    stream_suffix = f"_{stream.lower()}" if stream else ""
    out = save_dir / f"enu_histogram{stream_suffix}_combined_ci{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_enu_combined(
    bin_edges: np.ndarray,
    results: dict[str, np.ndarray],     # label -> hists_arr (n_throws, n_bins)
    n_throws: dict[str, int],
    save_dir: Path,
    stream: str = "",
    ci_method: str = "percentile",
    ci_levels: tuple[float, ...] = _DEFAULT_CI_LEVELS,
    smooth: bool = False,
) -> None:
    """
    Produce one combined plot per CI coverage level.
    Filenames: `enu_histogram_combined_ci<pct>[_<stream>].png`.
    """
    for level in ci_levels:
        _plot_enu_one_level(bin_edges, results, n_throws, save_dir,
                            ci_method=ci_method, level=float(level), smooth=smooth,
                            stream=stream)


def plot_correlation_matrix(
    hists_arr: np.ndarray,
    bin_edges: np.ndarray,
    label: str,
    save_dir: Path,
    stream: str = "",
) -> None:
    corr = np.corrcoef(hists_arr.T)   # [n_bins, n_bins]
    labels = _bin_labels(bin_edges)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    title_suffix = f" — {stream}" if stream else ""
    ax.set_title(f"Bin-content correlation — {label}{title_suffix}")
    fig.tight_layout()
    file_suffix = f"_{stream.lower()}" if stream else ""
    fig.savefig(save_dir / f"correlation_matrix{file_suffix}_{label.lower()}.png", dpi=150)
    plt.close(fig)


def plot_corner(
    hists_arr: np.ndarray,
    bin_edges: np.ndarray,
    label: str,
    save_dir: Path,
    stream: str = "",
    bins: int = 60,
) -> None:
    """Match training-time corner style: step diagonal, viridis hist2d w/ LogNorm."""
    n_bins = hists_arr.shape[1]
    labels = _bin_labels(bin_edges)

    fig, axes = plt.subplots(n_bins, n_bins, figsize=(2.2 * n_bins, 2.2 * n_bins))
    title_suffix = f" — {stream}" if stream else ""
    fig.suptitle(f"Corner plot — {label}{title_suffix}", y=1.01)

    for row in range(n_bins):
        for col in range(n_bins):
            ax = axes[row, col]
            if col > row:
                ax.axis("off")
            elif row == col:
                ax.hist(hists_arr[:, row], bins=bins, density=True, histtype="step")
                ax.set_xlabel(labels[row], fontsize=6)
            else:
                ax.hist2d(hists_arr[:, col], hists_arr[:, row],
                          bins=bins, norm=LogNorm(), cmap="viridis")
                ax.set_xlabel(labels[col], fontsize=6)
                ax.set_ylabel(labels[row], fontsize=6)
            ax.tick_params(labelsize=5)

    fig.tight_layout()
    file_suffix = f"_{stream.lower()}" if stream else ""
    fig.savefig(save_dir / f"corner{file_suffix}_{label.lower()}.png", dpi=150)
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
    #     Direct keys (llh_config, llh_cwd, …) take priority over the
    #     experiment.dataset sub-tree loaded from the training config.
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
    # E_nu binning
    # Priority: bin_edges_list in config > n_bins/enu_min/enu_max > default
    # ------------------------------------------------------------------
    enu_var = str(cfg.get("enu_var", "Enu"))
    if "bin_edges_list" in cfg and cfg.bin_edges_list is not None:
        bin_edges = np.array(list(cfg.bin_edges_list), dtype=np.float64)
    elif "n_bins" in cfg:
        n_bins  = int(cfg.n_bins)
        enu_min = float(cfg.get("enu_min", 0.0))
        enu_max = float(cfg.get("enu_max", 5.0))
        bin_edges = np.linspace(enu_min, enu_max, n_bins + 1)
    else:
        bin_edges = np.array(_DEFAULT_BIN_EDGES, dtype=np.float64)
    n_bins = len(bin_edges) - 1
    print(f"E_nu binning: {n_bins} bins, edges: {bin_edges}", flush=True)

    # Build Enu cache once — bin indices are constant across throws
    print("Building Enu cache (split by isRHC)...", flush=True)
    enu_cache = build_enu_cache(likelihood_sampler, bin_edges, enu_var)

    num_samples = int(cfg.num_samples)
    batch_size  = int(cfg.batch_size)
    save_every  = int(cfg.get("save_every", 1000))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    # CI / smoothing options for the combined plot
    ci_method = str(cfg.get("ci_method", "percentile"))
    ci_levels = tuple(float(x) for x in cfg.get("ci_levels", list(_DEFAULT_CI_LEVELS)))
    smooth    = bool(cfg.get("smooth", False))
    print(f"Combined plot: ci_method={ci_method}, levels={ci_levels}, smooth={smooth}", flush=True)

    # ------------------------------------------------------------------
    # Load MCMC throws up-front (all at once, no batching needed)
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
    # 2+3. Sampling loop — draw batches, propagate, collect histograms
    #       Histograms are stored per stream (FHC / RHC).
    # ------------------------------------------------------------------
    nf_per_stream:       dict[str, list[np.ndarray]] = {s: [] for s in STREAMS}
    gaussian_per_stream: dict[str, list[np.ndarray]] = {s: [] for s in STREAMS}
    mcmc_per_stream:     dict[str, list[np.ndarray]] = {s: [] for s in STREAMS}

    save_dir = Path(cfg.save_dir).expanduser() if "save_dir" in cfg else Path(".")
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "bin_edges.npy", bin_edges)

    active_parts = (["NF"] if use_nf else []) + (["Gaussian"] if use_gaussian else []) + (["MCMC"] if use_mcmc else [])
    print(f"\nStarting sampling loop: {num_samples} throws  [{' + '.join(active_parts)}]", flush=True)

    def _count(per_stream):
        return max((len(v) for v in per_stream.values()), default=0)

    def _nf_done():    return (not use_nf)      or _count(nf_per_stream)       >= num_samples
    def _gauss_done(): return (not use_gaussian) or _count(gaussian_per_stream) >= num_samples
    def _mcmc_done():  return (not use_mcmc)     or _count(mcmc_per_stream)     >= num_samples

    def _should_checkpoint(old_count: int, new_count: int) -> bool:
        if save_every <= 0:
            return False
        return (new_count // save_every) > (old_count // save_every)

    def _do_checkpoint():
        _checkpoint(save_dir, bin_edges,
                    nf_per_stream, gaussian_per_stream, mcmc_per_stream,
                    use_nf, use_gaussian, use_mcmc,
                    ci_method=ci_method, ci_levels=ci_levels, smooth=smooth)

    def _extend(per_stream, new):
        for s in STREAMS:
            per_stream[s].extend(new.get(s, []))

    with torch.no_grad():
        while not _nf_done() or not _gauss_done() or not _mcmc_done():

            # --- NF batch ---
            if not _nf_done():
                prev = _count(nf_per_stream)
                need = min(batch_size, num_samples - prev)
                z_nf, _ = nf_model.sample(need)
                x_nf = dataset.transform_eigen_space_to_data_space(z_nf).cpu().numpy()
                new_hists = _histograms_from_params(
                    likelihood_sampler, x_nf, enu_cache, n_bins,
                    f"NF {prev+1}–{prev+len(x_nf)}",
                )
                _extend(nf_per_stream, new_hists)
                print(f"NF:    {_count(nf_per_stream)}/{num_samples} valid throws", flush=True)
                if _should_checkpoint(prev, _count(nf_per_stream)):
                    _do_checkpoint()

            # --- Gaussian batch ---
            if not _gauss_done():
                prev = _count(gaussian_per_stream)
                need = min(batch_size, num_samples - prev)
                x_g = rng.multivariate_normal(bestfit, cov, size=need)
                new_hists = _histograms_from_params(
                    likelihood_sampler, x_g, enu_cache, n_bins,
                    f"Gauss {prev+1}–{prev+len(x_g)}",
                )
                _extend(gaussian_per_stream, new_hists)
                print(f"Gauss: {_count(gaussian_per_stream)}/{num_samples} valid throws", flush=True)
                if _should_checkpoint(prev, _count(gaussian_per_stream)):
                    _do_checkpoint()

            # --- MCMC batch ---
            if not _mcmc_done():
                prev = _count(mcmc_per_stream)
                need = min(batch_size, num_samples - prev)
                x_mc = mcmc_throws[prev:prev + need]
                new_hists = _histograms_from_params(
                    likelihood_sampler, x_mc, enu_cache, n_bins,
                    f"MCMC {prev+1}–{prev+len(x_mc)}",
                )
                _extend(mcmc_per_stream, new_hists)
                print(f"MCMC:  {_count(mcmc_per_stream)}/{num_samples} valid throws", flush=True)
                if _should_checkpoint(prev, _count(mcmc_per_stream)):
                    _do_checkpoint()

    # ------------------------------------------------------------------
    # 4. Summary: per-stream mean ± std, save to disk, produce plots
    # ------------------------------------------------------------------
    sources = (
        ("NF",       nf_per_stream,       use_nf),
        ("Gaussian", gaussian_per_stream, use_gaussian),
        ("MCMC",     mcmc_per_stream,     use_mcmc),
    )

    for stream in STREAMS:
        results:    dict[str, np.ndarray] = {}
        combined_n: dict[str, int] = {}

        for label, per_stream, use in sources:
            if not use:
                continue
            hists = per_stream.get(stream, [])[:num_samples]
            if not hists:
                continue
            hists_arr = np.array(hists, dtype=np.float64)
            mean_hist = hists_arr.mean(axis=0)
            std_hist  = hists_arr.std(axis=0)

            print(f"\n{'='*60}")
            print(f"  {label} [{stream}]  E_nu histogram  (mean ± std, {len(hists)} throws)")
            print(f"{'='*60}")
            for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                print(f"  [{lo:.2f}, {hi:.2f}) GeV:  {mean_hist[i]:.4f} ± {std_hist[i]:.4f}")

            tag = label.lower()
            ss  = stream.lower()
            np.save(save_dir / f"enu_histograms_{tag}_{ss}.npy", hists_arr)
            np.save(save_dir / f"enu_mean_{tag}_{ss}.npy",       mean_hist)
            np.save(save_dir / f"enu_std_{tag}_{ss}.npy",        std_hist)

            plot_correlation_matrix(hists_arr, bin_edges, label, save_dir, stream=stream)
            plot_corner(hists_arr, bin_edges, label, save_dir, stream=stream)
            print(f"  Correlation + corner plots saved for {label} [{stream}]", flush=True)

            results[label]    = hists_arr
            combined_n[label] = len(hists)

        if results:
            plot_enu_combined(bin_edges, results, combined_n, save_dir,
                              stream=stream,
                              ci_method=ci_method, ci_levels=ci_levels, smooth=smooth)
            print(f"  Combined E_nu histograms saved for {stream}.", flush=True)

    print(f"\nResults saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
