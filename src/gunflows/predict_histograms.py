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

def build_enu_cache(
    sampler,
    bin_edges: np.ndarray,
    enu_var: str = "Enu",
) -> tuple[list, np.ndarray]:
    """
    Called once before the sampling loop.
    Returns:
        events     — flat list of Event objects for all in-range MC events
        bin_indices — int32 array of pre-computed E_nu bin index per event
    Enu values never change between throws, so this lookup is done only once.
    """
    n_bins = len(bin_edges) - 1
    events: list = []
    bin_indices: list[int] = []

    for sp in sampler.likelihood_interface.getSamplePairList():
        model_sample = sp.model
        if not model_sample.isEnabled():
            continue
        for event in model_sample.getEventList():
            enu = event.getVariables().fetchVariable(enu_var).getVarAsDouble()
            idx = int(np.digitize(enu, bin_edges)) - 1
            if 0 <= idx < n_bins:
                events.append(event)
                bin_indices.append(idx)

    print(f"  Enu cache built: {len(events)} in-range MC events", flush=True)
    return events, np.array(bin_indices, dtype=np.int32)


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
    import uproot
    f = uproot.open(mcmc_chain)
    tree = f["FitterEngine/fit/MCMC"]   # uproot picks latest cycle
    n_total = tree.num_entries

    start = int(n_total * burnin_frac)
    stop  = min(n_total, start + max_steps) if max_steps is not None else n_total
    n_available = stop - start

    m = thin if thin is not None else max(1, n_available // n_samples)
    n_out = len(range(0, n_available, m)[:n_samples])
    print(f"  MCMC: {n_total} total steps, using [{start}:{stop}], "
          f"thin={m} → {n_out} throws", flush=True)

    pts_jagged = tree["Points"].array(library="np", entry_start=start, entry_stop=stop)
    pts_2d = np.stack(pts_jagged)       # (n_available, n_params)
    return pts_2d[::m][:n_samples]


# ---------------------------------------------------------------------------
# Per-sample inject + histogram
# ---------------------------------------------------------------------------

def _histograms_from_params(
    likelihood_sampler,
    params_array: np.ndarray,
    enu_events: list,
    enu_bin_indices: np.ndarray,
    n_bins: int,
    label: str,
) -> list[np.ndarray]:
    """
    Inject each row of params_array into GUNDAM, propagate, and collect
    E_nu histogram bin contents. Returns list of accepted histograms.
    enu_events/enu_bin_indices come from build_enu_cache() and are reused
    across throws to avoid re-fetching Enu values.
    """
    histograms = []
    for i, params in enumerate(params_array):
        nll, _, _ = likelihood_sampler.inject_params_and_compute_likelihood(
            params.tolist(), extend_continue=False
        )
        if nll == -1:
            print(f"  [{label} {i}] out of domain — skipped", flush=True)
            continue

        hist = fill_enu_histogram(enu_events, enu_bin_indices, n_bins)
        histograms.append(hist)
        print(f"  [{label} {len(histograms):4d}] NLL={nll:.4f}  hist={hist}", flush=True)

    return histograms


def _checkpoint(
    save_dir: Path,
    bin_edges: np.ndarray,
    nf_histograms: list,
    gaussian_histograms: list,
    mcmc_histograms: list,
    use_nf: bool,
    use_gaussian: bool,
    use_mcmc: bool,
) -> None:
    """Save intermediate npy arrays and regenerate plots."""
    label_hists = []
    if use_nf and nf_histograms:
        label_hists.append(("NF", nf_histograms))
    if use_gaussian and gaussian_histograms:
        label_hists.append(("Gaussian", gaussian_histograms))
    if use_mcmc and mcmc_histograms:
        label_hists.append(("MCMC", mcmc_histograms))

    combined_means: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    combined_n:     dict[str, int] = {}

    for label, histograms in label_hists:
        hists_arr = np.array(histograms, dtype=np.float64)
        mean_hist = hists_arr.mean(axis=0)
        std_hist  = hists_arr.std(axis=0)
        tag = label.lower()
        np.save(save_dir / f"enu_histograms_{tag}.npy", hists_arr)
        np.save(save_dir / f"enu_mean_{tag}.npy",       mean_hist)
        np.save(save_dir / f"enu_std_{tag}.npy",        std_hist)
        combined_means[label] = (mean_hist, std_hist)
        combined_n[label]     = len(histograms)

    if combined_means:
        plot_enu_combined(bin_edges, combined_means, combined_n, save_dir)
    print(f"  [checkpoint] saved {combined_n}", flush=True)


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


def plot_enu_combined(
    bin_edges: np.ndarray,
    results: dict[str, tuple[np.ndarray, np.ndarray]],  # label -> (mean, std)
    n_throws: dict[str, int],
    save_dir: Path,
) -> None:
    """
    Two-pad figure:
      Top    — E_nu spectrum normalized by bin width; Gaussian (red hatched)
               and NF (blue step-fill) with distinct, visible error bars.
      Bottom — relative uncertainty (std/mean) per bin for each distribution.
    """
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths  = bin_edges[1:] - bin_edges[:-1]

    fig, (ax, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.05)

    C_GAUSS = "#d62728"
    C_NF    = "#1f77b4"
    C_MCMC  = "#2ca02c"

    def _step_xy(edges, vals):
        x = np.concatenate([[edges[0]], np.repeat(edges[1:-1], 2), [edges[-1]]])
        y = np.repeat(vals, 2)
        return x, y

    # ---- Gaussian: hatched uncertainty band (mean±std) + step outline ----
    if "Gaussian" in results:
        mean_raw, std_raw = results["Gaussian"]
        mean = mean_raw / widths
        std  = std_raw  / widths
        n    = n_throws.get("Gaussian", "?")

        sx, sy_lo = _step_xy(bin_edges, np.maximum(0.0, mean - std))
        sx, sy_hi = _step_xy(bin_edges, mean + std)
        sx, sy    = _step_xy(bin_edges, mean)

        ax.fill_between(sx, sy_lo, sy_hi,
                        facecolor="none", edgecolor=C_GAUSS,
                        hatch="///", linewidth=0.6,
                        label=f"Gaussian ({n} throws)")
        ax.plot(sx, sy, color=C_GAUSS, linewidth=1.0)

        rel = np.where(mean_raw > 0, std_raw / mean_raw, 0.0)
        bx, by = _step_xy(bin_edges, rel)
        ax_bot.plot(bx, by, color=C_GAUSS, linewidth=1.2, label="Gaussian")

    # ---- NF: step line + cross markers (xerr=half-bin, yerr=std, no caps) ----
    if "NF" in results:
        mean_raw, std_raw = results["NF"]
        mean = mean_raw / widths
        std  = std_raw  / widths
        n    = n_throws.get("NF", "?")

        sx, sy = _step_xy(bin_edges, mean)
        ax.plot(sx, sy, color=C_NF, linewidth=1.5,
                label=f"NF ({n} throws)")
        ax.errorbar(centers, mean,
                    xerr=widths / 2, yerr=std,
                    fmt="+", color=C_NF,
                    elinewidth=1.0, capsize=0,
                    markersize=6, markeredgewidth=1.2)

        rel = np.where(mean_raw > 0, std_raw / mean_raw, 0.0)
        bx, by = _step_xy(bin_edges, rel)
        ax_bot.plot(bx, by, color=C_NF, linewidth=1.2, label="NF")

    # ---- MCMC: hatched band (backslash) + step outline ----
    if "MCMC" in results:
        mean_raw, std_raw = results["MCMC"]
        mean = mean_raw / widths
        std  = std_raw  / widths
        n    = n_throws.get("MCMC", "?")

        sx, sy_lo = _step_xy(bin_edges, np.maximum(0.0, mean - std))
        sx, sy_hi = _step_xy(bin_edges, mean + std)
        sx, sy    = _step_xy(bin_edges, mean)

        ax.fill_between(sx, sy_lo, sy_hi,
                        facecolor="none", edgecolor=C_MCMC,
                        hatch="\\\\", linewidth=0.6,
                        label=f"MCMC ({n} throws)")
        ax.plot(sx, sy, color=C_MCMC, linewidth=1.0)

        rel = np.where(mean_raw > 0, std_raw / mean_raw, 0.0)
        bx, by = _step_xy(bin_edges, rel)
        ax_bot.plot(bx, by, color=C_MCMC, linewidth=1.2, label="MCMC")

    ax.set_ylabel(r"Event yield / bin width  [GeV$^{-1}$]", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11)
    ax.tick_params(axis="both", labelsize=11)

    ax_bot.set_xlabel(r"$E_\nu^{\mathrm{rec}}$ [GeV]", fontsize=13)
    ax_bot.set_ylabel("Rel. unc.\n(std / mean)", fontsize=10)
    ax_bot.set_ylim(bottom=0)
    ax_bot.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_bot.legend(fontsize=9)
    ax_bot.tick_params(axis="both", labelsize=10)
    ax_bot.set_xlim(bin_edges[0], bin_edges[-1])

    fig.savefig(save_dir / "enu_histogram_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_matrix(
    hists_arr: np.ndarray,
    bin_edges: np.ndarray,
    label: str,
    save_dir: Path,
) -> None:
    corr = np.corrcoef(hists_arr.T)   # [n_bins, n_bins]
    labels = _bin_labels(bin_edges)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Bin-content correlation — {label}")
    fig.tight_layout()
    fig.savefig(save_dir / f"correlation_matrix_{label.lower()}.png", dpi=150)
    plt.close(fig)


def plot_corner(
    hists_arr: np.ndarray,
    bin_edges: np.ndarray,
    label: str,
    save_dir: Path,
) -> None:
    n_bins = hists_arr.shape[1]
    labels = _bin_labels(bin_edges)

    fig, axes = plt.subplots(n_bins, n_bins, figsize=(2.2 * n_bins, 2.2 * n_bins))
    fig.suptitle(f"Corner plot — {label}", y=1.01)

    for row in range(n_bins):
        for col in range(n_bins):
            ax = axes[row, col]
            if col > row:
                ax.axis("off")
            elif row == col:
                ax.hist(hists_arr[:, row], bins=12, color="steelblue", alpha=0.7)
                ax.set_xlabel(labels[row], fontsize=6)
            else:
                ax.hist2d(hists_arr[:, col], hists_arr[:, row],
                          bins=20, cmap="Blues")
                ax.set_xlabel(labels[col], fontsize=6)
                ax.set_ylabel(labels[row], fontsize=6)
            ax.tick_params(labelsize=5)

    fig.tight_layout()
    fig.savefig(save_dir / f"corner_{label.lower()}.png", dpi=150)
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
    print("Building Enu cache...", flush=True)
    enu_events, enu_bin_indices = build_enu_cache(likelihood_sampler, bin_edges, enu_var)

    num_samples = int(cfg.num_samples)
    batch_size  = int(cfg.batch_size)
    save_every  = int(cfg.get("save_every", 1000))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

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
    # ------------------------------------------------------------------
    nf_histograms:       list[np.ndarray] = []
    gaussian_histograms: list[np.ndarray] = []
    mcmc_histograms:     list[np.ndarray] = []

    save_dir = Path(cfg.save_dir).expanduser() if "save_dir" in cfg else Path(".")
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "bin_edges.npy", bin_edges)

    active_parts = (["NF"] if use_nf else []) + (["Gaussian"] if use_gaussian else []) + (["MCMC"] if use_mcmc else [])
    print(f"\nStarting sampling loop: {num_samples} throws  [{' + '.join(active_parts)}]", flush=True)

    def _nf_done():    return (not use_nf)      or len(nf_histograms)       >= num_samples
    def _gauss_done(): return (not use_gaussian) or len(gaussian_histograms) >= num_samples
    def _mcmc_done():  return (not use_mcmc)     or len(mcmc_histograms)     >= num_samples

    def _should_checkpoint(old_count: int, new_count: int) -> bool:
        if save_every <= 0:
            return False
        return (new_count // save_every) > (old_count // save_every)

    def _do_checkpoint():
        _checkpoint(save_dir, bin_edges,
                    nf_histograms, gaussian_histograms, mcmc_histograms,
                    use_nf, use_gaussian, use_mcmc)

    with torch.no_grad():
        while not _nf_done() or not _gauss_done() or not _mcmc_done():

            # --- NF batch ---
            if not _nf_done():
                prev = len(nf_histograms)
                need = min(batch_size, num_samples - prev)
                z_nf, _ = nf_model.sample(need)
                x_nf = dataset.transform_eigen_space_to_data_space(z_nf).cpu().numpy()
                new_hists = _histograms_from_params(
                    likelihood_sampler, x_nf,
                    enu_events, enu_bin_indices, n_bins,
                    f"NF {prev+1}–{prev+len(x_nf)}",
                )
                nf_histograms.extend(new_hists)
                print(f"NF:    {len(nf_histograms)}/{num_samples} valid throws", flush=True)
                if _should_checkpoint(prev, len(nf_histograms)):
                    _do_checkpoint()

            # --- Gaussian batch ---
            if not _gauss_done():
                prev = len(gaussian_histograms)
                need = min(batch_size, num_samples - prev)
                x_g = rng.multivariate_normal(bestfit, cov, size=need)
                new_hists = _histograms_from_params(
                    likelihood_sampler, x_g,
                    enu_events, enu_bin_indices, n_bins,
                    f"Gauss {prev+1}–{prev+len(x_g)}",
                )
                gaussian_histograms.extend(new_hists)
                print(f"Gauss: {len(gaussian_histograms)}/{num_samples} valid throws", flush=True)
                if _should_checkpoint(prev, len(gaussian_histograms)):
                    _do_checkpoint()

            # --- MCMC batch ---
            if not _mcmc_done():
                prev = len(mcmc_histograms)
                need = min(batch_size, num_samples - prev)
                x_mc = mcmc_throws[prev:prev + need]
                new_hists = _histograms_from_params(
                    likelihood_sampler, x_mc,
                    enu_events, enu_bin_indices, n_bins,
                    f"MCMC {prev+1}–{prev+len(x_mc)}",
                )
                mcmc_histograms.extend(new_hists)
                print(f"MCMC:  {len(mcmc_histograms)}/{num_samples} valid throws", flush=True)
                if _should_checkpoint(prev, len(mcmc_histograms)):
                    _do_checkpoint()

    # ------------------------------------------------------------------
    # 4. Summary: mean ± std per bin; save to disk
    # ------------------------------------------------------------------
    label_hists = []
    if use_nf:       label_hists.append(("NF",       nf_histograms[:num_samples]))
    if use_gaussian: label_hists.append(("Gaussian", gaussian_histograms[:num_samples]))
    if use_mcmc:     label_hists.append(("MCMC",     mcmc_histograms[:num_samples]))

    combined_means: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    combined_n:     dict[str, int] = {}

    for label, histograms in label_hists:
        hists_arr = np.array(histograms, dtype=np.float64)
        mean_hist = hists_arr.mean(axis=0)
        std_hist  = hists_arr.std(axis=0)

        print(f"\n{'='*60}")
        print(f"  {label}  E_nu histogram  (mean ± std, {len(histograms)} throws)")
        print(f"{'='*60}")
        for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            print(f"  [{lo:.2f}, {hi:.2f}) GeV:  {mean_hist[i]:.4f} ± {std_hist[i]:.4f}")

        tag = label.lower()
        np.save(save_dir / f"enu_histograms_{tag}.npy", hists_arr)
        np.save(save_dir / f"enu_mean_{tag}.npy",       mean_hist)
        np.save(save_dir / f"enu_std_{tag}.npy",        std_hist)

        plot_correlation_matrix(hists_arr, bin_edges, label, save_dir)
        plot_corner(hists_arr, bin_edges, label, save_dir)
        print(f"  Correlation + corner plots saved for {label}", flush=True)

        combined_means[label] = (mean_hist, std_hist)
        combined_n[label]     = len(histograms)

    plot_enu_combined(bin_edges, combined_means, combined_n, save_dir)
    print("  Combined E_nu histogram saved.", flush=True)

    print(f"\nResults saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
