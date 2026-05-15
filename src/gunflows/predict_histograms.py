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
# E_nu histogram filler
# ---------------------------------------------------------------------------

def fill_enu_histogram(
    sampler,
    bin_edges: np.ndarray,
    enu_var: str = "Enu",
) -> np.ndarray:
    """
    After propagateAndEvalLikelihood() has been called, iterate over all enabled
    MC model events and accumulate event weights into E_nu bins.

    Returns ndarray of shape (n_bins,).

    Requires GUNDAM Python bindings for:
        Sample.getEventList(), Event.getEventWeight(),
        Event.getVariables(), VariableCollection.fetchVariable(),
        VariableHolder.getVarAsDouble()
    """
    n_bins = len(bin_edges) - 1
    counts = np.zeros(n_bins, dtype=np.float64)

    for sp in sampler.likelihood_interface.getSamplePairList():
        model_sample = sp.model
        if not model_sample.isEnabled():
            continue
        for event in model_sample.getEventList():
            w = event.getEventWeight()
            enu = event.getVariables().fetchVariable(enu_var).getVarAsDouble()
            idx = int(np.digitize(enu, bin_edges)) - 1
            if 0 <= idx < n_bins:
                counts[idx] += w

    return counts


# ---------------------------------------------------------------------------
# Per-sample inject + histogram
# ---------------------------------------------------------------------------

def _histograms_from_params(
    likelihood_sampler,
    params_array: np.ndarray,
    bin_edges: np.ndarray,
    enu_var: str,
    label: str,
) -> list[np.ndarray]:
    """
    Inject each row of params_array into GUNDAM, propagate, and collect
    E_nu histogram bin contents. Returns list of accepted histograms.
    """
    histograms = []
    for i, params in enumerate(params_array):
        nll, _, _ = likelihood_sampler.inject_params_and_compute_likelihood(
            params.tolist(), extend_continue=False
        )
        if nll == -1:
            print(f"  [{label} {i}] out of domain — skipped", flush=True)
            continue

        try:
            hist = fill_enu_histogram(likelihood_sampler, bin_edges, enu_var)
        except AttributeError as exc:
            print(
                f"\nERROR: event-level GUNDAM access failed ({exc}).\n"
                "Ensure Sample.getEventList(), Event.getEventWeight(), "
                "Event.getVariables(), VariableCollection.fetchVariable(), and "
                "VariableHolder.getVarAsDouble() are compiled into the GUNDAM .so.",
                flush=True,
            )
            sys.exit(1)

        histograms.append(hist)
        print(f"  [{label} {len(histograms):4d}] NLL={nll:.4f}  hist={hist}", flush=True)

    return histograms


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _bin_labels(bin_edges: np.ndarray) -> list[str]:
    return [f"[{lo:.2f},{hi:.2f})" for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]


def plot_enu_histogram(
    bin_edges: np.ndarray,
    mean_hist: np.ndarray,
    std_hist: np.ndarray,
    label: str,
    save_dir: Path,
) -> None:
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths  = bin_edges[1:] - bin_edges[:-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(centers, mean_hist, width=widths * 0.85,
           yerr=std_hist, capsize=4, color="steelblue", alpha=0.7,
           error_kw=dict(elinewidth=1.2, ecolor="navy"))
    ax.set_xlabel(r"$E_\nu$ [GeV]")
    ax.set_ylabel("Event yield")
    ax.set_title(f"E_nu histogram — {label}  (mean ± std, {len(mean_hist)} bins)")
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    fig.tight_layout()
    fig.savefig(save_dir / f"enu_histogram_{label.lower()}.png", dpi=150)
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
                ax.scatter(hists_arr[:, col], hists_arr[:, row],
                           s=10, alpha=0.6, color="steelblue")
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

    use_nf      = bool(cfg.get("use_nf", True))
    use_gaussian = bool(cfg.get("use_gaussian", True))
    if not use_nf and not use_gaussian:
        raise ValueError("At least one of use_nf or use_gaussian must be True.")

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
    # E_nu binning (configurable, defaults to 8 bins 0–5 GeV)
    # ------------------------------------------------------------------
    n_bins    = int(cfg.get("n_bins",   8))
    enu_min   = float(cfg.get("enu_min", 0.0))
    enu_max   = float(cfg.get("enu_max", 5.0))
    enu_var   = str(cfg.get("enu_var",  "Enu"))
    bin_edges = np.linspace(enu_min, enu_max, n_bins + 1)
    print(f"E_nu binning: {n_bins} bins  [{enu_min}, {enu_max}] GeV", flush=True)

    num_samples = int(cfg.num_samples)
    batch_size  = int(cfg.batch_size)
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    # ------------------------------------------------------------------
    # 2+3. Sampling loop — draw batches, propagate, collect histograms
    # ------------------------------------------------------------------
    nf_histograms:       list[np.ndarray] = []
    gaussian_histograms: list[np.ndarray] = []

    active = ("NF + Gaussian" if use_nf and use_gaussian
              else "NF only" if use_nf else "Gaussian only")
    print(f"\nStarting sampling loop: {num_samples} throws  [{active}]", flush=True)

    def _nf_done():      return (not use_nf)      or len(nf_histograms)       >= num_samples
    def _gauss_done():   return (not use_gaussian) or len(gaussian_histograms) >= num_samples

    with torch.no_grad():
        while not _nf_done() or not _gauss_done():

            # --- NF batch ---
            if not _nf_done():
                need = min(batch_size, num_samples - len(nf_histograms))
                z_nf, _ = nf_model.sample(need)
                x_nf = dataset.transform_eigen_space_to_data_space(z_nf).cpu().numpy()
                new_hists = _histograms_from_params(
                    likelihood_sampler, x_nf, bin_edges, enu_var,
                    f"NF {len(nf_histograms)+1}–{len(nf_histograms)+len(x_nf)}",
                )
                nf_histograms.extend(new_hists)
                print(f"NF:    {len(nf_histograms)}/{num_samples} valid throws", flush=True)

            # --- Gaussian batch ---
            if not _gauss_done():
                need = min(batch_size, num_samples - len(gaussian_histograms))
                x_g = rng.multivariate_normal(bestfit, cov, size=need)
                new_hists = _histograms_from_params(
                    likelihood_sampler, x_g, bin_edges, enu_var,
                    f"Gauss {len(gaussian_histograms)+1}–{len(gaussian_histograms)+len(x_g)}",
                )
                gaussian_histograms.extend(new_hists)
                print(f"Gauss: {len(gaussian_histograms)}/{num_samples} valid throws", flush=True)

    # ------------------------------------------------------------------
    # 4. Summary: mean ± std per bin; save to disk
    # ------------------------------------------------------------------
    save_dir = Path(cfg.save_dir).expanduser() if "save_dir" in cfg else Path(".")
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "bin_edges.npy", bin_edges)

    results = []
    if use_nf:       results.append(("NF",       nf_histograms[:num_samples]))
    if use_gaussian: results.append(("Gaussian", gaussian_histograms[:num_samples]))

    for label, histograms in results:
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

        plot_enu_histogram(bin_edges, mean_hist, std_hist, label, save_dir)
        plot_correlation_matrix(hists_arr, bin_edges, label, save_dir)
        plot_corner(hists_arr, bin_edges, label, save_dir)
        print(f"  Plots saved for {label}", flush=True)

    print(f"\nResults saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
