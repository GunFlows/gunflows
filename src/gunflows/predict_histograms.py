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
from omegaconf import DictConfig, OmegaConf


NF_LOCAL = os.path.join(os.path.dirname(__file__), "normalizing-flows")
sys.path.append(os.path.abspath(NF_LOCAL))
from gunflows.utils.build_flow import build_base, build_flow_layers, build_model


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
# Main (hydra entry point)
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="predict_histograms", version_base=None)
def main(cfg: DictConfig) -> None:
    from gunflows.likelihood_sampler import LikelihoodSampler
    from gunflows.sample_mcmc_toy import build_sampling_dataset_target

    # ------------------------------------------------------------------
    # 1. Load checkpoint and training config
    # ------------------------------------------------------------------
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
    # 1. Initialise LikelihoodSampler (GUNDAM interface)
    # ------------------------------------------------------------------
    print("Initializing LikelihoodSampler...", flush=True)
    override_files = list(cfg.experiment.dataset.llh_overrides)
    override_files += list(cfg.get("llh_extra_overrides", []))
    likelihood_sampler = LikelihoodSampler(
        config_file=str(cfg.experiment.dataset.llh_config),
        override_files=override_files,
        data_is_asimov=bool(cfg.experiment.dataset.data_is_asimov),
        threads=int(cfg.experiment.sampler.threads) if hasattr(cfg.experiment, "sampler") else 1,
        llh_cwd=str(cfg.experiment.dataset.llh_cwd),
        light_mode=False,
    )

    bestfit = np.asarray(likelihood_sampler.postfit_parameter_values, dtype=np.float64)
    cov     = np.asarray(likelihood_sampler.postfit_covariance_matrix,  dtype=np.float64)
    print(f"Covariance matrix shape: {cov.shape}", flush=True)

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

    print(f"\nStarting sampling loop: {num_samples} throws each (NF + Gaussian)", flush=True)

    with torch.no_grad():
        while len(nf_histograms) < num_samples or len(gaussian_histograms) < num_samples:

            # --- NF batch ---
            if len(nf_histograms) < num_samples:
                need = min(batch_size, num_samples - len(nf_histograms))
                z_nf, _ = nf_model.sample(need)
                # NF samples are in standardised (eigen) space → convert to physical
                x_nf = dataset.transform_eigen_space_to_data_space(z_nf).cpu().numpy()
                new_hists = _histograms_from_params(
                    likelihood_sampler, x_nf, bin_edges, enu_var,
                    f"NF {len(nf_histograms)+1}–{len(nf_histograms)+len(x_nf)}",
                )
                nf_histograms.extend(new_hists)
                print(f"NF:    {len(nf_histograms)}/{num_samples} valid throws", flush=True)

            # --- Gaussian batch ---
            if len(gaussian_histograms) < num_samples:
                need = min(batch_size, num_samples - len(gaussian_histograms))
                x_g = rng.multivariate_normal(bestfit, cov, size=need)
                new_hists = _histograms_from_params(
                    likelihood_sampler, x_g, bin_edges, enu_var,
                    f"Gauss {len(gaussian_histograms)+1}–{len(gaussian_histograms)+len(x_g)}",
                )
                gaussian_histograms.extend(new_hists)
                print(f"Gauss: {len(gaussian_histograms)}/{num_samples} valid throws", flush=True)

    # Trim to exactly num_samples
    nf_histograms       = nf_histograms[:num_samples]
    gaussian_histograms = gaussian_histograms[:num_samples]

    # ------------------------------------------------------------------
    # 4. Summary: mean ± std per bin; save to disk
    # ------------------------------------------------------------------
    save_dir = Path(cfg.save_dir).expanduser() if "save_dir" in cfg else Path(".")
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "bin_edges.npy", bin_edges)

    for label, histograms in [("NF", nf_histograms), ("Gaussian", gaussian_histograms)]:
        hists_arr = np.array(histograms, dtype=np.float64)   # [N, n_bins]
        mean_hist = hists_arr.mean(axis=0)
        std_hist  = hists_arr.std(axis=0)

        print(f"\n{'='*60}")
        print(f"  {label}  E_nu histogram  (mean ± std, {len(histograms)} throws)")
        print(f"{'='*60}")
        for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            print(f"  [{lo:.2f}, {hi:.2f}) GeV:  {mean_hist[i]:.4f} ± {std_hist[i]:.4f}")

        tag = label.lower()
        np.save(save_dir / f"enu_histograms_{tag}.npy", hists_arr)   # [N, n_bins]
        np.save(save_dir / f"enu_mean_{tag}.npy",       mean_hist)    # [n_bins]
        np.save(save_dir / f"enu_std_{tag}.npy",        std_hist)     # [n_bins]

    print(f"\nResults saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
