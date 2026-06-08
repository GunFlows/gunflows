#!/usr/bin/env python3
# predict_sensitivity.py
#
# Extends predict_histograms.py to compute 2D oscillation sensitivity contours
# in the (sin²θ₂₃, Δm²₃₂) plane.
#
# Workflow:
#  1. Sample N parameter vectors from NF and Gaussian surrogates (same as
#     predict_histograms.py) and propagate through GUNDAM → Eν histograms.
#  2. Build oscillation reweighting factors w(b, θ_osc) for each grid point.
#  3. Compute the spectrum chi-square for every (grid_point, toy) pair.
#  4. Extract sensitivity contours via Feldman-Cousins and/or Wilks.
#  5. Plot 2D contours overlaying the NF and Gaussian surrogates.

from __future__ import annotations
import os, sys
from pathlib import Path

import hydra
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from omegaconf import DictConfig, OmegaConf
from scipy.stats import chi2 as chi2_dist

from gunflows.predict_histograms import (
    _maybe_build_nf,
    build_enu_cache,
    _histograms_from_params,
    pushd,
    STREAMS,
)
from gunflows.oscillation_probability import OscillationProbability


# ---------------------------------------------------------------------------
# Oscillation weight grid
# ---------------------------------------------------------------------------

def _build_osc_weights(
    bin_centers: np.ndarray,
    s23_vals: np.ndarray,
    dm2_vals: np.ndarray,
    osc_nom: OscillationProbability,
    min_ref_prob: float = 0.01,
) -> np.ndarray:
    """
    Precompute oscillation reweighting factors for all grid points.

    Returns w_all of shape (n_grid, n_bins).
    Grid ordering: gi = i_s23 * n_dm2 + i_dm2  (s23 is the outer loop).
    """
    n_s23 = len(s23_vals)
    n_dm2 = len(dm2_vals)
    n_bins = len(bin_centers)
    w_all = np.empty((n_s23 * n_dm2, n_bins), dtype=np.float64)
    for i_s23, s23 in enumerate(s23_vals):
        for i_dm2, dm2 in enumerate(dm2_vals):
            osc_g = OscillationProbability(
                sin2_theta23=float(s23),
                dm2_32=float(dm2),
                L_km=osc_nom.L_km,
            )
            gi = i_s23 * n_dm2 + i_dm2
            w_all[gi] = osc_g.weight_ratio(bin_centers, osc_nom, min_ref_prob)
    return w_all


# ---------------------------------------------------------------------------
# Chi-square helpers
# ---------------------------------------------------------------------------

def _estimate_N_raw(
    N_obs: np.ndarray,
    bin_centers: np.ndarray,
    osc_nom: "OscillationProbability",
    p_floor: float = 0.005,
) -> np.ndarray:
    """
    Estimate raw (unweighted) MC event count per bin.

    N_raw(b) ≈ N_obs(b) / P_surv(E_b, θ_nominal)

    Valid when <w_sys> ≈ 1 at best-fit parameters.
    p_floor prevents N_raw from diverging near the oscillation minimum.
    """
    p_surv = osc_nom.survival_prob(bin_centers)
    return N_obs / np.maximum(p_surv, p_floor)


def _chi2_bb_obs(
    N_obs_fhc: np.ndarray,
    N_obs_rhc: np.ndarray,
    w_all: np.ndarray,
    N_raw_fhc: np.ndarray,
    N_raw_rhc: np.ndarray,
) -> np.ndarray:
    """
    Barlow-Beeston spectrum chi-square for the Asimov observed spectrum.

    Data = prediction base = N_obs (after FD scaling).
    Prediction at grid point: μ = N_obs * w.

        χ²_BB(b, θ) = (N_obs − μ)² / (N_obs + μ²/N_raw)

    Consistent with _chi2_bb_toys: both use N_obs as data and N_obs*w as prediction.
    Returns shape (n_grid,).
    """
    def _term(N_obs, w, N_raw):
        mu  = N_obs[None, :] * w                          # (n_grid, n_bins)
        num = (N_obs[None, :] - mu) ** 2
        den = np.maximum(N_obs[None, :], 1e-9) + mu ** 2 / np.maximum(N_raw[None, :], 1e-9)
        return np.sum(num / den, axis=-1)                  # (n_grid,)

    return _term(N_obs_fhc, w_all, N_raw_fhc) + _term(N_obs_rhc, w_all, N_raw_rhc)


def _chi2_bb_toys(
    hists_fhc: np.ndarray,
    hists_rhc: np.ndarray,
    N_obs_fhc: np.ndarray,
    N_obs_rhc: np.ndarray,
    w_all: np.ndarray,
    N_raw_fhc: np.ndarray,
    N_raw_rhc: np.ndarray,
    chunk_size: int = 500,
) -> np.ndarray:
    """
    Barlow-Beeston spectrum chi-square for each (grid_point, toy) pair.

    Data = N_toy (after FD scaling).  Prediction = N_obs * w (after FD scaling).

        χ²_BB(b, θ, t) = (N_toy − N_obs·w)² / (N_toy + (N_obs·w)²/N_raw)

    Returns shape (n_grid, N_toys). Chunked over toys to bound peak memory.
    """
    n_grid = w_all.shape[0]
    N_toys = hists_fhc.shape[0]
    chi2_out = np.empty((n_grid, N_toys), dtype=np.float64)

    for t_start in range(0, N_toys, chunk_size):
        t_end = min(t_start + chunk_size, N_toys)
        for hists, N_obs, N_raw in [
            (hists_fhc, N_obs_fhc, N_raw_fhc),
            (hists_rhc, N_obs_rhc, N_raw_rhc),
        ]:
            mu   = N_obs[None, None, :] * w_all[:, None, :]          # (n_grid, chunk, n_bins)
            num  = (hists[None, t_start:t_end, :] - mu) ** 2
            den  = (np.maximum(hists[None, t_start:t_end, :], 1e-9)
                    + mu ** 2 / np.maximum(N_raw[None, None, :], 1e-9))
            c    = np.sum(num / den, axis=-1)                         # (n_grid, chunk)
            if hists is hists_fhc:
                chi2_out[:, t_start:t_end] = c
            else:
                chi2_out[:, t_start:t_end] += c

    return chi2_out


# ---------------------------------------------------------------------------
# Sensitivity computation
# ---------------------------------------------------------------------------

def compute_sensitivity(
    hists_fhc: np.ndarray,
    hists_rhc: np.ndarray,
    N_obs_fhc: np.ndarray,
    N_obs_rhc: np.ndarray,
    w_all: np.ndarray,
    N_raw_fhc: np.ndarray,
    N_raw_rhc: np.ndarray,
    ci_levels: tuple[float, ...],
    chunk_size: int = 500,
) -> dict:
    """
    Compute FC and Wilks sensitivity for one surrogate (NF or Gaussian).

    Test statistic (same as standard Δχ² profiled over the grid):
        T(θ, t) = χ²_spec(θ, t) − min_{θ'} χ²_spec(θ', t)

    FC threshold at grid point θ, level α:
        T_crit_FC(θ, α) = percentile_α { T(θ, t) over toys t }

    Wilks threshold: fixed at chi2.ppf(α, df=2).

    Returns a dict with keys:
        T_obs        : (n_grid,) test statistic for observed (Asimov) data
        T_toys       : (n_grid, N_toys) test statistic for each toy
        T_crit_fc    : (n_grid, n_levels) FC critical values
        in_contour_fc    : (n_grid, n_levels) bool — inside FC contour
        in_contour_wilks : (n_grid, n_levels) bool — inside Wilks contour
        wilks_thresholds : (n_levels,)
    """
    print("  Computing BB chi-square for toy experiments...", flush=True)
    chi2_toys = _chi2_bb_toys(
        hists_fhc, hists_rhc, N_obs_fhc, N_obs_rhc, w_all,
        N_raw_fhc, N_raw_rhc, chunk_size,
    )

    chi2_obs_raw = _chi2_bb_obs(N_obs_fhc, N_obs_rhc, w_all, N_raw_fhc, N_raw_rhc)

    # Profile: subtract per-toy minimum over grid
    chi2_toys_min = chi2_toys.min(axis=0)   # (N_toys,)
    T_toys = chi2_toys - chi2_toys_min[None, :]  # (n_grid, N_toys)
    T_obs  = chi2_obs_raw - chi2_obs_raw.min()   # (n_grid,)

    # FC thresholds
    levels_pct = np.asarray(ci_levels, dtype=np.float64) * 100.0
    T_crit_fc = np.percentile(T_toys, levels_pct, axis=1).T  # (n_grid, n_levels)
    in_contour_fc = T_obs[:, None] < T_crit_fc

    # Wilks thresholds (2 DOF)
    wilks_thresholds = chi2_dist.ppf(ci_levels, df=2)
    in_contour_wilks = T_obs[:, None] < wilks_thresholds[None, :]

    print(f"  FC: {in_contour_fc[:, 0].sum()} / {len(T_obs)} grid points inside 1σ", flush=True)
    print(f"  Wilks: {in_contour_wilks[:, 0].sum()} / {len(T_obs)} grid points inside 1σ", flush=True)

    return {
        "T_obs": T_obs,
        "T_toys": T_toys,
        "T_crit_fc": T_crit_fc,
        "in_contour_fc": in_contour_fc,
        "in_contour_wilks": in_contour_wilks,
        "wilks_thresholds": wilks_thresholds,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_SIGMA_LABELS = ["1σ", "2σ", "3σ"]
_NF_COLORS    = ["#08519c", "#2171b5", "#6baed6"]
_GAUSS_COLORS = ["#a50f15", "#cb181d", "#fc8d59"]


def _draw_binary_contours(
    ax: plt.Axes,
    S23: np.ndarray,
    DM2: np.ndarray,
    in_contour_2d: np.ndarray,
    colors: list[str],
    linestyle: str,
    label_prefix: str,
    sigma_labels: list[str],
) -> list[Line2D]:
    """Draw filled-region contour from a boolean (n_s23, n_dm2, n_levels) array."""
    handles: list[Line2D] = []
    for il in range(in_contour_2d.shape[-1]):
        mask = in_contour_2d[:, :, il].astype(float)
        try:
            ax.contour(
                S23, DM2, mask, levels=[0.5],
                colors=[colors[il]], linewidths=2.0, linestyles=[linestyle],
            )
        except Exception:
            pass
        handles.append(
            Line2D([], [], color=colors[il], lw=2, ls=linestyle,
                   label=f"{label_prefix} {sigma_labels[il]}")
        )
    return handles


def plot_sensitivity_contours(
    s23_vals: np.ndarray,
    dm2_vals: np.ndarray,
    sens_nf: dict | None,
    sens_gauss: dict | None,
    ci_levels: tuple[float, ...],
    osc_nom: OscillationProbability,
    save_dir: Path,
) -> None:
    """
    Produce three output figures:
      sensitivity_contours_fc.png       — FC contours (NF vs Gaussian)
      sensitivity_contours_wilks.png    — Wilks contours (same T_obs, fixed thresholds)
      sensitivity_contours_combined.png — side-by-side FC + Wilks
    """
    n_s23 = len(s23_vals)
    n_dm2 = len(dm2_vals)
    n_lvl = len(ci_levels)
    sigma_labels = _SIGMA_LABELS[:n_lvl]

    S23, DM2 = np.meshgrid(s23_vals, dm2_vals, indexing="ij")

    best_fit_kwargs = dict(marker="*", s=300, c="k", zorder=10)

    def _annotate(ax):
        ax.scatter([osc_nom.sin2_theta23], [osc_nom.dm2_32], **best_fit_kwargs)
        ax.set_xlabel(r"$\sin^2\theta_{23}$", fontsize=13)
        ax.set_ylabel(r"$\Delta m^2_{32}\ [\mathrm{eV}^2]$", fontsize=13)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # ── FC plot ──────────────────────────────────────────────────────────────
    fig_fc, ax_fc = plt.subplots(figsize=(7, 6))
    handles_fc: list[Line2D] = []
    if sens_nf is not None:
        mask_nf = sens_nf["in_contour_fc"].reshape(n_s23, n_dm2, n_lvl)
        h = _draw_binary_contours(ax_fc, S23, DM2, mask_nf,
                                   _NF_COLORS, "solid", "NF FC", sigma_labels)
        handles_fc.extend(h)
    if sens_gauss is not None:
        mask_gauss = sens_gauss["in_contour_fc"].reshape(n_s23, n_dm2, n_lvl)
        h = _draw_binary_contours(ax_fc, S23, DM2, mask_gauss,
                                   _GAUSS_COLORS, "dashed", "Gaussian FC", sigma_labels)
        handles_fc.extend(h)
    handles_fc.append(
        Line2D([], [], marker="*", color="k", lw=0, markersize=12, label="T2K best-fit")
    )
    _annotate(ax_fc)
    ax_fc.set_title("Sensitivity — Feldman–Cousins", fontsize=12)
    ax_fc.legend(handles=handles_fc, fontsize=9, loc="upper right")
    fig_fc.tight_layout()
    fig_fc.savefig(save_dir / "sensitivity_contours_fc.png", dpi=150)
    plt.close(fig_fc)
    print("Saved sensitivity_contours_fc.png", flush=True)

    # ── Wilks plot ───────────────────────────────────────────────────────────
    # T_obs is the same for NF and Gaussian (same Asimov data); use whichever is available
    sens_ref = sens_nf if sens_nf is not None else sens_gauss
    if sens_ref is not None:
        T_obs_2d = sens_ref["T_obs"].reshape(n_s23, n_dm2)
        wilks_thr = sens_ref["wilks_thresholds"]

        fig_wk, ax_wk = plt.subplots(figsize=(7, 6))
        handles_wk: list[Line2D] = []
        for il, (thr, color, slbl) in enumerate(zip(wilks_thr, _NF_COLORS, sigma_labels)):
            try:
                cs = ax_wk.contour(S23, DM2, T_obs_2d, levels=[float(thr)],
                                   colors=[color], linewidths=2.0)
                cs.collections[0].set_label(f"Wilks {slbl}  (Δχ²={thr:.2f})")
            except Exception:
                pass
            handles_wk.append(
                Line2D([], [], color=color, lw=2, label=f"Wilks {slbl}  (Δχ²={thr:.2f})")
            )
        handles_wk.append(
            Line2D([], [], marker="*", color="k", lw=0, markersize=12, label="T2K best-fit")
        )
        _annotate(ax_wk)
        ax_wk.set_title("Sensitivity — Wilks (2 DOF)", fontsize=12)
        ax_wk.legend(handles=handles_wk, fontsize=9)
        fig_wk.tight_layout()
        fig_wk.savefig(save_dir / "sensitivity_contours_wilks.png", dpi=150)
        plt.close(fig_wk)
        print("Saved sensitivity_contours_wilks.png", flush=True)

    # ── Combined side-by-side ─────────────────────────────────────────────────
    fig_c, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: FC
    handles_l: list[Line2D] = []
    if sens_nf is not None:
        mask_nf = sens_nf["in_contour_fc"].reshape(n_s23, n_dm2, n_lvl)
        h = _draw_binary_contours(ax_l, S23, DM2, mask_nf,
                                   _NF_COLORS, "solid", "NF", sigma_labels)
        handles_l.extend(h)
    if sens_gauss is not None:
        mask_gauss = sens_gauss["in_contour_fc"].reshape(n_s23, n_dm2, n_lvl)
        h = _draw_binary_contours(ax_l, S23, DM2, mask_gauss,
                                   _GAUSS_COLORS, "dashed", "Gaussian", sigma_labels)
        handles_l.extend(h)
    handles_l.append(Line2D([], [], marker="*", color="k", lw=0, markersize=12, label="T2K best-fit"))
    _annotate(ax_l)
    ax_l.set_title("Feldman–Cousins", fontsize=12)
    ax_l.legend(handles=handles_l, fontsize=8, loc="upper right")

    # Right: Wilks (independent of surrogate choice)
    if sens_ref is not None:
        handles_r: list[Line2D] = []
        for il, (thr, color_nf, color_ga, slbl) in enumerate(
            zip(wilks_thr, _NF_COLORS, _GAUSS_COLORS, sigma_labels)
        ):
            try:
                ax_r.contour(S23, DM2, T_obs_2d, levels=[float(thr)],
                             colors=[color_nf], linewidths=2.0)
            except Exception:
                pass
            handles_r.append(Line2D([], [], color=color_nf, lw=2,
                                    label=f"Wilks {slbl} (Δχ²={thr:.2f})"))
        handles_r.append(Line2D([], [], marker="*", color="k", lw=0, markersize=12, label="T2K best-fit"))
        _annotate(ax_r)
        ax_r.set_title("Wilks / Gaussian approx. (2 DOF)", fontsize=12)
        ax_r.legend(handles=handles_r, fontsize=8)

    fig_c.suptitle(
        r"T2K sensitivity: NF vs Gaussian surrogate  "
        r"($\nu_\mu$ disappearance, 2-flavor)",
        fontsize=12,
    )
    fig_c.tight_layout()
    fig_c.savefig(save_dir / "sensitivity_contours_combined.png", dpi=150)
    plt.close(fig_c)
    print("Saved sensitivity_contours_combined.png", flush=True)


# ---------------------------------------------------------------------------
# Checkpoint helper (histograms only; no E_nu combined plot)
# ---------------------------------------------------------------------------

def _checkpoint_sens(
    save_dir: Path,
    nf_per_stream: dict[str, list],
    gaussian_per_stream: dict[str, list],
    use_nf: bool,
    use_gaussian: bool,
) -> None:
    for label, per_stream, use in (
        ("nf", nf_per_stream, use_nf),
        ("gauss", gaussian_per_stream, use_gaussian),
    ):
        if not use:
            continue
        for stream in STREAMS:
            hists = per_stream.get(stream, [])
            if not hists:
                continue
            arr = np.array(hists, dtype=np.float64)
            ss = stream.lower()
            np.save(save_dir / f"sensitivity_histograms_{label}_{ss}.npy", arr)
    print(f"  [checkpoint] saved histograms ({len(nf_per_stream.get('FHC', []))} NF / "
          f"{len(gaussian_per_stream.get('FHC', []))} Gauss throws)", flush=True)


# ---------------------------------------------------------------------------
# Main (Hydra entry point)
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="predict_sensitivity", version_base=None)
def main(cfg: DictConfig) -> None:
    from gunflows.likelihood_sampler import LikelihoodSampler
    from gunflows.sample_mcmc_toy import build_sampling_dataset_target

    use_nf       = bool(cfg.get("use_nf", True))
    use_gaussian = bool(cfg.get("use_gaussian", True))
    if not use_nf and not use_gaussian:
        raise ValueError("At least one of use_nf or use_gaussian must be True.")

    # ------------------------------------------------------------------
    # 1a. Load checkpoint + training config (needed when use_nf=True)
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
        llh_config     = str(cfg.llh_config)
        llh_overrides  = list(cfg.get("llh_overrides", []))
        data_is_asimov = bool(cfg.get("data_is_asimov", True))
        llh_cwd        = str(cfg.get("llh_cwd", ".")) if cfg.get("llh_cwd") else None
        threads        = int(cfg.get("threads", 1))
    else:
        llh_config     = str(cfg.experiment.dataset.llh_config)
        llh_overrides  = list(cfg.experiment.dataset.llh_overrides)
        data_is_asimov = bool(cfg.experiment.dataset.data_is_asimov)
        llh_cwd        = str(cfg.experiment.dataset.llh_cwd)
        threads        = int(cfg.experiment.sampler.threads) if hasattr(cfg.experiment, "sampler") else 1

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
    cov     = np.asarray(likelihood_sampler.postfit_covariance_matrix, dtype=np.float64)
    print(f"Post-fit covariance shape: {cov.shape}", flush=True)

    # ------------------------------------------------------------------
    # 1c. Build NF model
    # ------------------------------------------------------------------
    nf_model = None
    dataset  = None
    if use_nf:
        dataset  = build_sampling_dataset_target(cfg, bestfit, cov)
        nf_model = _maybe_build_nf(cfg, dataset, best_ckpt)
        print("NF model loaded.", flush=True)

    # ------------------------------------------------------------------
    # E_nu binning & cache
    # ------------------------------------------------------------------
    from gunflows.predict_histograms import _DEFAULT_BIN_EDGES
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
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    print(f"E_nu: {n_bins} bins, edges: {bin_edges}", flush=True)

    print("Building Enu cache...", flush=True)
    enu_cache = build_enu_cache(likelihood_sampler, bin_edges, enu_var)

    # ------------------------------------------------------------------
    # Sampling config
    # ------------------------------------------------------------------
    num_samples = int(cfg.num_samples)
    batch_size  = int(cfg.get("batch_size", 512))
    save_every  = int(cfg.get("save_every", 500))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    save_dir = Path(cfg.save_dir).expanduser() if "save_dir" in cfg else Path(".")
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "bin_edges.npy", bin_edges)

    # ------------------------------------------------------------------
    # 2. Sampling loop
    # ------------------------------------------------------------------
    nf_per_stream:       dict[str, list[np.ndarray]] = {s: [] for s in STREAMS}
    gaussian_per_stream: dict[str, list[np.ndarray]] = {s: [] for s in STREAMS}

    def _count(per_stream: dict) -> int:
        return max((len(v) for v in per_stream.values()), default=0)

    def _nf_done():    return (not use_nf)      or _count(nf_per_stream)       >= num_samples
    def _gauss_done(): return (not use_gaussian) or _count(gaussian_per_stream) >= num_samples

    def _should_checkpoint(old_n: int, new_n: int) -> bool:
        return save_every > 0 and (new_n // save_every) > (old_n // save_every)

    def _extend(per_stream: dict, new: dict) -> None:
        for s in STREAMS:
            per_stream[s].extend(new.get(s, []))

    print(f"\nSampling loop: {num_samples} throws  "
          f"[{'NF' if use_nf else ''}{'+ Gaussian' if use_gaussian else ''}]", flush=True)

    with torch.no_grad():
        while not _nf_done() or not _gauss_done():

            # --- NF batch ---
            if not _nf_done():
                prev = _count(nf_per_stream)
                need = min(batch_size, num_samples - prev)
                z_nf, _ = nf_model.sample(need)
                x_nf = dataset.transform_eigen_space_to_data_space(z_nf).cpu().numpy()
                new_hists = _histograms_from_params(
                    likelihood_sampler, x_nf, enu_cache, n_bins,
                    f"NF {prev + 1}–{prev + len(x_nf)}",
                )
                _extend(nf_per_stream, new_hists)
                now = _count(nf_per_stream)
                print(f"NF: {now}/{num_samples} valid throws", flush=True)
                if _should_checkpoint(prev, now):
                    _checkpoint_sens(save_dir, nf_per_stream, gaussian_per_stream,
                                     use_nf, use_gaussian)

            # --- Gaussian batch ---
            if not _gauss_done():
                prev = _count(gaussian_per_stream)
                need = min(batch_size, num_samples - prev)
                x_g = rng.multivariate_normal(bestfit, cov, size=need)
                new_hists = _histograms_from_params(
                    likelihood_sampler, x_g, enu_cache, n_bins,
                    f"Gauss {prev + 1}–{prev + len(x_g)}",
                )
                _extend(gaussian_per_stream, new_hists)
                now = _count(gaussian_per_stream)
                print(f"Gauss: {now}/{num_samples} valid throws", flush=True)
                if _should_checkpoint(prev, now):
                    _checkpoint_sens(save_dir, nf_per_stream, gaussian_per_stream,
                                     use_nf, use_gaussian)

    # Final save of histograms
    _checkpoint_sens(save_dir, nf_per_stream, gaussian_per_stream, use_nf, use_gaussian)

    # ------------------------------------------------------------------
    # 3. Oscillation sensitivity analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("Oscillation sensitivity analysis", flush=True)
    print("=" * 60, flush=True)

    osc_cfg = cfg.get("osc_nominal", {})
    osc_nom = OscillationProbability(
        sin2_theta23=float(osc_cfg.get("sin2_theta23", 0.512)),
        dm2_32=float(osc_cfg.get("dm2_32", 2.45e-3)),
        L_km=float(osc_cfg.get("L_km", 295.0)),
        sin2_2theta13=float(osc_cfg.get("sin2_2theta13", 0.0851)),
    )
    print(f"Nominal oscillation: {osc_nom}", flush=True)

    grid_cfg = cfg.get("grid", {})
    s23_range = list(grid_cfg.get("sin2_theta23_range", [0.3, 0.7]))
    s23_n     = int(grid_cfg.get("sin2_theta23_n", 51))
    dm2_range = list(grid_cfg.get("dm2_32_range", [1.5e-3, 3.5e-3]))
    dm2_n     = int(grid_cfg.get("dm2_32_n", 51))

    s23_vals = np.linspace(s23_range[0], s23_range[1], s23_n)
    dm2_vals = np.linspace(dm2_range[0], dm2_range[1], dm2_n)
    n_grid = s23_n * dm2_n
    print(f"Grid: {s23_n} × {dm2_n} = {n_grid} points", flush=True)
    print(f"  sin²θ₂₃ ∈ [{s23_range[0]}, {s23_range[1]}]", flush=True)
    print(f"  Δm²₃₂   ∈ [{dm2_range[0]:.2e}, {dm2_range[1]:.2e}] eV²", flush=True)

    min_ref_prob  = float(cfg.get("min_ref_prob", 0.01))
    chunk_size    = int(cfg.get("sensitivity_chunk_size", 500))
    ci_levels_raw = cfg.get("ci_levels", [0.6827, 0.9545, 0.9973])
    ci_levels     = tuple(float(x) for x in ci_levels_raw)

    print("Building oscillation weight grid...", flush=True)
    w_all = _build_osc_weights(bin_centers, s23_vals, dm2_vals, osc_nom, min_ref_prob)
    np.save(save_dir / "sensitivity_w_all.npy", w_all)
    np.save(save_dir / "sensitivity_s23_vals.npy", s23_vals)
    np.save(save_dir / "sensitivity_dm2_vals.npy", dm2_vals)
    print(f"Weight grid shape: {w_all.shape}", flush=True)

    # ── FD/ND geometric scale factor ──────────────────────────────────────
    fd_cfg   = cfg.get("fd_scale", {})
    L_ND_m   = float(fd_cfg.get("L_ND_m",  280.0))
    L_FD_m   = float(fd_cfg.get("L_FD_m",  295_000.0))
    M_FD_t   = float(fd_cfg.get("M_FD_t",  22_500.0))
    M_ND_t   = float(fd_cfg.get("M_ND_t",  8.0))
    p_floor  = float(fd_cfg.get("p_floor", 0.005))
    f_scale  = (L_ND_m / L_FD_m) ** 2 * M_FD_t / M_ND_t
    print(f"\nFD/ND scale factor: f = {f_scale:.5f}  ({100*f_scale:.3f}%)", flush=True)

    # Convert to arrays, apply FD scaling so chi-squares use realistic statistics
    def _scaled(per_stream, stream):
        return np.array(per_stream[stream][:num_samples], dtype=np.float64) * f_scale

    hists_nf_fhc = _scaled(nf_per_stream, "FHC") if use_nf else None
    hists_nf_rhc = _scaled(nf_per_stream, "RHC") if use_nf else None
    hists_g_fhc  = _scaled(gaussian_per_stream, "FHC") if use_gaussian else None
    hists_g_rhc  = _scaled(gaussian_per_stream, "RHC") if use_gaussian else None

    # Observed (Asimov) FD spectrum = scaled mean of NF (or Gaussian) toys
    ref_fhc   = hists_nf_fhc if use_nf else hists_g_fhc
    ref_rhc   = hists_nf_rhc if use_nf else hists_g_rhc
    N_obs_fhc = ref_fhc.mean(axis=0)
    N_obs_rhc = ref_rhc.mean(axis=0)

    # Estimate raw MC counts for Barlow-Beeston correction
    N_raw_fhc = _estimate_N_raw(N_obs_fhc, bin_centers, osc_nom, p_floor)
    N_raw_rhc = _estimate_N_raw(N_obs_rhc, bin_centers, osc_nom, p_floor)

    np.save(save_dir / "sensitivity_N_obs_fhc.npy", N_obs_fhc)
    np.save(save_dir / "sensitivity_N_obs_rhc.npy", N_obs_rhc)
    print(f"FD Asimov spectrum: FHC total={N_obs_fhc.sum():.1f}, RHC total={N_obs_rhc.sum():.1f}", flush=True)

    sens_nf    = None
    sens_gauss = None

    if use_nf:
        print("\nComputing NF sensitivity (BB, FD-scaled)...", flush=True)
        sens_nf = compute_sensitivity(
            hists_nf_fhc, hists_nf_rhc, N_obs_fhc, N_obs_rhc,
            w_all, N_raw_fhc, N_raw_rhc, ci_levels, chunk_size,
        )
        np.save(save_dir / "sensitivity_T_obs.npy",               sens_nf["T_obs"])
        np.save(save_dir / "sensitivity_T_crit_fc_nf.npy",        sens_nf["T_crit_fc"])
        np.save(save_dir / "sensitivity_in_contour_fc_nf.npy",    sens_nf["in_contour_fc"])
        np.save(save_dir / "sensitivity_in_contour_wilks_nf.npy", sens_nf["in_contour_wilks"])

    if use_gaussian:
        print("\nComputing Gaussian sensitivity (BB, FD-scaled)...", flush=True)
        sens_gauss = compute_sensitivity(
            hists_g_fhc, hists_g_rhc, N_obs_fhc, N_obs_rhc,
            w_all, N_raw_fhc, N_raw_rhc, ci_levels, chunk_size,
        )
        np.save(save_dir / "sensitivity_T_crit_fc_gauss.npy",        sens_gauss["T_crit_fc"])
        np.save(save_dir / "sensitivity_in_contour_fc_gauss.npy",    sens_gauss["in_contour_fc"])
        np.save(save_dir / "sensitivity_in_contour_wilks_gauss.npy", sens_gauss["in_contour_wilks"])

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    print("\nPlotting sensitivity contours...", flush=True)
    plot_sensitivity_contours(
        s23_vals, dm2_vals, sens_nf, sens_gauss,
        ci_levels, osc_nom, save_dir,
    )

    print(f"\nResults saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
