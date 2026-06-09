#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Title: ess_plotting.py
#  Author: Lorenzo Giannessi
#  Description:
#   Shared helpers to turn ESS-vs-epoch results into plots. Used both by
#   effective_sample_size.py (live, while sampling) and by plot_ess_from_json.py
#   (offline re-plot from the saved json, so aesthetics can be tweaked without
#   re-running the expensive sampling).
#
#   Two simple linear (y = A*x + b) epoch -> x-axis conversions are provided:
#     * epoch -> wall-clock time [hours]:  A = time_ref_hours / time_ref_epochs,
#                                          b = 0.
#     * epoch -> number of LH samplings:   b = samplings_base (samples already
#                                          present at epoch 0),
#                                          A = sum_generated / samplings_ref_epoch.
# =============================================================================

from __future__ import annotations
from pathlib import Path
import glob
import json
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Epoch -> x-axis conversions (linear, y = A*epoch + b)
# -----------------------------------------------------------------------------
def epoch_to_time_hours(epochs, time_ref_epochs: float, time_ref_hours: float):
    """Wall-clock training time in hours as a linear function of epoch (b=0)."""
    epochs = np.asarray(epochs, dtype=np.float64)
    if time_ref_epochs is None or float(time_ref_epochs) <= 0:
        return np.full_like(epochs, np.nan)
    a = float(time_ref_hours) / float(time_ref_epochs)
    return a * epochs


def sum_generated_from_progress(progress_glob: str) -> float:
    """Sum the 'generated' field over all progress_*.json files matched."""
    total = 0.0
    for fp in sorted(glob.glob(str(progress_glob))):
        try:
            with open(fp) as f:
                total += float(json.load(f).get("generated", 0) or 0)
        except Exception:
            continue
    return float(total)


def epoch_to_n_samplings(epochs, samplings_base: float, sum_generated: float,
                         samplings_ref_epoch: float):
    """Number of LH samplings as a linear function of epoch.

    b = samplings_base (samples already present at epoch 0);
    A = sum_generated / samplings_ref_epoch (the per-epoch generation rate over
    the training span).
    """
    epochs = np.asarray(epochs, dtype=np.float64)
    if samplings_ref_epoch is None or float(samplings_ref_epoch) <= 0:
        a = 0.0
    else:
        a = float(sum_generated) / float(samplings_ref_epoch)
    return float(samplings_base) + a * epochs


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
# Relative ESS definition, shown as a text box on every plot.
_RESS_FORMULA = r"$rESS = \dfrac{1}{N}\,\dfrac{\left(\sum w\right)^2}{\sum w^2}$"


def _plot_one(x, y, gauss_mask, xlabel, ylabel, title, out_path,
              color="#2563eb", point_label="NF checkpoints",
              gauss_label="Gaussian (epoch 0)", log_y=True, show_formula=True):
    """Single rESS-vs-x plot (log y by default).

    NF checkpoints are drawn as a connected line; the Gaussian (epoch 0) point
    is a triangle, joined to the first NF checkpoint by a segment.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    gauss_mask = np.asarray(gauss_mask, dtype=bool)
    nf_mask = ~gauss_mask

    # sort the NF points by x for a clean connecting line
    order = np.argsort(x[nf_mask]) if nf_mask.any() else np.array([], dtype=int)
    xs = x[nf_mask][order] if nf_mask.any() else np.array([])
    ys = y[nf_mask][order] if nf_mask.any() else np.array([])

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    if nf_mask.any():
        ax.plot(xs, ys, marker="o", markersize=6, linewidth=1.8,
                color=color, label=point_label, zorder=3)

    if gauss_mask.any():
        gx = x[gauss_mask]
        gy = y[gauss_mask]
        # connect the Gaussian point to the first NF checkpoint
        if nf_mask.any():
            ax.plot([gx[0], xs[0]], [gy[0], ys[0]], color=color,
                    linewidth=1.8, zorder=4)
        ax.scatter(gx, gy, marker="^", s=160, color=color,
                   edgecolor="k", linewidth=0.6, zorder=5, label=gauss_label)

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=10, framealpha=0.9, loc="best")

    if show_formula:
        ax.text(0.97, 0.05, _RESS_FORMULA, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.8, edgecolor="gray"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_ess_plots(results: dict, out_dir, num_samples=None) -> list[Path]:
    """Produce up to 6 ESS plots from a results dict.

    results must contain "epochs", "ess", "ess_filtered"; optionally
    "time_hours" and "n_samplings" (same length as "epochs"). For each of the
    two ESS variants (non-filtered / filtered) up to three x-axes are drawn
    (epoch, time, LH samplings) when the corresponding array is available.

    Returns the list of written file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = np.asarray(results.get("epochs", []), dtype=np.float64)
    ess = np.asarray(results.get("ess", []), dtype=np.float64)
    essf = np.asarray(results.get("ess_filtered", []), dtype=np.float64)
    if epochs.size == 0:
        return []

    time_hours = results.get("time_hours", None)
    n_samp = results.get("n_samplings", None)
    time_hours = np.asarray(time_hours, dtype=np.float64) if time_hours is not None else None
    n_samp = np.asarray(n_samp, dtype=np.float64) if n_samp is not None else None

    gauss_mask = epochs == 0

    ns = "" if num_samples is None else f" ({int(num_samples)} samples)"
    ylabel = "rESS (log scale)"

    # (suffix, y, color, point_label, title_prefix)
    ess_variants = [
        ("", ess, "#2563eb", "NF checkpoints", "Relative effective sample size"),
        ("_filtered", essf, "#ea580c", "NF checkpoints (filtered)",
         "Relative effective sample size (filtered)"),
    ]
    # (key, x, xlabel, title_noun) -- only included when x is valid
    x_variants = [("epoch", epochs, "Epoch", "epoch")]
    if time_hours is not None and time_hours.size == epochs.size and np.isfinite(time_hours).any():
        x_variants.append(("time", time_hours, "Training time [hours]", "training time"))
    if n_samp is not None and n_samp.size == epochs.size and np.isfinite(n_samp).any():
        x_variants.append(("samplings", n_samp, "Number of LH samplings", "number of LH samplings"))

    written: list[Path] = []
    for suffix, yvals, color, plabel, tprefix in ess_variants:
        if yvals.size != epochs.size:
            continue
        for xkey, xvals, xlabel, tnoun in x_variants:
            out_path = out_dir / f"ess{suffix}_vs_{xkey}.png"
            _plot_one(
                xvals, yvals, gauss_mask,
                xlabel=xlabel, ylabel=ylabel,
                title=f"{tprefix} vs {tnoun}{ns}",
                out_path=out_path, color=color, point_label=plabel,
            )
            written.append(out_path)
    return written
