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
from matplotlib.ticker import FuncFormatter


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
              gauss_label="Gaussian (epoch 0)", log_y=True, show_formula=True,
              y_percent=False, log_x=False, show_title=True, label_fontsize=12):
    """Single rESS-vs-x plot (log y by default).

    NF checkpoints are drawn as a connected line; the Gaussian (epoch 0) point
    is a triangle, joined to the first NF checkpoint by a segment.

    If ``y_percent`` is True the y values are shown as percentages (x100).
    If ``log_x`` is True the x axis is logarithmic; points with x<=0 (e.g. the
    Gaussian at epoch/time 0) are dropped since they cannot sit on a log axis.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y_percent:
        y = y * 100.0
    gauss_mask = np.asarray(gauss_mask, dtype=bool)
    nf_mask = ~gauss_mask

    # On a log x-axis, x<=0 cannot be shown -> drop those points.
    pos = x > 0 if log_x else np.ones_like(x, dtype=bool)
    nf_sel = nf_mask & pos
    gauss_sel = gauss_mask & pos

    # sort the NF points by x for a clean connecting line
    order = np.argsort(x[nf_sel]) if nf_sel.any() else np.array([], dtype=int)
    xs = x[nf_sel][order] if nf_sel.any() else np.array([])
    ys = y[nf_sel][order] if nf_sel.any() else np.array([])

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    if nf_sel.any():
        ax.plot(xs, ys, marker="o", markersize=6, linewidth=1.8,
                color=color, label=point_label, zorder=3)

    if gauss_sel.any():
        gx = x[gauss_sel]
        gy = y[gauss_sel]
        # connect the Gaussian point to the first NF checkpoint
        if nf_sel.any():
            ax.plot([gx[0], xs[0]], [gy[0], ys[0]], color=color,
                    linewidth=1.8, zorder=4)
        ax.scatter(gx, gy, marker="^", s=160, color=color,
                   edgecolor="k", linewidth=0.6, zorder=5, label=gauss_label)

    if log_y:
        ax.set_yscale("log")
        if y_percent:
            # plain (non-exponential) tick labels on the log axis, e.g. 40, 60, 100
            plain = FuncFormatter(lambda v, _: f"{v:g}")
            ax.yaxis.set_major_formatter(plain)
            ax.yaxis.set_minor_formatter(plain)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if show_title:
        ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.tick_params(axis="both", labelsize=max(10, label_fontsize - 2))
    ax.legend(fontsize=11, framealpha=0.9, loc="best")

    if show_formula:
        ax.text(0.97, 0.05, _RESS_FORMULA, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.8, edgecolor="gray"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_ess_plots(results: dict, out_dir, num_samples=None, y_percent=False,
                   show_title=True, label_fontsize=12, also_loglog=False) -> list[Path]:
    """Produce ESS plots from a results dict.

    results must contain "epochs", "ess", "ess_filtered"; optionally
    "time_hours" and "n_samplings" (same length as "epochs"). For each of the
    two ESS variants (non-filtered / filtered) up to three x-axes are drawn
    (epoch, time, LH samplings) when the corresponding array is available.

    If ``also_loglog`` is True, an additional log-x ("_loglog") version of every
    plot is produced (the Gaussian point at x=0 is dropped on those).

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
    ylabel = "rESS [%] (log scale)" if y_percent else "rESS (log scale)"

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
            # linear-x, plus an optional log-x ("_loglog") counterpart
            modes = [("", False)]
            if also_loglog:
                modes.append(("_loglog", True))
            for msuffix, log_x in modes:
                out_path = out_dir / f"ess{suffix}_vs_{xkey}{msuffix}.png"
                _plot_one(
                    xvals, yvals, gauss_mask,
                    xlabel=xlabel, ylabel=ylabel,
                    title=f"{tprefix} vs {tnoun}{ns}",
                    out_path=out_path, color=color, point_label=plabel,
                    y_percent=y_percent, log_x=log_x,
                    show_title=show_title, label_fontsize=label_fontsize,
                )
                written.append(out_path)
    return written
