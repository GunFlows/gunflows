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
# Paper style (mirrors make_paper_plots.py _apply_style / COLORS / _ax_fontsize)
# -----------------------------------------------------------------------------
# Palette: NF blue, Gaussian red, MCMC green (matplotlib tab colors).
PAPER_COLORS = {"NF": "#1f77b4", "Gaussian": "#d62728", "MCMC": "#2ca02c"}


def _paper_rc() -> dict:
    """rcParams matching make_paper_plots.py (serif/CM, in-ticks, etc.).

    Returns a dict to be used with ``plt.rc_context`` so the global state (and
    the main effective_sample_size.py output) is never mutated.
    """
    usetex = False
    try:
        import subprocess
        subprocess.run(["latex", "--version"], capture_output=True, check=True)
        usetex = True
    except Exception:
        pass
    rc = {
        "text.usetex":          usetex,
        "mathtext.fontset":     "cm",
        "font.family":          "serif",
        "font.size":            16,
        "axes.labelsize":       16,
        "xtick.labelsize":      14,
        "ytick.labelsize":      14,
        "legend.fontsize":      14,
        "legend.frameon":       False,
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
        "axes.linewidth":       1.4,
        "axes.labelpad":        10,
        "lines.linewidth":      2.0,
        "figure.dpi":           150,
        "savefig.dpi":          200,
        "savefig.bbox":         "tight",
    }
    if usetex:
        rc["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
    return rc


def _ax_fontsize(ax, label_fs: int, tick_fs: int | None = None,
                 legend_fs: int | None = None) -> None:
    """Force explicit font sizes on a single axes (copied from make_paper_plots.py)."""
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
              color="#2563eb", gauss_color=None, point_label="NF checkpoints",
              gauss_label="Gaussian (epoch 0)", log_y=True, show_formula=True,
              y_percent=False, log_x=False, show_title=True, label_fontsize=12,
              paper_style=False):
    """Single rESS-vs-x plot (log y by default).

    NF checkpoints are drawn as a connected line; the Gaussian (epoch 0) point
    is a triangle, joined to the first NF checkpoint by a segment.

    If ``y_percent`` is True the y values are shown as percentages (x100).
    If ``log_x`` is True the x axis is logarithmic; points with x<=0 (e.g. the
    Gaussian at epoch/time 0) are dropped since they cannot sit on a log axis.
    If ``paper_style`` is True the figure is rendered with make_paper_plots.py
    rcParams (serif/CM fonts, in-ticks, ...) via a local rc_context.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y_percent:
        y = y * 100.0
    gauss_mask = np.asarray(gauss_mask, dtype=bool)
    nf_mask = ~gauss_mask
    if gauss_color is None:
        gauss_color = color

    # On a log x-axis, x<=0 cannot be shown -> drop those points.
    pos = x > 0 if log_x else np.ones_like(x, dtype=bool)
    nf_sel = nf_mask & pos
    gauss_sel = gauss_mask & pos

    # sort the NF points by x for a clean connecting line
    order = np.argsort(x[nf_sel]) if nf_sel.any() else np.array([], dtype=int)
    xs = x[nf_sel][order] if nf_sel.any() else np.array([])
    ys = y[nf_sel][order] if nf_sel.any() else np.array([])

    rc = _paper_rc() if paper_style else {}
    with plt.rc_context(rc=rc):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))

        if nf_sel.any():
            ax.plot(xs, ys, marker="o", markersize=6, linewidth=1.8,
                    color=color, label=point_label, zorder=3)

        if gauss_sel.any():
            gx = x[gauss_sel]
            gy = y[gauss_sel]
            # connect the Gaussian point to the first NF checkpoint
            if nf_sel.any():
                ax.plot([gx[0], xs[0]], [gy[0], ys[0]], color=gauss_color,
                        linewidth=1.8, zorder=4)
            ax.scatter(gx, gy, marker="^", s=160, color=gauss_color,
                       edgecolor="k", linewidth=0.6, zorder=5, label=gauss_label)

        # Gaussian points that cannot sit on a log x-axis (x<=0): show their rESS
        # as a horizontal dashed reference line instead of a (missing) point.
        if log_x:
            dropped = gauss_mask & ~pos
            for k, gy_val in enumerate(y[dropped]):
                ax.axhline(gy_val, color=gauss_color, linestyle="--", linewidth=1.5,
                           zorder=2, label=(gauss_label if k == 0 else None))

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
        ax.legend(loc="best", framealpha=(None if paper_style else 0.9))

        if show_formula:
            ax.text(0.97, 0.05, _RESS_FORMULA, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=label_fontsize - 2,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              alpha=0.8, edgecolor="gray"))

        # In paper style, force explicit sizes after layout (paper convention).
        if paper_style:
            _ax_fontsize(ax, label_fontsize)

        fig.tight_layout()
        if paper_style:
            fig.savefig(out_path)  # dpi/bbox come from the paper rcParams
        else:
            fig.savefig(out_path, dpi=150)
        plt.close(fig)


def make_ess_plots(results: dict, out_dir, num_samples=None, y_percent=False,
                   show_title=True, label_fontsize=12, also_loglog=False,
                   paper_style=False, fmt="png") -> list[Path]:
    """Produce ESS plots from a results dict.

    results must contain "epochs", "ess", "ess_filtered"; optionally
    "time_hours" and "n_samplings" (same length as "epochs"). For each of the
    two ESS variants (non-filtered / filtered) up to three x-axes are drawn
    (epoch, time, LH samplings) when the corresponding array is available.

    If ``also_loglog`` is True, an additional log-x ("_loglog") version of every
    plot is produced (the Gaussian point at x=0 is dropped on those).
    If ``paper_style`` is True, the make_paper_plots.py style + palette is used.
    ``fmt`` is the output image extension (e.g. "png", "pdf").

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

    # Colors: paper palette (NF blue / Gaussian red; filtered uses MCMC green)
    # when paper_style, else the previous scheme.
    if paper_style:
        nf_color, nf_color_f = PAPER_COLORS["NF"], PAPER_COLORS["MCMC"]
        gauss_color = PAPER_COLORS["Gaussian"]
    else:
        nf_color, nf_color_f = "#2563eb", "#ea580c"
        gauss_color = None  # same as NF color
    # (suffix, y, nf_color, point_label, title_prefix)
    ess_variants = [
        ("", ess, nf_color, "NF checkpoints", "Relative effective sample size"),
        ("_filtered", essf, nf_color_f, "NF checkpoints (filtered)",
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
                out_path = out_dir / f"ess{suffix}_vs_{xkey}{msuffix}.{fmt}"
                _plot_one(
                    xvals, yvals, gauss_mask,
                    xlabel=xlabel, ylabel=ylabel,
                    title=f"{tprefix} vs {tnoun}{ns}",
                    out_path=out_path, color=color, gauss_color=gauss_color,
                    point_label=plabel,
                    y_percent=y_percent, log_x=log_x,
                    show_title=show_title, label_fontsize=label_fontsize,
                    paper_style=paper_style,
                )
                written.append(out_path)
    return written
