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
from matplotlib.ticker import (FuncFormatter, FixedLocator, NullLocator,
                               NullFormatter, LogLocator, MaxNLocator)


# -----------------------------------------------------------------------------
# Paper style (mirrors make_paper_plots.py _apply_style / COLORS / _ax_fontsize)
# -----------------------------------------------------------------------------
# Palette: NF blue, Gaussian red, MCMC green (matplotlib tab colors).
PAPER_COLORS = {"NF": "#1f77b4", "Gaussian": "#d62728", "MCMC": "#2ca02c"}


def _paper_rc(usetex=None) -> dict:
    """rcParams matching make_paper_plots.py (serif/CM, in-ticks, etc.).

    Returns a dict to be used with ``plt.rc_context`` so the global state (and
    the main effective_sample_size.py output) is never mutated.

    ``usetex``: None -> auto-detect a ``latex`` binary; True/False -> force.
    Forcing False (the replot default) makes output identical on every machine
    and avoids LaTeX-only quirks (e.g. '%' starting a comment).
    """
    if usetex is None:
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


def _fmt_millions(v) -> str:
    """Format a count in millions, e.g. 2.5e6 -> '2.5M'."""
    return f"{float(v) / 1e6:.3g}M"


_MILLIONS_FMT = FuncFormatter(lambda v, _: _fmt_millions(v))


def _fmt_hm(v) -> str:
    """Format decimal hours as hours/minutes, e.g. 1.67 -> '1h40m', 0.2 -> '12m'."""
    total_min = int(round(float(v) * 60.0))
    h, m = divmod(total_min, 60)
    if h > 0 and m > 0:
        return f"{h}h{m:02d}m"
    if h > 0:
        return f"{h}h"
    return f"{m}m"


_HM_FMT = FuncFormatter(lambda v, _: _fmt_hm(v))


def _plot_one(x, y, gauss_mask, xlabel, ylabel, title, out_path,
              color="#2563eb", gauss_color=None, point_label="NF checkpoints",
              gauss_label="Gaussian (epoch 0)", log_y=True, show_formula=True,
              y_percent=False, log_x=False, show_title=True, label_fontsize=16,
              paper_style=False, usetex=None, secondary_xaxes=None,
              x_minor_ticks=False, x_major_formatter=None,
              figsize=(8.0, 5.0), square_box=False):
    """Single rESS-vs-x plot (log y by default).

    NF checkpoints are drawn as a connected line; the Gaussian (epoch 0) point
    is a triangle, joined to the first NF checkpoint by a segment.

    If ``y_percent`` is True the y values are shown as percentages (x100).
    If ``log_x`` is True the x axis is logarithmic; points with x<=0 (e.g. the
    Gaussian at epoch/time 0) are dropped since they cannot sit on a log axis.
    If ``paper_style`` is True the figure is rendered with make_paper_plots.py
    rcParams (serif/CM fonts, in-ticks, ...) via a local rc_context.
    ``secondary_xaxes``: optional list of dicts, each
    {"label", "forward", "inverse", "location"}, drawn as extra x-axes on top
    (e.g. show epoch, time and #LH-samplings together).
    ``x_minor_ticks``: show minor tick marks on the x-axis (and secondaries).
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

    rc = _paper_rc(usetex=usetex) if paper_style else {}
    with plt.rc_context(rc=rc):
        fig, ax = plt.subplots(figsize=figsize)

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
        # as a triangle pinned to the LEFT axis (blended transform: x in axes
        # fraction, y in data) at the Gaussian level -- a discrete reference, not
        # a full-width line.
        if log_x:
            dropped = gauss_mask & ~pos
            trans = ax.get_yaxis_transform()       # x = axes-fraction, y = data
            for k, gy_val in enumerate(y[dropped]):
                ax.scatter([0.0], [gy_val], transform=trans, marker="^", s=160,
                           color=gauss_color, edgecolor="k", linewidth=0.6,
                           zorder=6, clip_on=False,
                           label=(gauss_label if k == 0 else None))

        if log_y:
            ax.set_yscale("log")
            if y_percent:
                # plain (non-exponential) tick labels on the log axis, e.g. 40, 60, 100
                plain = FuncFormatter(lambda v, _: f"{v:g}")
                ax.yaxis.set_major_formatter(plain)
                ax.yaxis.set_minor_formatter(plain)
        if log_x:
            ax.set_xscale("log")
            # When the x range spans less than a decade (e.g. #LH-samplings,
            # 2.5-3.3M), matplotlib labels every sub-decade tick and they
            # overlap -> use a handful of evenly log-spaced ticks instead.
            xpos = x[pos]
            if xpos.size and np.isfinite(xpos).all() and xpos.min() > 0 \
                    and (xpos.max() / xpos.min()) < 10.0:
                ticks = np.geomspace(xpos.min(), xpos.max(), 5)
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2g}"))
        # Optional custom primary-x formatter (e.g. millions for #LH-samplings);
        # applied last so it overrides the defaults above.
        if x_major_formatter is not None:
            ax.xaxis.set_major_formatter(x_major_formatter)
        # With usetex, a bare '%' starts a LaTeX comment and eats the rest of the
        # string -> escape it.
        def _esc(s):
            return s.replace("%", r"\%") if plt.rcParams.get("text.usetex") else s
        ax.set_xlabel(_esc(xlabel), fontsize=label_fontsize)
        ax.set_ylabel(_esc(ylabel), fontsize=label_fontsize)
        if show_title:
            ax.set_title(_esc(title), fontsize=13)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.tick_params(axis="both", labelsize=max(10, label_fontsize - 2))
        # Legend pinned to the middle height on the right edge. This also keeps
        # it clear of the rESS formula box (anchored at the lower-right corner).
        ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.65),
                  framealpha=(None if paper_style else 0.9))

        # rESS formula box on the canvas -- commented out per request.
        # if show_formula:
        #     ax.text(0.97, 0.05, _RESS_FORMULA, transform=ax.transAxes,
        #             ha="right", va="bottom", fontsize=label_fontsize - 2,
        #             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
        #                       alpha=0.8, edgecolor="gray"))

        # Minor tick MARKS (no labels) on the primary x-axis, e.g. for log-log.
        if x_minor_ticks and log_x:
            ax.xaxis.set_minor_locator(
                LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1)))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis="x", which="minor", bottom=True, top=False,
                           length=4, width=1.0)

        # Extra x-axes (e.g. epoch / time / #LH-samplings shown together).
        secax_list = []
        # On a log primary, a linear secondary quantity bunches up where the
        # log axis compresses. Drive the secondary tick POSITIONS from
        # log-spaced primary values so they stay evenly spread and aligned.
        _xpos = x[pos]
        _log_ticks = (np.geomspace(_xpos.min(), _xpos.max(), 5)
                      if (log_x and _xpos.size and _xpos.min() > 0) else None)
        for spec in (secondary_xaxes or []):
            secax = ax.secondary_xaxis(spec.get("location", 1.0),
                                       functions=(spec["forward"], spec["inverse"]))
            secax.set_xlabel(_esc(spec["label"]), fontsize=label_fontsize)
            _sfmt = spec.get("formatter")  # callable(v)->str, optional
            if log_x and _log_ticks is not None:
                vals = np.asarray(spec["forward"](_log_ticks), dtype=float)
                secax.set_xticks(vals)
                secax.set_xticklabels([(_sfmt(v) if _sfmt else f"{v:.3g}") for v in vals])
                secax.xaxis.set_minor_locator(NullLocator())
            elif _sfmt is not None:
                secax.xaxis.set_major_formatter(FuncFormatter(lambda v, _, f=_sfmt: f(v)))
            secax_list.append(secax)

        # In paper style, force explicit sizes after layout (paper convention).
        # Tick labels a touch smaller than the default (label-2) so dense log
        # y-axes (e.g. 4..100) don't crowd.
        if paper_style:
            tick_fs = label_fontsize - 4
            _ax_fontsize(ax, label_fontsize, tick_fs=tick_fs)
            # On log axes the crowded labels (e.g. 4,5,6,...) are MINOR ticks,
            # which _ax_fontsize/tick_params(which="major") don't touch.
            ax.tick_params(axis="both", which="minor", labelsize=tick_fs)
            for t in (ax.get_xticklabels(minor=True) + ax.get_yticklabels(minor=True)):
                t.set_fontsize(tick_fs)
            for secax in secax_list:
                secax.xaxis.label.set_size(label_fontsize)
                secax.tick_params(axis="x", labelsize=tick_fs)

        if square_box:
            # Square data box (excluding the stacked top x-axes).
            ax.set_box_aspect(1)

        fig.tight_layout()
        if paper_style:
            fig.savefig(out_path)  # dpi/bbox come from the paper rcParams
        else:
            fig.savefig(out_path, dpi=150)
        plt.close(fig)


def make_ess_plots(results: dict, out_dir, num_samples=None, y_percent=False,
                   show_title=True, label_fontsize=16, also_loglog=False,
                   paper_style=False, fmt="png", usetex=None,
                   also_combined=False) -> list[Path]:
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
    else:
        nf_color, nf_color_f = "#2563eb", "#ea580c"
    # Gaussian drawn in the same color as its NF curve (per-variant).
    gauss_color = None  # None -> _plot_one uses the NF color
    # (suffix, y, nf_color, point_label, title_prefix)
    ess_variants = [
        ("", ess, nf_color, "NF checkpoints", "Relative effective sample size"),
        ("_filtered", essf, nf_color_f, "NF checkpoints (filtered)",
         "Relative effective sample size (filtered)"),
    ]
    # (key, x, xlabel, title_noun, x_formatter) -- only included when x is valid
    x_variants = [("epoch", epochs, "Epoch", "epoch", None)]
    if time_hours is not None and time_hours.size == epochs.size and np.isfinite(time_hours).any():
        x_variants.append(("time", time_hours, "Training time", "training time", _HM_FMT))
    if n_samp is not None and n_samp.size == epochs.size and np.isfinite(n_samp).any():
        x_variants.append(("samplings", n_samp, "Number of LH samplings",
                           "number of LH samplings", _MILLIONS_FMT))

    written: list[Path] = []
    for suffix, yvals, color, plabel, tprefix in ess_variants:
        if yvals.size != epochs.size:
            continue
        for xkey, xvals, xlabel, tnoun, xfmt in x_variants:
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
                    paper_style=paper_style, usetex=usetex,
                    x_major_formatter=xfmt,
                    x_minor_ticks=(xkey == "time" and log_x),
                )
                written.append(out_path)

    # ---- combined plot: epoch + time + #LH-samplings on stacked x-axes -------
    # Requires both time and samplings, which are affine in epoch -> use linear
    # fits to build the secondary-axis transforms.
    have_combo = (
        also_combined
        and time_hours is not None and time_hours.size == epochs.size
        and n_samp is not None and n_samp.size == epochs.size
        and epochs.size >= 2
    )
    if have_combo:
        a_t, b_t = np.polyfit(epochs, time_hours, 1)
        a_s, b_s = np.polyfit(epochs, n_samp, 1)

        def _mk(a, b):
            fwd = (lambda e, a=a, b=b: a * np.asarray(e, dtype=float) + b)
            inv = (lambda v, a=a, b=b: (np.asarray(v, dtype=float) - b) / a
                   if a != 0 else np.zeros_like(np.asarray(v, dtype=float)))
            return fwd, inv

        ft, it = _mk(a_t, b_t)
        fs, isf = _mk(a_s, b_s)
        sec_specs = [
            {"label": "Training time", "forward": ft, "inverse": it,
             "location": 1.0, "formatter": _fmt_hm},
            {"label": "Number of LH samplings", "forward": fs, "inverse": isf,
             "location": 1.17, "formatter": _fmt_millions},
        ]

        for suffix, yvals, color, plabel, tprefix in ess_variants:
            if yvals.size != epochs.size:
                continue
            for msuffix, log_x in ([("", False), ("_loglog", True)] if also_loglog else [("", False)]):
                out_path = out_dir / f"ess{suffix}_combined{msuffix}.{fmt}"
                _plot_one(
                    epochs, yvals, gauss_mask,
                    xlabel="Epoch", ylabel=ylabel,
                    title=f"{tprefix}{ns}",
                    out_path=out_path, color=color, gauss_color=gauss_color,
                    point_label=plabel,
                    y_percent=y_percent, log_x=log_x,
                    show_title=show_title, label_fontsize=label_fontsize,
                    paper_style=paper_style, usetex=usetex,
                    secondary_xaxes=sec_specs, x_minor_ticks=log_x,
                    figsize=(7.0, 8.0), square_box=True,
                )
                written.append(out_path)
    return written
