#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Title: plot_ess_from_json.py
#  Author: Lorenzo Giannessi
#  Description:
#   Re-draw the ESS plots from a saved ess_vs_epoch.json, so the aesthetics can
#   be tweaked without re-running the (expensive) ESS sampling.
#
#   Produces the same 6 plots as effective_sample_size.py:
#     {non-filtered, filtered} x {epoch, time [hours], #LH samplings}
#
#   Usage:
#     python -m gunflows.plot_ess_from_json json_path=/path/to/ess_vs_epoch.json
#     # optionally: out_dir=/somewhere num_samples=5000
#
#   Multiple json files may be given; their epochs are merged into one curve.
#   The FIRST file has priority for duplicated epochs:
#     python -m gunflows.plot_ess_from_json 'json_path=[/p/a.json,/p/b.json]'
#     python -m gunflows.plot_ess_from_json json_path=/p/a.json,/p/b.json
# =============================================================================

from __future__ import annotations
import json
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from ess_plotting import (
    make_ess_plots,
    epoch_to_time_hours,
    epoch_to_n_samplings,
    sum_generated_from_progress,
)


def _json_paths(cfg) -> list[str]:
    """Resolve cfg.json_path into a list of absolute paths.

    Accepts a single path, a Hydra list (json_path=[a,b]) or a comma-separated
    string (json_path=a,b).
    """
    jp = cfg.json_path
    if isinstance(jp, str):
        parts = [p for p in jp.split(",") if str(p).strip()]
    else:  # ListConfig / list
        parts = [str(p) for p in jp]
    return [os.path.abspath(str(p)) for p in parts]


def _ensure_conversions(results: dict, cfg) -> dict:
    """Make sure a per-file results dict has time_hours and n_samplings arrays."""
    epochs = results.get("epochs", [])
    conv = results.get("conversion", {})

    def _pick(name, default):
        v = getattr(cfg, name, None)
        return v if v is not None else conv.get(name, default)

    if cfg.recompute_conversions or "time_hours" not in results:
        time_ref_epochs = float(_pick("time_ref_epochs", 56000))
        time_ref_hours = float(_pick("time_ref_hours", 23.0 + 20.0 / 60.0))
        results["time_hours"] = epoch_to_time_hours(
            epochs, time_ref_epochs, time_ref_hours).tolist()

    if cfg.recompute_conversions or "n_samplings" not in results:
        samplings_base = float(_pick("samplings_base", 2_500_000))
        samplings_ref_epoch = float(_pick("samplings_ref_epoch", 0) or 0)
        sum_generated = conv.get("sum_generated", None)
        prog_glob = getattr(cfg, "samplings_progress_glob", None)
        if sum_generated is None or prog_glob:
            sum_generated = sum_generated_from_progress(prog_glob) if prog_glob else 0.0
        if samplings_ref_epoch <= 0 and epochs:
            samplings_ref_epoch = float(max(epochs))
        results["n_samplings"] = epoch_to_n_samplings(
            epochs, samplings_base, sum_generated, samplings_ref_epoch).tolist()
    return results


def _merge_results(results_list: list[dict]) -> dict:
    """Merge several per-file results by epoch. Earlier files win on duplicates."""
    keys = ["ess", "ess_filtered", "time_hours", "n_samplings"]
    by_epoch: dict[float, dict] = {}
    for res in results_list:  # in priority order (first file first)
        eps = res.get("epochs", [])
        for i, ep in enumerate(eps):
            k = float(ep)
            if k in by_epoch:
                continue  # first file already provided this epoch
            rec = {"epoch": ep}
            for key in keys:
                arr = res.get(key)
                rec[key] = (arr[i] if (arr is not None and i < len(arr)) else None)
            by_epoch[k] = rec
    items = sorted(by_epoch.values(), key=lambda r: float(r["epoch"]))
    merged = {"epochs": [r["epoch"] for r in items]}
    for key in keys:
        merged[key] = [r[key] for r in items]
    # carry over metadata from the first file
    if results_list:
        for meta in ("num_samples", "conversion"):
            if meta in results_list[0]:
                merged[meta] = results_list[0][meta]
    return merged


@hydra.main(config_path="../../configs", config_name="plot_ess", version_base=None)
def main(cfg: DictConfig) -> None:
    json_paths = _json_paths(cfg)
    for p in json_paths:
        if not os.path.isfile(p):
            raise RuntimeError(f"json file not found: {p}")
    if not json_paths:
        raise RuntimeError("no json_path given")

    out_dir = getattr(cfg, "out_dir", None)
    out_dir = os.path.abspath(str(out_dir)) if out_dir else os.path.dirname(json_paths[0])

    results_list = []
    for p in json_paths:
        with open(p) as f:
            results_list.append(_ensure_conversions(json.load(f), cfg))

    if len(results_list) == 1:
        results = results_list[0]
    else:
        results = _merge_results(results_list)
        print(f"Merged {len(json_paths)} json files (first wins on duplicate epochs) "
              f"-> {len(results['epochs'])} epochs: {results['epochs']}", flush=True)

    num_samples = getattr(cfg, "num_samples", None)
    if num_samples is None:
        num_samples = results.get("num_samples", None)

    y_percent = bool(getattr(cfg, "y_percent", True))
    show_title = bool(getattr(cfg, "show_title", False))
    label_fontsize = int(getattr(cfg, "label_fontsize", 16))
    also_loglog = bool(getattr(cfg, "also_loglog", True))
    paper_style = bool(getattr(cfg, "paper_style", True))
    fmt = str(getattr(cfg, "fmt", "png"))
    # usetex: "auto" -> detect latex; true/false -> force. Default false so the
    # output is identical on every machine (and '%' needs no special handling).
    _usetex_cfg = getattr(cfg, "usetex", False)
    if isinstance(_usetex_cfg, str) and _usetex_cfg.lower() == "auto":
        usetex = None
    else:
        usetex = bool(_usetex_cfg)
    also_combined = bool(getattr(cfg, "also_combined", True))
    written = make_ess_plots(
        results, out_dir, num_samples=num_samples, y_percent=y_percent,
        show_title=show_title, label_fontsize=label_fontsize, also_loglog=also_loglog,
        paper_style=paper_style, fmt=fmt, usetex=usetex, also_combined=also_combined,
    )
    print(f"Wrote {len(written)} ESS plots to {out_dir}:", flush=True)
    for p in written:
        print(f"  {Path(p).name}", flush=True)


if __name__ == "__main__":
    main()
