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


@hydra.main(config_path="../../configs", config_name="plot_ess", version_base=None)
def main(cfg: DictConfig) -> None:
    json_path = os.path.abspath(str(cfg.json_path))
    if not os.path.isfile(json_path):
        raise RuntimeError(f"json file not found: {json_path}")

    out_dir = getattr(cfg, "out_dir", None)
    out_dir = os.path.abspath(str(out_dir)) if out_dir else os.path.dirname(json_path)

    with open(json_path) as f:
        results = json.load(f)

    num_samples = getattr(cfg, "num_samples", None)
    if num_samples is None:
        num_samples = results.get("num_samples", None)

    # If the json predates the time/samplings storage (or the user wants to
    # recompute with different constants), derive the x-axes from the epochs.
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

    y_percent = bool(getattr(cfg, "y_percent", True))
    show_title = bool(getattr(cfg, "show_title", False))
    label_fontsize = int(getattr(cfg, "label_fontsize", 16))
    also_loglog = bool(getattr(cfg, "also_loglog", True))
    written = make_ess_plots(
        results, out_dir, num_samples=num_samples, y_percent=y_percent,
        show_title=show_title, label_fontsize=label_fontsize, also_loglog=also_loglog,
    )
    print(f"Wrote {len(written)} ESS plots to {out_dir}:", flush=True)
    for p in written:
        print(f"  {Path(p).name}", flush=True)


if __name__ == "__main__":
    main()
