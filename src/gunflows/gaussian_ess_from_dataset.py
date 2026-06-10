#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Title: gaussian_ess_from_dataset.py
#  Author: Lorenzo Giannessi
#  Description:
#   Compute the relative ESS of the GAUSSIAN surrogate directly from the
#   original training dataset (the batches in the training run's
#   `starting_folder`), without any likelihood evaluation.
#
#   Each batch_*.npz stores (see make_initial_dataset.py / check_initial_dataset.py):
#     - log_p : full target NLL          (= -log p)
#     - log_q : baseline/Gaussian NLL    (= -log q)
#   so the importance weight of the Gaussian proposal is
#     w ∝ p/q = exp(log_q - log_p)
#   (same convention as the Gaussian-ESS block in effective_sample_size.py).
#
#   Writes an ess_vs_epoch.json with a SINGLE entry (epoch 0), in the same
#   schema as effective_sample_size.py / plot_ess_from_json.py, so it can be
#   plotted or merged with the NF ESS jsons:
#     epochs=[0], ess=[..], ess_filtered=[..], time_hours=[0.0],
#     n_samplings=[N], num_samples=N      (N = total throws in the dataset)
#
#   Usage:
#     python -m gunflows.gaussian_ess_from_dataset training_folder=/path/to/run
# =============================================================================

from __future__ import annotations
import glob
import json
import os
import re
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


def _find_starting_folder(train_cfg) -> str:
    """Locate 'starting_folder' in the training config (prefer experiment.dataset)."""
    try:
        sf = train_cfg.experiment.dataset.starting_folder
        if sf is not None:
            return str(sf)
    except Exception:
        pass
    # Fallback: recursive search for the first 'starting_folder' key.
    container = OmegaConf.to_container(train_cfg, resolve=False)

    def _search(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "starting_folder" and v:
                    return v
                found = _search(v)
                if found:
                    return found
        elif isinstance(node, list):
            for v in node:
                found = _search(v)
                if found:
                    return found
        return None

    sf = _search(container)
    if sf is None:
        raise RuntimeError("Could not find 'starting_folder' in the training config.")
    return str(sf)


def _ess(log_w: np.ndarray) -> float:
    """Kish ESS from log-weights (scale-invariant; subtract max for stability)."""
    lw = np.asarray(log_w, dtype=np.float64)
    lw = lw[np.isfinite(lw)]
    if lw.size == 0:
        return 0.0
    w = np.exp(lw - lw.max())
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    return (s1 * s1 / s2) if s2 > 0 else 0.0


@hydra.main(config_path="../../configs", config_name="gaussian_ess", version_base=None)
def main(cfg: DictConfig) -> None:
    training_folder = os.path.abspath(str(cfg.training_folder))
    save_dir = os.path.abspath(str(cfg.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    train_cfg_path = os.path.join(training_folder, ".hydra", "config.yaml")
    if not os.path.isfile(train_cfg_path):
        raise RuntimeError(f"Training config not found: {train_cfg_path}")
    train_cfg = OmegaConf.load(train_cfg_path)

    starting_folder = _find_starting_folder(train_cfg)
    print(f"training_folder : {training_folder}", flush=True)
    print(f"starting_folder : {starting_folder}", flush=True)
    print(f"save_dir        : {save_dir}", flush=True)
    if not os.path.isdir(starting_folder):
        raise RuntimeError(f"starting_folder does not exist: {starting_folder}")

    batches = sorted(
        glob.glob(os.path.join(starting_folder, "batch*.npz")),
        key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)),
    )
    if not batches:
        raise RuntimeError(f"No batch*.npz files found in {starting_folder}")
    max_batches = getattr(cfg, "max_batches", None)
    if max_batches is not None and int(max_batches) > 0:
        batches = batches[: int(max_batches)]
    print(f"Found {len(batches)} batch files.", flush=True)

    log_p_all, log_q_all = [], []
    for bp in batches:
        z = np.load(bp, allow_pickle=True)
        if "log_p" not in z or "log_q" not in z:
            raise RuntimeError(
                f"{bp} is missing 'log_p'/'log_q' (keys: {list(z.keys())}). "
                "Cannot compute the Gaussian ESS from this dataset."
            )
        log_p_all.append(np.asarray(z["log_p"], dtype=np.float64).reshape(-1))
        log_q_all.append(np.asarray(z["log_q"], dtype=np.float64).reshape(-1))

    log_p = np.concatenate(log_p_all)   # full target NLL  (= -log p)
    log_q = np.concatenate(log_q_all)   # baseline NLL     (= -log q)
    n_total = int(log_p.size)
    print(f"Loaded {n_total} throws total.", flush=True)

    # IS log-weight of the Gaussian proposal: log(p/q) = log_q - log_p.
    log_w = log_q - log_p
    finite = np.isfinite(log_w)
    n_finite = int(finite.sum())
    log_w = log_w[finite]

    ess = _ess(log_w)

    # Filtered ESS: drop the extreme 0.1% tails of the log-weights.
    q_lo = float(getattr(cfg, "quantile_lo", 0.001))
    q_hi = float(getattr(cfg, "quantile_hi", 0.999))
    lo, hi = np.quantile(log_w, q_lo), np.quantile(log_w, q_hi)
    fmask = (log_w >= lo) & (log_w <= hi)
    n_filt = int(fmask.sum())
    ess_f = _ess(log_w[fmask])

    ess_frac = ess / n_finite if n_finite > 0 else 0.0
    ess_f_frac = ess_f / n_filt if n_filt > 0 else 0.0
    print(f"Gaussian ESS         : {ess:.1f} / {n_finite}  (rESS = {ess_frac:.4g})", flush=True)
    print(f"Gaussian ESS filtered: {ess_f:.1f} / {n_filt}  (rESS = {ess_f_frac:.4g})", flush=True)

    # Single-entry json, same schema as effective_sample_size.py.
    results = {
        "epochs": [0],
        "ess": [ess_frac],
        "ess_filtered": [ess_f_frac],
        "time_hours": [0.0],
        "n_samplings": [n_total],
        "num_samples": n_total,
        "conversion": {  # carried for schema-compatibility; unused for a single point
            "samplings_base": float(n_total),
            "samplings_ref_epoch": 0.0,
            "sum_generated": 0.0,
            "time_ref_epochs": 0.0,
            "time_ref_hours": 0.0,
        },
        "source": "gaussian_surrogate_from_dataset",
        "starting_folder": starting_folder,
        "n_batches": len(batches),
    }
    out_json = Path(save_dir) / "ess_vs_epoch.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
