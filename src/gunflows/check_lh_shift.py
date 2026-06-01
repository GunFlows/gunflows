#!/usr/bin/env python3
"""
Diagnostic for the detector-systematics marginal shift seen in NF-reweighted plots.

Uses the same LikelihoodSampler code path as the reweighting in sample_mcmc_toy.py.

Workflow:
  1. Initialize LikelihoodSampler from the training config (Hydra).
  2. Read best-fit parameter values (self.postfit_parameter_values).
  3. Print parameter list with indices and group flags  -> verify ordering.
  4. Inject best-fit values and print  (stat, penalty, total)  NLL.
  5. Shift one detector param to a target value (e.g., its NF-reweighted peak)
     and re-evaluate.  Repeat for several detector params + a control linear param.

Run (inside the same container bindings as sample_and_compare.sh):

    apptainer exec ... python -s -m gunflows.check_lh_shift \\
        --config-path ${GUNFLOWS}/configs --config-name sample_mcmc_nf_toyOA
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

NF_LOCAL = os.path.join(os.path.dirname(__file__), "..", "normalizing-flows")
sys.path.append(os.path.abspath(NF_LOCAL))

import re
import torch
from gunflows.likelihood_sampler import LikelihoodSampler
from gunflows.utils.build_flow import build_base, build_flow_layers, build_model
from gunflows.sample_mcmc_toy import (
    build_sampling_dataset_target,
    eval_nf_neglogq_on_physical_points,
)


# Shifts to test (param name fragment -> target value). Targets read off NF-reweighted marginal peaks.
DETECTOR_SHIFTS = [
    ("Detector Systematics/#10", 1.07),
    ("Detector Systematics/#10", 0.96),   # symmetric negative shift
    ("Detector Systematics/#20", 1.07),
    ("Detector Systematics/#20", 0.98),
    ("Detector Systematics/#36", 1.07),
    ("Detector Systematics/#36", 0.975),
]
# Control: a linear systematic — marginals showed those agree perfectly (NF == NF-reweighted == Gauss).
LINEAR_SHIFTS = [
    ("Linear Systematics/#30", 1.10),
]


def _short(s: str, n: int = 50) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 3] + "..."


def _find_index(names, fragment):
    matches = [i for i, n in enumerate(names) if fragment in n]
    if len(matches) != 1:
        raise RuntimeError(f"name fragment {fragment!r}: {len(matches)} matches "
                           f"(showing first 5: {matches[:5]})")
    return matches[0]


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    print(f"pwd: {os.getcwd()}", flush=True)
    print(f"training_folder: {cfg.training_folder}", flush=True)

    # Merge the training config (provides cfg.experiment.*), same as sample_mcmc_toy.py
    train_cfg_path = os.path.join(str(cfg.training_folder), ".hydra", "config.yaml")
    if not os.path.isfile(train_cfg_path):
        raise RuntimeError(f"Training config not found: {train_cfg_path}")
    train_cfg = OmegaConf.load(train_cfg_path)
    cfg = OmegaConf.merge(train_cfg, cfg)

    # ------------------------------------------------------------------
    # 1. Construct LikelihoodSampler exactly like sample_mcmc_toy.py does
    # ------------------------------------------------------------------
    print("\n=== Initialising LikelihoodSampler ===", flush=True)
    sampler = LikelihoodSampler(
        config_file=cfg.experiment.dataset.llh_config,
        override_files=cfg.experiment.dataset.llh_overrides,
        data_is_asimov=cfg.experiment.dataset.data_is_asimov,
        threads=int(cfg.experiment.sampler.threads),
        llh_cwd=cfg.experiment.dataset.llh_cwd,
        light_mode=False,
    )

    names = list(sampler.get_parameter_names())
    bestfit = np.asarray(sampler.postfit_parameter_values, dtype=np.float64).reshape(-1)
    prior_vals = np.asarray(sampler.prior_parameter_values, dtype=np.float64).reshape(-1)
    postfit_cov = np.asarray(sampler.postfit_covariance_matrix, dtype=np.float64)
    n = len(names)
    print(f"\nNumber of parameters: {n}", flush=True)

    # ------------------------------------------------------------------
    # 2. Print parameter list (verify ordering)
    # ------------------------------------------------------------------
    print("\n=== Parameter list (idx, group, name, prior, bestfit, postfit sigma) ===", flush=True)
    for i, name in enumerate(names):
        if 0 <= i < 60:
            grp = "[LIN]"
        elif 60 <= i < 100:
            grp = "[DET]"
        else:
            grp = "[SPL]"
        sigma = np.sqrt(max(0.0, float(postfit_cov[i][i])))
        print(f"  {i:3d}  {grp}  {_short(name, 50):50s}  "
              f"prior={prior_vals[i]:.4f}  bestfit={bestfit[i]:+.5f}  sigma_postfit={sigma:.4f}")

    # ------------------------------------------------------------------
    # 3. Evaluate LH at best-fit point
    # ------------------------------------------------------------------
    print("\n=== Evaluate LH at best-fit point ===", flush=True)
    logp_bf, nll_stat_bf, nll_syst_bf = sampler.inject_params_and_compute_likelihood(
        bestfit.copy(), extend_continue=False,
    )
    logp_bf = float(logp_bf); nll_stat_bf = float(nll_stat_bf); nll_syst_bf = float(nll_syst_bf)
    print(f"  returned 'logp' (total NLL?) = {logp_bf:.6f}")
    print(f"  nll_stat                     = {nll_stat_bf:.6f}")
    print(f"  nll_syst (penalty)           = {nll_syst_bf:.6f}")
    print(f"  stat + penalty               = {nll_stat_bf + nll_syst_bf:.6f}")
    print(f"  ⇒ 'logp' == stat+penalty?    "
          f"{'YES' if abs(logp_bf - (nll_stat_bf + nll_syst_bf)) < 1e-3 else 'NO — MISMATCH'}")
    print(f"  (ROOT bestFitStats reference: total≈1348.00  stat≈1340.95  penalty≈7.05)")

    # ------------------------------------------------------------------
    # 4b. Load latest NF model and compute log q_NF at the best-fit
    # ------------------------------------------------------------------
    print("\n=== Loading latest NF checkpoint and evaluating log q_NF ===", flush=True)
    training_folder = str(cfg.training_folder)
    ckpt_folder = os.path.join(training_folder, "checkpoints")
    pat = re.compile(r"sampler_epoch(\d+)\.pt")
    max_file, max_ep = None, -1
    for fname in os.listdir(ckpt_folder):
        m = pat.match(fname)
        if m and int(m.group(1)) > max_ep:
            max_ep, max_file = int(m.group(1)), fname
    if max_file is None:
        raise RuntimeError(f"No NF checkpoint found in {ckpt_folder}")
    ckpt_path = Path(os.path.join(ckpt_folder, max_file))
    print(f"  Using latest NF model: {ckpt_path}")

    dataset = build_sampling_dataset_target(cfg, bestfit, postfit_cov)
    dim_spline = len(dataset.phase_space_dim)
    base = build_base(cfg.experiment.model.total_dim)
    tail_bounds = torch.ones(dim_spline) * cfg.experiment.model.tail_bound
    flows = build_flow_layers(
        cfg.experiment.model.nflows,
        dim_spline,
        cfg.experiment.model.hidden,
        cfg.experiment.model.nlayers,
        cfg.experiment.model.nbins,
        tail_bounds,
        n_context=cfg.experiment.model.total_dim - dim_spline,
    )
    nf_model = build_model(
        base, flows, dataset,
        cfg.experiment.model.context_transform,
        cfg.experiment.model.freeze_covflow,
        n_context_flows=cfg.experiment.model.n_context_flows,
        hidden_dim=cfg.experiment.model.hidden_dim,
        n_hidden_layers=cfg.experiment.model.n_hidden_layers,
    )
    device = str(cfg.get("device", "cuda"))
    nf_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    nf_model = nf_model.to(device).eval()

    def _neglogq(vec):
        return float(eval_nf_neglogq_on_physical_points(
            nf_model, dataset, vec.reshape(1, -1), batch_size=1, device=device
        )[0])

    neglogq_bf = _neglogq(bestfit)
    logq_bf = -neglogq_bf
    rw_bf = -logq_bf - logp_bf
    print(f"  -log q_NF at best-fit       = {neglogq_bf:.4f}   (log q_NF = {logq_bf:.4f})")
    print(f"  reweight value rw = -logq - NLL at best-fit  = {rw_bf:.4f}")


    # ------------------------------------------------------------------
    # 4. Shift one parameter at a time
    # ------------------------------------------------------------------
    def _shift_and_eval(label, idx, target):
        vec = bestfit.copy()
        old = float(vec[idx])
        vec[idx] = float(target)
        logp, ns, np_ = sampler.inject_params_and_compute_likelihood(vec, extend_continue=False)
        logp, ns, np_ = float(logp), float(ns), float(np_)
        d_total = logp - logp_bf
        d_stat = ns - nll_stat_bf
        d_pen = np_ - nll_syst_bf
        print(f"\n--- {label}: idx={idx}  '{_short(names[idx], 50)}' ---")
        print(f"  bestfit value = {old:+.5f}   ->  shifted to {target:+.5f}   (Δ = {target-old:+.5f})")
        print(f"  total NLL   :  {logp_bf:>12.5f}  ->  {logp:>12.5f}   Δ = {d_total:+.5f}")
        print(f"  stat NLL    :  {nll_stat_bf:>12.5f}  ->  {ns:>12.5f}   Δ = {d_stat:+.5f}")
        print(f"  penalty NLL :  {nll_syst_bf:>12.5f}  ->  {np_:>12.5f}   Δ = {d_pen:+.5f}")
        neglogq_shift = _neglogq(vec)
        logq_shift = -neglogq_shift
        d_neglogq = neglogq_shift - neglogq_bf
        rw_shift = -logq_shift - float(logp)
        d_rw = rw_shift - rw_bf
        print(f"  -log q_NF   :  {neglogq_bf:>12.5f}  ->  {neglogq_shift:>12.5f}   Δ = {d_neglogq:+.5f}")
        print(f"  rw = -logq - NLL :  {rw_bf:>9.5f}  ->  {rw_shift:>9.5f}   Δ = {d_rw:+.5f}")
        if d_rw > 0:
            import math as _m
            print(f"    ⇒ shifted point has HIGHER IS weight than best-fit (×{_m.exp(d_rw):.3g})")
            print(f"      → NF under-represents the shifted region relative to the LH → reweighting will lift it.")
        elif d_rw < 0:
            import math as _m
            print(f"    ⇒ shifted point has LOWER IS weight than best-fit (×{_m.exp(d_rw):.3g})")
            print(f"      → NF over-represents the shifted region; reweighting pushes mass away from it.")
        if d_total < 0:
            print(f"   ⇒ shifted point has LOWER total NLL  ⇒ Minuit best-fit IS NOT the joint min.")
        else:
            print(f"   ⇒ shifted point has HIGHER total NLL  ⇒ best-fit is min along this 1D direction.")
            print(f"     If the NF-reweighted marginal nonetheless peaks here, the issue is in")
            print(f"     the reweighting code (weight formula / parameter ordering / convention).")

    for frag, target in DETECTOR_SHIFTS:
        idx = _find_index(names, frag)
        _shift_and_eval("DETECTOR shift", idx, target)

    for frag, target in LINEAR_SHIFTS:
        idx = _find_index(names, frag)
        _shift_and_eval("CONTROL (linear) shift", idx, target)

    # ------------------------------------------------------------------
    # 5. Sanity: re-inject best-fit and verify reproducibility
    # ------------------------------------------------------------------
    print("\n=== Sanity: re-inject best-fit a second time ===", flush=True)
    logp2, ns2, np2 = sampler.inject_params_and_compute_likelihood(bestfit.copy(), extend_continue=False)
    print(f"  total = {float(logp2):.6f}  (first call: {logp_bf:.6f})  Δ = {float(logp2)-logp_bf:+.2e}")
    print(f"  stat  = {float(ns2):.6f}  (first call: {nll_stat_bf:.6f})")
    print(f"  pen   = {float(np2):.6f}  (first call: {nll_syst_bf:.6f})")


if __name__ == "__main__":
    main()
