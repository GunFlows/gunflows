#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp & KL Importance Losses
Author: Mathias El Baz
Date: 28/01/2025
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

__all__ = [
    "exp_forward", "exp_reverse", "exp_symmetric",
    "kl_symmetric", "absolute_kl_symmetric",
]


def _update_log_norm(model, log_p, log_ref, momentum: float = 0.99):
    # log_ref MUST be the PROPOSAL log-density log_g (model-independent), NOT the
    # model log_q. log_norm estimates the target's log-evidence; calibrating it to
    # the untrained model underflows the importance weights (w_f, w_r -> 0) so the
    # loss and gradient collapse to 0. Using log_g centers w_f ~ O(1).
    with torch.no_grad():
        est = torch.median(log_p - log_ref)
        if not torch.isfinite(est):
            return
        if bool(model.log_norm_ready):
            model.log_norm.mul_(momentum).add_((1.0 - momentum) * est)
        else:
            model.log_norm.copy_(est)
            model.log_norm_ready.fill_(True)

def _cap_logw(log_w: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.exp(torch.clamp(log_w, max=float(np.log(cap))))

def _ess(w: torch.Tensor) -> float:
    return (w.sum() ** 2 / w.pow(2).sum()).item()

def _print_symmetric_diag(name: str, fwd_loss, rev_loss, log_pq, w_f, w_r) -> None:
    print(f"Forward {name} loss: {fwd_loss.item()}")
    print(f"Reverse {name} loss: {rev_loss.item()}")
    print(f"Mean log_pq: {log_pq.mean().item()}")
    print(f"Mean w_forward: {w_f.mean().item()}")
    print(f"Mean w_reverse: {w_r.mean().item()}")

def _extras(log_pq, w_f, w_r) -> dict:
    q_pq = torch.quantile(log_pq, 0.999)
    mask_pq = log_pq <= q_pq
    return {
        "ess": _ess(torch.exp(log_pq[mask_pq])),
        "mean_w_forward": w_f[mask_pq].mean().item(),
        "mean_w_reverse": w_r[mask_pq].mean().item(),
    }

def _common(model, dataset, idx): # Check with L if it is the NLL or the log(p)
    device = next(model.parameters()).device
    zb, ctx, log_g, log_p = dataset.log_prob(idx)

    zb = zb.to(device)
    ctx = ctx.to(device)
    log_g = log_g.to(device)
    log_p = log_p.to(device)

    log_q = model.log_prob(zb, ctx).unsqueeze(1)
    return (
        zb,
        ctx,
        log_g.unsqueeze(1),
        log_p.unsqueeze(1),
        log_q,
    )


def _diag_plot(
    log_p: torch.Tensor,
    log_nf: torch.Tensor,
    save_dir: str | Path,
    tag: str,
    stage: int = 0,
    mcmc_mask: torch.Tensor | None = None,
    log_norm=0.0,
):
    if mcmc_mask is not None:
        keep = ~mcmc_mask.reshape(-1)
        if not bool(keep.any()):
            return
        log_p = log_p[keep]
        log_nf = log_nf[keep]

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    x1 = -log_p.squeeze(1).detach().cpu().numpy()
    y1 = -log_nf.squeeze(1).detach().cpu().numpy()
    if np.isnan(y1).all():
        y1 = x1
    finite = np.isfinite(x1) & np.isfinite(y1)
    if not finite.any():
        print(f"Warning: no finite (-log(p), -log(NF)) pairs at stage {stage}; skipping diagnostic plot.")
        return
    if not finite.all():
        print(f"Warning: {int((~finite).sum())} non-finite (-log(p), -log(NF)) pairs at stage {stage}; excluded from diagnostic plot.")
    x1 = x1[finite]
    y1 = y1[finite]
    med = np.median(x1)
    spread = max(np.quantile(np.abs(x1 - med), 0.99), np.quantile(np.abs(y1 - med), 0.99))
    spread = max(spread, 1e-6)
    x_lower, x_upper = med - spread, med + spread

    fig, ax = plt.subplots(figsize=(8, 6))
    h1 = ax.hist2d(
        x1,
        y1,
        bins=50,
        norm=LogNorm(),
        cmap="viridis",
        range=[[x_lower, x_upper], [x_lower, x_upper]],
    )
    plt.colorbar(h1[3], ax=ax)
    ax.plot([x_lower, x_upper], [x_lower, x_upper], "r--", linewidth=1)
    ax.set_title(f"-log(p) vs -log(NF)  (stage {stage})")
    ax.set_xlabel("-log(p)")
    ax.set_ylabel("-log(NF)")

    fig.tight_layout()
    fig.savefig(save_dir / f"NLLH_comparison_{tag}.png")
    plt.close(fig)

    ln = float(log_norm) if not torch.is_tensor(log_norm) else float(log_norm.detach())
    log_pq = (log_p.squeeze(1) - ln - log_nf.squeeze(1)).detach().cpu().numpy()
    bad = ~np.isfinite(log_pq)
    if bad.any():
        print(f"Warning: {int(bad.sum())} non-finite values found in log_pq at stage {stage}. Replacing with 0.")
    log_pq[bad] = 0
    w_spread = max(np.quantile(np.abs(log_pq), 0.99), 1e-6)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(log_pq, bins=50, density=True, alpha=0.7, range=(-w_spread, w_spread), label="log(p) - log(NF)")
    ax.axvline(0.0, color="r", linestyle="--", linewidth=1)
    ax.set_title(f"Histogram of log weights (stage {stage})")
    ax.set_xlabel("Log Weight")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / f"weights_histogram_{tag}.png")
    plt.close(fig)


def exp_forward(
    model,
    dataset,
    idx,
    *,
    cap=np.exp(500),
    return_extra=False,
    validation=False,
    save_dir=".",
):
    _, _, log_g , log_p, log_q = _common(model, dataset, idx)

    log_pq = log_p - model.log_norm - log_q

    log_w_f = log_p - model.log_norm - log_g
    w_f = _cap_logw(log_w_f, cap).detach()

    loss = torch.mean(w_f * log_pq**2)

    if validation:
        mcmc_mask = dataset.is_mcmc_at(idx) if hasattr(dataset, "is_mcmc_at") else None
        _diag_plot(log_p, log_q, save_dir, "exp_for", mcmc_mask=mcmc_mask)

    return loss 


def exp_reverse(
    model,
    dataset,
    idx,
    *,
    cap=np.exp(500),
    return_extra=False,
    validation=False,
    save_dir=".",
):
    _, _, log_g , log_p, log_q = _common(model, dataset, idx)

    log_pq = log_p - model.log_norm - log_q

    log_w_r = log_q - log_g
    w_r = _cap_logw(log_w_r, cap).detach()

    loss = torch.mean(w_r * log_pq**2)

    if validation:
        mcmc_mask = dataset.is_mcmc_at(idx) if hasattr(dataset, "is_mcmc_at") else None
        _diag_plot(log_p, log_q, save_dir, "exp_rev", mcmc_mask=mcmc_mask)

    return loss 


def exp_symmetric(
    model,
    dataset,
    idx,
    stage,
    *,
    a=1.0,
    b=1.0,
    cap_f=np.exp(10),
    cap_r=np.exp(500),
    return_extra=False,
    validation=False,
    save_dir=".",
):
    _, _, log_g , log_p, log_q = _common(model, dataset, idx)

    log_pq = log_p - model.log_norm - log_q

    log_w_f = log_p - model.log_norm - log_g
    w_f = _cap_logw(log_w_f, cap_f)

    log_w_r = log_q - log_g
    w_r = _cap_logw(log_w_r, cap_r)

    loss = torch.mean(a * w_f * log_pq**2 + b * w_r * log_pq**2)

    if validation:
        mcmc_mask = dataset.is_mcmc_at(idx) if hasattr(dataset, "is_mcmc_at") else None
        _diag_plot(log_p, log_q, save_dir, "exp_sym", stage, mcmc_mask=mcmc_mask)
        _print_symmetric_diag(
            "Exp", torch.mean(w_f * log_pq**2), torch.mean(w_r * log_pq**2), log_pq, w_f, w_r
        )

    if not return_extra:
        return loss
    return loss, _extras(log_pq, w_f, w_r)


def kl_symmetric(
    model,
    dataset,
    idx,
    stage,
    *,
    a=1.0,
    b=1.0,
    cap_f=np.exp(50),
    cap_r=np.exp(500),
    return_extra=False,
    validation=False,
    save_dir=".",
):
    _, _, log_g , log_p, log_q = _common(model, dataset, idx)

    _update_log_norm(model, log_p, log_g)  # calibrate against the PROPOSAL, not the model
    log_pq = log_p - model.log_norm - log_q

    log_w_f = log_p - model.log_norm - log_g
    w_f = _cap_logw(log_w_f, cap_f)

    log_w_r = log_q - log_g
    w_r = _cap_logw(log_w_r, cap_r)

    loss = torch.mean(a * w_f * log_pq - b * w_r * log_pq)

    if validation:
        plot_idx = dataset.latest_idx() if hasattr(dataset, "latest_idx") else idx
        _, _, _, log_p_l, log_q_l = _common(model, dataset, plot_idx)
        mcmc_mask = dataset.is_mcmc_at(plot_idx) if hasattr(dataset, "is_mcmc_at") else None
        _diag_plot(log_p_l, log_q_l, save_dir, "kl_sym", stage, mcmc_mask=mcmc_mask, log_norm=model.log_norm)
        _print_symmetric_diag(
            "KL", torch.mean(w_f * log_pq), -torch.mean(w_r * log_pq), log_pq, w_f, w_r
        )

    if not return_extra:
        return loss
    return loss, _extras(log_pq, w_f, w_r)


def absolute_kl_symmetric(
    model,
    dataset,
    idx,
    stage,
    *,
    a=1.0,
    b=1.0,
    cap_f=np.exp(50),
    cap_r=np.exp(500),
    return_extra=False,
    validation=False,
    save_dir=".",
):
    _, _, log_g , log_p, log_q = _common(model, dataset, idx)

    log_pq = log_p - model.log_norm - log_q

    log_w_f = log_p - model.log_norm - log_g
    w_f = _cap_logw(log_w_f, cap_f)

    log_w_r = log_q - log_g
    w_r = _cap_logw(log_w_r, cap_r)

    loss = torch.mean(torch.abs(a * w_f * log_pq - b * w_r * log_pq))

    if validation:
        mcmc_mask = dataset.is_mcmc_at(idx) if hasattr(dataset, "is_mcmc_at") else None
        _diag_plot(log_p, log_q, save_dir, "abs_kl_sym", stage, mcmc_mask=mcmc_mask)
        _print_symmetric_diag(
            "KL", torch.mean(w_f * log_pq), -torch.mean(w_r * log_pq), log_pq, w_f, w_r
        )

    if not return_extra:
        return loss
    return loss, _extras(log_pq, w_f, w_r)