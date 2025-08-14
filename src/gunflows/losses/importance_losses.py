#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp & KL Importance Losses
Author: Mathias El Baz
Date: 28/01/2025
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
# TODO: To be cleaned after we figure out a working training configuration
__all__ = [
    "exp_forward", "exp_reverse", "exp_symmetric",
    "kl_forward", "kl_reverse", "kl_symmetric",
]

def _cap_logw(log_w: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.exp(torch.clamp(log_w, max=float(np.log(cap))))

def _ess(w: torch.Tensor) -> float:
    return (w.sum() ** 2 / w.pow(2).sum()).item()

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
    log_g: torch.Tensor,
    save_dir: str | Path,
    tag: str,
    stage: int,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    x1 = -log_p.squeeze(1).detach().cpu().numpy()
    y1 = -log_nf.squeeze(1).detach().cpu().numpy()
    x2 = -log_p.squeeze(1).detach().cpu().numpy()
    y2 = -log_g.squeeze(1).detach().cpu().numpy()

    x_lower = np.quantile(x1, 0.0)
    x_upper = np.quantile(x1, 0.9999)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # if y is filled with NAn
    if np.isnan(y1).all():
        y1 = x1
    if np.isnan(y2).all():
        y2 = x2

    h1 = axs[0].hist2d(
        x1,
        y1,
        bins=50,
        norm=LogNorm(),
        cmap="viridis",
        # range=[[x_lower, x_upper], [x_lower, x_upper]],
    )
    # plt.colorbar(h1[3], ax=axs[0])
    axs[0].plot([x_lower, x_upper], [x_lower, x_upper], "r--", linewidth=1)
    axs[0].set_title("-log(p) vs -log(NF)")
    axs[0].set_xlabel("-log(p)")
    axs[0].set_ylabel("-log(NF)")

    h2 = axs[1].hist2d(
        x2,
        y2,
        bins=50,
        norm=LogNorm(),
        cmap="viridis",
        # range=[[x_lower, x_upper], [x_lower, x_upper]],
    )
    # plt.colorbar(h2[3], ax=axs[1])
    axs[1].plot([x_lower, x_upper], [x_lower, x_upper], "r--", linewidth=1)
    axs[1].set_title("-log(p) vs -log(g)")
    axs[1].set_xlabel("-log(p)")
    axs[1].set_ylabel("-log(g)")

    fig.tight_layout()
    fig.savefig(save_dir / f"NLLH_comparison_{tag}_{stage}.png")
    plt.close(fig)
    log_w_f=log_p.squeeze(1).detach().cpu().numpy() - log_g.squeeze(1).detach().cpu().numpy()
    if np.isnan(log_w_f).any():
        print(f"Warning: NaN values found in log_w_f at stage {stage}. Replacing with 0.")
    log_w_f[np.isnan(log_w_f)] = 0
    log_w_r=log_p.squeeze(1).detach().cpu().numpy() - log_nf.squeeze(1).detach().cpu().numpy()
    if np.isnan(log_w_r).any():
        print(f"Warning: NaN values found in log_w_r at stage {stage}. Replacing with 0.")
    log_w_r[np.isnan(log_w_r)] = 0
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        log_w_f,
        bins=50,
        density=True,
        alpha=0.5,
        label="log(p) - log(g)",
    )
    ax.hist(
        log_w_r,
        bins=50,
        density=True,
        alpha=0.5,
        label="log(p) - log(NF)",
    )
    ax.set_title("Histogram of log weights")
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
        _diag_plot(log_p, log_q, log_g, save_dir, "exp_for")

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
        _diag_plot(log_p, log_q, log_g, save_dir, "exp_rev")

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
    w_r = _cap_logw(log_w_r, cap_r).detach()

    loss = torch.mean(a * w_f * log_pq**2 + b * w_r * log_pq**2)

    if validation:
        _diag_plot(log_p, log_q, log_g, save_dir, "exp_sym", stage)
        print(f"Forward Exp loss: {torch.mean(w_f * log_pq**2).item()}")
        print(f"Reverse Exp loss: {torch.mean(w_r * log_pq**2).item()}")
        print(f"Mean log_pq: {log_pq.mean().item()}")
        print(f"Mean w_forward: {w_f.mean().item()}")
        print(f"Mean w_reverse: {w_r.mean().item()}")

    if not return_extra:
        return loss
    
    q_pq = torch.quantile(log_pq, 0.999)
    mask_pq = log_pq <= q_pq
    w_f = w_f[mask_pq]
    w_r = w_r[mask_pq]
    return loss, {
        "ess": _ess(torch.exp(log_pq[mask_pq])),
        "mean_w_forward": w_f.mean().item(),
        "mean_w_reverse": w_r.mean().item(),
    }


def kl_forward(
    model,
    dataset,
    idx,
    *,
    cap=np.exp(500),
    return_extra=False,
    validation=False,
    save_dir=".",
):
    _, _, log_g_all, log_g_cond, log_p, log_q = _common(model, dataset, idx)
    log_w = log_p - log_g_all
    w = _cap_logw(log_w, cap)
    diff = log_p - log_q - log_g_cond
    loss = torch.mean(w * diff)

    if validation:
        _diag_plot(log_p, log_q + log_g_cond, log_g_all, save_dir, "kl_fwd")

    if not return_extra:
        return loss
    return loss, {"ess": _ess(torch.exp(log_w)), "mean_w": w.mean().item()}


def kl_reverse(
    model,
    dataset,
    idx,
    *,
    cap=np.exp(500),
    return_extra=False,
    validation=False,
    save_dir=".",
):
    _, _, log_g_all, log_g_cond, log_p, log_q = _common(model, dataset, idx)
    log_w = log_q + log_g_cond - log_g_all
    w = _cap_logw(log_w, cap)
    diff = log_q + log_g_cond - log_p
    loss = torch.mean(w * diff)

    if validation:
        _diag_plot(log_p, log_q + log_g_cond, log_g_all, save_dir, "kl_rev")

    if not return_extra:
        return loss
    return loss, {"ess": _ess(torch.exp(log_w)), "mean_w": w.mean().item()}


def kl_symmetric(
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
    w_r = _cap_logw(log_w_r, cap_r).detach()

    loss = torch.mean(a * w_f * log_pq - b * w_r * log_pq)

    if validation:
        _diag_plot(log_p, log_q, log_g, save_dir, "kl_sym", stage)
        print(f"Forward KL loss: {torch.mean(w_f * log_pq).item()}")
        print(f"Reverse KL loss: {-torch.mean(w_r * log_pq).item()}")
        print(f"Mean log_pq: {log_pq.mean().item()}")
        print(f"Mean w_forward: {w_f.mean().item()}")
        print(f"Mean w_reverse: {w_r.mean().item()}")

    if not return_extra:
        return loss
    
    q_pq = torch.quantile(log_pq, 0.999)
    mask_pq = log_pq <= q_pq
    w_f = w_f[mask_pq]
    w_r = w_r[mask_pq]
    return loss, {
        "ess": _ess(torch.exp(log_pq[mask_pq])),
        "mean_w_forward": w_f.mean().item(),
        "mean_w_reverse": w_r.mean().item(),
    }
