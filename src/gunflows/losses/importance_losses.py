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

__all__ = [
    "exp_forward", "exp_reverse", "exp_symmetric",
    "kl_forward", "kl_reverse", "kl_symmetric",
]

def _cap_logw(log_w: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.exp(torch.clamp(log_w, max=float(np.log(cap))))

def _ess(w: torch.Tensor) -> float:
    return (w.sum() ** 2 / w.pow(2).sum()).item()

def _common(model, dataset, idx):
    device = next(model.parameters()).device
    zb, ctx, log_g_all, log_g_cond, log_p = dataset.log_prob(idx)

    zb = zb.to(device)
    ctx = ctx.to(device)
    log_g_all = log_g_all.to(device)
    log_g_cond = log_g_cond.to(device)
    log_p = log_p.to(device)

    log_q = model.log_prob(zb, ctx).unsqueeze(1)
    return (
        zb,
        ctx,
        log_g_all.unsqueeze(1),
        log_g_cond.unsqueeze(1),
        log_p.unsqueeze(1),
        log_q,
    )


def _diag_plot(
    log_p: torch.Tensor,
    log_nf: torch.Tensor,
    log_g: torch.Tensor,
    save_dir: str | Path,
    tag: str,
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

    h1 = axs[0].hist2d(
        x1,
        y1,
        bins=50,
        norm=LogNorm(),
        cmap="viridis",
        range=[[x_lower, x_upper], [x_lower, x_upper]],
    )
    plt.colorbar(h1[3], ax=axs[0])
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
        range=[[x_lower, x_upper], [x_lower, x_upper]],
    )
    plt.colorbar(h2[3], ax=axs[1])
    axs[1].plot([x_lower, x_upper], [x_lower, x_upper], "r--", linewidth=1)
    axs[1].set_title("-log(p) vs -log(g)")
    axs[1].set_xlabel("-log(p)")
    axs[1].set_ylabel("-log(g)")

    fig.tight_layout()
    fig.savefig(save_dir / f"NLLH_comparison_{tag}.png")
    plt.close(fig)

    # Plot subplot histogram of weights log(p) - log(g) and log(p) - log(NF)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        -log_p.squeeze(1).detach().cpu().numpy() + log_g.squeeze(1).detach().cpu().numpy(),
        bins=50,
        density=True,
        alpha=0.5,
        label="-log(p) + log(g)",
    )
    ax.hist(
        -log_p.squeeze(1).detach().cpu().numpy() + log_nf.squeeze(1).detach().cpu().numpy(),
        bins=50,
        density=True,
        alpha=0.5,
        label="-log(p) + log(NF)",
    )
    ax.set_title("Histogram of weights")
    ax.set_xlabel("Weight")
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
    _, _, log_g_all, log_g_cond, log_p, log_q = _common(model, dataset, idx)
    log_w = log_p - model.log_norm - log_g_all
    w = _cap_logw(log_w, cap)
    diff = log_p - model.log_norm - log_q - log_g_cond
    loss = torch.mean(w * diff**2)

    if validation:
        _diag_plot(log_p, log_q + log_g_cond, log_g_all, save_dir, "exp_fwd")

    if not return_extra:
        return loss
    return loss, {"ess": _ess(torch.exp(log_w)), "mean_w": w.mean().item()}


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
    _, _, log_g_all, log_g_cond, log_p, log_q = _common(model, dataset, idx)
    log_w = log_q + log_g_cond - log_g_all
    w = _cap_logw(log_w, cap)
    diff = log_q + log_g_cond - log_p
    loss = torch.mean(w * diff**2)

    if validation:
        _diag_plot(log_p, log_q + log_g_cond, log_g_all, save_dir, "exp_rev")

    if not return_extra:
        return loss
    return loss, {"ess": _ess(torch.exp(log_w)), "mean_w": w.mean().item()}


def exp_symmetric(
    model,
    dataset,
    idx,
    *,
    a=1.0,
    b=1.0,
    cap=np.exp(500),
    return_extra=False,
    validation=False,
    save_dir=".",
):
    _, _, log_g_all, log_g_cond, log_p, log_q = _common(model, dataset, idx)

    log_w_f = log_p - model.log_norm - log_g_all
    w_f = _cap_logw(log_w_f, cap)
    diff_f = log_p - model.log_norm - log_q

    log_w_r = log_q - log_g_all
    w_r = _cap_logw(log_w_r, cap).detach()
    diff_r = -diff_f

    loss = torch.mean(a * w_f * diff_f**2 + b * w_r * diff_r**2)

    if validation:
        _diag_plot(log_p, log_q, log_g_all, save_dir, "exp_sym")
        print(f"Forward exp loss: {torch.mean(w_f * diff_f**2).item()}")
        print(f"Reverse exp loss: {torch.mean(w_r * diff_r**2).item()}")

    if not return_extra:
        return loss
    return loss, {
        "ess_forward": _ess(torch.exp(log_w_f)),
        "ess_reverse": _ess(torch.exp(log_w_r)),
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
    *,
    a=1.0,
    b=1.0,
    cap=np.exp(500),
    return_extra=False,
    validation=False,
    save_dir=".",
):
    _, _, log_g_all, _, log_p, log_q = _common(model, dataset, idx)

    log_w_f = log_p - log_g_all
    w_f = _cap_logw(log_w_f, cap)
    diff_f = log_p - log_q

    log_w_r = log_q - log_g_all
    w_r = _cap_logw(log_w_r, cap)
    diff_r = log_q - log_p

    loss = torch.mean(a * w_f * diff_f + b * w_r * diff_r)

    if validation:
        _diag_plot(log_p, log_q, log_g_all, save_dir, "kl_sym")
        print(f"Forward KL loss: {torch.mean(w_f * diff_f).item()}")
        print(f"Reverse KL loss: {torch.mean(w_r * diff_r).item()}")

    if not return_extra:
        return loss
    return loss, {
        "ess_forward": _ess(torch.exp(log_w_f)),
        "ess_reverse": _ess(torch.exp(log_w_r)),
        "mean_w_forward": w_f.mean().item(),
        "mean_w_reverse": w_r.mean().item(),
    }
