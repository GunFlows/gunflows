#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp & KL Importance Losses
Author: Mathias El Baz
Date: 28/01/2025
Description:
  Exponential and KL-like losses (forward, reverse, symmetric), with optional diagnostics.
"""
import torch
import numpy as np

__all__ = [
    "exp_forward", "exp_reverse", "exp_symmetric",
    "kl_forward", "kl_reverse", "kl_symmetric"
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
    return (zb, ctx,
            log_g_all.unsqueeze(1),
            log_g_cond.unsqueeze(1),
            log_p.unsqueeze(1),
            log_q)

def exp_forward(model, dataset, idx, cap=np.exp(500), return_extra=False, **_):
    _, _, log_g_all, log_g_cond, log_p, log_q = _common(model, dataset, idx)
    log_w = log_p - model.log_norm - log_g_all
    w = _cap_logw(log_w, cap)
    diff = (log_p - model.log_norm - log_q - log_g_cond)
    loss = torch.mean(w * diff ** 2)
    if not return_extra:
        return loss
    return loss, {"ess": _ess(torch.exp(log_w)), "mean_w": w.mean().item()}

def exp_reverse(model, dataset, idx, cap=np.exp(500), return_extra=False, **_):
    _, _, log_g_all, log_g_cond, log_p, log_q = _common(model, dataset, idx)
    log_w = log_q + log_g_cond - log_g_all
    w = _cap_logw(log_w, cap)
    diff = (log_q + log_g_cond - log_p)
    loss = torch.mean(w * diff ** 2)
    if not return_extra:
        return loss
    return loss, {"ess": _ess(torch.exp(log_w)), "mean_w": w.mean().item()}

def exp_symmetric(model, dataset, idx, a=1.0, b=1.0, cap=np.exp(500), return_extra=False, **_):
    _, _, log_g_all, log_g_cond, log_p, log_q = _common(model, dataset, idx)
    log_w_f = log_p - model.log_norm - log_g_all
    w_f = _cap_logw(log_w_f, cap)
    diff_f = (log_p - model.log_norm - log_q)

    log_w_r = log_q - log_g_all
    w_r = _cap_logw(log_w_r, cap)
    diff_r = (log_q - log_p + model.log_norm)

    loss = torch.mean(a * w_f * diff_f ** 2 + b * w_r * diff_r ** 2)
    if not return_extra:
        return loss
    return loss, {
        "ess_forward": _ess(torch.exp(log_w_f)),
        "ess_reverse": _ess(torch.exp(log_w_r)),
        "mean_w_forward": w_f.mean().item(),
        "mean_w_reverse": w_r.mean().item(),
    }

def kl_forward(model, dataset, idx, cap=np.exp(500), return_extra=False, **_):
    _, _, log_g_all, log_g_cond, log_p, log_q = _common(model, dataset, idx)
    log_w = log_p - log_g_all
    w = _cap_logw(log_w, cap)
    diff = (log_p - log_q - log_g_cond)
    loss = torch.mean(w * diff)
    if not return_extra:
        return loss
    return loss, {"ess": _ess(torch.exp(log_w)), "mean_w": w.mean().item()}

def kl_reverse(model, dataset, idx, cap=np.exp(500), return_extra=False, **_):
    _, _, log_g_all, log_g_cond, log_p, log_q = _common(model, dataset, idx)
    log_w = log_q + log_g_cond - log_g_all
    w = _cap_logw(log_w, cap)
    diff = (log_q + log_g_cond - log_p)
    loss = torch.mean(w * diff)
    if not return_extra:
        return loss
    return loss, {"ess": _ess(torch.exp(log_w)), "mean_w": w.mean().item()}

def kl_symmetric(model, dataset, idx, a=1.0, b=1.0, cap=np.exp(500), return_extra=False, **_):
    _, _, log_g_all, _, log_p, log_q = _common(model, dataset, idx)
    log_w_f = log_p - log_g_all
    w_f = _cap_logw(log_w_f, cap)
    diff_f = (log_p - log_q)

    log_w_r = log_q - log_g_all
    w_r = _cap_logw(log_w_r, cap)
    diff_r = (log_q - log_p)

    loss = torch.mean(a * w_f * diff_f + b * w_r * diff_r)
    if not return_extra:
        return loss
    return loss, {
        "ess_forward": _ess(torch.exp(log_w_f)),
        "ess_reverse": _ess(torch.exp(log_w_r)),
        "mean_w_forward": w_f.mean().item(),
        "mean_w_reverse": w_r.mean().item(),
    }
