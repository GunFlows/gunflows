#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: CovFlow
Author: Mathias El Baz
Date: 28/01/2025
Description:
  Linear whitening layer with trainable mean/Cholesky (frozen by default).
"""

import torch
import torch.nn as nn

class CovFlow(nn.Module):
    def __init__(self, target, device):
        super().__init__()
        D = target.cholesky.shape[0]
        self.device = device
        lower_init = torch.tril(target.cholesky).clone()
        diag_init = torch.log(torch.diagonal(lower_init).clamp_min(1e-6))
        lower_init = lower_init - torch.diag_embed(torch.diagonal(lower_init)) + torch.diag_embed(diag_init)
        self.param_tril = nn.Parameter(lower_init)
        self.mu = nn.Parameter(torch.zeros(D, device=device))
        self.list_dim_conditionnal = target.list_dim_conditionnal
        print(f"CovFlow initialized with conditional dimensions ranging from {self.list_dim_conditionnal[0]} to {self.list_dim_conditionnal[-1]}")
        self.phase_space_dim = target.phase_space_dim
        print(f"CovFlow initialized with phase space dimensions ranging from {self.phase_space_dim[0]} to {self.phase_space_dim[-1]}")

    def freeze_params(self):
        self.param_tril.requires_grad_(False)
        self.mu.requires_grad_(False)

    def get_cholesky(self):
        lower = torch.tril(self.param_tril)
        diag_exp = torch.diag_embed(torch.diagonal(lower).exp())
        return lower - torch.diag_embed(torch.diagonal(lower)) + diag_exp

    def forward(self, x):
        L = self.get_cholesky()
        x_centered = x - self.mu
        out = x_centered @ L.T
        c = out[:, self.list_dim_conditionnal]
        z = out[:, self.phase_space_dim]
        log_det = torch.logdet(L) * torch.ones(x.size(0), device=x.device)
        return z, c, log_det

    def inverse(self, z, context):
        L = self.get_cholesky()
        D = L.shape[0]
        out = torch.zeros(z.size(0), D, device=z.device, dtype=z.dtype)
        out[:, self.list_dim_conditionnal] = context
        out[:, self.phase_space_dim] = z
        x_centered = torch.linalg.solve(L, out.T).T
        x = x_centered + self.mu
        log_det = -torch.logdet(L) * torch.ones(x.size(0), device=x.device)
        return x, log_det
