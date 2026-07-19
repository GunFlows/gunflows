#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: SystematicFlow - Conditional Normalizing Flow
Author: Mathias El Baz
Date: 28/01/2025
Description:
  Conditional NF using a fixed Gaussian dist (CovFlow) and optional
  non-linear context transform (ContextFlow). No loss functions here.
"""

import torch
import torch.nn as nn
from normflows.core import NormalizingFlow
from .cov_flow import CovFlow
from .context_flow import ContextFlow

class SystematicFlow(NormalizingFlow):
    def __init__(self, base, flows, target, context_transform=True, n_context_flows=12,
                 n_hidden_layers=2, hidden_dim=64, freeze_covflow=False, device=None):
        super().__init__(base, flows, target)
        self.context_transform = context_transform
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        d_ctx = len(target.list_dim_conditionnal)
        self.register_buffer("log_norm", torch.tensor(0.0))
        self.register_buffer("log_norm_ready", torch.tensor(False))
        self.CovFlow = CovFlow(target, self.device)
        if freeze_covflow:
            print("Freezing CovFlow parameters")
            self.CovFlow.freeze_params()
        if self.context_transform:
            print(f"Using ContextFlow with {n_context_flows} flows, {n_hidden_layers} hidden layers, and hidden dim {hidden_dim}")
            self.ContextFlow = ContextFlow(n_context_flows, n_hidden_layers, hidden_dim, d_ctx)

    def forward(self, u, context=None):
        z, context, _ = self.CovFlow(u)
        if self.context_transform:
            context, _ = self.ContextFlow(context)
        for f in self.flows:
            z, _ = f(z, context=context)
        return z

    def forward_and_log_det(self, u, context=None):
        z, context, log_det = self.CovFlow(u)
        if self.context_transform:
            context, log_d = self.ContextFlow(context)
            log_det += log_d
        for f in self.flows:
            z, log_d = f(z, context=context)
            log_det += log_d
        return z, log_det

    def inverse(self, x, context):
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x, context=context)
        if self.context_transform:
            context, _ = self.ContextFlow.inverse(context)
        u, _ = self.CovFlow.inverse(x, context)
        return u

    def inverse_and_log_det(self, x, context):
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x, context=context)
            log_det += log_d
        if self.context_transform:
            context, log_d = self.ContextFlow.inverse(context)
            log_det += log_d
        u, log_d = self.CovFlow.inverse(x, context)
        log_det += log_d
        return u, log_det

    def log_prob(self, x, context):
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_d = self.flows[i].inverse(z, context=context)
            log_q += log_d
        if self.context_transform:
            context, log_d = self.ContextFlow.inverse(context)
            log_q += log_d
        u, log_d = self.CovFlow.inverse(z, context)
        log_q += log_d
        log_q += self.q0.log_prob(u)
        return log_q

    def sample(self, num_samples=1):
        u, log_q = self.q0(num_samples)
        z, context, log_det = self.CovFlow(u)
        log_q -= log_det
        if self.context_transform:
            context, log_det = self.ContextFlow(context)
            log_q -= log_det
        for f in self.flows:
            z, log_det = f(z, context=context)
            log_q -= log_det
        out = torch.zeros(z.size(0), z.size(1) + context.size(1), device=z.device, dtype=z.dtype)
        out[:, self.CovFlow.list_dim_conditionnal] = context
        out[:, self.CovFlow.phase_space_dim] = z
        return out, log_q

    def sample_before_after_flow(self, num_samples=1):
        u, log_q = self.q0(num_samples)
        z_base, context, log_det = self.CovFlow(u)
        log_q -= log_det
        if self.context_transform:
            context, log_det = self.ContextFlow(context)
            log_q -= log_det
        for f in self.flows:
            z, log_det = f(z_base, context=context)
            log_q -= log_det
        out_flow = torch.zeros(z.size(0), z.size(1) + context.size(1), device=z.device, dtype=z.dtype)
        out_flow[:, self.CovFlow.list_dim_conditionnal] = context
        out_flow[:, self.CovFlow.phase_space_dim] = z
        out_base = torch.zeros(z_base.size(0), z_base.size(1) + context.size(1), device=z_base.device, dtype=z_base.dtype)
        out_base[:, self.CovFlow.list_dim_conditionnal] = context
        out_base[:, self.CovFlow.phase_space_dim] = z_base
        return out_flow, out_base, log_q
