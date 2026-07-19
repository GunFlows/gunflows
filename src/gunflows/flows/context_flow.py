#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: ContextFlow
Author: Mathias El Baz
Date: 28/01/2025
Description:
  RealNVP-style affine coupling stack acting on the context dimensions.
"""

import torch
import torch.nn as nn
import normflows as nf
from gunflows.utils.nets import MLP  

class ContextFlow(nn.Module):
    def __init__(self, num_flows, num_hidden_layers, hidden_dim, context_dim):
        super().__init__()
        flows = []
        for _ in range(num_flows):
            mlp_structure = [context_dim - context_dim // 2] + [hidden_dim] * num_hidden_layers + [2 * (context_dim // 2)]
            param_map = MLP(mlp_structure, init_zeros=True)
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            flows.append(nf.flows.Permute(context_dim, mode='swap'))
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for f in self.flows:
            z, log_det = f(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for f in reversed(self.flows):
            z, log_det = f.inverse(z)
            log_det_tot += log_det
        return z, log_det_tot
