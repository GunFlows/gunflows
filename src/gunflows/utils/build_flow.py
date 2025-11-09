#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Flow Builders
Author: Mathias El Baz
Date: 28/01/2025
Description:
  Helpers to construct base distribution, flow blocks and the SystematicFlow.
"""
from typing import Sequence
import torch
import normflows as nf

from gunflows.flows.systematic_flow import SystematicFlow


def build_base(dim: int) -> nf.distributions.DiagGaussian:
    return nf.distributions.DiagGaussian(dim, trainable=False)


def build_flow_layers(
    nflows: int,
    dim_spline: int,
    hidden: int,
    nlayers: int,
    nbins: int,
    tail_bounds: torch.Tensor,
    n_context: int,
) -> Sequence[nf.flows.Flow]:
    return [
        nf.flows.AutoregressiveRationalQuadraticSpline(
            dim_spline,
            nlayers,
            hidden,
            num_context_channels=n_context,
            num_bins=nbins,
            tail_bound=tail_bounds,
            permute_mask=True,
        )
        for _ in range(nflows)
    ]


def build_model(
    base,
    flows,
    target,
    context_transform: bool = True,
    freeze_covflow: bool = False,
    n_context_flows: int = 12,
    n_hidden_layers: int = 2,
    hidden_dim: int = 64,
    device=None,
) -> SystematicFlow:
    # Pass an explicit device to SystematicFlow so callers can control where the
    # model is instantiated (important to avoid allocating temporarily on GPU).
    model = SystematicFlow(
        base,
        flows,
        target,
        context_transform=context_transform,
        freeze_covflow=freeze_covflow,
        n_context_flows=n_context_flows,
        n_hidden_layers=n_hidden_layers,
        hidden_dim=hidden_dim,
        device=device,
    )
    return model
