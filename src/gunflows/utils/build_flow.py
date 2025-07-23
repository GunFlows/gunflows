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
) -> SystematicFlow:
    model = SystematicFlow(base, flows, target, context_transform=context_transform)
    return model
