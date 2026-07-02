#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title : mcmc
Author: Mathias El Baz
Date  : 2026-07-02

Hydra entry point for parallel MCMC sampling over a pluggable likelihood
backend. The actual engine (proposals, parallel tempering, independent
chains, adaptation, plotting) lives in gunflows.likelihood_sampler.mcmc_engine
and is backend-agnostic; the likelihood implementation is resolved from
cfg.likelihood.sampler_target (see configs/mcmc.yaml).
"""
from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from gunflows.likelihood_sampler.mcmc_engine import run_mcmc


@hydra.main(config_path="../configs", config_name="mcmc", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        base_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
    except Exception:
        base_dir = os.path.abspath(os.getcwd())
    run_mcmc(cfg, base_dir)


if __name__ == "__main__":
    main()
