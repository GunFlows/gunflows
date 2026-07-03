#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Title: train.py
#  Author: Mathias El Baz
#  Date: 28/01/2025
#  Description:
#       Hydra entry point: build dataset/model/optim/scheduler, instantiate
#       experiment trainer and run training.
# =============================================================================
from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from gunflows.utils.build_flow import build_base, build_flow_layers, build_model


if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

if not OmegaConf.has_resolver("len"):
    OmegaConf.register_new_resolver("len", lambda x: len(x))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if "experiment" in cfg:
        cfg = cfg.experiment

    torch.manual_seed(cfg.seed if "seed" in cfg else 42)

    dataset = instantiate(cfg.dataset)
    mode = "files" if (hasattr(cfg, "data") and cfg.data.get("starting_folder", None)) else "sampler"
    print(f"Dataset mode: {mode}.")
    print(f"Dataset loaded with {len(dataset)} samples.")

    base = build_base(cfg.model.total_dim)
    tail_bounds = torch.ones(cfg.model.dim_spline) * cfg.model.tail_bound
    flows = build_flow_layers(
        cfg.model.nflows,
        cfg.model.dim_spline,
        cfg.model.hidden,
        cfg.model.nlayers,
        cfg.model.nbins,
        tail_bounds,
        n_context=cfg.model.total_dim - cfg.model.dim_spline,
    )
    model = build_model(
        base,
        flows,
        dataset,
        cfg.model.context_transform,
        cfg.model.freeze_covflow,
        cfg.model.n_context_flows,
        cfg.model.n_hidden_layers,
        cfg.model.hidden_dim,
        device=cfg.trainer.device,
    ).to(cfg.trainer.device)
    print(f"Model built with {len(flows)} flow layers.")

    optimizer = instantiate(cfg.optim, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer) if "scheduler" in cfg else None

    trainer = instantiate(
        cfg.trainer,
        cfg=cfg,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        _recursive_=False,
    )
    print("Trainer instantiated.")
    trainer.train()


if __name__ == "__main__":
    main()
