#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: OA2022 Trainer
Author: Mathias El Baz
Date: 28/01/2025
Description:
  Training loop for OA2022 experiment. Loss, epochs, early stop, etc. are pulled
  from cfg.trainer.* (experiment config).
"""
from __future__ import annotations
import time
import numpy as np
import torch
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt

from gunflows.trainer.base_trainer import BaseTrainer
import gunflows.losses.importance_losses as IL

LOSS_MAP = {
    "exp_forward":   IL.exp_forward,
    "exp_reverse":   IL.exp_reverse,
    "exp_symmetric": IL.exp_symmetric,
    "kl_forward":    IL.kl_forward,
    "kl_reverse":    IL.kl_reverse,
    "kl_symmetric":  IL.kl_symmetric,
}


class OA2022Trainer(BaseTrainer):
    def __init__(self, cfg: DictConfig, model, dataset, optimizer, scheduler, **kwargs):
        super().__init__(cfg)
        tcfg = cfg.trainer
        self.device = torch.device(tcfg.device)
        self.model = model.to(self.device)
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler if scheduler is not None else None

        self.epochs = tcfg.epochs
        self.batch_size = tcfg.batch_size
        self.val_every = tcfg.val_every

        es = tcfg.early_stop
        self.patience = es.patience
        self.min_delta = es.min_delta
        self.min_epoch = es.min_epoch

        self.loss_train = LOSS_MAP[tcfg.loss.name_train]
        self.loss_val   = LOSS_MAP[tcfg.loss.name_val]
        self.loss_kwargs = dict(tcfg.loss.kwargs)
        self.loss_kwargs.setdefault("return_extra", False)

        seed_split = tcfg.get("seed", cfg.get("seed", 42))
        rng = np.random.default_rng(seed_split)
        self.val_idx = rng.choice(len(dataset), size=tcfg.num_val, replace=False)
        self.train_idx = np.setdiff1d(np.arange(len(dataset)), self.val_idx)

        self.wait = 0
        self.best_ratio = -float("inf")
        self.train_losses = []
        self.val_losses = []
        self.ratio_losses = []
        self.ess_vals = []
        self.ess_ratios = []
        self.mathias_scores = []
        self.epochs_val = []

        run_dir = HydraConfig.get().runtime.output_dir
        ckpt_root = cfg.get("paths", {}).get("checkpoints_dir", f"{run_dir}/checkpoints")
        self.ckpt_dir = Path(ckpt_root)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.plot_path = Path(run_dir) / "training_curves.png"

    def train(self) -> None:
        start = time.time()
        for epoch in range(self.epochs):
            self._train_batch()
            if epoch % self.val_every == 0:
                self._validate_epoch(epoch)
            if self.wait >= self.patience and epoch > self.min_epoch:
                print(f"[Early Stop] epoch={epoch}")
                break
        print(f"Finished in {time.time() - start:.1f}s")

    def _train_batch(self) -> None:
        idx = np.random.choice(self.train_idx, self.batch_size, replace=False)
        loss = self.loss_train(self.model, self.dataset, idx, **self.loss_kwargs)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.train_losses.append(loss.item())

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> None:
        self.epochs_val.append(epoch)
        val_kwargs = dict(self.loss_kwargs)
        val_kwargs["return_extra"] = True
        out = self.loss_val(self.model, self.dataset, self.val_idx, **val_kwargs)

        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            loss_val, extras = out
        else:
            loss_val, extras = out, {}

        ratio = extras.get("loss_ratio", loss_val.item())
        ess_f = extras.get("ess_forward", None)
        ess_r = extras.get("ess_reverse", None)
        ess_ratio = extras.get("ess_ratio", 1.0)
        mscore = extras.get("mathias_scores", None)

        baseline = self.ratio_losses[0] if self.ratio_losses else ratio
        norm_ratio = ratio / baseline if baseline != 0 else 0.0

        self.val_losses.append(loss_val.item())
        self.ratio_losses.append(norm_ratio)
        self.ess_vals.append((ess_f, ess_r))
        self.ess_ratios.append(ess_ratio)
        self.mathias_scores.append(mscore)

        improved = norm_ratio > self.best_ratio + self.min_delta
        if improved:
            self.best_ratio = norm_ratio
            self.wait = 0
            self._checkpoint(best=True)
        else:
            self.wait += 1
        self._checkpoint(best=False)

        print(f"[Val] epoch={epoch:05d} val_loss={loss_val.item():.3e} ratio={norm_ratio:.3f} ess_ratio={ess_ratio:.3f}")

        self._plot_curves()

    def _checkpoint(self, best: bool = False) -> None:
        tag = "best" if best else "last"
        torch.save(self.model.state_dict(), self.ckpt_dir / f"{tag}_model.pth")

    def _plot_curves(self) -> None:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("log10(|loss|)")
        x_train = np.arange(len(self.train_losses))
        ax1.plot(x_train, np.log10(np.abs(self.train_losses)), label="train_loss")
        x_val = np.array(self.epochs_val)
        if len(self.val_losses):
            ax1.plot(x_val, np.log10(np.abs(self.val_losses)), label="val_loss")
        ax2 = ax1.twinx()
        ax2.set_ylabel("ESS ratio")
        if len(self.ess_ratios):
            ax2.plot(x_val, self.ess_ratios, color="green", label="ess_ratio")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        plt.tight_layout()
        fig.savefig(self.plot_path)
        plt.close(fig)
