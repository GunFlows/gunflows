#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    "exp_forward": IL.exp_forward,
    "exp_reverse": IL.exp_reverse,
    "exp_symmetric": IL.exp_symmetric,
    "kl_forward": IL.kl_forward,
    "kl_reverse": IL.kl_reverse,
    "kl_symmetric": IL.kl_symmetric,
}


class OA2022Trainer(BaseTrainer):
    def __init__(
        self,
        cfg: DictConfig,
        model,
        dataset,
        optimizer,
        scheduler,
        **kwargs,
    ):
        super().__init__(cfg)
        tcfg = cfg.trainer
        self.device = torch.device(tcfg.device)
        self.model = model.to(self.device)
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = tcfg.epochs
        self.batch_size = tcfg.batch_size
        self.val_every = tcfg.val_every

        es = tcfg.early_stop
        self.patience = es.patience
        self.min_delta = es.min_delta
        self.min_epoch = es.min_epoch

        self.loss_train = LOSS_MAP[tcfg.loss.name_train]
        self.loss_val = LOSS_MAP[tcfg.loss.name_val]
        self.loss_kwargs = dict(tcfg.loss.kwargs)

        seed_split = tcfg.get("seed", cfg.get("seed", 42))
        rng = np.random.default_rng(seed_split)
        self.val_idx = rng.choice(len(dataset), size=tcfg.num_val, replace=False)
        self.train_idx = np.setdiff1d(np.arange(len(dataset)), self.val_idx)

        self.wait = 0
        self.best_loss = float("inf")
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.ess_f: list[float | None] = []
        self.ess_r: list[float | None] = []
        self.epochs_val: list[int] = []

        run_dir = HydraConfig.get().runtime.output_dir
        ckpt_root = cfg.get("paths", {}).get("checkpoints_dir", f"{run_dir}/checkpoints")
        self.ckpt_dir = Path(ckpt_root)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir = self.ckpt_dir / "img"
        self.img_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        start = time.time()
        for epoch in range(self.epochs):
            self._train_batch()
            if epoch % self.val_every == 0:
                self._validate_epoch(epoch)
            if self.wait >= self.patience and epoch > self.min_epoch:
                break
        print(f"Finished in {time.time() - start:.1f}s")

    def _train_batch(self) -> None:
        idx = np.random.choice(self.train_idx, self.batch_size, replace=False)
        loss = self.loss_train(
            self.model, self.dataset, idx, **self.loss_kwargs, return_extra=False
        )
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
        val_kwargs.update(return_extra=True, validation=True, save_dir=self.img_dir)
        loss_val, extras = self.loss_val(
            self.model, self.dataset, self.val_idx, **val_kwargs
        )
        ess_f = extras.get("ess_forward")
        ess_r = extras.get("ess_reverse")

        self.val_losses.append(loss_val.item())
        self.ess_f.append(ess_f)
        self.ess_r.append(ess_r)

        improved = loss_val < self.best_loss + self.min_delta
        if improved:
            self.best_loss = loss_val
            self.wait = 0
            self._checkpoint(best=True)
        else:
            self.wait += 1
        self._checkpoint(best=False)

        print(
            f"Epoch={epoch:05d} val_loss={loss_val.item():.3e} ess={ess_f:.3f}"
        )
        self._plot_curves()

    def _checkpoint(self, best: bool = False) -> None:
        tag = "best" if best else "last"
        torch.save(self.model.state_dict(), self.ckpt_dir / f"{tag}_model.pth")

    def _plot_curves(self) -> None:
        if not self.epochs_val:
            return

        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(self.train_losses, label="train loss (iter)", color="tab:blue")
        ax1.plot(self.epochs_val, self.val_losses, "o-", label="val loss", color="tab:orange")
        ax1.set_yscale("log")
        ax1.set_xlabel("iteration / epoch")
        ax1.set_ylabel("loss")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(self.epochs_val, self.ess_f, "s-", label="ESS", color="tab:green")
        ax2.set_ylabel("ESS")
        ax2.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(self.img_dir / "training_curves.png")
        plt.close(fig)
