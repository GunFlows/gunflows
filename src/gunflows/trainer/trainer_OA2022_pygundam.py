#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import time, copy
import numpy as np
import torch
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

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


class OA2022PyGundamTrainer(BaseTrainer):
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

        self.warmup_k = int(tcfg.get("warmup_k", 1000))
        self.stage_len = int(tcfg.get("stage_len", 1000))
        self.use_best_for_sampling = bool(tcfg.get("use_best_for_sampling", True))

        es = tcfg.early_stop
        self.patience = es.patience
        self.min_delta = es.min_delta
        self.min_epoch = es.min_epoch

        self.loss_train = LOSS_MAP[tcfg.loss.name_train]
        self.loss_val = LOSS_MAP[tcfg.loss.name_val]
        self.loss_kwargs = dict(tcfg.loss.kwargs)

        seed_split = tcfg.get("seed", cfg.get("seed", 42))
        self._split_rng = np.random.default_rng(seed_split)
        self._reset_split(len(dataset), tcfg.num_val)

        self.wait = 0
        self.best_loss = float("inf")
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.ess: list[float | None] = []
        self.epochs_val: list[int] = []

        run_dir = HydraConfig.get().runtime.output_dir
        ckpt_root = cfg.get("paths", {}).get("checkpoints_dir", f"{run_dir}/checkpoints")
        self.ckpt_dir = Path(ckpt_root)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._next_trigger = self.warmup_k
        self._stage = 0

    def train(self) -> None:
        start = time.time()
        for epoch in range(self.epochs):
            if epoch % self.val_every == 0:
                self._validate_epoch(epoch)

            self._train_batch()

            self._maybe_refresh_dataset()
            self._maybe_trigger_sampling(epoch)

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
        val_kwargs.update(return_extra=True, validation=True, save_dir=self.ckpt_dir)
        loss_val, extras = self.loss_val(
            self.model, self.dataset, self.val_idx, **val_kwargs
        )
        ess = extras.get("ess")

        self.val_losses.append(loss_val.item())
        self.ess.append(ess)

        improved = loss_val < self.best_loss + self.min_delta
        if improved:
            self.best_loss = loss_val
            self.wait = 0
            self._checkpoint(best=True)
        else:
            self.wait += 1
        self._checkpoint(best=False)

        print(f"Epoch={epoch:05d} val_loss={loss_val.item():.3e} ess={ess:.3f}")

    def _checkpoint(self, best: bool = False) -> None:
        tag = "best" if best else "last"
        torch.save(self.model.state_dict(), self.ckpt_dir / f"{tag}_model.pth")

    def _save_full_model_for_sampling(self, epoch: int, use_best: bool) -> Path:
        m = copy.deepcopy(self.model).cpu()
        if use_best and (self.ckpt_dir / "best_model.pth").is_file():
            sd = torch.load(self.ckpt_dir / "best_model.pth", map_location="cpu")
            m.load_state_dict(sd)
        out = self.ckpt_dir / f"sampler_epoch{epoch:05d}.pt"
        torch.save(m, out)
        return out

    def _maybe_trigger_sampling(self, epoch: int) -> None:
        if epoch + 1 == self._next_trigger:
            ckpt = self._save_full_model_for_sampling(epoch=epoch + 1, use_best=self.use_best_for_sampling)
            if hasattr(self.dataset, "request_switch_to_nf"):
                self.dataset.request_switch_to_nf(str(ckpt))
                print(f"[stream] requested NF sampling from {ckpt}")
            self._stage += 1
            self._next_trigger = self.warmup_k + self._stage * self.stage_len

    def _maybe_refresh_dataset(self) -> None:
        if hasattr(self.dataset, "refresh_if_ready") and self.dataset.refresh_if_ready(plot_grid=False):
            n = len(self.dataset)
            nv = min(getattr(self.cfg.trainer, "num_val"), max(1, n - 1))
            self._reset_split(n, nv)
            print(f"[stream] dataset swapped: N={n}, re-split train/val.")

    def _reset_split(self, n_total: int, n_val_req: int) -> None:
        if n_total < 2:
            raise ValueError("Dataset too small to split.")
        n_val = min(n_val_req, n_total - 1)
        self.val_idx = self._split_rng.choice(n_total, size=n_val, replace=False)
        self.train_idx = np.setdiff1d(np.arange(n_total), self.val_idx)
