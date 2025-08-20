#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title : OA2022TrainerPyGundam
Author: Mathias El Baz
Date  : 2025-08-14

Schedule-driven training with periodic validation and early stopping.
Saves last/best checkpoints, exports sampler-ready checkpoints on a warmup/stage
schedule, signals the dataset to switch, and optionally logs validation to CSV.
"""

from __future__ import annotations
import time, json, csv
from pathlib import Path
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from gunflows.trainer.base_trainer import BaseTrainer
import gunflows.losses.importance_losses as IL

LOSS_MAP = {
    "exp_forward":   IL.exp_forward,
    "exp_reverse":   IL.exp_reverse,
    "exp_symmetric": IL.exp_symmetric,
    "kl_forward":    IL.kl_forward,
    "kl_reverse":    IL.kl_reverse,
    "kl_symmetric":  IL.kl_symmetric,
    "absolute_kl_symmetric": IL.absolute_kl_symmetric,
}

class OA2022TrainerPyGundam(BaseTrainer):
    def __init__(self, cfg: DictConfig, model, dataset, optimizer, scheduler, **kwargs):
        super().__init__(cfg)

        tcfg = cfg.trainer
        self.device = torch.device(tcfg.device)
        self.model = model.to(self.device)
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.epochs     = int(tcfg.epochs)
        self.batch_size = int(tcfg.batch_size)
        self.val_every  = int(tcfg.val_every)

        self.warmup_k   = int(tcfg.get("warmup_k", 1000))
        self.stage_len  = int(tcfg.get("stage_len", 1000))
        self.use_best_for_sampling = bool(tcfg.get("use_best_for_sampling", True))
        self.sample_chunk_size     = int(tcfg.get("sample_chunk_size", 0))
        self.save_full_nf          = bool(tcfg.get("save_full_nf", True))
        self.stage_log_csv         = Path(tcfg.get("stage_log_csv")) if "stage_log_csv" in tcfg else None
        if self.stage_log_csv and not self.stage_log_csv.exists():
            self.stage_log_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.stage_log_csv.open("w", newline="") as f:
                csv.writer(f).writerow(["epoch","stage","val_loss","ess","ckpt"])

        es = tcfg.early_stop
        self.patience  = int(es.patience)
        self.min_delta = float(es.min_delta)
        self.min_epoch = int(es.min_epoch)

        self.loss_train = LOSS_MAP[str(tcfg.loss.name_train)]
        self.loss_val   = LOSS_MAP[str(tcfg.loss.name_val)]
        self.loss_kwargs = dict(tcfg.loss.kwargs)
        if "cap_f" in tcfg.loss: self.loss_kwargs["cap_f"] = float(tcfg.loss.cap_f)
        if "cap_r" in tcfg.loss: self.loss_kwargs["cap_r"] = float(tcfg.loss.cap_r)

        seed_split = int(tcfg.get("seed", cfg.get("seed", 42)))
        self._split_rng = np.random.default_rng(seed_split)
        self._reset_split(len(dataset), int(tcfg.num_val))

        self.wait = 0
        self.best_loss = float("inf")
        self.train_losses, self.val_losses, self.ess, self.epochs_val = [], [], [], []

        run_dir = HydraConfig.get().runtime.output_dir
        ckpt_root = cfg.get("paths", {}).get("checkpoints_dir", f"{run_dir}/checkpoints")
        self.ckpt_dir = Path(ckpt_root); self.ckpt_dir.mkdir(parents=True, exist_ok=True)

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
        loss = self.loss_train(self.model, self.dataset, idx, self.dataset.stage, **self.loss_kwargs, return_extra=False)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.train_losses.append(float(loss.item()))

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> None:
        self.epochs_val.append(epoch)
        vkw = dict(self.loss_kwargs); vkw.update(return_extra=True, validation=True, save_dir=self.ckpt_dir)
        loss_val, extras = self.loss_val(self.model, self.dataset, self.val_idx, self.dataset.stage, **vkw)
        ess = extras.get("ess", float("nan"))
        self.val_losses.append(float(loss_val.item())); self.ess.append(float(ess))
        improved = loss_val < self.best_loss + self.min_delta
        if improved:
            self.best_loss = float(loss_val); self.wait = 0; self._checkpoint(best=True)
        else:
            self.wait += 1
        self._checkpoint(best=False)
        if self.stage_log_csv:
            with self.stage_log_csv.open("a", newline="") as f:
                csv.writer(f).writerow([epoch, self._stage, float(loss_val.item()), float(ess), ""])
        print(f"Epoch={epoch:05d} val_loss={loss_val.item():.3e} ess={ess:.3f}")

    def _checkpoint(self, best: bool = False) -> None:
        tag = "best" if best else "last"
        torch.save(self.model.state_dict(), self.ckpt_dir / f"{tag}_model.pth")

    def _save_full_model_for_sampling(self, epoch: int, use_best: bool) -> Path:
        base = self.ckpt_dir / f"sampler_epoch{epoch:05d}"
        if use_best and (self.ckpt_dir / "best_model.pth").is_file():
            state = torch.load(self.ckpt_dir / "best_model.pth", map_location="cpu")
        else:
            state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state, base.with_suffix(".pt"))
        if self.save_full_nf and "model" in self.cfg:
            meta = OmegaConf.to_container(self.cfg.model, resolve=True)
            with open(base.with_suffix(".json"), "w") as f:
                json.dump(meta, f)
        return base.with_suffix(".pt")

    def _maybe_trigger_sampling(self, epoch: int) -> None:
        if epoch + 1 == self._next_trigger:
            ckpt = self._save_full_model_for_sampling(epoch=epoch + 1, use_best=self.use_best_for_sampling)
            if hasattr(self.dataset, "request_switch_to_nf"):
                self.dataset.request_switch_to_nf(str(ckpt))
                print(f"[stream] requested NF sampling from {ckpt}")
            if self.stage_log_csv:
                with self.stage_log_csv.open("a", newline="") as f:
                    csv.writer(f).writerow([epoch + 1, self._stage + 1, "", "", str(ckpt)])
            self._stage += 1
            self._next_trigger = self.warmup_k + self._stage * self.stage_len

    def _maybe_refresh_dataset(self) -> None:
        if hasattr(self.dataset, "refresh_if_ready") and self.dataset.refresh_if_ready(plot_grid=False):
            n = len(self.dataset)
            nv = min(int(getattr(self.cfg.trainer, "num_val")), max(1, n - 1))
            self._reset_split(n, nv)
            print(f"[stream] dataset swapped: N={n}, re-split train/val.")

    def _reset_split(self, n_total: int, n_val_req: int) -> None:
        if n_total < 2:
            raise ValueError("Dataset too small to split.")
        n_val = min(n_val_req, n_total - 1)
        self.val_idx = self._split_rng.choice(n_total, size=n_val, replace=False)
        self.train_idx = np.setdiff1d(np.arange(n_total), self.val_idx)
