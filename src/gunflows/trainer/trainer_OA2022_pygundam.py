#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import time
import numpy as np
import torch
import multiprocessing as mp
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt

from gunflows.trainer.base_trainer import BaseTrainer
from gunflows.dataset.systematic_dataset import SystematicDataset
from gunflows.sampler.nf_llh_sampler import NFSamplerProcess
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
        self.cfg = cfg
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
        self.ess: list[float | None] = []
        self.epochs_val: list[int] = []

        run_dir = HydraConfig.get().runtime.output_dir
        ckpt_root = cfg.get("paths", {}).get("checkpoints_dir", f"{run_dir}/checkpoints")
        self.ckpt_dir = Path(ckpt_root)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir = self.ckpt_dir / "img"
        self.img_dir.mkdir(parents=True, exist_ok=True)

        self.gen_start_epoch = tcfg.get("gen_start_epoch", 1000)
        self.gen_batch_size = tcfg.get("gen_batch_size", 1_000_000)
        self.current_ckpt = cfg.paths.nf_ckpt

        self.data_q = mp.Queue(maxsize=2)
        self.cmd_q = mp.Queue()
        self.stop_evt = mp.Event()
        self.gen_proc = None

    def train(self) -> None:
        start = time.time()
        for epoch in range(self.epochs):
            self._train_batch()

            if epoch == self.gen_start_epoch:
                self._start_generator()

            if self.gen_proc is not None:
                self._swap_if_ready()

            if epoch % self.val_every == 0:
                self._validate_epoch(epoch)

            if self.wait >= self.patience and epoch > self.min_epoch:
                break
        self.stop_evt.set()
        if self.gen_proc is not None:
            self.gen_proc.join()
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
        self._plot_curves()

    def _checkpoint(self, best: bool = False) -> None:
        tag = "best" if best else "last"
        path = self.ckpt_dir / f"{tag}_model.pth"
        torch.save(self.model.state_dict(), path)
        if best and self.gen_proc is not None:
            self.current_ckpt = str(path)
            self.cmd_q.put(f"reload:{self.current_ckpt}")

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
        fig.tight_layout()
        fig.savefig(self.img_dir / "training_curves.png")
        plt.close(fig)

    def _start_generator(self) -> None:
        if self.gen_proc is not None:
            return
        self.gen_proc = NFSamplerProcess(
            nf_ckpt=self.current_ckpt,
            n_points=self.gen_batch_size,
            llh_config=self.cfg.likelihood.config,
            llh_overrides=self.cfg.likelihood.get("overrides", []),
            phase_space_dim=self.dataset.phase_space_dim,
            data_q=self.data_q,
            cmd_q=self.cmd_q,
            stop_evt=self.stop_evt,
            seed=self.cfg.get("seed", 42),
        )
        self.gen_proc.start()

    def _swap_if_ready(self) -> bool:
        swapped = False
        while not self.data_q.empty():
            d = self.data_q.get_nowait()
            self.dataset.replace_from_dict(d)
            seed_split = self.cfg.trainer.get("seed", self.cfg.get("seed", 42))
            rng = np.random.default_rng(seed_split)
            self.val_idx = rng.choice(len(self.dataset), size=self.cfg.trainer.num_val, replace=False)
            self.train_idx = np.setdiff1d(np.arange(len(self.dataset)), self.val_idx)
            swapped = True
        return swapped
