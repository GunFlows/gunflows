#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title : SystematicDatasetOA2022
Author: Mathias El Baz
Date  : 2025-08-14

- Loads parameter batches from a *starting* folder and/or batches produced
  on-the-fly by a background sampler (written as `batch*.npz`).
- Standardizes each batch with its own (mean, cov): x_std = (x - mean) / sqrt(diag(cov)).
- For starting batches only, fixes log_q into standardized space by:
      log_q <- (log_q - mean(log_q)) + 0.5*logdet(cov_std) + D/2*(log(2π) + 1)
- Aligns log_p to log_q using one of:
    * shift_mode="per_batch" : shift each batch before concatenation
    * shift_mode="global"    : shift once after concatenation
    * shift_mode="none"      : no alignment
  The alignment is two-step: median shift, then an additional mean-weight shift:
      extra = log( mean( exp(log_q - log_p) ) ) after trimming the top 0.5%.
- Provides a PyTorch Dataset interface plus a small plotting helper.

- SystematicDatasetOA2022(..., shift_mode="per_batch", mean_shift=True)
- refresh_if_ready(): merge newly produced batches (from sampler queue)
- request_switch_to_nf(path): tell sampler to switch NF checkpoint
- sampler_status(), close(), getters, __len__, __getitem__
- num_samplers > 1 launches several parallel NFSamplerProcess workers
"""

import glob, os, json, numpy as np, torch, logging, multiprocessing as mp, queue as _q
from pathlib import Path
from collections import deque
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from gunflows.likelihood_sampler.nf_llh_sampler import NFSamplerProcess

__all__ = ["SystematicDatasetOA2022"]


def _plot_grid(samples, mean, weights, cov, names, n, out_dir, phase_dims=None, stage=0):
    n_total = samples.shape[1]
    n = min(int(n), n_total)

    if weights is None or not np.all(np.isfinite(weights)):
        weights = np.ones(samples.shape[0])
    w = weights / np.clip(np.sum(weights), 1e-12, None)

    # Standardise using provided mean and covariance to remain insensitive to cuts
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    z = (samples - mean) / std

    ks = np.zeros(n_total)
    for i in range(n_total):
        x = z[:, i]
        m = np.isfinite(x) & np.isfinite(w)
        if not np.any(m):
            ks[i] = 0.0
            continue
        x_i = x[m]
        w_i = w[m]
        w_i = w_i / np.clip(np.sum(w_i), 1e-12, None)
        order = np.argsort(x_i)
        x_sorted = x_i[order]
        w_sorted = w_i[order]
        cdf = np.cumsum(w_sorted)
        cdf_norm = 0.5 * (1.0 + torch.erf(torch.from_numpy(x_sorted) / np.sqrt(2.0))).numpy()
        ks[i] = np.max(np.abs(cdf - cdf_norm))

    candidates = np.array(phase_dims if phase_dims is not None else np.arange(n_total))
    selected = candidates[np.argsort(-ks[candidates])[:n]]

    samples = samples[:, selected]
    mean = mean[selected]
    names = np.array(names)[selected]
    cov = cov[np.ix_(selected, selected)]

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                x = samples[:, i]
                ax.hist(x, bins=60, weights=weights, density=True, histtype="step")
                mu_i = mean[i]
                sigma = np.sqrt(max(cov[i, i], 1e-12))
                xs = np.linspace(mu_i - 3*sigma, mu_i + 3*sigma, 200)

                ax.plot(xs, (1.0/(np.sqrt(2*np.pi)*sigma))*np.exp(-0.5*((xs - mu_i)/sigma)**2))
            else:
                ax.hist2d(samples[:, j], samples[:, i], weights=weights, bins=60, norm=LogNorm())
            if i == n - 1:
                ax.set_xlabel(names[j], fontsize=7)
            if j == 0:
                ax.set_ylabel(names[i], fontsize=7)
            ax.tick_params(axis="both", labelsize=6)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"grid_{stage}.png", dpi=150)
    plt.close(fig)


class SystematicDatasetOA2022(Dataset):
    def __init__(
        self,
        phase_space_dim,
        starting_folder=None,
        with_sampler=False,
        nf_ckpt=None,
        gen_batch_size=10000,
        llh_config=None,
        llh_overrides=None,
        llh_cwd=None,
        seed=42,
        queue_size=2,
        plot_grid=True,
        out_dir="plots",
        timeout=6000.0,
        data_dir=None,
        threads=6,
        num_samplers=1,
        data_is_asimov=True,
        model_cfg=None,
        max_batches=10,
        shift_mode="per_batch",            # "per_batch", "global", or "none"
        mean_shift=True,                   # include the second (mean-weight) shift
    ):
        super().__init__()
        self.phase_space_dim = list(phase_space_dim)
        self.out_dir = Path(out_dir)
        self.timeout = float(timeout)
        self.save_dir = Path(data_dir) if data_dir else None
        self.max_batches = int(max(1, max_batches))
        self.shift_mode = str(shift_mode).lower()
        assert self.shift_mode in {"per_batch", "global", "none"}
        self.mean_shift = bool(mean_shift)
        self.num_samplers = int(max(1, num_samplers))

        self._gen_procs = []
        self._data_q = None
        self._cmd_queues = None
        self._stop_evt = None

        self._logger = None
        self._batches = deque(maxlen=self.max_batches)
        self._seen_paths = set()
        self._starting_paths = set()
        self.stage = 0

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self._logger = logging.getLogger(f"dataset.{os.getpid()}")
            self._logger.setLevel(logging.INFO)
            self._logger.handlers.clear()
            fh = logging.FileHandler(self.save_dir / "dataset.log")
            fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
            self._logger.addHandler(fh)

        if starting_folder:
            self._seed_from_folder(starting_folder, tag="starting", plot_grid=plot_grid)

        if with_sampler:
            if llh_config is None:
                raise ValueError("llh_config must be provided when with_sampler=True")
            self._start_sampler(
                nf_ckpt=nf_ckpt,
                n_points=gen_batch_size,
                llh_config=llh_config,
                llh_overrides=llh_overrides or [],
                llh_cwd=llh_cwd,
                seed=seed,
                queue_size=queue_size,
                save_dir=str(self.save_dir) if self.save_dir else None,
                write_every=gen_batch_size,
                threads=threads,
                data_is_asimov=data_is_asimov,
                model_cfg=model_cfg,
                num_samplers=self.num_samplers,
            )
            if not starting_folder:
                if self.save_dir is not None:
                    self._wait_for_first_file(self.save_dir, timeout=self.timeout)
                    self._seed_from_folder(str(self.save_dir), tag="runtime", plot_grid=plot_grid)
                else:
                    self._wait_and_merge(plot_grid=plot_grid, timeout=self.timeout)

        if not starting_folder and not with_sampler:
            raise ValueError("Either starting_folder must be set or with_sampler=True")

    def __len__(self):
        return self.nsample

    def __getitem__(self, idx):
        return self.data_spline[idx, :], self.data_cond[idx, :], -self.log_q[idx], -self.log_p[idx]

    def log_prob(self, idx):
        return self.__getitem__(idx)

    def get_cov(self): return self.cov
    def get_true_cov(self): return self.true_cov
    def get_cov_sub(self, dims): return self.cov[np.ix_(dims, dims)]
    def get_mean(self): return self.mean
    def transform_eigen_space_to_data_space(self, x): return x * self.std_per_dim + self.mean

    def refresh_if_ready(self, plot_grid=True):
        updated = False
        if self._data_q is not None:
            added = False
            while True:
                try:
                    msg = self._data_q.get_nowait()
                except _q.Empty:
                    break
                if isinstance(msg, dict) and "from_file" in msg:
                    self._append_batch_path(msg["from_file"], tag="runtime")
                    added = True
            if added:
                self.stage += 1
                self._rebuild_merged(plot_grid=plot_grid)
                updated = True
        return updated

    def request_switch_to_nf(self, nf_ckpt):
        if self._cmd_queues:
            for q in self._cmd_queues:
                q.put(f"reload:{nf_ckpt}")

    def sampler_status(self):
        statuses = []
        for p in self._gen_procs:
            statuses.append({
                "alive": bool(p and p.is_alive()),
                "pid": int(p.pid) if p else None,
                "exitcode": int(p.exitcode) if p and p.exitcode is not None else None,
            })
        return statuses

    def close(self):
        if self._stop_evt is not None:
            self._stop_evt.set()
        for p in self._gen_procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        self._gen_procs = []
        self._data_q = None
        self._cmd_queues = None
        self._stop_evt = None

    def __del__(self):
        try: self.close()
        except Exception: pass

    def _log(self, msg):
        if self._logger: self._logger.info(msg)
        else: print(msg)

    def _latest_paths(self, folder, k):
        files = sorted(glob.glob(os.path.join(folder, "batch*.npz")))
        if not files:
            raise FileNotFoundError(f"No files found in {folder} matching 'batch*.npz'")
        return files[-k:]

    def _append_batch_path(self, path, tag):
        r = os.path.realpath(path)
        if r in self._seen_paths: return
        self._seen_paths.add(r)
        if tag == "starting": self._starting_paths.add(r)
        self._batches.append((path, tag))

    def _seed_from_folder(self, folder, tag, plot_grid):
        for p in self._latest_paths(folder, self.max_batches):
            self._append_batch_path(p, tag=tag)
        self._rebuild_merged(plot_grid=plot_grid)

    def _rebuild_merged(self, plot_grid=True):
        if not self._batches:
            raise FileNotFoundError("No batches available to merge.")

        data_std_list, log_p_list, log_q_list = [], [], []
        meta_last = None

        latest_samples_phys = None
        latest_mean_phys = None
        latest_cov_phys = None
        latest_titles = None
        latest_logw = None

        for path, tag in list(self._batches):
            ld = np.load(path, allow_pickle=True)
            is_starting = (os.path.realpath(path) in self._starting_paths)

            data_phys = torch.tensor(ld["data"], dtype=torch.float32)
            log_p_phys = torch.tensor(ld["log_p"], dtype=torch.float32)
            cov_ref   = torch.tensor(ld["cov"],  dtype=torch.float32)
            mean_ref  = torch.tensor(ld["mean"], dtype=torch.float32)
            titles_ref = ld["par_names"]
            bestfit_nll = float((ld["bestfit_nll"].item() if hasattr(ld["bestfit_nll"], "item") else ld["bestfit_nll"])) if "bestfit_nll" in ld else 0.0

            std_per_dim = torch.sqrt(torch.diag(cov_ref))
            data_std = (data_phys - mean_ref) / std_per_dim
            Dinv = torch.diag(1.0 / std_per_dim)
            cov_std = Dinv @ cov_ref @ Dinv
            chol_std = torch.linalg.cholesky(cov_std + 1e-6 * torch.eye(cov_std.shape[0], dtype=cov_std.dtype))

            if "log_q" in ld and ld["log_q"] is not None:
                log_q_phys = torch.tensor(ld["log_q"], dtype=torch.float32)
            else:
                L = torch.linalg.cholesky(cov_std)
                diff = data_std
                y = torch.linalg.solve_triangular(L, diff.T, upper=False)
                quad = (y * y).sum(dim=0)
                const_g = data_phys.shape[1] * np.log(2.0 * np.pi)
                log_q_phys = 0.5 * (quad + 2.0 * torch.log(torch.diag(L)).sum() + const_g)

            if is_starting:
                log_det_covstd = torch.logdet(cov_std)
                D = data_phys.shape[1]
                const = 0.5 * D * (np.log(2 * np.pi) + 1.0)
                log_q_adj = (log_q_phys - log_q_phys.mean()) + 0.5 * log_det_covstd + const
            else:
                log_q_adj = log_q_phys

            log_p_adj = log_p_phys - bestfit_nll

            if self.shift_mode == "per_batch":
                med = torch.median(log_q_adj - log_p_adj)
                log_p_adj = log_p_adj + med
                if self.mean_shift:
                    w = torch.exp(log_q_adj - log_p_adj)
                    q = torch.quantile(w, 0.995)
                    w = w[w <= q].clamp_min(1e-40)
                    log_p_adj = log_p_adj + torch.log(w.mean())

            data_std_list.append(data_std)
            log_q_list.append(log_q_adj)
            log_p_list.append(log_p_adj)

            meta_last = {
                "cov": cov_std,
                "true_cov": cov_std,
                "mean": mean_ref,
                "titles": titles_ref,
                "chol": chol_std,
                "std_per_dim": std_per_dim,
                "cov_phys": cov_ref,
            }
            latest_samples_phys = data_phys.detach().cpu().numpy()
            latest_mean_phys = mean_ref.detach().cpu().numpy()
            latest_cov_phys = cov_ref.detach().cpu().numpy()
            latest_titles = titles_ref
            latest_logw = (log_q_adj - log_p_adj).detach().cpu().numpy()

        self.data  = torch.cat(data_std_list, dim=0)
        self.log_q = torch.cat(log_q_list,  dim=0)
        self.log_p = torch.cat(log_p_list,  dim=0)

        if self.shift_mode == "global":
            shift = torch.median(self.log_q - self.log_p)
            self.log_p = self.log_p + shift
            if self.mean_shift:
                w = torch.exp(self.log_q - self.log_p)
                q = torch.quantile(w, 0.995)
                w = w[w <= q].clamp_min(1e-40)
                self.log_p = self.log_p + torch.log(w.mean())

        self.cov = meta_last["cov"]; self.true_cov = meta_last["true_cov"]
        self.mean = meta_last["mean"]; self.titles = meta_last["titles"]
        self.cholesky = meta_last["chol"]; self.std_per_dim = meta_last["std_per_dim"]
        self.cov_phys = meta_last["cov_phys"]

        self.nsample, self.ndim = self.data.shape
        self.list_dim_conditionnal = [i for i in range(self.ndim) if i not in self.phase_space_dim]
        self.data_cond = self.data[:, self.list_dim_conditionnal]
        self.data_spline = self.data[:, self.phase_space_dim]

        self._log(f"Merged {len(self._batches)} batches → {self.nsample} samples")

        if plot_grid:
            logw = (self.log_q - self.log_p).reshape(-1)
            q = torch.quantile(logw, 0.99)
            mask = logw <= q
            logw = logw[mask] - torch.logsumexp(logw[mask], dim=0)
            samples_phys = (self.data * self.std_per_dim + self.mean)[mask, :].detach().cpu().numpy()
            mean_phys = self.mean.detach().cpu().numpy()
            cov_phys = self.cov_phys.detach().cpu().numpy()
            weights_np = torch.exp(logw).detach().cpu().numpy()
            _plot_grid(
                samples_phys,
                mean_phys,
                weights_np,
                cov_phys,
                self.titles, min(10,len(self.phase_space_dim)), self.out_dir, self.phase_space_dim, self.stage,
            )
            _plot_grid(
                samples_phys,
                mean_phys,
                np.ones_like(weights_np),
                cov_phys,
                self.titles, min(10,len(self.phase_space_dim)), self.out_dir, self.phase_space_dim, f"{self.stage}_unweighted",
            )

            if latest_samples_phys is not None:
                lw = latest_logw
                q_last = np.quantile(lw, 0.99)
                m_last = lw <= q_last
                lw = lw[m_last] - np.log(np.sum(np.exp(lw[m_last])) + 1e-40)
                print(f"Current stage {self.stage}")
                _plot_grid(
                    latest_samples_phys[m_last, :],
                    latest_mean_phys,
                    np.exp(lw),
                    latest_cov_phys,
                    latest_titles, min(10,len(self.phase_space_dim)), self.out_dir, self.phase_space_dim, f"{self.stage}_latest",
                )
                _plot_grid(
                    latest_samples_phys[m_last, :],
                    latest_mean_phys,
                    np.ones_like(lw),
                    latest_cov_phys,
                    latest_titles, min(10,len(self.phase_space_dim)), self.out_dir, self.phase_space_dim, f"{self.stage}_latest_unweighted",
                )

    def _start_sampler(self, nf_ckpt, n_points, llh_config, llh_overrides, llh_cwd, seed, queue_size, save_dir=None, write_every=None, threads=6, data_is_asimov=True, model_cfg=None, num_samplers=1):
        self._data_q = mp.Queue(maxsize=queue_size)
        self._stop_evt = mp.Event()
        self._gen_procs = []
        self._cmd_queues = []
        for i in range(int(num_samplers)):
            cmd_q = mp.Queue()
            p = NFSamplerProcess(
                nf_ckpt=nf_ckpt,
                n_points=n_points,
                llh_config=llh_config,
                llh_overrides=llh_overrides,
                phase_space_dim=self.phase_space_dim,
                data_q=self._data_q,
                cmd_q=cmd_q,
                stop_evt=self._stop_evt,
                seed=seed + i,
                llh_cwd=llh_cwd,
                save_dir=str(save_dir) if save_dir else None,
                write_every=int(write_every) if write_every else None,
                threads=threads,
                data_is_asimov=data_is_asimov,
                model_cfg=model_cfg,
                worker_id=i,
            )
            p.start()
            self._gen_procs.append(p)
            self._cmd_queues.append(cmd_q)

    def _wait_and_merge(self, timeout=600.0, plot_grid=True):
        assert self._data_q is not None and self._gen_procs
        try:
            msg = self._data_q.get(timeout=timeout)
        except Exception as e:
            if not any(p.is_alive() for p in self._gen_procs):
                codes = [p.exitcode for p in self._gen_procs]
                raise RuntimeError(f"Sampler exited with codes {codes}") from e
            raise
        if isinstance(msg, dict) and "from_file" in msg:
            self._append_batch_path(msg["from_file"], tag="runtime")
            self.stage += 1
            self._rebuild_merged(plot_grid=plot_grid)

    def _wait_for_first_file(self, folder, timeout):
        import time
        deadline = time.time() + float(timeout)
        pattern = os.path.join(str(folder), "batch*.npz")
        while time.time() < deadline:
            if glob.glob(pattern): return
            time.sleep(0.5)
        raise TimeoutError(f"No batch*.npz found in {folder} before timeout ({timeout}s).")
