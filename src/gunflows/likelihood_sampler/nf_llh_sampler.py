#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing as mp, queue as _q, os, math
from contextlib import contextmanager
import numpy as np
import torch

@contextmanager
def pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class NFSamplerProcess(mp.Process):
    def __init__(self, nf_ckpt, n_points, llh_config, llh_overrides,
                 phase_space_dim, data_q, cmd_q, stop_evt, seed=123,
                 llh_cwd: str | None = None,
                 threads: int = 6,
                 data_is_asimov: bool = False,
                 nf_chunk_size: int = 32768,           # chunked NF sampling like sample.py
                 device: str = "cpu",
                 model_cfg: dict | None = None):       # optional: if you ever need to rebuild from state_dict
        super().__init__(daemon=True)
        self.nf_ckpt = nf_ckpt
        self.n_points = int(n_points)
        self.llh_config = llh_config
        self.llh_overrides = llh_overrides or []
        self.phase_space_dim = phase_space_dim
        self.data_q = data_q
        self.cmd_q = cmd_q
        self.stop_evt = stop_evt
        self.seed = seed
        base_cwd = (llh_cwd or os.path.dirname(os.path.abspath(llh_config))).strip()
        if not os.path.isdir(base_cwd):
            raise FileNotFoundError(base_cwd)
        self.llh_cwd = base_cwd
        self.threads = int(threads)
        self.data_is_asimov = bool(data_is_asimov)
        self.nf_chunk_size = int(nf_chunk_size)
        self.device = device
        self.model_cfg = model_cfg or {}

    def _load_llh(self):
        from gunflows.likelihood_sampler.likelihoodSampler import LikelihoodSampler
        with pushd(self.llh_cwd):
            return LikelihoodSampler(
                self.llh_config,
                override_files=self.llh_overrides,
                threads=self.threads,
                data_is_asimov=self.data_is_asimov,
                seed=self.seed,
            )

    def _load_nf_full(self):
        m = torch.load(self.nf_ckpt, map_location="cpu")
        m.to("cpu").eval()
        return m

    @staticmethod
    def _to_phys(x_std: torch.Tensor, std: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        return x_std * std + mean

    @torch.no_grad()
    def run(self):
        torch.set_grad_enabled(False)
        rng = torch.Generator().manual_seed(self.seed)

        llh = self._load_llh()
        cov_ref = np.asarray(llh.postfit_covariance_matrix, dtype=np.float32)
        mean_ref = np.asarray(llh.postfit_parameter_values, dtype=np.float32)
        par_names = llh.get_parameter_names()
        bestfit = float(getattr(llh, "likelihood_at_bestfit", 0.0))

        cov_t = torch.as_tensor(cov_ref)
        mean_t = torch.as_tensor(mean_ref)
        std_t = torch.sqrt(torch.diag(cov_t))
        eps = 1e-6
        L_phys = torch.linalg.cholesky(cov_t + eps * torch.eye(cov_t.shape[0], dtype=cov_t.dtype))

        Dinv = torch.diag(1.0 / std_t)
        S = Dinv @ cov_t @ Dinv
        L_std = torch.linalg.cholesky(S + eps * torch.eye(S.shape[0], dtype=S.dtype))
        logdet_S = 2.0 * torch.log(torch.diag(L_std)).sum()
        const = cov_t.shape[0] * np.log(2.0 * np.pi)

        use_nf = self.nf_ckpt is not None
        nf_model = self._load_nf_full() if use_nf else None

        while not self.stop_evt.is_set():
            if use_nf:
                need = self.n_points
                xs_std, logqs = [], []
                while need > 0:
                    b = min(self.nf_chunk_size, need)
                    z, logq = nf_model.sample(b)   # R1 space
                    xs_std.append(z.cpu())
                    logqs.append(logq.cpu())
                    need -= b
                x_std = torch.cat(xs_std, dim=0)
                log_q = torch.cat(logqs, dim=0).numpy().astype(np.float32)
                x_phys = self._to_phys(x_std, std_t, mean_t)
            else:
                z = torch.randn(self.n_points, mean_t.numel(), generator=rng)
                x_phys = z @ L_phys.T + mean_t
                x_std = (x_phys - mean_t) / std_t
                y = torch.linalg.solve_triangular(L_std, x_std.T, upper=False)
                quad = (y * y).sum(dim=0)
                log_q = (-0.5 * (quad  + const)).cpu().numpy().astype(np.float32)

            x_np = x_phys.cpu().numpy().astype(np.float32)

            log_p = np.empty(self.n_points, dtype=np.float32)
            with pushd(self.llh_cwd):
                for i in range(self.n_points):
                    llh.inject_parameter_values(x_np[i].tolist())
                    nll = llh.compute_stat_likelihood() + llh.compute_syst_likelihood()
                    log_p[i] = -float(nll)

            payload = {
                "data": x_np,            # physical space
                "log_p": log_p,          # true log-likelihood
                "log_q": log_q,          # NF/logq or MVN in R1 space
                "cov": cov_ref,
                "mean": mean_ref,
                "par_names": par_names,
                "bestfit_nll": bestfit,
            }
            self.data_q.put(payload)

            try:
                cmd = self.cmd_q.get_nowait()
                if cmd.startswith("reload:"):
                    self.nf_ckpt = cmd.split("reload:", 1)[1]
                    nf_model = self._load_nf_full()
                    use_nf = True
                elif cmd == "mode:cov":
                    use_nf = False
                elif cmd == "mode:nf" and self.nf_ckpt is not None:
                    use_nf = True
            except _q.Empty:
                pass
