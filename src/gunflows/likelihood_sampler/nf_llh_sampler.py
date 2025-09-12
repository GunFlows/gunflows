#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title : NFSamplerProcess
Author: Mathias El Baz
Date  : 2025-08-14

Description
-----------
- Spawns a background process that generates parameter samples either from:
  * a Normalizing Flow checkpoint ("NF" mode), or
  * a reference Gaussian N(mean, cov) defined by the likelihood ("COV" mode).
- Evaluates the negative log-likelihood (NLL) for each sample via the provided
  likelihood interface and writes batches as compressed NPZ files:
    {data, log_p, log_q, cov, mean, par_names, bestfit_nll, from_cov, mode}
- Feeds the dataset process with file notifications via a multiprocessing Queue.

- class NFSamplerProcess(...):
    __init__(...):     wire up config and I/O.
    run():             main loop; sampling + likelihood eval + batch writes.
    request protocol:  via cmd_q you can send:
                           "reload:/path/to/new_nf.pt"
                           "mode:cov" / "mode:nf"


- In COV mode, we compute the proposal log-density correctly:
      log q(x) = -0.5 * [ (x-μ)^T Σ^{-1} (x-μ) + D*log(2π) + logdet(Σ) ]
  using the Cholesky factor of Σ to get both the quadratic form and logdet.
- In NF mode, we assume the NF's `sample(b)` returns (z, logq) where `logq`
  is the log-density in the model space; we negate if needed to get log q(x).
"""

import multiprocessing as mp, queue as _q, os, sys, importlib, traceback, tempfile
from contextlib import contextmanager
import numpy as np
import torch
import logging, time, json

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
                 nf_chunk_size: int = 32768,
                 device: str = "cpu",
                 model_cfg: dict | None = None,
                 save_dir: str | None = None,
                 write_every: int | None = None,
                 log_every: int = 100,
                 rethrow: bool = True,
                 worker_id: int = 0):
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
        base_cwd = (llh_cwd or os.path.dirname(os.path.abspath(llh_config))) if llh_config else (llh_cwd or ".")
        self.llh_cwd = base_cwd
        self.threads = int(threads)
        self.data_is_asimov = bool(data_is_asimov)
        self.nf_chunk_size = int(nf_chunk_size)
        self.device = device
        self.model_cfg = model_cfg or {}
        self.save_dir = save_dir
        self.write_every = int(write_every) if write_every else int(n_points)
        self.log_every = int(log_every)
        self._logger = None
        self._batch_counter = 0
        self._log_path = None
        self._progress_path = None
        self._cov_ref = None
        self._mean_ref = None
        self.rethrow = bool(rethrow)
        self.worker_id = int(worker_id)

        print(f"NFSamplerProcess[{self.worker_id}] initialized: nf_ckpt={self.nf_ckpt} n_points={self.n_points} llh_config={self.llh_config} llh_cwd={self.llh_cwd} threads={self.threads} asimov={self.data_is_asimov} device={self.device} save_dir={self.save_dir} write_every={self.write_every} log_every={self.log_every} rethrow={self.rethrow}")

    def _sd(self):
        sd = self.save_dir if self.save_dir else os.environ.get("TMPDIR", "/tmp")
        os.makedirs(sd, exist_ok=True)
        return sd

    def _setup_io(self):
        sd = self._sd()
        self._log_path = os.path.join(sd, f"sampling_{self.worker_id}.log")
        self._progress_path = os.path.join(sd, f"progress_{self.worker_id}.json")
        sys.stdout = open(os.path.join(sd, f"stdout_{self.worker_id}.log"), "a", buffering=1)
        sys.stderr = open(os.path.join(sd, f"stderr_{self.worker_id}.log"), "a", buffering=1)
        self._logger = logging.getLogger(f"sampling.{os.getpid()}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.handlers.clear()
        fh = logging.FileHandler(self._log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self._logger.addHandler(fh)

    def _write_progress(self, generated:int, mode:str, status:str="running", err:str|None=None):
        payload = {"generated": int(generated), "files": int(self._batch_counter), "mode": mode, "pid": os.getpid(), "status": status, "ts": time.time(), "worker_id": int(self.worker_id)}
        if err is not None:
            payload["error"] = err
        with open(self._progress_path, "w") as f:
            json.dump(payload, f)

    def _save_batch_npz(self, data_chunk, log_p_chunk, log_q_chunk,
                        cov_ref, mean_ref, par_names, bestfit, mode):
        sd = self._sd()
        fname = f"batch{self._batch_counter:06d}_s{self.worker_id:02d}.npz"
        final_path = os.path.join(sd, fname)
        with tempfile.NamedTemporaryFile(dir=sd, delete=False, suffix=".npz") as tf:
            np.savez_compressed(
                tf.name,
                data=data_chunk,
                log_p=log_p_chunk,
                log_q=log_q_chunk,
                cov=cov_ref,
                mean=mean_ref,
                par_names=par_names,
                bestfit_nll=bestfit,
                from_cov=np.array(1 if mode == "COV" else 0, dtype=np.int8),
                logq_is_physical=np.array(1, dtype=np.int8),
                mode=np.array(mode)
            )
            tmp_path = tf.name
        os.replace(tmp_path, final_path)
        self._batch_counter += 1
        if self._logger:
            self._logger.info(f"Wrote {final_path} n={len(data_chunk)}")
        try:
            self.data_q.put({"from_file": final_path}, timeout=0.01)
        except Exception:
            pass

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

    def _build_model_from_meta(self, meta: dict):
        mod = importlib.import_module("gunflows.utils.build_flow")
        build_base = getattr(mod, "build_base")
        build_flow_layers = getattr(mod, "build_flow_layers")
        build_model = getattr(mod, "build_model")
        total_dim   = int(meta["total_dim"])
        dim_spline  = int(meta["dim_spline"])
        nflows      = int(meta["nflows"])
        hidden      = int(meta["hidden"])
        nlayers     = int(meta["nlayers"])
        nbins       = int(meta["nbins"])
        tail_bound  = float(meta["tail_bound"])
        ctx_tr      = bool(meta.get("context_transform", True))
        freeze_cf   = bool(meta.get("freeze_covflow", True))
        n_ctx_flows = int(meta.get("n_context_flows", 12))
        n_hidden    = int(meta.get("n_hidden_layers", 2))
        hidden_dim  = int(meta.get("hidden_dim", 64))

        base = build_base(total_dim)
        tail_bounds = torch.ones(dim_spline) * tail_bound
        flows = build_flow_layers(nflows, dim_spline, hidden, nlayers, nbins, tail_bounds, n_context=(total_dim - dim_spline))

        cov_t  = torch.as_tensor(self._cov_ref,  dtype=torch.float32)
        mean_t = torch.as_tensor(self._mean_ref, dtype=torch.float32)
        std_t  = torch.sqrt(torch.diag(cov_t))
        Dinv = torch.diag(1.0 / std_t)
        S = Dinv @ cov_t @ Dinv
        chol = torch.linalg.cholesky(S + 1e-6 * torch.eye(S.shape[0], dtype=S.dtype))

        class TargetShim: pass
        t = TargetShim()
        t.list_dim_conditionnal = [i for i in range(total_dim) if i not in self.phase_space_dim]
        t.phase_space_dim = self.phase_space_dim
        t.cholesky = chol
        t.std_per_dim = std_t
        t.mean = mean_t
        t.cov = S
        t.true_cov = S

        return build_model(base, flows, t, ctx_tr, freeze_cf, n_ctx_flows, n_hidden, hidden_dim)

    def _resolve_callable(self, target: str):
        try:
            mod = importlib.import_module(target)
            for name in ("create","create_model","build","build_model","make","make_model","factory","Model","model"):
                fn = getattr(mod, name, None)
                if callable(fn):
                    return fn, f"{target}.{name}"
            raise TypeError(f"Target '{target}' is a module with no callable factory")
        except ModuleNotFoundError:
            if "." not in target:
                raise
            module_name, attr_name = target.rsplit(".", 1)
            mod = importlib.import_module(module_name)
            attr = getattr(mod, attr_name, None)
            if attr is None:
                raise AttributeError(f"'{module_name}' has no attribute '{attr_name}'")
            if not callable(attr):
                raise TypeError(f"Target '{target}' resolved to non-callable object of type {type(attr)}")
            return attr, target

    def _load_nf_full(self):
        obj = torch.load(self.nf_ckpt, map_location="cpu")
        if hasattr(obj, "state_dict") and callable(getattr(obj, "eval", None)):
            m = obj
            m.to("cpu").eval()
            return m
        if isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values()):
            meta_path = os.path.splitext(self.nf_ckpt)[0] + ".json"
            meta = {}
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            need = {"total_dim","dim_spline","nflows","hidden","nlayers","nbins","tail_bound"}
            if need.issubset(set(meta.keys())):
                if self._logger: self._logger.info("Rebuilding NF from meta hyperparameters")
                m = self._build_model_from_meta(meta)
                m.load_state_dict(obj, strict=True)
                m.to("cpu").eval()
                return m
            target = meta.get("_target_") or (self.model_cfg.get("_target_") if isinstance(self.model_cfg, dict) else None)
            if target:
                ctor, used = self._resolve_callable(target)
                kwargs = {k: v for k, v in meta.items() if k != "_target_"}
                if isinstance(self.model_cfg, dict):
                    for k, v in self.model_cfg.items():
                        kwargs.setdefault(k, v)
                if self._logger: self._logger.info(f"Instantiating NF with target={used}")
                m = ctor(**kwargs)
                if hasattr(m, "load_state_dict"):
                    m.load_state_dict(obj, strict=True)
                m.to("cpu").eval()
                return m
            raise RuntimeError("Cannot reconstruct NF: provide build hyperparameters in JSON or a valid _target_.")
        raise TypeError(f"Unsupported checkpoint object type: {type(obj)}")

    @torch.no_grad()
    def run(self):
        try:
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            torch.set_num_threads(1)
            self._setup_io()
            if self._logger: self._logger.info("Sampler starting")
            self._write_progress(0, "unknown", status="starting")
            torch.set_grad_enabled(False)
            rng = torch.Generator().manual_seed(self.seed)

            llh = self._load_llh() if self.llh_config else None
            if llh is not None:
                cov_ref = np.asarray(llh.postfit_covariance_matrix, dtype=np.float32)
                mean_ref = np.asarray(llh.postfit_parameter_values, dtype=np.float32)
                par_names = llh.get_parameter_names()
                bestfit = float(getattr(llh, "likelihood_at_bestfit", 0.0))
            else:
                raise RuntimeError("No likelihood configuration provided, cannot sample from COV.")
            self._cov_ref = cov_ref
            self._mean_ref = mean_ref

            cov_t  = torch.as_tensor(cov_ref)
            mean_t = torch.as_tensor(mean_ref)
            L_phys = torch.linalg.cholesky(cov_t)
            logdet_cov = 2.0 * torch.log(torch.diag(L_phys)).sum()
            const = cov_t.shape[0] * np.log(2.0 * np.pi)

            use_nf = bool(self.nf_ckpt) and os.path.isfile(self.nf_ckpt)
            nf_model = self._load_nf_full() if use_nf else None
            mode = "NF" if use_nf else "COV"
            self._write_progress(0, mode, status="running")
            if self._logger: self._logger.info(f"NF mode: {use_nf} path={self.nf_ckpt}")
            generated_total = 0

            def sample_candidates(k: int):
                if use_nf and nf_model is not None:
                    xs, lqs = [], []
                    need = k
                    while need > 0:
                        b = min(self.nf_chunk_size, need)
                        z, logq = nf_model.sample(b)
                        xs.append(z.cpu())
                        lqs.append(-logq.cpu())
                        need -= b
                    x_std = torch.cat(xs, 0)
                    std = torch.sqrt(torch.diag(cov_t))
                    x_phys = x_std * std + mean_t
                    log_q = torch.cat(lqs, 0).numpy().astype(np.float32)
                    return x_phys, log_q
                else:
                    z = torch.randn(k, mean_t.numel(), generator=rng)
                    x_phys = z @ L_phys.T + mean_t
                    std = torch.sqrt(torch.diag(cov_t))
                    Dinv = torch.diag(1.0 / std)
                    cov_std = Dinv @ cov_t @ Dinv
                    L_std = torch.linalg.cholesky(cov_std)
                    logdet_covstd = 2.0 * torch.log(torch.diag(L_std)).sum()
                    diff = (x_phys - mean_t).T
                    y = torch.linalg.solve_triangular(L_phys, diff, upper=False)
                    quad = (y * y).sum(dim=0)
                    log_q = (0.5 * (quad + const + logdet_covstd)).cpu().numpy().astype(np.float32)
                    return x_phys, log_q

            while not self.stop_evt.is_set():
                reload_path, mode_cmd = None, None
                while True:
                    try:
                        cmd = self.cmd_q.get_nowait()
                    except _q.Empty:
                        break
                    if cmd.startswith("reload:"): reload_path = cmd.split("reload:", 1)[1]
                    elif cmd in ("mode:cov", "mode:nf"): mode_cmd = cmd

                if reload_path is not None and reload_path != self.nf_ckpt:
                    self.nf_ckpt = reload_path
                    use_nf = os.path.isfile(self.nf_ckpt)
                    nf_model = self._load_nf_full() if use_nf else None
                    mode = "NF" if use_nf else "COV"
                    if self._logger: self._logger.info(f"Reload → mode={mode} path={self.nf_ckpt}")
                    self._write_progress(generated_total, mode, status="running")
                elif mode_cmd == "mode:cov":
                    use_nf, nf_model, mode = False, None, "COV"
                    self._write_progress(generated_total, mode, status="running")
                elif mode_cmd == "mode:nf" and os.path.isfile(self.nf_ckpt):
                    use_nf, nf_model, mode = True, self._load_nf_full(), "NF"
                    self._write_progress(generated_total, mode, status="running")

                bsz = max(1, int(self.write_every or self.n_points))
                acc_x, acc_lq, acc_lp = [], [], []

                while len(acc_x) < bsz and not self.stop_evt.is_set():
                    need = bsz - len(acc_x)
                    print(f"Generating {need} samples (total={generated_total + len(acc_x)}) mode={mode}")
                    x_phys_t, log_q_np = sample_candidates(need)

                    if use_nf and nf_model is not None:
                        pass

                    x_np = x_phys_t.cpu().numpy().astype(np.float32)
                    with pushd(self.llh_cwd if self.llh_config else "."):
                        for i in range(x_np.shape[0]):
                            if self.stop_evt.is_set(): break
                            if self.llh_config:
                                nll, _, _ = llh.inject_params_and_compute_likelihood(x_np[i].tolist(), extend_continue=False)
                            else:
                                raise NotImplementedError("Likelihood computation not implemented")
                            if nll == -1:
                                if self.rethrow:
                                    continue
                            print("Index:", i, "Log Q:", log_q_np[i], "NLL:", nll)
                            acc_x.append(x_np[i])
                            acc_lq.append(float(log_q_np[i]))
                            acc_lp.append(float(nll))

                if not acc_x:
                    continue

                x_np_final  = np.asarray(acc_x, dtype=np.float32)[:bsz]
                log_q_final = np.asarray(acc_lq, dtype=np.float32)[:bsz]
                log_p_final = np.asarray(acc_lp, dtype=np.float32)[:bsz]

                self._save_batch_npz(
                    x_np_final, log_p_final, log_q_final,
                    cov_ref, mean_ref, par_names, bestfit, mode=mode
                )
                generated_total += x_np_final.shape[0]
                self._write_progress(generated_total, mode, status="running")
                if self._logger: self._logger.info(f"Batch complete total={generated_total} mode={mode}")
                time.sleep(0.05)

            if self._logger: self._logger.info("Sampler stopping")
            self._write_progress(generated_total, mode, status="finished")
        except Exception:
            tb = traceback.format_exc()
            try:
                with open(os.path.join(self._sd(), f"stderr_{self.worker_id}.log"), "a") as f:
                    f.write(tb + "\n")
                self._write_progress(0, "unknown", status="error", err=str(tb.splitlines()[-1]))
            except Exception:
                pass
