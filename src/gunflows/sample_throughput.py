#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Title: sample_throughput.py
#  Author: Lorenzo Giannessi
#  Description:
#   Simple timing test. Takes a training folder (like effective_sample_size.py),
#   loads the LATEST-epoch NF model, samples N events from it, then evaluates the
#   likelihood on those samples. Reports timing and CPU-hour throughput:
#     - samples per CPU-hour          (NF sampling only)
#     - (samples + evaluation) per CPU-hour  (sampling + LH evaluation)
#   where CPU-hours = wall_time[h] * n_cpus (allocated CPUs, billing convention).
#   Writes a json and a bar plot.
#
#   Usage:
#     python -m gunflows.sample_throughput training_folder=/path/to/run num_samples=100000
# =============================================================================

from __future__ import annotations
import math, time, os, sys, json
from pathlib import Path
import re
import multiprocessing as mp

import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from sample_mcmc_toy import _abspath
from sample_mcmc import check_parameters_limits, sample_check_append

NF_LOCAL = os.path.join(os.path.dirname(__file__), "..", "normalizing-flows")
sys.path.append(os.path.abspath(NF_LOCAL))

from gunflows.utils.build_flow import build_base, build_flow_layers, build_model
from gunflows.likelihood_sampler import LikelihoodSampler


def redirect_fds(path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)


def init_worker(cfg, logdir):
    global _sampler
    worker_index = mp.current_process()._identity[0] - 1
    redirect_fds(os.path.join(logdir, f"worker_{worker_index}.log"))
    print(f"Worker {worker_index} starting")
    _sampler = LikelihoodSampler(
        config_file=cfg.experiment.dataset.llh_config,
        override_files=cfg.experiment.dataset.llh_overrides,
        data_is_asimov=cfg.experiment.dataset.data_is_asimov,
        threads=cfg.experiment.sampler.threads,
        llh_cwd=cfg.experiment.dataset.llh_cwd,
        light_mode=False,
    )
    print(f"Worker {worker_index} initialized.")


def worker(v):
    logp, _, _ = _sampler.inject_params_and_compute_likelihood(values=v, extend_continue=False, verbose=0)
    return logp


def _n_cpus() -> int:
    """Allocated CPUs (SLURM convention), used for CPU-hour accounting."""
    for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(k)
        if v and v.isdigit():
            return int(v)
    return os.cpu_count() or 1


@hydra.main(config_path="../../configs", config_name="sample_throughput", version_base=None)
def main(cfg: DictConfig) -> None:
    training_folder = _abspath(str(cfg.training_folder))
    save_dir = _abspath(str(cfg.save_dir))
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg_path = os.path.join(training_folder, ".hydra", "config.yaml")
    if not os.path.isfile(train_cfg_path):
        raise RuntimeError(f"Training config not found: {train_cfg_path}")
    train_cfg = OmegaConf.load(train_cfg_path)
    cfg = OmegaConf.merge(train_cfg, cfg)
    cfg.experiment.dataset.max_batches = 1
    cfg.experiment.dataset.with_sampler = False
    cfg.experiment.dataset.plot_grid = False

    torch.manual_seed(int(getattr(cfg, "seed", 0)))
    num_samples = int(cfg.num_samples)
    batch_size = int(cfg.batch_size)
    n_cpus = _n_cpus()

    print(f"training_folder: {training_folder}", flush=True)
    print(f"save_dir: {save_dir}", flush=True)
    print(f"num_samples: {num_samples}  n_cpus: {n_cpus}  device: {cfg.device}", flush=True)

    # likelihood interface (main process) + parameter limits
    print("Initializing likelihood interface...", flush=True)
    likelihood_sampler = LikelihoodSampler(
        config_file=cfg.experiment.dataset.llh_config,
        override_files=cfg.experiment.dataset.llh_overrides,
        data_is_asimov=cfg.experiment.dataset.data_is_asimov,
        threads=cfg.experiment.sampler.threads,
        llh_cwd=cfg.experiment.dataset.llh_cwd,
        light_mode=False,
    )
    nf_param_names = list(likelihood_sampler.get_parameter_names())
    parameter_limits = {n: likelihood_sampler.get_parameter_limits(n) for n in nf_param_names}

    dataset = instantiate(cfg.experiment.dataset)
    dim_spline = len(dataset.phase_space_dim)

    # --- latest-epoch checkpoint ---------------------------------------------
    ckpt_folder = os.path.join(training_folder, "checkpoints")
    pattern = re.compile(r"sampler_epoch(\d+)\.pt$")
    ckpts = [(int(m.group(1)), f) for f in os.listdir(ckpt_folder)
             if (m := pattern.match(f))]
    if not ckpts:
        raise RuntimeError(f"No sampler_epoch*.pt checkpoints in {ckpt_folder}")
    latest_epoch, fname = max(ckpts, key=lambda t: t[0])
    ckpt_path = Path(ckpt_folder) / fname
    print(f"Latest NF checkpoint: {fname} (epoch {latest_epoch})", flush=True)

    base = build_base(cfg.experiment.model.total_dim)
    tail_bounds = torch.ones(dim_spline) * cfg.experiment.model.tail_bound
    flows = build_flow_layers(
        cfg.experiment.model.nflows, dim_spline, cfg.experiment.model.hidden,
        cfg.experiment.model.nlayers, cfg.experiment.model.nbins, tail_bounds,
        n_context=cfg.experiment.model.total_dim - dim_spline,
    )
    model = build_model(
        base, flows, dataset, cfg.experiment.model.context_transform,
        cfg.experiment.model.freeze_covflow,
        n_context_flows=cfg.experiment.model.n_context_flows,
        hidden_dim=cfg.experiment.model.hidden_dim,
        n_hidden_layers=cfg.experiment.model.n_hidden_layers,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
    model = model.to(cfg.device).eval()
    print("NF model loaded.", flush=True)

    use_gpu = str(cfg.device).startswith("cuda") and torch.cuda.is_available()

    # GPU warm-up (untimed): trigger CUDA context / kernel autotune so the timed
    # sampling reflects steady-state throughput, not one-off start-up overhead.
    print("Warming up the sampler (untimed)...", flush=True)
    t_warm0 = time.perf_counter()
    with torch.no_grad():
        _ = model.sample(min(int(batch_size), 512))
    if use_gpu:
        torch.cuda.synchronize()
    sample_warmup_s = time.perf_counter() - t_warm0

    # --- 1) timed sampling ----------------------------------------------------
    print(f"Sampling {num_samples} events from NF model...", flush=True)
    samples_nf, logqs = [], []
    need_total = int(num_samples)
    t_sample0 = time.perf_counter()
    with torch.no_grad():
        while len(samples_nf) < need_total:
            b = min(batch_size, need_total - len(samples_nf))
            remain = b
            while remain > 5:
                take = sample_check_append(
                    sample_from_nf=True, batch_size=remain, model=model, dataset=dataset,
                    parameter_limits=parameter_limits, samples=samples_nf,
                    return_probs=True, logqs=logqs, mean=None, cov=None)
                remain -= take
                if take == 0:
                    break
            if remain > 0:
                for _ in range(remain):
                    z, logq = model.sample(1)
                    z = z.to("cpu")
                    phys_z = dataset.transform_eigen_space_to_data_space(z).detach().cpu().numpy()[0]
                    while not check_parameters_limits(phys_z, parameter_limits):
                        z, logq = model.sample(1)
                        z = z.to("cpu")
                        phys_z = dataset.transform_eigen_space_to_data_space(z).detach().cpu().numpy()[0]
                    samples_nf.append(phys_z)
                    logqs.append(float(logq.detach().cpu().numpy()[0]))
    if use_gpu:
        torch.cuda.synchronize()  # sampling is async on GPU -> sync before stopping the clock
    t_sample = time.perf_counter() - t_sample0
    samples_nf = np.asarray(samples_nf[:need_total])
    print(f"Sampling done: {samples_nf.shape} in {t_sample:.2f} s", flush=True)

    # --- 2) likelihood evaluation (steady-state; init/warm-up excluded) -------
    print("Evaluating likelihood on the sampled points...", flush=True)
    pool_init_s = 0.0
    if int(cfg.llh_workers) > 0:
        workers_log_dir = out_dir / "llh_workers_logs"
        workers_log_dir.mkdir(parents=True, exist_ok=True)
        # Pool creation + a small warm-up map fully initialize every worker's
        # GUNDAM/LikelihoodSampler. This fixed cost is timed SEPARATELY and is
        # NOT part of the throughput.
        t_init0 = time.perf_counter()
        pool = mp.Pool(processes=int(cfg.llh_workers), initializer=init_worker,
                       initargs=(cfg, workers_log_dir))
        n_warm = min(len(samples_nf), int(cfg.llh_workers))
        if n_warm > 0:
            pool.map(worker, samples_nf[:n_warm], chunksize=1)
        pool_init_s = time.perf_counter() - t_init0
        # steady-state evaluation of all N points
        t_eval0 = time.perf_counter()
        _ = pool.map(worker, samples_nf, chunksize=32)
        t_eval = time.perf_counter() - t_eval0
        pool.close()
        pool.join()
        n_eval = int(samples_nf.shape[0])
    else:
        # sequential: warm up one eval (excluded), then time the rest
        t_init0 = time.perf_counter()
        if len(samples_nf) > 0:
            likelihood_sampler.inject_params_and_compute_likelihood(samples_nf[0], extend_continue=False)
        pool_init_s = time.perf_counter() - t_init0
        t_eval0 = time.perf_counter()
        for v in samples_nf[1:]:
            likelihood_sampler.inject_params_and_compute_likelihood(v, extend_continue=False)
        t_eval = time.perf_counter() - t_eval0
        n_eval = max(0, int(samples_nf.shape[0]) - 1)
    print(f"Evaluation done: {t_eval:.2f} s for {n_eval} evals "
          f"(init/warm-up excluded: {pool_init_s:.2f} s)", flush=True)

    # --- throughput -----------------------------------------------------------
    # Resource accounting: <resource>-hours = wall[h] * #units (allocated).
    # Sampling runs on the GPU (when device=cuda) -> per GPU-hour.
    # Likelihood evaluation runs on the CPU worker pool -> per CPU-hour.
    N = int(samples_nf.shape[0])
    n_gpus = (max(torch.cuda.device_count(), 1) if use_gpu else 0)
    smp_units = n_gpus if use_gpu else n_cpus
    smp_res = "GPU" if use_gpu else "CPU"

    smp_res_hours = (t_sample / 3600.0) * smp_units
    samples_per_smp_hour = N / smp_res_hours if smp_res_hours > 0 else float("nan")

    cpu_h_eval = (t_eval / 3600.0) * n_cpus
    evals_per_cpu_hour = n_eval / cpu_h_eval if cpu_h_eval > 0 else float("nan")

    # combined end-to-end (sample + evaluate one throw), steady-state
    total_ss = t_sample + t_eval
    samples_eval_per_smp_hour = N / ((total_ss / 3600.0) * smp_units) if total_ss > 0 and smp_units > 0 else float("nan")
    samples_eval_per_cpu_hour = N / ((total_ss / 3600.0) * n_cpus) if total_ss > 0 else float("nan")

    print(f"sampling: {samples_per_smp_hour:.4g} samples / {smp_res}-hour", flush=True)
    print(f"evaluation: {evals_per_cpu_hour:.4g} evals / CPU-hour", flush=True)
    print(f"sample+eval: {samples_eval_per_smp_hour:.4g} / {smp_res}-hour, "
          f"{samples_eval_per_cpu_hour:.4g} / CPU-hour", flush=True)

    results = {
        "training_folder": training_folder,
        "epoch": int(latest_epoch),
        "n_samples": N,
        "n_eval": int(n_eval),
        "n_cpus": n_cpus,
        "n_gpus": int(n_gpus),
        "device": str(cfg.device),
        "sampling_resource": smp_res,
        # times (overhead excluded from the rates below)
        "sample_time_s": float(t_sample),
        "eval_time_s": float(t_eval),
        "total_time_s": float(total_ss),
        "sample_warmup_s": float(sample_warmup_s),
        "eval_init_warmup_s": float(pool_init_s),
        # throughput on the relevant compute resource
        "samples_per_gpu_hour": float(samples_per_smp_hour) if use_gpu else None,
        "samples_per_cpu_hour": float(samples_per_smp_hour) if not use_gpu else None,
        "evals_per_cpu_hour": float(evals_per_cpu_hour),
        "samples_plus_eval_per_gpu_hour": float(samples_eval_per_smp_hour) if use_gpu else None,
        "samples_plus_eval_per_cpu_hour": float(samples_eval_per_cpu_hour),
    }
    json_path = out_dir / "sample_throughput.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Wrote {json_path}", flush=True)

    # --- bar plot of the timing ----------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    labels = ["sampling", "evaluation", "eval init\n(excluded)"]
    vals = [t_sample, t_eval, pool_init_s]
    bars = ax.bar(labels, vals, color=["#2563eb", "#ea580c", "#9ca3af"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}s",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("wall time [s]")
    ax.set_title(
        f"NF epoch {latest_epoch}, N={N}  ({n_gpus} GPU, {n_cpus} CPU)\n"
        f"sampling: {samples_per_smp_hour:.3g} samples/{smp_res}-h   |   "
        f"eval: {evals_per_cpu_hour:.3g} evals/CPU-h",
        fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_throughput.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dir / 'sample_throughput.png'}", flush=True)


if __name__ == "__main__":
    main()
