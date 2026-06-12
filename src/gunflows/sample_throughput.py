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
    t_sample = time.perf_counter() - t_sample0
    samples_nf = np.asarray(samples_nf[:need_total])
    print(f"Sampling done: {samples_nf.shape} in {t_sample:.2f} s", flush=True)

    # --- 2) timed likelihood evaluation --------------------------------------
    print("Evaluating likelihood on the sampled points...", flush=True)
    t_eval0 = time.perf_counter()
    if int(cfg.llh_workers) > 0:
        workers_log_dir = out_dir / "llh_workers_logs"
        workers_log_dir.mkdir(parents=True, exist_ok=True)
        pool = mp.Pool(processes=int(cfg.llh_workers), initializer=init_worker,
                       initargs=(cfg, workers_log_dir))
        _ = pool.map(worker, samples_nf, chunksize=32)
        pool.close()
        pool.join()
    else:
        for v in samples_nf:
            likelihood_sampler.inject_params_and_compute_likelihood(v, extend_continue=False)
    t_eval = time.perf_counter() - t_eval0
    print(f"Evaluation done in {t_eval:.2f} s", flush=True)

    # --- throughput (CPU-hours = wall[h] * n_cpus) ---------------------------
    N = int(samples_nf.shape[0])
    cpu_h_sample = (t_sample / 3600.0) * n_cpus
    cpu_h_total = ((t_sample + t_eval) / 3600.0) * n_cpus
    samples_per_cpu_h = N / cpu_h_sample if cpu_h_sample > 0 else float("nan")
    samples_eval_per_cpu_h = N / cpu_h_total if cpu_h_total > 0 else float("nan")

    print(f"samples per CPU-hour (sampling only)      : {samples_per_cpu_h:.4g}", flush=True)
    print(f"(samples+evaluation) per CPU-hour (total) : {samples_eval_per_cpu_h:.4g}", flush=True)

    results = {
        "training_folder": training_folder,
        "epoch": int(latest_epoch),
        "n_samples": N,
        "n_cpus": n_cpus,
        "device": str(cfg.device),
        "sample_time_s": float(t_sample),
        "eval_time_s": float(t_eval),
        "total_time_s": float(t_sample + t_eval),
        "cpu_hours_sample": float(cpu_h_sample),
        "cpu_hours_total": float(cpu_h_total),
        "samples_per_cpu_hour": float(samples_per_cpu_h),
        "samples_plus_eval_per_cpu_hour": float(samples_eval_per_cpu_h),
    }
    json_path = out_dir / "sample_throughput.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Wrote {json_path}", flush=True)

    # --- bar plot of the timing ----------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    bars = ax.bar(["sampling", "evaluation", "total"],
                  [t_sample, t_eval, t_sample + t_eval],
                  color=["#2563eb", "#ea580c", "#6b7280"])
    for b, v in zip(bars, [t_sample, t_eval, t_sample + t_eval]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}s",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("wall time [s]")
    ax.set_title(
        f"NF epoch {latest_epoch}, N={N}, {n_cpus} CPU\n"
        f"sampling: {samples_per_cpu_h:.3g} samples/CPU-h   |   "
        f"+eval: {samples_eval_per_cpu_h:.3g} samples/CPU-h",
        fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_throughput.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dir / 'sample_throughput.png'}", flush=True)


if __name__ == "__main__":
    main()
