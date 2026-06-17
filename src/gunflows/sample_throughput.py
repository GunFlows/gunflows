#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Title: sample_throughput.py
#  Author: Lorenzo Giannessi
#  Description:
#   Simple timing test. Takes a training folder (like effective_sample_size.py),
#   loads the LATEST-epoch NF model, and times three phases on N events:
#     1) NF sampling            (draw z ; GPU)        -> samples per GPU-hour
#     2) NF density evaluation  (log q ; GPU)         -> density evals per GPU-hour
#     3) LH evaluation          (GUNDAM ; CPU pool)   -> LH evals per CPU-hour
#   (model.sample() returns log q jointly, so phase 1 already yields the density
#   at the drawn points; phase 2 times the standalone density eval on points.)
#   Warm-up / pool-init overheads are measured separately and EXCLUDED from the
#   rates. <resource>-hours = wall_time[h] * #units (allocated). Writes json + plot.
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


def nf_logprob_physical(model, dataset, x_phys, batch_size, device):
    """Evaluate the NF log-density log q(x) on physical-space points (GPU forward).

    Mirrors the NF-density evaluation in sample_mcmc_toy.py: invert the
    eigen->data map, split into phase/context dims, call model.log_prob.
    """
    phase_dims = list(dataset.phase_space_dim)
    cond_dims = list(dataset.list_dim_conditionnal)
    n = int(x_phys.shape[0])
    out = np.empty(n, dtype=np.float64)
    dev = torch.device(device)
    with torch.no_grad():
        mean = dataset.mean.to(device=dev, dtype=torch.float32)
        std = dataset.std_per_dim.to(device=dev, dtype=torch.float32)
        for start in range(0, n, int(batch_size)):
            end = min(n, start + int(batch_size))
            xb = torch.as_tensor(np.asarray(x_phys[start:end]), dtype=torch.float32, device=dev)
            x_eig = (xb - mean) / std
            logq = model.log_prob(x_eig[:, phase_dims], context=x_eig[:, cond_dims])
            out[start:end] = logq.detach().to(device="cpu", dtype=torch.float64).numpy().reshape(-1)
    return out


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
    # Use the allocated CPUs for torch intra-op threads (matters for device=cpu,
    # so the CPU-hour accounting reflects all CPUs actually working).
    try:
        torch.set_num_threads(int(n_cpus))
    except Exception:
        pass

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
    # NOTE: model.sample() also returns log q at the drawn points, so this
    # "sampling" already yields the density at the samples for free. The phase
    # below times the NF density evaluation as a standalone operation (log_prob
    # on given points, the inverse pass) so the two costs can be compared.

    # --- 2) NF probability-density evaluation (log q), GPU --------------------
    print("Evaluating NF probability density (log q) on the sampled points...", flush=True)
    # GPU warm-up (untimed)
    t_dwarm0 = time.perf_counter()
    _ = nf_logprob_physical(model, dataset,
                            samples_nf[:min(int(samples_nf.shape[0]), batch_size)],
                            batch_size, cfg.device)
    if use_gpu:
        torch.cuda.synchronize()
    density_warmup_s = time.perf_counter() - t_dwarm0
    t_density0 = time.perf_counter()
    _ = nf_logprob_physical(model, dataset, samples_nf, batch_size, cfg.device)
    if use_gpu:
        torch.cuda.synchronize()
    t_density = time.perf_counter() - t_density0
    print(f"NF density eval done in {t_density:.2f} s", flush=True)

    # --- NF (sampling + density) throughput ----------------------------------
    # <resource>-hours = wall[h] * #units (allocated). NF sampling and NF density
    # eval run on the GPU (device=cuda) -> per GPU-hour (fall back to CPU if no GPU).
    N = int(samples_nf.shape[0])
    n_gpus = (max(torch.cuda.device_count(), 1) if use_gpu else 0)
    gpu_units = n_gpus if use_gpu else n_cpus
    gpu_res = "GPU" if use_gpu else "CPU"

    def _per_hour(n, t, units):
        h = (t / 3600.0) * units
        return (n / h) if h > 0 else float("nan")

    samples_per_gpu_h = _per_hour(N, t_sample, gpu_units)
    density_per_gpu_h = _per_hour(N, t_density, gpu_units)
    sample_plus_density_per_gpu_h = _per_hour(N, t_sample + t_density, gpu_units)

    json_path = out_dir / "sample_throughput.json"
    plot_path = out_dir / "sample_throughput.png"

    def _save(results, labels, vals, colors, title):
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        bars = ax.bar(labels, vals, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2g}s",
                    ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("wall time [s]")
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    print(f"sampling        : {samples_per_gpu_h:.4g} samples / {gpu_res}-hour", flush=True)
    print(f"NF density eval : {density_per_gpu_h:.4g} evals / {gpu_res}-hour", flush=True)
    print(f"sample+density  : {sample_plus_density_per_gpu_h:.4g} / {gpu_res}-hour", flush=True)

    results = {
        "training_folder": training_folder,
        "epoch": int(latest_epoch),
        "n_samples": N,
        "n_cpus": n_cpus,
        "n_gpus": int(n_gpus),
        "device": str(cfg.device),
        "gpu_resource": gpu_res,
        "sample_time_s": float(t_sample),
        "density_time_s": float(t_density),
        "sample_warmup_s": float(sample_warmup_s),
        "density_warmup_s": float(density_warmup_s),
        "samples_per_gpu_hour": float(samples_per_gpu_h),
        "nf_density_evals_per_gpu_hour": float(density_per_gpu_h),
        "sample_plus_density_per_gpu_hour": float(sample_plus_density_per_gpu_h),
        # LH fields filled in after the (slow) LH loop below
        "n_lh_eval": None,
        "lh_time_s": None,
        "lh_init_warmup_s": None,
        "lh_evals_per_cpu_hour": None,
    }
    # Write the NF (sampling+density) results NOW, before the slow LH loop.
    _save(results,
          ["NF sampling", "NF density\n(log q)"], [t_sample, t_density],
          ["#2563eb", "#16a34a"],
          f"NF epoch {latest_epoch}, N={N}  ({n_gpus} GPU, {n_cpus} CPU)\n"
          f"sampling {samples_per_gpu_h:.3g} / density {density_per_gpu_h:.3g} "
          f"samples·{gpu_res}-h$^{{-1}}$")
    print(f"Wrote NF (sampling+density) results to {json_path} (LH still pending)", flush=True)

    # --- 3) likelihood (GUNDAM) evaluation (steady-state; init excluded) ------
    print("Evaluating the likelihood (GUNDAM) on the sampled points...", flush=True)
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
        t_lh0 = time.perf_counter()
        _ = pool.map(worker, samples_nf, chunksize=32)
        t_lh = time.perf_counter() - t_lh0
        pool.close()
        pool.join()
        n_lh = int(samples_nf.shape[0])
    else:
        # sequential: warm up one eval (excluded), then time the rest
        t_init0 = time.perf_counter()
        if len(samples_nf) > 0:
            likelihood_sampler.inject_params_and_compute_likelihood(samples_nf[0], extend_continue=False)
        pool_init_s = time.perf_counter() - t_init0
        t_lh0 = time.perf_counter()
        for v in samples_nf[1:]:
            likelihood_sampler.inject_params_and_compute_likelihood(v, extend_continue=False)
        t_lh = time.perf_counter() - t_lh0
        n_lh = max(0, int(samples_nf.shape[0]) - 1)
    print(f"LH eval done: {t_lh:.2f} s for {n_lh} evals "
          f"(init/warm-up excluded: {pool_init_s:.2f} s)", flush=True)

    # --- update results with the LH metric ------------------------------------
    lh_per_cpu_h = _per_hour(n_lh, t_lh, n_cpus)
    print(f"LH (GUNDAM) eval : {lh_per_cpu_h:.4g} evals / CPU-hour", flush=True)
    results.update({
        "n_lh_eval": int(n_lh),
        "lh_time_s": float(t_lh),
        "lh_init_warmup_s": float(pool_init_s),
        "lh_evals_per_cpu_hour": float(lh_per_cpu_h),
    })
    _save(results,
          ["NF sampling", "NF density\n(log q)", "LH eval\n(GUNDAM)"],
          [t_sample, t_density, t_lh],
          ["#2563eb", "#16a34a", "#ea580c"],
          f"NF epoch {latest_epoch}, N={N}  ({n_gpus} GPU, {n_cpus} CPU)\n"
          f"sampling {samples_per_gpu_h:.3g} / density {density_per_gpu_h:.3g} "
          f"samples·{gpu_res}-h$^{{-1}}$  |  LH {lh_per_cpu_h:.3g} /CPU-h")
    print(f"Updated results (incl. LH) in {json_path}", flush=True)


if __name__ == "__main__":
    main()
