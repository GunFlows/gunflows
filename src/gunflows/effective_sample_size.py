#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Title: sample_mcmc.py
#  Author: Lorenzo Giannessi
#  Date: 29/01/2026
#  Description:
#   Compute the ESS of the NF model as a function of time, then compare to the MCMC throws
# =============================================================================

from __future__ import annotations
import math, time, os, sys, json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import multiprocessing as mp

import re
import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from concurrent.futures import ThreadPoolExecutor, as_completed
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra.utils import instantiate
from matplotlib.colors import LogNorm
from sample_mcmc_toy import _abspath, _strip_common_prefixes
from sample_mcmc import check_parameters_limits, sample_check_append


NF_LOCAL = os.path.join(os.path.dirname(__file__), "..", "normalizing-flows")
sys.path.append(os.path.abspath(NF_LOCAL))

from gunflows.utils.build_flow import build_base, build_flow_layers, build_model
from gunflows.likelihood_sampler import LikelihoodSampler, pygundam_utils

import ROOT # to read the MCMC chain ROOT file

def redirect_fds(path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)

    # redirect stdout (1) and stderr (2)
    os.dup2(fd, 1)
    os.dup2(fd, 2)

    os.close(fd)

    # also update Python wrappers
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)

def init_worker(cfg,logdir):
    global _sampler
    worker_index = mp.current_process()._identity[0] - 1

    pid = os.getpid()
    logfile = os.path.join(logdir, f"worker_{worker_index}.log")

    # redirect stdout / stderr
    redirect_fds(logfile)

    print(f"Worker {worker_index} starting")
    _sampler = LikelihoodSampler(config_file=cfg.experiment.dataset.llh_config,
        override_files=cfg.experiment.dataset.llh_overrides,
        data_is_asimov=cfg.experiment.dataset.data_is_asimov,
        threads=cfg.experiment.sampler.threads,
        llh_cwd=cfg.experiment.dataset.llh_cwd,
        light_mode=False,
    )
    print(f"Worker {worker_index} initialized.")

def worker(v):
    t0 = time.perf_counter()
    logp, _, _ = _sampler.inject_params_and_compute_likelihood(values=v, extend_continue=False, verbose=0)
    # print(f"computed LH in {time.perf_counter()-t0:.2f} s. NLL/2: {logp}", flush=True)
    return logp


@hydra.main(config_path="../../configs", config_name="effective_sample_size", version_base=None)
def main(cfg: DictConfig) -> None:
    training_folder = _abspath(str(cfg.training_folder))
    mcmc_root = _abspath(str(cfg.mcmc_chain))
    save_dir = _abspath(str(cfg.save_dir))

    print(f"PWD (hydra chdir): {os.getcwd()}", flush=True)
    print(f"training_folder: {training_folder}", flush=True)
    print(f"mcmc_chain: {mcmc_root}", flush=True)
    print(f"save_dir: {save_dir}", flush=True)

    train_cfg_path = os.path.join(training_folder, ".hydra", "config.yaml")
    if not os.path.isfile(train_cfg_path):
        raise RuntimeError(f"Training config not found: {train_cfg_path}")

    train_cfg = OmegaConf.load(train_cfg_path)
    cfg = OmegaConf.merge(train_cfg, cfg)

    cfg.experiment.dataset.max_batches = 1
    cfg.experiment.dataset.with_sampler = False
    cfg.experiment.dataset.plot_grid = False

    seed = int(getattr(cfg, "seed", 0))
    torch.manual_seed(seed)


    # create output directories
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "marginals"
    img_dir.mkdir(parents=True, exist_ok=True)
    corr2d_dir = out_dir / "corr2d"
    corr2d_dir.mkdir(parents=True, exist_ok=True)

    # initialize likelihood interface
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
    nf_param_names_short = [_strip_common_prefixes(n) for n in nf_param_names]
    parameter_limits: dict[str, tuple[float, float]] = {n: likelihood_sampler.get_parameter_limits(n) for n in nf_param_names}

    bestfit_parameter_values = np.asarray(likelihood_sampler.postfit_parameter_values, dtype=np.float64).reshape(-1)
    postfit_covariance = np.asarray(likelihood_sampler.postfit_covariance_matrix, dtype=np.float64)

    dataset = instantiate(cfg.experiment.dataset)
    dim_spline = len(dataset.phase_space_dim)


    ess_list = []
    ess_filtered_list = []
    epoch_list = []

    if (cfg.llh_workers > 0):
        print(f"Initializing {cfg.llh_workers} ({mp.cpu_count()}) workers to compute LH values in parallel.", flush=True)
        # compute LH with multiple threads
        workers_log_dir = out_dir / "llh_workers_logs"
        workers_log_dir.mkdir(parents=True, exist_ok=True)
        pool = mp.Pool(processes=cfg.llh_workers, initializer=init_worker, initargs=(cfg, workers_log_dir))

    # start a loop where at each iteration you pickup a checkpoint, sample from it, compute ESS, make some plots, then store the results
    ckpt_folder = os.path.join(training_folder, "checkpoints")
    pattern = re.compile(r"sampler_epoch(\d+)000.pt")
    tot_models = 0
    for fname in os.listdir(ckpt_folder):
        m = pattern.match(fname)
        if m:
            tot_models += 1
    print(f"Total NF tot_models found: {tot_models}", flush=True)
    for fname in os.listdir(ckpt_folder):
        m = pattern.match(fname)
        if m:
            print(f"Found NF model file: {fname}", flush=True)
            ep = int(m.group(1))
        else:
            continue
        ckpt_path = Path(os.path.join(ckpt_folder, fname))
        print("Using NF model:", ckpt_path, flush=True)

        base = build_base(cfg.experiment.model.total_dim)
        tail_bounds = torch.ones(dim_spline) * cfg.experiment.model.tail_bound
        flows = build_flow_layers(
            cfg.experiment.model.nflows,
            dim_spline,
            cfg.experiment.model.hidden,
            cfg.experiment.model.nlayers,
            cfg.experiment.model.nbins,
            tail_bounds,
            n_context=cfg.experiment.model.total_dim - dim_spline,
        )
        model = build_model(
            base,
            flows,
            dataset,
            cfg.experiment.model.context_transform,
            cfg.experiment.model.freeze_covflow,
            n_context_flows=cfg.experiment.model.n_context_flows,
            hidden_dim=cfg.experiment.model.hidden_dim,
            n_hidden_layers=cfg.experiment.model.n_hidden_layers,
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
        model = model.to(cfg.device).eval()
        print("NF model loaded.", flush=True)

        # sample from NF in physical space
        num_samples = int(cfg.num_samples)
        batch_size = int(cfg.batch_size)

        print(f"Sampling {num_samples} events from NF model...",flush=True)    
        t0 = time.time()
        # Sample from NF (vectorized): sample batches at once, filter vectorially
        batches = math.ceil(num_samples / batch_size)
        samples_nf, logqs = [], []
        start = time.time()
        need_total = int(num_samples)
        with torch.no_grad():
            while len(samples_nf) < need_total:
                need = need_total - len(samples_nf)
                b = min(int(batch_size), need)
                if (cfg.verbose>=1): print(f" NF sampling. {need} throws to go. Sampling {b} now ...", flush=True)            
                remain = b
                while remain > 5:
                    take = sample_check_append(sample_from_nf=True, batch_size=remain, model=model, dataset=dataset, parameter_limits=parameter_limits, samples=samples_nf, return_probs=True, logqs=logqs, mean=None, cov=None)
                    remain -= take
                    if take == 0:
                        break  # avoid infinite loop if no samples accepted
                # for any remaining samples not accepted, fall back to single-sample retry
                if remain > 0:
                    if (cfg.verbose>=2): print(f" Need {remain} more samples after batch filtering. Sampling individually...", flush=True)
                    for _ in range(remain):
                        z, logq = model.sample(1)
                        z = z.to('cpu')
                        phys_z = dataset.transform_eigen_space_to_data_space(z).detach().cpu().numpy()[0]
                        logq_np = float(logq.detach().cpu().numpy()[0])
                        while not check_parameters_limits(phys_z, parameter_limits):
                            # print(f"  -debug- single sample not physical, resampling...", flush=True)
                            z, logq = model.sample(1)
                            z = z.to('cpu')
                            phys_z = dataset.transform_eigen_space_to_data_space(z).detach().cpu().numpy()[0]
                            logq_np = float(logq.detach().cpu().numpy()[0])
                        samples_nf.append(phys_z)
                        logqs.append(logq_np)
                        if (cfg.verbose>=2): print(f" Sampled individual throw. {_+1}/{remain}", flush=True)
                if (cfg.verbose>=1): print(f"Total samples collected: {len(samples_nf)}/{need_total}", flush=True)

        # samples_nf already contains physical-space numpy arrays (appended above);
        # just convert the list to a single NumPy array instead of transforming again.
        samples_nf = np.asarray(samples_nf)

        # Prepare optional weights for NF histogram
        logq_nf = None
        w = np.asarray(logqs)
        if w.ndim > 1:
            w = w.reshape(-1)
        # Ensure weights length matches number of NF samples
        logq_nf = w[: samples_nf.shape[0]]        
        
        print(f"NF sampling done: {samples_nf.shape} in {time.time()-t0:.1f} s", flush=True)

        # scan through the nf samples, compute the nll and compare it to the nf weight (logq_nf)
        iter = 0
        reweight_nf_to_lh = []
        lh_values = []
        start = time.time()
        print("Computing reweighting factors from NF to LH...",flush=True)

        if (cfg.llh_workers > 0):
            print(f"Using {cfg.llh_workers} ({mp.cpu_count()}) workers to compute LH values in parallel.", flush=True)
            # compute LH with multiple threads
            lh_values = pool.map(worker, samples_nf, chunksize=32)
            reweight_nf_to_lh = [-logq - logp for logp, logq in zip(lh_values, logq_nf)]
        else:
            print("Computing LH values sequentially.", flush=True)
            for nf_vector, logq in zip(samples_nf, logq_nf):
                logp,nll_stat,nll_syst = likelihood_sampler.inject_params_and_compute_likelihood(nf_vector,extend_continue=False)
                if (iter % max(1, num_samples // 100) == 0):
                        if (cfg.verbose >= 3):    
                            print(f"iter {iter} NLL/2: {logp}, log_q_nf: {logq}", flush=True)
                iter += 1
                reweight_nf_to_lh.append(-logq - logp)
                lh_values.append(-logp)

        print(f"Computed reweighting factors for {len(reweight_nf_to_lh)} NF samples.",flush=True)
        end = time.time()
        print(f"Time to compute LH values: {end - start:.1f}s", flush=True)

        # Normalize reweighting factors
        if reweight_nf_to_lh:
            median_reweight = np.median(reweight_nf_to_lh)
            reweight_nf_to_lh = (np.array(reweight_nf_to_lh)-median_reweight)
            # shift the median of the likelihood values and log_q_nf accordingly
            median_lh = np.median(lh_values)
            lh_values = np.array(lh_values) - median_lh
            median_logq = np.median(logq_nf)
            logq_nf = logq_nf - median_logq
        # compute variance
        variance_reweight = np.var(reweight_nf_to_lh)
        # compute variance after removing 0.001 quantiles
        lower_bound = np.quantile(reweight_nf_to_lh, 0.001)
        upper_bound = np.quantile(reweight_nf_to_lh, 0.999)
        outlier_mask = (reweight_nf_to_lh >= lower_bound) & (reweight_nf_to_lh <= upper_bound)
        filtered_reweights = reweight_nf_to_lh[outlier_mask]
        variance_filtered = np.var(filtered_reweights)
        # compute effective sample size
        weights = np.exp(reweight_nf_to_lh)
        effective_sample_size = np.sum(weights) ** 2 / np.sum(weights ** 2)
        filtered_weights = np.exp(filtered_reweights)
        effective_sample_size_filtered = np.sum(filtered_weights) ** 2 / np.sum(filtered_weights ** 2)
        print(f"Effective sample size (NF to LH): {effective_sample_size} / {len(reweight_nf_to_lh)}", flush=True)
        print(f"Effective sample size (NF to LH, filtered): {effective_sample_size_filtered} / {len(filtered_reweights)}", flush=True)
        epoch_list.append(ep)
        ess_list.append(effective_sample_size/len(reweight_nf_to_lh))
        ess_filtered_list.append(effective_sample_size_filtered/len(filtered_reweights))

        # sort lists by epoch
        sorted_indices = np.argsort(epoch_list)
        epoch_list = [epoch_list[i] for i in sorted_indices]
        ess_list = [ess_list[i] for i in sorted_indices]
        ess_filtered_list = [ess_filtered_list[i] for i in sorted_indices]

        # save intermediate results to json
        results = {
            "epochs": epoch_list,
            "ess": ess_list,
            "ess_filtered": ess_filtered_list,
        }
        json_path = out_dir / "ess_vs_epoch.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)

        # plot ESS vs epoch
        plt.figure(figsize=(8,6))
        plt.plot(epoch_list, ess_list, marker='o', label='ESS')
        plt.plot(epoch_list, ess_filtered_list, marker='o', label='ESS (filtered)')
        plt.xlabel('Epoch')
        plt.ylabel('Effective Sample Size')
        plt.title(f'Effective Sample Size vs Training Epoch ({num_samples} samples)')
        # plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt_path = out_dir / "ess_vs_epoch.png"
        plt.savefig(plt_path)
        plt.close()

        print(f"Completed {len(epoch_list)}/{tot_models} models", flush=True)

    



    print("Finished looping over checkpoints.")




if __name__ == "__main__":
    main()



