#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Title: sample_mcmc.py
#  Author: Lorenzo Giannessi
#  Date: 15/09/2025
#  Description:
#    Sample from Normalizing Flow model and compare with MCMC samples
# =============================================================================

from __future__ import annotations
import math, time, os, sys, json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

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

NF_LOCAL = os.path.join(os.path.dirname(__file__), "..", "normalizing-flows")
sys.path.append(os.path.abspath(NF_LOCAL))

from gunflows.utils.build_flow import build_base, build_flow_layers, build_model
from gunflows.likelihood_sampler import LikelihoodSampler, pygundam_utils

import ROOT # to read the MCMC chain ROOT file



def index_mcmc_to_nf(mcmc_branch_name, translator_array, nf_names):
    parameter_name = None
    # find the corresponding index in the NF samples
    if mcmc_branch_name.startswith("ndd_"):
        return int(mcmc_branch_name.split("_")[1]) + 100, f"DetSyst_{int(mcmc_branch_name.split('_')[1])}" # det syst start at 100 in the NF samples
    elif mcmc_branch_name.startswith("xsec_"):
        if mcmc_branch_name == "xsec_74": 
            return None, None  # skip problematic one (I think this is EB alpha, only present in MCMC)
        index = int(mcmc_branch_name.split("_")[1])
        if index > 74:
            index -= 1  # adjust index to skip xsec_74
        parameter_name = translator_array[index]
        # print(f"Mapping MCMC branch {mcmc_branch_name} to parameter name {parameter_name}")
        if parameter_name.startswith("b_"):
            return int(parameter_name.split("_")[1]), f"FluxSyst_{int(parameter_name.split('_')[1])}"  # xsec_100..xsec_174 are the flux syst in the NF samples
        # find the name of this parameter in the NF samples
        # handle special cases
        if parameter_name == "EB_dial_O_nubar":
            parameter_name = "EB_bin_O_nubar"
        if parameter_name == "EB_dial_O_nu":
            parameter_name = "EB_bin_O_nu"
        if parameter_name == "EB_dial_C_nubar":
            parameter_name = "EB_bin_C_nubar"
        if parameter_name == "EB_dial_C_nu":
            parameter_name = "EB_bin_C_nu"
        for i, nf_name in enumerate(nf_names):
            pattern = re.compile(r"#\d+_(.*)")
            if pattern.match(nf_name) is not None:
                if parameter_name == pattern.match(nf_name).group(1):
                    # print(f" -> Mapping {parameter_name} to {nf_name}")
                    return i, pattern.match(nf_name).group(1)
    # print(f"Warning: unrecognized MCMC branch name {mcmc_branch_name}.")
    return None, parameter_name            


@hydra.main(config_path="../../configs", config_name="sample", version_base=None)
def main(cfg: DictConfig) -> None:

    # Get the folder with all the training information from one folder. This allows to have everything consistent
    training_folder = cfg.training_folder
    print("Training folder:", training_folder,flush=True)
    # override config with config.yaml file found in the training folder
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(os.path.join(training_folder, ".hydra", "config.yaml")))

    # just one batch, necessary to load the data.
    cfg.experiment.dataset.max_batches = 1
    cfg.experiment.dataset.with_sampler = False
    # not needed here
    cfg.experiment.dataset.plot_grid = False


    # print out the whole config for debugging
    print("Full config:")
    print(OmegaConf.to_yaml(cfg),flush=True)


    torch.manual_seed(cfg.seed)
    # find latest checkpoint if not specified
    ckpt_folder = os.path.join(training_folder, "checkpoints")
    pattern = re.compile(r"sampler_epoch(\d+)\.pt")

    max_file = None
    max_epoch = -1

    for fname in os.listdir(ckpt_folder):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                max_file = fname

    if max_file:
        ckpt_path = Path(os.path.join(ckpt_folder, max_file))
        print("Using latest NF model:", ckpt_path)
    else:
        print(f"No checkpoints found with pattern {pattern} in {ckpt_folder}") 

    # ckpt_path = Path(os.path.join(ckpt_folder, "last_model.pth"))

    # Only write to a user-specified directory. Do not auto-create timestamped folders.
    if not cfg.save_dir:
        save_dir = training_folder + "/samples/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        save_dir = cfg.save_dir
    out_dir = Path(save_dir).expanduser()
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    # # dummy img to test the output folder
    # dummy_img_path = img_dir / "test.png"
    # plt.figure()
    # plt.plot([0,1,2], [0,1,0])
    # plt.savefig(dummy_img_path)
    # plt.close()
    # print(f"Test image saved to {dummy_img_path}")

    print(f"device: {cfg.device}",flush=True)

    # Load translator file for MCMC chain
    print(f"Loading translator file {cfg.mcmc_translator}...",flush=True)
    nf_translator = ROOT.TFile.Open(cfg.mcmc_translator)
    translator_array = nf_translator.Get("xsec_param_names")
    translator_array = [str(translator_array.At(i)) for i in range(translator_array.GetEntries())]
    print(f"Translator array has {len(translator_array)} entries.",flush=True)
    for i, name in enumerate(translator_array):
        print(f"  {i}: {name}",flush=True)



    dataset = instantiate(cfg.experiment.dataset)
    phase_dims = dataset.phase_space_dim
    dim_spline = len(phase_dims)
    nf_names = [dataset.titles[i].split("/")[-1] for i in range(cfg.experiment.model.total_dim)]
    

    # check correct functioning of the mapping function
    if cfg.mcmc_chain is not None:
            # open root file and read the tree
            f_mcmc = ROOT.TFile.Open(cfg.mcmc_chain)
            tree = f_mcmc.Get("posteriors")
            mcmc_entries = int(tree.GetEntries())
            # Build list of branches to read and precompute mapping
            all_branches = [br.GetName() for br in tree.GetListOfBranches()]
            for b in all_branches:
                idx, name = index_mcmc_to_nf(b, translator_array, nf_names)
                if idx is not None:
                    print(f"Mapping MCMC branch {b} to NF index {idx} ({name})")
                else:
                    print(f"Skipping MCMC branch {b} ({name})")
    
    # now print nf names with their indices
    print("NF parameter names and their indices:")
    for i, name in enumerate(nf_names):
        print(f"  {i}: {name}")


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
    print(f"Building NF model...",flush=True)
    model = build_model(base, 
                        flows, 
                        dataset, 
                        cfg.experiment.model.context_transform, 
                        cfg.experiment.model.freeze_covflow,
                        n_context_flows=cfg.experiment.model.n_context_flows,
                        hidden_dim=cfg.experiment.model.hidden_dim,
                        n_hidden_layers=cfg.experiment.model.n_hidden_layers
                        )
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
    model = model.to(cfg.device).eval()

    print(f"Built NF model. ",flush=True)

    print(f"Sampling {cfg.num_samples} events from NF model...",flush=True)    
    # Sample from NF
    batches = math.ceil(cfg.num_samples / cfg.batch_size)
    samples_nf, logqs = [], []
    start = time.time()
    with torch.no_grad():
        for _ in range(batches):
            z, logq = model.sample(cfg.batch_size)
            samples_nf.append(z.cpu().numpy())
            if cfg.return_probs:
                logqs.append(logq.cpu().numpy())
    samples_nf = np.concatenate(samples_nf, 0)[: cfg.num_samples]
    if cfg.return_probs:
        logqs = np.concatenate(logqs, 0)[: cfg.num_samples]

    samples_nf = dataset.transform_eigen_space_to_data_space(torch.from_numpy(samples_nf)).numpy()
    # Prepare optional weights for NF histogram
    weights_nf = None
    if cfg.return_probs:
        w = np.asarray(logqs)
        if w.ndim > 1:
            w = w.reshape(-1)
        # Ensure weights length matches number of NF samples
        weights_nf = w[: samples_nf.shape[0]]


    print(f"Sampled {len(samples_nf)} events from NF in {time.time()-start:.1f} seconds",flush=True)


    # Sample from gaussian (postfit covariance matrix)
    # Load llh interface
    likelihood_sampler = LikelihoodSampler(
            config_file=cfg.experiment.dataset.llh_config,
            override_files=cfg.experiment.dataset.llh_overrides,
            data_is_asimov=cfg.experiment.dataset.data_is_asimov,
            threads=cfg.experiment.sampler.threads,
            llh_cwd=cfg.experiment.dataset.llh_cwd
        ) # random seed not used in this mode (throwing outside gundam)

    print("Gundam likelihood interface initialized.",flush=True)
    bestfit_parameter_values = likelihood_sampler.postfit_parameter_values
    postfit_covariance = likelihood_sampler.postfit_covariance_matrix

    # Sample from gaussian (postfit covariance matrix)
    print(f"Sampling from covariance matrix for {cfg.num_samples} samples...",flush=True)
    start_time = time.time()
    gaus_throws = np.random.multivariate_normal(mean=bestfit_parameter_values, cov=postfit_covariance, size=cfg.num_samples)
    end_time = time.time()
    print(f"Done sampling from covariance matrix in {end_time - start_time:.2f} seconds.",flush=True)

    # scan through the nf samples, compute the nll and compare it to the nf weight (weights_nf)
    iter = 0
    reweight_nf_to_lh = []
    for nf_vector, weight in zip(samples_nf, weights_nf):
        nll,nll_stat,nll_syst = likelihood_sampler.inject_params_and_compute_likelihood(nf_vector,extend_continue=False)
        # print(f"iter {iter} NLL: {nll}, (log q_nf): {weight}", flush=True)
        iter += 1
        reweight_nf_to_lh.append(weight - nll)

    # Normalize reweighting factors
    if reweight_nf_to_lh:
        median_reweight = np.median(reweight_nf_to_lh)
        reweight_nf_to_lh = (np.array(reweight_nf_to_lh)-median_reweight)
        print(f"Reweighting factors (NF to LH): {reweight_nf_to_lh}")

    # Sample from MCMC chain
    if cfg.mcmc_chain is not None:
        # open root file and read the tree
        f_mcmc = ROOT.TFile.Open(cfg.mcmc_chain)
        tree = f_mcmc.Get("posteriors")
        mcmc_entries = int(tree.GetEntries())
        n_take = int(min(cfg.num_samples, mcmc_entries))
        print(f"MCMC chain has {mcmc_entries} entries. Sampling {n_take} unique entries.", flush=True)

        # Build list of branches to read and precompute mapping
        all_branches = [br.GetName() for br in tree.GetListOfBranches()]

        branch_map = {}
        for b in all_branches:
            idx, name = index_mcmc_to_nf(b, translator_array, nf_names)
            if idx is not None:
                branch_map[b] = (idx, name)
        selected_branches = list(branch_map.keys())
        print(f"Using {len(selected_branches)}/{len(all_branches)} branches after mapping.", flush=True)

        # Restrict IO to just the needed branches
        tree.SetBranchStatus("*", 0)
        for b in selected_branches:
            tree.SetBranchStatus(b, 1)

        if cfg.randomize_mcmc:
            # Sample unique entry indices once
            rng = np.random.default_rng(cfg.seed)
            indices = rng.choice(mcmc_entries, size=n_take, replace=False)
            # sort them to minimize random disk access
            indices = np.sort(indices)
        else:
            indices = np.arange(n_take)
            print(f"Taking first {n_take} entries from MCMC chain (no randomization).", flush=True)
        # Pre-allocate storage per branch
        mcmc_data = {b: np.empty(n_take, dtype=float) for b in selected_branches}

        print("Reading selected entries from MCMC TTree (might take some time)...", flush=True)
        t0 = time.time()
        for i, entry in enumerate(indices):
            tree.GetEntry(int(entry))
            # print(f"{i}: Reading entry {entry}")
            for b in selected_branches:
                mcmc_data[b][i] = getattr(tree, b)
            if (i + 1) % max(1, n_take // 10) == 0:
                print(f"  {i + 1}/{n_take} entries read", flush=True)
        print(f"Done reading in {time.time() - t0:.2f}s", flush=True)

        # Plotting function
        def plot_one(branch_name: str):
            index_nf, meaningful_name = branch_map[branch_name]
            mcmc_values = mcmc_data[branch_name]
            nf_values = samples_nf[:, index_nf]
            gaus_values = gaus_throws[:, index_nf]

            fig = plt.figure(figsize=(6, 4))
            # unify bin width for the three histograms
            bin_width =  (max(mcmc_values.max(), nf_values.max(), gaus_values.max()) - min(mcmc_values.min(), nf_values.min(), gaus_values.min())) / 50
            bins = np.arange(min(mcmc_values.min(), nf_values.min(), gaus_values.min()), max(mcmc_values.max(), nf_values.max(), gaus_values.max()) + bin_width, bin_width)
            plt.hist(mcmc_values, bins=bins, histtype='step', label='MCMC', color='blue')
            # Plot weighted NF only if weights are available and correctly shaped
            # if (weights_nf is not None) and (np.size(weights_nf) == nf_values.shape[0]):
            #     w = weights_nf
            #     if hasattr(w, "ndim") and w.ndim > 1:
            #         w = w.reshape(-1)
            #     plt.hist(nf_values, bins=bins, histtype='step', color='red', weights=w, label='NF (weighted)')
            plt.hist(nf_values, bins=bins, histtype='step', color='black', label='NF (unweighted)', alpha=0.5)
            plt.hist(gaus_values, bins=bins, histtype='step', color='green', label='Gaussian', alpha=0.7)
            plt.legend()
            plt.xlabel(meaningful_name)
            plt.ylabel("a.u.")
            plt.title(f"Marginal of {meaningful_name}    Entries: {len(mcmc_values)}")
            plt.grid()
            out_path = img_dir / f"MCMC_marginal_{meaningful_name}.png"
            plt.savefig(out_path)
            plt.close(fig)
            return meaningful_name

        # Parallelize plotting (reading remains single-threaded)
        # Resolve plotting worker count from config if present
        cfg_workers = None
        try:
            cfg_workers = int(cfg.plot_workers) or 0
        except Exception:
            cfg_workers = 0
        print("Plotting marginals in parallel...", flush=True)
        count_marginals = 0
        max_workers = cfg_workers if cfg_workers and cfg_workers > 0 else min(8, os.cpu_count())
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(plot_one, b): b for b in selected_branches}
            for fut in as_completed(futures):
                name = fut.result()
                count_marginals += 1
                if count_marginals % 20 == 0:
                    print(f"  Plotted {count_marginals}/{len(selected_branches)}", flush=True)
        print(f"Plotted {count_marginals} marginals.", flush=True)


if __name__ == "__main__":
    main()