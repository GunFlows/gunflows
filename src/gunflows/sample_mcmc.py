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

# check the validity of a single throw
def check_parameters_limits(param_vector, limits_dictionary):
    par_names = list(limits_dictionary.keys())
    limits_vector = [limits_dictionary[name] for name in par_names]
    for i, val in enumerate(param_vector):
        low, high = limits_vector[i]
        if np.isnan(low):
            low = -np.inf
        if np.isnan(high):
            high = np.inf
        if val < low or val > high:
            # print(f"-debug- parameter {par_names[i]} with value {val} out of limits {limits}")
            return False
    return True

#check the validity of an array of throws, returns a mask of booleans
def check_parameters_array_limits(array_of_param_vector, limits_dictionary):
    # returns a mask of booleans
    par_names = list(limits_dictionary.keys())
    limits_vector = [limits_dictionary[name] for name in par_names]
    mask = np.ones(array_of_param_vector.shape[0], dtype=bool) # a vector of 1s as long as the number of samples
    for i, limits in enumerate(limits_vector):
        low, high = limits
        if np.isnan(low):
            low = -np.inf
        if np.isnan(high):
            high = np.inf
        vals = array_of_param_vector[:, i]
        mask &= (vals >= low) & (vals <= high)
        # print(f"    -debug- {i} mask sum: {mask.sum()} after checking limits {limits} for parameter {par_names[i]}")
        # print(f"         -debug- {vals} ")
    return mask

def sample_check_append(batch_size, sample_from_nf, model, dataset, parameter_limits, samples, return_probs, logqs, mean, cov):
    # Ensure consistent tensor/ndarray types and explicit CPU placement.
    if sample_from_nf and model is not None:
        # draw a batch (model.sample returns torch tensors)
        z_batch, lq_batch = model.sample(batch_size)
        # move to CPU and detach from autograd
        z_batch = z_batch.detach().to(dtype=torch.float32, device="cpu")
        # make sure log-probs become a 1-D numpy array
        lq_np = lq_batch.detach().to(dtype=torch.float32, device="cpu").cpu().numpy()
        lq_np = np.asarray(lq_np).reshape(-1)
        # transform entire batch to physical/data space using dataset helper (expects a torch.Tensor)
        phys_batch = dataset.transform_eigen_space_to_data_space(z_batch)
        phys_np = phys_batch.detach().cpu().numpy().astype(np.float32)
    else:
        # sample from covariance matrix -- coerce mean/cov to CPU torch tensors
        mean_t = torch.as_tensor(mean, dtype=torch.float32, device="cpu")
        cov_t = torch.as_tensor(cov, dtype=torch.float32, device="cpu")
        const = mean_t.shape[0] * np.log(2.0 * np.pi)
        z = torch.randn((batch_size, mean_t.shape[0]), dtype=torch.float32, device="cpu")
        L_phys = torch.linalg.cholesky(cov_t)
        phys_t = z @ L_phys.T + mean_t
        phys_np = phys_t.detach().cpu().numpy().astype(np.float32)
        if return_probs:
            # compute log-probabilities in tensor space, then convert
            std = torch.sqrt(torch.diag(cov_t))
            Dinv = torch.diag(1.0 / std)
            cov_std = Dinv @ cov_t @ Dinv
            L_std = torch.linalg.cholesky(cov_std)
            logdet_covstd = 2.0 * torch.log(torch.diag(L_std)).sum()
            diff = phys_t - mean_t
            # solve triangular system with cholesky of physical cov
            y = torch.linalg.solve_triangular(L_phys, diff.T, upper=False)
            quad = (y * y).sum(dim=0)
            lq_np = (0.5 * (quad + const + logdet_covstd)).detach().cpu().numpy().astype(np.float32)

    # mask is a numpy boolean array (check_parameters_array_limits expects numpy input)
    mask = check_parameters_array_limits(phys_np, parameter_limits)

    model_name = "NF" if sample_from_nf else "Gaussian"
    print(f" Sampled batch of {batch_size} from {model_name}. Accepting {np.sum(mask)}/{batch_size} samples within physical limits.", flush=True)
    accepted_idx = np.nonzero(mask)[0]
    take = min(len(accepted_idx), batch_size)
    # append accepted samples (up to needed)
    for idx in accepted_idx[:take]:
        samples.append(phys_np[idx])
        if return_probs:
            # lq_np should be a 1-D numpy array (handled above), guard indexing
            logqs.append(float(np.asarray(lq_np).reshape(-1)[idx]))

    return take
            


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

    # Load llh interface
    likelihood_sampler = LikelihoodSampler(
            config_file=cfg.experiment.dataset.llh_config,
            override_files=cfg.experiment.dataset.llh_overrides,
            data_is_asimov=cfg.experiment.dataset.data_is_asimov,
            threads=cfg.experiment.sampler.threads,
            llh_cwd=cfg.experiment.dataset.llh_cwd,
            light_mode=False
        ) # random seed not used in this mode (throwing happens outside lh_sampler)

    print("Gundam likelihood interface initialized.",flush=True)
    bestfit_parameter_values = likelihood_sampler.postfit_parameter_values
    postfit_covariance = likelihood_sampler.postfit_covariance_matrix
    parameter_names = likelihood_sampler.get_parameter_names()
    # obtain parameters limits from likelihood sampler
    parameter_limits = {}
    parameter_limits_vector = []
    for name in parameter_names:
        limits = likelihood_sampler.get_parameter_limits(name) # USE THIS!
        physical_limits = likelihood_sampler.get_parameter_physical_range(name)
        parameter_limits[name] = limits
        parameter_limits_vector.append(limits)
        print(f"  {name}: {limits} - {physical_limits}",flush=True)
    print("Parameter limits obtained from likelihood sampler.",flush=True)


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
    # Sample from NF (vectorized): sample batches at once, filter vectorially
    batches = math.ceil(cfg.num_samples / cfg.batch_size)
    samples_nf, logqs = [], []
    start = time.time()
    need_total = int(cfg.num_samples)
    with torch.no_grad():
        while len(samples_nf) < need_total:
            need = need_total - len(samples_nf)
            b = min(int(cfg.batch_size), need)
            print(f" NF sampling. {need} throws to go. Sampling {b} now ...", flush=True)            
            remain = b
            while remain > 5:
                take = sample_check_append(sample_from_nf=True, batch_size=remain, model=model, dataset=dataset, parameter_limits=parameter_limits, samples=samples_nf, return_probs=cfg.return_probs, logqs=logqs, mean=None, cov=None)
                remain -= take
                if take == 0:
                    break  # avoid infinite loop if no samples accepted
            # for any remaining samples not accepted, fall back to single-sample retry
            if remain > 0:
                print(f" Need {remain} more samples after batch filtering. Sampling individually...", flush=True)
                for _ in range(remain):
                    z, logq = model.sample(1)
                    z = z.to('cpu')
                    phys_z = dataset.transform_eigen_space_to_data_space(z).detach().cpu().numpy()[0]
                    if cfg.return_probs:
                        logq_np = float(logq.detach().cpu().numpy()[0])
                    else:
                        logq_np = None
                    while not check_parameters_limits(phys_z, parameter_limits):
                        # print(f"  -debug- single sample not physical, resampling...", flush=True)
                        z, logq = model.sample(1)
                        z = z.to('cpu')
                        phys_z = dataset.transform_eigen_space_to_data_space(z).detach().cpu().numpy()[0]
                        if cfg.return_probs:
                            logq_np = float(logq.detach().cpu().numpy()[0])
                    samples_nf.append(phys_z)
                    if cfg.return_probs:
                        logqs.append(logq_np)
                    print(f" Sampled individual throw. {_+1}/{remain}", flush=True)
            print(f"Total samples collected: {len(samples_nf)}/{need_total}", flush=True)

    # samples_nf already contains physical-space numpy arrays (appended above);
    # just convert the list to a single NumPy array instead of transforming again.
    samples_nf = np.asarray(samples_nf)

    # Prepare optional weights for NF histogram
    logq_nf = None
    if cfg.return_probs:
        w = np.asarray(logqs)
        if w.ndim > 1:
            w = w.reshape(-1)
        # Ensure weights length matches number of NF samples
        logq_nf = w[: samples_nf.shape[0]]


    print(f"Sampled {len(samples_nf)} events from NF in {time.time()-start:.1f} seconds",flush=True)

    # Sample from gaussian (postfit covariance matrix)
    print(f"Sampling from covariance matrix for {cfg.num_samples} samples...",flush=True)
    start_time = time.time()

    gaus_throws = []
    gaus_logqs = []
    while len(gaus_throws) < need_total:
            need = need_total - len(gaus_throws)
            b = min(int(cfg.batch_size), need)
            print(f" Gaussian sampling. {need} throws to go. Sampling {b} now ...", flush=True)            
            remain = b
            while remain > 5:
                take = sample_check_append(sample_from_nf=False, batch_size=remain, parameter_limits=parameter_limits, 
                                           model=None, dataset=None, 
                                           mean=bestfit_parameter_values, cov=postfit_covariance, 
                                           samples=gaus_throws, return_probs=True, logqs=gaus_logqs)
                remain -= take
                if take == 0:
                    break  # avoid infinite loop if no samples accepted
            # for any remaining samples not accepted, fall back to single-sample retry
            if remain > 0:
                print(f" Need {remain} more samples after batch filtering. Sampling individually...", flush=True)
                for _ in range(remain):
                    phys_z = np.random.multivariate_normal(mean=bestfit_parameter_values, cov=postfit_covariance, size=1)[0]
                    while not check_parameters_limits(phys_z, parameter_limits):
                        # print(f"  -debug- single sample not physical, resampling...", flush=True)
                        phys_z = np.random.multivariate_normal(mean=bestfit_parameter_values, cov=postfit_covariance, size=1)[0]
                    gaus_throws.append(phys_z)
                    print(f" Sampled individual throw. {_+1}/{remain}", flush=True)
            print(f"Total samples collected: {len(gaus_throws)}/{need_total}", flush=True)

    
    gaus_throws = np.asarray(gaus_throws)
    end_time = time.time()
    print(f"Done sampling from covariance matrix in {end_time - start_time:.2f} seconds.",flush=True)

    # scan through the nf samples, compute the nll and compare it to the nf weight (logq_nf)
    iter = 0
    reweight_nf_to_lh = []
    print("Computing reweighting factors from NF to LH...",flush=True)
    for nf_vector, logq in zip(samples_nf, logq_nf):
       logp,nll_stat,nll_syst = likelihood_sampler.inject_params_and_compute_likelihood(nf_vector,extend_continue=False)
       if (iter % max(1, cfg.num_samples // 100) == 0):
            print(f"iter {iter} NLL: {logp}, log_q_nf: {logq}", flush=True)
       iter += 1
       reweight_nf_to_lh.append(logq - logp)
    print(f"Computed reweighting factors for {len(reweight_nf_to_lh)} NF samples.",flush=True)

    # Normalize reweighting factors
    # if reweight_nf_to_lh:
    #     median_reweight = np.median(reweight_nf_to_lh)
    #     reweight_nf_to_lh = (np.array(reweight_nf_to_lh)-median_reweight)
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
            print(f"Taking {n_take} random unique entries from MCMC chain.", flush=True)
            # Sample unique entry indices once
            rng = np.random.default_rng(cfg.seed)
            indices = rng.choice(mcmc_entries, size=n_take, replace=False)
            # sort them to minimize random disk access
            indices = np.sort(indices)
        else:
            indices = np.arange(mcmc_entries - n_take, mcmc_entries)
            print(f"Taking last {n_take} entries from MCMC chain (no randomization).", flush=True)
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

        # NF samples:
        ## samples_nf[:, index_nf]
        # NF weights:
        ## reweight_nf_to_lh
        # Gaussian samples:
        ## gaus_throws[:, index_nf]
        # MCMC samples:
        ## mcmc_data[branch_name]

        # Plotting function
        print("Finally, plotting marginals...", flush=True)
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
            # if (logq_nf is not None) and (np.size(logq_nf) == nf_values.shape[0]):
            #     w = logq_nf
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
