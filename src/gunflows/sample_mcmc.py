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
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra.utils import instantiate
from matplotlib.colors import LogNorm

NF_LOCAL = os.path.join(os.path.dirname(__file__), "..", "normalizing-flows")
sys.path.append(os.path.abspath(NF_LOCAL))

from gunflows.utils.build_flow import build_base, build_flow_layers, build_model

import ROOT # to read the MCMC chain ROOT file



def index_mcmc_to_nf(mcmc_branch_name, translator_array, nf_names):
    # find the corresponding index in the NF samples
    if mcmc_branch_name.startswith("ndd_"):
        return int(mcmc_branch_name.split("_")[1]) + 100, f"DetSyst_{int(mcmc_branch_name.split('_')[1])}" # det syst start at 100 in the NF samples
    elif mcmc_branch_name.startswith("xsec_"):
        parameter_name = translator_array[int(mcmc_branch_name.split("_")[1])]
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
                    print(f" -> Mapping {parameter_name} to {nf_name}")
                    return i, pattern.match(nf_name).group(1)
    # print(f"Warning: unrecognized MCMC branch name {mcmc_branch_name}.")
    return None, None            


@hydra.main(config_path="../../configs", config_name="sample", version_base=None)
def main(cfg: DictConfig) -> None:

    # Get the folder with all the training information from one folder. This allows to have everything consistent
    training_folder = cfg.training_folder
    print("Training folder:", training_folder)
    # override config with config.yaml file found in the training folder
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(os.path.join(training_folder, ".hydra", "config.yaml")))

    # just one batch, necessary to load the data.
    # TODO: use sampler, so that the GUNDAM LH interace is loaded
    cfg.experiment.dataset.max_batches = 1
    cfg.experiment.dataset.with_sampler = False
    # not needed here
    cfg.experiment.dataset.plot_grid = False


    # print out the whole config for debugging
    print("Full config:")
    print(OmegaConf.to_yaml(cfg))


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

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = (
        Path(cfg.save_dir).expanduser()
        if cfg.save_dir is not None
        else ckpt_path.parent.parent / "samples" / "test"  #replace "test" with ts in the official version
    )
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    # # dummy img to test the output folder
    # dummy_img_path = img_dir / "test.png"
    # plt.figure()
    # plt.plot([0,1,2], [0,1,0])
    # plt.savefig(dummy_img_path)
    # plt.close()
    # print(f"Test image saved to {dummy_img_path}")

    print(f"device: {cfg.device}")



    dataset = instantiate(cfg.experiment.dataset)
    phase_dims = dataset.phase_space_dim
    dim_spline = len(phase_dims)
    nf_names = [dataset.titles[i].split("/")[-1] for i in range(cfg.experiment.model.total_dim)]
    


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
    print(f"Building NF model...")
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

    print(f"Built NF model. ")
    
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

    print(f"Sampled {len(samples_nf)} events from NF in {time.time()-start:.1f} seconds")

    # Load translator file
    print(f"Loading translator file {cfg.mcmc_translator}...")
    nf_translator = ROOT.TFile.Open(cfg.mcmc_translator)
    translator_array = nf_translator.Get("xsec_param_names")
    translator_array = [str(translator_array.At(i)) for i in range(translator_array.GetEntries())]
    print(f"Translator array has {len(translator_array)} entries.")

    # Check the matching between MCMC xsec_0..73 and NF parameters
    n_matches = 0
    for i in range(74):
        parameter_name = translator_array[i]
        # handle special cases
        if parameter_name == "EB_dial_O_nubar":
            parameter_name = "EB_bin_O_nubar"
        if parameter_name == "EB_dial_O_nu":
            parameter_name = "EB_bin_O_nu"
        if parameter_name == "EB_dial_C_nubar":
            parameter_name = "EB_bin_C_nubar"
        if parameter_name == "EB_dial_C_nu":
            parameter_name = "EB_bin_C_nu"
        # print(f"Mapping for MCMC xsec_{i}:")
        matches = 0
        for nf_name in nf_names:
            pattern = re.compile(r"#\d+_(.*)")
            if pattern.match(nf_name) is not None:
                if parameter_name == pattern.match(nf_name).group(1):
                    print(f"Mapping {translator_array[i]} to {nf_name}")
                    matches += 1
        if matches == 0:
            print(f"  No matches found for {translator_array[i]}")
        elif matches > 1:
            print("  Warning: multiple matches found!")
        if matches == 1:
            n_matches += 1
    
    print(f"Total of {n_matches} unique matches found between MCMC xsec_0..73 and NF parameters.")
            

    # Sample from MCMC chain
    if cfg.mcmc_chain is not None:
        # open root file and read the tree
        f_mcmc = ROOT.TFile.Open(cfg.mcmc_chain)
        tree = f_mcmc.Get("posteriors")
        mcmc_entries = tree.GetEntries()
        print(f"MCMC chain has {mcmc_entries} entries. Requested {cfg.num_samples} samples.")
        # tree structure: each branch is one variable.
        # ndd_0..ndd_551 are the detector syst. 
        # xsec_0..xsec_174 are the cross-section AND flux systematics (xsec until 73 (included), flux from 74 to 173)
        # debug: just draw the marginal of each variable
        branch_names = [br.GetName() for br in tree.GetListOfBranches()]
        count_marginals = 0
        for branch_name in branch_names:
            if branch_name == "xsec_174":
                print("I don't know what to map xsec_174 to.")
                continue
            mcmc_values = []
            for i in range( min(cfg.num_samples, mcmc_entries) ):
                # take a random entry
                tree.GetEntry(i)
                mcmc_values.append( getattr(tree, branch_name) )
            mcmc_values = np.array(mcmc_values)

            index_nf, meaningful_name = index_mcmc_to_nf(branch_name, translator_array, nf_names)
            if index_nf is None:
                print(f"Skipping {branch_name} as it has no correspondence in the NF samples.")
                continue
            nf_values = samples_nf[:, index_nf]

            plt.figure(figsize=(6,4))
            plt.hist(mcmc_values, bins=50, histtype='step')
            plt.hist(nf_values, bins=50, histtype='step', color='red')
            plt.legend(["MCMC", "NF"])
            plt.xlabel(meaningful_name)
            plt.ylabel("a.u.")
            plt.title(f"MCMC marginal of {meaningful_name}. entries: {len(mcmc_values)}")
            plt.grid()
            plt.savefig(img_dir / f"MCMC_marginal_{meaningful_name}.png")
            print(f"Saved MCMC marginal of {meaningful_name}")
            plt.close()
            count_marginals += 1
        print(f"Plotted {count_marginals} marginals.")


if __name__ == "__main__":
    main()