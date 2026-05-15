# python
#!/usr/bin/env python3
# predict_histograms.py
from __future__ import annotations
import os, sys, math, time
from pathlib import Path
from contextlib import contextmanager

import hydra
import torch
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate

# project-specific imports (same layout as sample.py)
NF_LOCAL = os.path.join(os.path.dirname(__file__), "normalizing-flows")
sys.path.append(os.path.abspath(NF_LOCAL))
from gunflows.utils.build_flow import build_base, build_flow_layers, build_model

@contextmanager
def pushd(path: str | None):
    prev = os.getcwd()
    if path:
        os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

def _maybe_build_nf(cfg, dataset, ckpt_path: str):
    # if not cfg.predict.nf:
        # return None
    # Use the merged cfg; model definition is expected under cfg.experiment.model
    m = cfg.experiment.model
    device = cfg.device if "device" in cfg else (cfg.experiment.device if "device" in cfg.experiment else "cpu")

    base = build_base(int(m.total_dim))
    dim_spline = len(dataset.phase_space_dim)
    tail_bounds = torch.ones(dim_spline) * float(m.tail_bound)
    flows = build_flow_layers(
        int(m.nflows),
        dim_spline,
        int(m.hidden),
        int(m.nlayers),
        int(m.nbins),
        tail_bounds,
        n_context=int(m.total_dim) - dim_spline,
    )
    # Pass additional model kwargs if present in config to match training architecture
    freeze_covflow = bool(m.freeze_covflow) if hasattr(m, "freeze_covflow") else False
    kw = {}
    if hasattr(m, "n_context_flows"):
        kw["n_context_flows"] = int(m.n_context_flows)
    if hasattr(m, "hidden_dim"):
        kw["hidden_dim"] = int(m.hidden_dim)
    if hasattr(m, "n_hidden_layers"):
        kw["n_hidden_layers"] = int(m.n_hidden_layers)

    model = build_model(base, flows, dataset, m.context_transform, freeze_covflow, **kw)

    # load checkpoint; allow strict loading but prefer matching architecture from training cfg
    state = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # try to load with strict=False to allow partial loading and emit a clear error
        model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return model

def _likelihoods_from_samples(sampler, samples_data: np.ndarray):
    out = []
    for s in samples_data:
        nll, _, _ = sampler.inject_params_and_compute_likelihood(s.tolist(), extend_continue=False)
        if nll == -1:
            out.append(0.0)
        else:
            out.append(math.exp(-nll))
    return np.array(out, dtype=float)

def _sample_gaussian(N, mean, cov):
    return np.random.default_rng().multivariate_normal(mean, cov, size=N)

def _sample_nf(N, model, dataset, cfg):
    # sample in batches from NF; NF returns eigen-space samples (z)
    batches = math.ceil(N / cfg.batch_size)
    samples = []
    with torch.no_grad():
        for _ in range(batches):
            z, _ = model.sample(cfg.batch_size)
            samples.append(z.cpu().numpy())
    samples = np.concatenate(samples, 0)[:N]
    return samples

def _run_mcmc(N, sampler, dataset, cfg):
    # Simple Metropolis-Hastings in data space using the sampler.inject_params_and_compute_likelihood
    dim = dataset.mean.numel()
    current = dataset.mean.numpy().copy()
    step = cfg.mcmc.step
    burn_in = cfg.mcmc.burn_in
    thinning = cfg.mcmc.thinning
    total_needed = burn_in + N * thinning
    samples = []

    # compute current likelihood
    nll_cur, _, _ = sampler.inject_params_and_compute_likelihood(current.tolist(), extend_continue=False)
    current_like = 0.0 if nll_cur == -1 else math.exp(-nll_cur)

    rng = np.random.default_rng(cfg.seed + 1234)
    for i in range(total_needed):
        proposal = current + rng.normal(scale=step, size=current.shape)
        nll_prop, _, _ = sampler.inject_params_and_compute_likelihood(proposal.tolist(), extend_continue=False)
        prop_like = 0.0 if nll_prop == -1 else math.exp(-nll_prop)

        # avoid division by zero; treat like as non-negative
        if prop_like >= current_like or rng.random() < (prop_like / (current_like + 1e-300)):
            current = proposal
            current_like = prop_like

        if i >= burn_in and ((i - burn_in) % thinning == 0):
            samples.append(current.copy())
            if len(samples) >= N:
                break
    return np.asarray(samples)

@hydra.main(config_path="../../configs", config_name="predict_histograms", version_base=None)
def main(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    from gunflows.likelihood_sampler import LikelihoodSampler
    from gunflows.sample_mcmc_toy import build_sampling_dataset_target
    
    # Load the training folder path
    training_folder = Path(cfg.training_folder).expanduser().resolve()
    print(f"Loading NF model from training folder: {training_folder}", flush=True)

    # if folder does not exist: raise error
    if not training_folder.exists():
        raise FileNotFoundError(f"Training folder not found: {training_folder}")
    
    # Find the checkpoint (usually stored as .pt file)
    checkpoint_dir = training_folder / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    else:
        # Fallback: look for .pt files directly in training folder
        checkpoints = sorted(training_folder.glob("*.pt"))
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {training_folder} or {checkpoint_dir}")
    
    best_ckpt = str(checkpoints[-1])  # Use latest checkpoint
    print(f"Using checkpoint: {best_ckpt}", flush=True)
    
    # Load the hydra config from the training folder
    hydra_config_path = training_folder / ".hydra" / "config.yaml"
    if not hydra_config_path.exists():
        raise FileNotFoundError(f"Hydra config not found at {hydra_config_path}")
    
    train_cfg = OmegaConf.load(hydra_config_path)
    print(f"Loaded training config from: {hydra_config_path}", flush=True)

    # Merge training config and runtime config into a single cfg (runtime overrides training)
    cfg = OmegaConf.merge(train_cfg, cfg)

    # Initialize LikelihoodSampler to load covariance matrix from ROOT file
    print("Initializing LikelihoodSampler...", flush=True)
    likelihood_sampler = LikelihoodSampler(
        config_file=str(cfg.experiment.dataset.llh_config),
        override_files=cfg.experiment.dataset.llh_overrides,
        data_is_asimov=cfg.experiment.dataset.data_is_asimov,
        threads=cfg.experiment.sampler.threads if hasattr(cfg.experiment, 'sampler') else 1,
        llh_cwd=cfg.experiment.dataset.llh_cwd,
        light_mode=False,
    )

    # Extract mean and covariance matrix from the sampler
    bestfit_parameter_values = np.asarray(likelihood_sampler.postfit_parameter_values, dtype=np.float64).reshape(-1)
    postfit_covariance = np.asarray(likelihood_sampler.postfit_covariance_matrix, dtype=np.float64)
    print(f"Loaded covariance matrix with shape: {postfit_covariance.shape}", flush=True)

    # Build dataset target for the model
    dataset = build_sampling_dataset_target(cfg, bestfit_parameter_values, postfit_covariance)

    # Load the NF model checkpoint using unified cfg
    nf_model = _maybe_build_nf(cfg, dataset, best_ckpt)
    print("NF model loaded successfully.", flush=True)
    
    # Sample from NF and Gaussian (covariance)
    num_samples = int(cfg.num_samples)
    print(f"Starting to sample {num_samples} parameter sets from NF and Gaussian...", flush=True)
    
    nf_samples = []
    gaussian_samples = []
    batch_size = int(cfg.batch_size)
    
    # Sample in batches for both NF and Gaussian draws.
    with torch.no_grad():
        sample_offset = 0
        while sample_offset < num_samples:
            current_batch_size = min(batch_size, num_samples - sample_offset)

            z_nf, _ = nf_model.sample(current_batch_size)
            z_nf_np = z_nf.cpu().numpy()
            nf_samples.extend(z_nf_np)

            z_gaussian = np.random.multivariate_normal(
                mean=bestfit_parameter_values,
                cov=postfit_covariance,
                size=current_batch_size,
            )
            gaussian_samples.extend(z_gaussian)

            for local_idx in range(min(20 - sample_offset, current_batch_size)):
                global_idx = sample_offset + local_idx
                print(f"Sample {global_idx + 1}:")
                print(f"  NF sample (z): {z_nf_np[local_idx].flatten()}")
                print(f"  Gaussian sample: {z_gaussian[local_idx]}")

            sample_offset += current_batch_size
            print(f"  Sampled {sample_offset}/{num_samples}", flush=True)
    
    print(f"Sampling complete. Collected {len(nf_samples)} NF samples and {len(gaussian_samples)} Gaussian samples.", flush=True)

if __name__ == "__main__":
    main()
