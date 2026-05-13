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

def _maybe_build_nf(cfg, dataset):
    if not cfg.predict.nf:
        return None
    base = build_base(cfg.model.total_dim)
    dim_spline = len(dataset.phase_space_dim)
    tail_bounds = torch.ones(dim_spline) * cfg.model.tail_bound
    flows = build_flow_layers(
        cfg.model.nflows,
        dim_spline,
        cfg.model.hidden,
        cfg.model.nlayers,
        cfg.model.nbins,
        tail_bounds,
        n_context=cfg.model.total_dim - dim_spline,
    )
    model = build_model(base, flows, dataset, cfg.model.context_transform)
    model.load_state_dict(torch.load(cfg.ckpt, map_location=cfg.device))
    model = model.to(cfg.device).eval()
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
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset = instantiate(cfg.dataset)
    phase_dims = dataset.phase_space_dim
    mean_eig = dataset.mean.numpy()  # in data-space or eigen-space depending on dataset

    # Determine eigen-space mean/cov:
    try:
        cov_eig = dataset.get_true_cov().numpy()
    except Exception:
        cov_eig = np.eye(len(phase_dims))

    # Build NF model if requested
    nf_model = _maybe_build_nf(cfg, dataset)

    # Load likelihood sampler immediately (required)
    if not getattr(cfg, "likelihood", None) or not cfg.likelihood.get("config", None):
        raise RuntimeError("A likelihood configuration must be provided at cfg.likelihood.config")
    from gunflows.likelihood_sampler.likelihoodSampler import LikelihoodSampler
    lh_cfg = cfg.likelihood
    cwd = lh_cfg.get("cwd", None)
    with pushd(cwd or os.path.dirname(lh_cfg.config)):
        sampler = LikelihoodSampler(
            lh_cfg.config,
            override_files=lh_cfg.get("overrides", []),
            threads=lh_cfg.get("threads", 1),
            data_is_asimov=lh_cfg.get("asimov", True),
            seed=cfg.seed,
        )

    N = int(cfg.predict.n_samples)

    # For gaussian and NF the sampling is done in eigen space and then transformed to data space
    if cfg.predict.gaussian:
        z = _sample_gaussian(N, mean=np.zeros(len(phase_dims)), cov=cov_eig)
        # if dataset expects eigen-space input for transform:
        try:
            samples_data = dataset.transform_eigen_space_to_data_space(torch.from_numpy(z)).numpy()
        except Exception:
            # if transform not available, assume mean_eig is already data-space and z are data-space
            samples_data = z
        likes = _likelihoods_from_samples(sampler, samples_data)
        for i, L in enumerate(likes):
            print(f"gaussian\t{L:.6e}")

    if cfg.predict.nf:
        if nf_model is None:
            raise RuntimeError("NF sampling requested but model could not be built. Check cfg.")
        z = _sample_nf(N, nf_model, dataset, cfg)
        try:
            samples_data = dataset.transform_eigen_space_to_data_space(torch.from_numpy(z)).numpy()
        except Exception:
            samples_data = z
        likes = _likelihoods_from_samples(sampler, samples_data)
        for i, L in enumerate(likes):
            print(f"nf\t{L:.6e}")

    if cfg.predict.mcmc:
        # MCMC operates in data-space here and uses the sampler directly.
        samples_data = _run_mcmc(N, sampler, dataset, cfg)
        likes = _likelihoods_from_samples(sampler, samples_data)
        for i, L in enumerate(likes):
            print(f"mcmc\t{L:.6e}")

if __name__ == "__main__":
    main()
