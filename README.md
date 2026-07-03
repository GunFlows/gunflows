# gunflows

Train and sample conditional normalizing flows for high-dimensional likelihoods.

`gunflows` is `pip install`-able and has **no dependency on GUNDAM or ROOT**. The
likelihood a flow is trained against is resolved at runtime from a dotted path
(`sampler_target`, e.g. `apps.gundam.likelihoodSampler.LikelihoodSampler`), so the
core library never imports a concrete backend. GUNDAM/ROOT is one such backend,
used for the physics analyses this repo was built for; `apps/toyllh` is a second,
GUNDAM-free backend that exists purely to demonstrate (and test) that gunflows
works against any likelihood implementing the same small interface.

## Install

```bash
pip install .          # from a checkout of this repo
```

Requires Python >= 3.9. Pulls in `torch`, `hydra-core`, `omegaconf`, `numpy`,
`scipy`, `matplotlib`, `normflows`. No GUNDAM, no ROOT, no container needed for
the core library or the toy-likelihood demo.

To run against GUNDAM instead, you additionally need a GUNDAM/ROOT environment
(see `env/` for the Apptainer image used on the cluster, and `setup_scripts/`
for baobab-specific install helpers) — `apps/gundam` is the only place in the
codebase that imports GUNDAM/ROOT directly.

## Quickstart

See `examples/toy_llh_walkthrough.ipynb` for a runnable, step-by-step notebook
(dataset → model → train → sample) using the GUNDAM-free `ToyLLH` backend.

Minimal shape of the API:

```python
from gunflows.dataset import StreamingDataset
from gunflows.utils.build_flow import build_base, build_flow_layers, build_model
from gunflows.losses.importance_losses import kl_symmetric

dataset = StreamingDataset(
    phase_space_dim=list(range(50, 60)),
    with_sampler=True,
    sampler_target="apps.toyllh.likelihood.ToyLLH",
    llh_config="toy",
)
base = build_base(dataset.ndim)
flows = build_flow_layers(nflows=8, dim_spline=10, hidden=128, nlayers=1, nbins=12,
                           tail_bounds=..., n_context=dataset.ndim - 10)
model = build_model(base, flows, dataset, device="cpu")
```

Or use the Hydra CLI entry points directly from a repo checkout:

```bash
python -m apps.train experiment=toy_llh                 # train
python -m apps.sample experiment=toy_llh ...             # sample from a checkpoint
python -m apps.mcmc likelihood.sampler_target=apps.toyllh.likelihood.ToyLLH ...
```

`apps/` (the CLI entry points, Hydra configs, and the two likelihood backends)
is not part of the pip package — it's meant to be run from a repo checkout,
importing the installed `gunflows` library. This mirrors how you'd plug in your
own likelihood: write a class matching
`gunflows.likelihood_sampler.base.LikelihoodSamplerProtocol` and point
`sampler_target` at it.

## Repo layout

```
src/gunflows/           the pip-installable library
  dataset/              StreamingDataset (background sampler workers + on-disk batches)
  flows/                SystematicFlow: CovFlow (fixed Gaussian base) + ContextFlow + spline flows
  losses/                importance-weighted forward/reverse/symmetric KL losses
  trainer/               StreamingTrainer: epoch loop, dataset refresh/re-split, NF-bootstrap staging
  likelihood_sampler/    NFSamplerProcess (background worker), MCMC engine, backend protocol
  utils/                 flow-building helpers

apps/                    Hydra CLI entry points + likelihood backends (not pip-installed)
  train.py, sample.py, mcmc.py
  gundam/                GUNDAM/ROOT-backed LikelihoodSampler (the only GUNDAM import site)
  toyllh/                GUNDAM-free demo likelihood (60 iid dims: 50 Gaussian + 10 skew-normal)

configs/                 Hydra config groups (dataset/model/trainer/experiment/...)
examples/                toy_llh_walkthrough.ipynb — quickstart notebook
tests/                   pytest suite
bash/                    SLURM submit scripts for cluster training runs
```

## Pluggable likelihood interface

Any object with this shape can be used as `sampler_target`, without `src/gunflows`
importing it by name (see `gunflows.likelihood_sampler.base`):

- `get_parameter_names() -> list[str]`
- `inject_params_and_compute_likelihood(params, extend_continue=False) -> (nll, _, _)`
- `postfit_parameter_values`, `postfit_covariance_matrix` — used as the reference
  point/covariance for the initial Gaussian proposal and for standardization

`apps/toyllh/likelihood.py` is the minimal reference implementation of this
interface; `apps/gundam/likelihoodSampler.py` is the GUNDAM/ROOT one.

## Tests

```bash
pytest tests/
```
