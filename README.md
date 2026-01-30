# GunFlows

GunFlows provides a small framework for training conditional normalizing flows. The code relies on `hydra` for configuration and uses the `normflows` library to build the flow layers.

## Features

- **SystematicDataset** for loading training data stored in `.npz` files.
- Flow components `CovFlow`, `ContextFlow` and `SystematicFlow` implemented with PyTorch.
- Several importance–weighted losses available in `gunflows.losses`.
- Hydra powered configuration under `configs/` with an example experiment `oa2022`.

## Quick start

1. Install dependencies (PyTorch, hydra-core, omegaconf, matplotlib, normflows):
   ```bash
   pip install torch hydra-core omegaconf matplotlib normflows
   ```
2. Export the path to the dataset:
   ```bash
   export DATA_DIR=/path/to/npz/files
   ```
3. Launch training with the default configuration:
   ```bash
   python -m gunflows.train
   ```
   Model checkpoints are stored in `outputs/<run>/checkpoints`.

Configuration files can be overridden from the command line, e.g.:
```bash
python -m gunflows.train trainer.device=cpu optim.lr=1e-4
```

## Project structure

- `configs/` – Hydra configuration tree
- `src/gunflows/` – library code
  - `dataset/` – dataset loaders
  - `flows/` – flow components
  - `losses/` – importance-based loss functions
  - `trainer/` – training loops
  - `utils/` – helpers to build the model

---

## Hyperparameter tuning (Optuna + Slurm)

There is an Optuna-based hyperparameter tuning pipeline under `hparam_tuning/` that launches many training runs on the cluster via Slurm and collects them into a single Optuna study.

### Layout

Inside `hparam_tuning/` you will find:

- `launch_tuning.sh` – top-level Slurm job that drives the whole tuning campaign (CPU node).
- `run_array.sh` – GPU Slurm array job; each task runs one Optuna worker (one trial per task).
- `worker.py` – Python worker that:
  - samples hyperparameters from `search_space.yaml` via Optuna,
  - launches training inside the Apptainer container (calling `gunflows.train`),
  - parses the validation loss from the logs and reports it back to Optuna.
- `search_space.yaml` – definition of the hyperparameter search space (keys like `experiment.optim.lr`, `experiment.model.nflows`, etc.).
- `merge_stage.py` – merges per-job SQLite databases into a single master Optuna database, deduplicating trials and running diagnostics.
- `diagnostics.py` – generates diagnostic plots (contour plots + manual importance scores) from the merged study.
- `databases/<STUDY>/` – per-study data:
  - `databases/<STUDY>/<STUDY>.db` – master Optuna SQLite DB.
  - `databases/<STUDY>/tmp_dbs/` – temporary DBs and flag files for each stage and worker.
  - `databases/<STUDY>/figs/` – diagnostic plots for that study.

### Dependencies for tuning

In addition to the dependencies needed for training, the tuning pipeline requires:

- Python packages (installed in the environment where you run `python worker.py` / `python diagnostics.py`):
  ```bash
  pip install --user optuna pyyaml matplotlib scipy numpy


## License

This project is distributed without a specific license file.
