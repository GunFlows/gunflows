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

## License

This project is distributed without a specific license file.
