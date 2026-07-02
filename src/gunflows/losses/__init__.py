# src/gunflows/losses/__init__.py
from .importance_losses import (
    exp_forward,
    exp_reverse,
    exp_symmetric,
    kl_symmetric,
    absolute_kl_symmetric
)

LOSS_MAP = {
    "exp_forward":   exp_forward,
    "exp_reverse":   exp_reverse,
    "exp_symmetric": exp_symmetric,
    "kl_symmetric":  kl_symmetric,
    "absolute_kl_symmetric": absolute_kl_symmetric,
}

__all__ = [
    "exp_forward",
    "exp_reverse",
    "exp_symmetric",
    "kl_symmetric",
    "absolute_kl_symmetric",
    "LOSS_MAP",
]

