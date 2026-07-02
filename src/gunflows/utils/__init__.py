"""
build_flow needs gunflows.flows.systematic_flow, while gunflows.flows needs
gunflows.utils.nets — importing build_flow eagerly here creates a circular
import that only "works" if callers happen to import gunflows.utils before
gunflows.flows. Load it lazily instead so import order doesn't matter.
"""
from .nets import MLP

__all__ = ["build_base", "build_flow_layers", "build_model", "MLP"]


def __getattr__(name):
    if name in ("build_base", "build_flow_layers", "build_model"):
        from . import build_flow
        return getattr(build_flow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
