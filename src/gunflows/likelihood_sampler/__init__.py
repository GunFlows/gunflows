"""
Backend-agnostic sampling infrastructure. NFSamplerProcess drives a
background sampling loop against *some* likelihood implementation, resolved
at runtime from a dotted path (see base.py) - it never imports a concrete
backend like GUNDAM by name. Concrete backends (e.g. apps.gundam) live
outside src/gunflows.
"""
from .nf_llh_sampler import NFSamplerProcess

__all__ = ["NFSamplerProcess"]
