# src/gunflows/__init__.py
import os
import sys

# normflows lives in the git submodule at src/normalizing-flows rather than
# being pip-installed; make it importable for every gunflows entry point.
_NF_LOCAL = os.path.join(os.path.dirname(__file__), "..", "normalizing-flows")
sys.path.append(os.path.abspath(_NF_LOCAL))

__all__ = []
