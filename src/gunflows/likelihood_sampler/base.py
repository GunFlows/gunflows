#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title : base
Author: Mathias El Baz
Date  : 2026-07-02

Interface NFSamplerProcess expects from a likelihood-sampler backend.

This is a structural (duck-typed) contract, not a base class to inherit
from: any object with this shape works, resolved at runtime from a dotted
path (see NFSamplerProcess.sampler_target) so src/gunflows never imports a
concrete backend by name. apps.gundam.likelihoodSampler.LikelihoodSampler is
the GUNDAM/ROOT implementation; swapping likelihoods means writing a new
class with this shape and pointing sampler_target at it.
"""
from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

__all__ = ["LikelihoodSamplerProtocol", "resolve_target", "pushd"]


def resolve_target(target: str):
    """Resolve 'pkg.mod.ClassName' to the class/callable, without src/gunflows
    ever importing the concrete module by name."""
    module_name, attr_name = target.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr_name)


@contextmanager
def pushd(path: str | None):
    prev = os.getcwd()
    if path:
        os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@runtime_checkable
class LikelihoodSamplerProtocol(Protocol):
    postfit_covariance_matrix: object
    postfit_parameter_values: object
    likelihood_at_bestfit: float

    def __init__(
        self,
        config_file: str,
        override_files: list[str] | None = ...,
        threads: int = ...,
        data_is_asimov: bool = ...,
        seed: int = ...,
    ) -> None: ...

    def get_parameter_names(self) -> list[str]: ...

    def inject_params_and_compute_likelihood(
        self, params: list[float], extend_continue: bool = ...
    ) -> tuple[float, object, object]: ...
