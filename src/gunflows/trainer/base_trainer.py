#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Base Trainer
Author: Mathias El Baz
Date: 28/01/2025
Description:
  Abstract training interface; all concrete trainers derive from this class.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from omegaconf import DictConfig

class BaseTrainer(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def train(self) -> None: ...
