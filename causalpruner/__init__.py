from .base import best_device, Pruner
from .sgd_pruner import get_sgd_pruner, SGDPruner
from .trainer import DataConfig, EpochConfig, PrunerConfig, SGDPrunerConfig, TrainerConfig
from .trainer import Trainer, SGDPrunerTrainer
from .trainer import AverageMeter

from typing import Optional

import torch.nn as nn
import torch.optim as optim


def get_causal_pruner(
        model: nn.Module, optimizer: optim.Optimizer, checkpoint_dir: str,
        momentum: bool = False, **kwargs) -> Optional[Pruner]:
    if isinstance(optimizer, optim.SGD):
        return get_sgd_pruner(
            model, checkpoint_dir, momentum, **kwargs)
    raise NotImplementedError(
        f"CausalPruner is not implemented for optimizer of type {type(optim)}")
