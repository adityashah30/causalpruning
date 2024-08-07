from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import shutil
from typing import Union

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def best_device(identifier: Union[None, int] = None) -> torch.device:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if identifier is not None:
        device = f'{device}:{identifier}'
    return torch.device(device)


@dataclass
class PrunerConfig:
    pruner: str
    model: nn.Module
    checkpoint_dir: str
    start_clean: bool
    eval_after_epoch: bool
    reset_weights: bool
    device: Union[str, torch.device]


class Pruner(ABC):

    _SUPPORTED_MODULES = [
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
    ]

    _MODULES_TO_RESET = [
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
    ]

    @staticmethod
    def is_module_supported(module: nn.Module) -> bool:
        for supported_module in Pruner._SUPPORTED_MODULES:
            if isinstance(module, supported_module):
                return True
        return False

    @staticmethod
    def is_module_to_be_reset(module: nn.Module) -> bool:
        for module_to_reset in Pruner._MODULES_TO_RESET:
            if isinstance(module, module_to_reset):
                return True
        return False

    @staticmethod
    def apply_identity_masks_to_model(model: nn.Module) -> None:
        for module in model.modules():
            if not Pruner.is_module_supported(module):
                continue
            prune.identity(module, 'weight')

    def __init__(self, config: PrunerConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.iteration = -1

        self.modules_dict = nn.ModuleDict()
        self.reset_modules_dict = nn.ModuleDict()
        for name, module in self.config.model.named_modules():
            if self.is_module_supported(module):
                self.modules_dict[name] = module
            if self.is_module_to_be_reset(module):
                self.reset_modules_dict[name] = module

        self.params = []
        for module_name, module in self.modules_dict.items():
            if hasattr(module, 'weight'):
                self.params.append(module_name)

        self.checkpoint_dir = config.checkpoint_dir
        if config.start_clean and os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.init_model_path = os.path.join(self.checkpoint_dir, 'init.ckpt')
        self.loss_checkpoint_dir = os.path.join(
            self.checkpoint_dir, 'loss')
        os.makedirs(self.loss_checkpoint_dir, exist_ok=True)
        self.param_checkpoint_dirs = dict()
        for param in self.params:
            self.param_checkpoint_dirs[param] = os.path.join(
                self.checkpoint_dir, param)
            os.makedirs(self.param_checkpoint_dirs[param],
                        exist_ok=True)

    def __str__(self) -> str:
        return self.config.pruner

    def __repr__(self) -> str:
        return str(self)

    @abstractmethod
    def compute_masks(self) -> None:
        raise NotImplementedError(
            "Pruner is an abstract class. Use an appropriate derived class.")

    @torch.no_grad
    def apply_masks(self) -> None:
        for param in self.params:
            module = self.modules_dict[param]
            prune.remove(module, 'weight')

    @torch.no_grad
    def apply_identity_masks(self) -> None:
        for param in self.params:
            module = self.modules_dict[param]
            prune.identity(module, 'weight')

    @torch.no_grad
    def remove_masks(self) -> None:
        for _, module in self.modules_dict.items():
            setattr(module, 'weight', module.weight_orig)
            delattr(module, 'weight_orig')
            delattr(module, 'weight_mask')
            for k, hook in module._forward_pre_hooks.items():
                if isinstance(hook, prune.BasePruningMethod):
                    del module._forward_pre_hooks[k]

    def provide_loss(self, loss: float) -> None:
        pass

    @torch.no_grad
    def start_pruning(self) -> None:
        torch.save(self.config.model.state_dict(), self.init_model_path)

    @torch.no_grad
    def start_iteration(self) -> None:
        self.iteration += 1
        iteration_name = f'{self.iteration}'
        loss_dir = os.path.join(self.loss_checkpoint_dir, iteration_name)
        os.makedirs(loss_dir, exist_ok=True)
        for param in self.params:
            param_dir = os.path.join(
                self.param_checkpoint_dirs[param], iteration_name)
            os.makedirs(param_dir, exist_ok=True)
        self.counter = 0

    @torch.no_grad
    def reset_weights(self) -> None:
        self.config.model.zero_grad(set_to_none=True)
        if not self.config.reset_weights:
            return
        masks = dict()
        for name, module in self.modules_dict.items():
            if hasattr(module, 'weight_mask'):
                masks[name] = getattr(module, 'weight_mask')
        self.remove_masks()
        self.config.model.load_state_dict(torch.load(self.init_model_path))
        for name, module in self.modules_dict.items():
            if name in masks:
                prune.custom_from_mask(module, 'weight', masks[name])
        for module in self.reset_modules_dict.values():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
