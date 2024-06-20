from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import shutil
from typing import Union

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def best_device() -> torch.device:
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    return device


@dataclass
class PrunerConfig:
    pruner: str
    model: nn.Module
    checkpoint_dir: str
    start_clean: bool
    device: Union[str, torch.device]


class Pruner(ABC):

    @staticmethod
    def is_module_supported(module: nn.Module) -> bool:
        return True

    @staticmethod
    def get_children(model: nn.Module, root: str) -> dict[str, nn.Module]:
        children = dict()
        for name, module in model.named_children():
            root_name = root + '_' + name
            if isinstance(module, nn.Sequential):
                children.update(Pruner.get_children(module, root_name))
            else:
                children[root_name] = module
        return children

    def __init__(self, config: PrunerConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.iteration = -1

        self.modules_dict = nn.ModuleDict()
        children = self.get_children(self.config.model, 'model')
        for name, module in children.items():
            if self.is_module_supported(module):
                self.modules_dict[name] = module

        self.params = nn.ParameterList()
        for module_name, module in self.modules_dict.items():
            if hasattr(module, 'weight'):
                self.params.append(module_name)

        self.checkpoint_dir = config.checkpoint_dir
        if config.start_clean and os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

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
    def remove_masks(self) -> None:
        for _, module in self.modules_dict.items():
            setattr(module, 'weight', module.weight_orig)
            delattr(module, 'weight_orig')
            delattr(module, 'weight_mask')
            for k, hook in module._forward_pre_hooks.items():
                if isinstance(hook, prune.BasePruningMethod):
                    del module._forward_pre_hooks[k]

    def provide_loss(self, loss: torch.Tensor) -> None:
        pass

    @torch.no_grad
    def start_pruning(self) -> None:
        for param in self.params:
            param_dir = os.path.join(
                self.param_checkpoint_dirs[param], 'initial')
            os.makedirs(param_dir, exist_ok=True)
            module = self.modules_dict[param]
            torch.save(torch.flatten(module.weight.detach().clone()),
                       os.path.join(param_dir, 'ckpt.initial'))

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
        for param in self.params:
            initial_param_path = os.path.join(
                self.param_checkpoint_dirs[param],
                'initial/ckpt.initial')
            initial_param = torch.load(initial_param_path)
            weight = self.modules_dict[param].weight
            initial_param = initial_param.reshape_as(weight)
            weight.data = initial_param
