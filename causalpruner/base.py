from abc import ABC, abstractmethod
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


class Pruner(ABC):

    @staticmethod
    def is_module_supported(module: nn.Module) -> bool:
        return True

    def __init__(self, model: nn.Module, checkpoint_dir: str,
                 start_clean: bool,
                 device: Union[str, torch.device] = best_device()):
        super().__init__()
        self.device = device

        self.modules_dict = nn.ModuleDict()
        for name, module in model.named_children():
            if self.is_module_supported(module):
                self.modules_dict[name] = module

        self.params = nn.ParameterList()
        for module_name, module in self.modules_dict.items():
            if hasattr(module, 'weight'):
                self.params.append(module_name)

        self.checkpoint_dir = checkpoint_dir
        if start_clean and os.path.exists(self.checkpoint_dir):
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

    @abstractmethod
    def compute_masks(self) -> None:
        raise NotImplementedError(
            "Pruner is an abstract class. Use an appropriate derived class.")

    def apply_masks(self) -> None:
        for param in self.params:
            module = self.modules_dict[param]
            prune.remove(module, 'weight')

    def remove_masks(self) -> None:
        for _, module in self.modules_dict.items():
            setattr(module, 'weight', module.weight_orig)
            delattr(module, 'weight_orig')
            delattr(module, 'weight_mask')
            for k, hook in module._forward_pre_hooks.items():
                if isinstance(hook, prune.BasePruningMethod):
                    del module._forward_pre_hooks[k]

    def start_pruning(self) -> None:
        for param in self.params:
            param_dir = os.path.join(
                self.param_checkpoint_dirs[param], 'initial')
            os.makedirs(param_dir, exist_ok=True)
            module = self.modules_dict[param]
            torch.save(torch.flatten(module.weight.detach().clone()),
                       os.path.join(param_dir, 'ckpt.initial'))

    def reset_weights(self) -> None:
        for param in self.params:
            initial_param_path = os.path.join(
                self.param_checkpoint_dirs[param],
                'initial/ckpt.initial')
            initial_param = torch.load(initial_param_path)
            with torch.no_grad():
                weight = self.modules_dict[param].weight
                initial_param = initial_param.reshape_as(weight)
                weight.data = initial_param
