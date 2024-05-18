from abc import ABC, abstractmethod
from typing import Union

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def best_device() -> torch.device:
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    return device


class Pruner(ABC, nn.Module):

    @staticmethod
    def is_module_supported(module: nn.Module) -> bool:
        return True

    def __init__(
            self, model: nn.Module,
            device: Union[str, torch.device] = best_device()):
        super().__init__()
        self.device = device

        self.modules_dict = nn.ModuleDict()
        for name, module in model.named_children():
            if self.is_module_supported(module):
                self.modules_dict[name] = module

        self.params_dict = nn.ParameterDict()
        for module_name, module in self.modules_dict.items():
            for param_name, param in module.named_parameters():
                if 'weight' not in param_name:
                    continue
                self.params_dict[module_name] = param

    @abstractmethod
    def compute_masks(self) -> None:
        raise NotImplementedError(
            "Pruner is an abstract class. Use an appropriate derived class.")

    def remove_masks(self) -> None:
        for _, module in self.modules_dict.items():
            setattr(module, 'weight', module.weight_orig)
            delattr(module, 'weight_orig')
            delattr(module, 'weight_mask')
            for k, hook in module._forward_pre_hooks.items():
                if isinstance(hook, prune.BasePruningMethod):
                    del module._forward_pre_hooks[k]
