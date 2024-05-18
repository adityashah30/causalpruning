from causalpruner.base import Pruner, best_device

from abc import abstractmethod
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim


class CausalWeightsTrainer(nn.Module):

    def __init__(self, model_weights: torch.Tensor, *,
                 device: Union[str, torch.device] = best_device()):
        super().__init__()
        self.device = device
        flattened_dims = np.prod(model_weights.size(), dtype=int)
        self.layer = nn.Linear(
            flattened_dims, 1, bias=False, device=self.device)
        self.optimizer = optim.Adam(self.layer.parameters())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layer(X)

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        self.layer.train()
        Y_hat = torch.squeeze(self.forward(X), dim=1)
        Y = torch.flatten(Y)
        loss = F.mse_loss(Y_hat, Y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_weights(self) -> torch.Tensor:
        return torch.flatten(self.layer.weight.detach().clone())


class CausalPruner(Pruner):

    _SUPPORTED_MODULES = [
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
    ]

    @staticmethod
    def is_module_supported(module: nn.Module) -> bool:
        for supported_module in CausalPruner._SUPPORTED_MODULES:
            if isinstance(module, supported_module):
                return True
        return False

    def __init__(self, model: nn.Module, *,
                 prune_threshold: float = 1e-3,
                 device: Union[str, torch.device] = best_device()):
        super().__init__(model, device)

        self.prune_threshold = prune_threshold
        self.counter = 0

        self.causal_weights_trainers = nn.ModuleDict()
        for param_name, param in self.params_dict.items():
            self.causal_weights_trainers[param_name] = CausalWeightsTrainer(
                param, device=self.device)

    @abstractmethod
    def provide_loss(self, loss: torch.Tensor) -> None:
        pass

    @abstractmethod
    def train_pruning_weights(self) -> None:
        pass

    def get_flattened_weight(self, name: str) -> torch.Tensor:
        weight = self.params_dict[name].detach().clone()
        flattened_weight = torch.flatten(weight)
        return flattened_weight

    def get_mask(self, name: str) -> torch.Tensor:
        trainer = self.causal_weights_trainers[name]
        weights = trainer.get_weights()
        return torch.where(torch.abs(weights) <= self.prune_threshold, 0, 1)

    def compute_masks(self) -> None:
        for module_name, module in self.modules_dict.items():
            mask = self.get_mask(module_name)
            mask = torch.reshape(mask, self.params_dict[module_name].size())
            prune.custom_from_mask(module, 'weight', mask)
