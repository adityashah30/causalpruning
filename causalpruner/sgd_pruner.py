from causalpruner import base

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Optional


class SGDPruner(base.CausalPruner):

    def __init__(self, model: nn.Module, prune_threshold: float = 1e-3):
        super().__init__(model)
        self.prune_threshold = prune_threshold
        self.counter = 0

    def _get_flattened_weight(self, name: str) -> torch.Tensor:
        weight = self.params_dict[name].detach().clone()
        flattened_weight = torch.flatten(weight)
        return flattened_weight

    def get_mask(self, name: str) -> torch.Tensor:
        trainer = self.causal_weights_trainers[name]
        weights = trainer.get_weights()
        return torch.where(weights <= self.prune_threshold, 0, 1)

    def compute_masks(self) -> None:
        for module_name, module in self.modules_dict.items():
            mask = self.get_mask(module_name)
            mask = torch.reshape(mask, self.params_dict[module_name].size())
            prune.custom_from_mask(module, 'weight', mask)

    def remove_masks(self) -> None:
        for _, module in self.modules_dict.items():
            setattr(module, 'weight', module.weight_orig)
            delattr(module, 'weight_orig')
            delattr(module, 'weight_mask')


class OnlineSGDPruner(SGDPruner):

    def __init__(self, model: nn.Module, *, prune_threshold: float = 1e-3,
                 num_epochs_batched: int = 16, causal_weights_num_epochs: int = 10):
        super().__init__(model, prune_threshold)
        self.num_epochs_batched = num_epochs_batched
        self.causal_weights_num_epochs = causal_weights_num_epochs
        self.weights = dict()
        device = torch.device('cpu')
        for param_name, param in self.params_dict.items():
            device = param.device
            param_size = np.prod(param.size(), dtype=int)
            self.weights[param_name] = torch.zeros(
                (self.num_epochs_batched, param_size), device=device)
        self.losses = torch.zeros(self.num_epochs_batched, device=device)

    def provide_loss(self, loss: torch.Tensor) -> None:
        self.losses[self.counter] = loss.detach().clone()
        for param_name in self.params_dict:
            self.weights[param_name][
                self.counter] = self._get_flattened_weight(param_name)
        self.counter += 1
        if self.counter % self.num_epochs_batched == 0:
            self._train_pruning_weights()

    def _train_pruning_weights(self) -> None:
        self.counter = 0
        delta_losses = torch.diff(self.losses)
        for param_name in self.params_dict:
            weights = self.weights[param_name]
            delta_weights = torch.diff(weights, dim=0)
            delta_weights_squared = torch.pow(delta_weights, 2)
            trainer = self.causal_weights_trainers[param_name]
            trainer.fit(delta_weights_squared, delta_losses,
                        self.causal_weights_num_epochs)


class DiskSGDPruner(SGDPruner):

    def __init__(
            self, model: nn.Module, *, prune_threshold=1e-3,
            checkpoint_dir: str = ''):
        super().__init__(model, prune_threshold)
        self.checkpoint_dir = checkpoint_dir

    def compute_masks(self) -> None:
        pass


def get_sgd_pruner(
        model: nn.Module, *, online: bool = True,
        checkpoint_dir: str = '') -> Optional[SGDPruner]:
    if online:
        return OnlineSGDPruner(model)
    # else:
    #     assert checkpoint_dir != ''
    #     return DiskSGDPruner(model, checkpoint_dir)
    return None
