from causalpruner.base import Pruner, PrunerConfig, best_device

from abc import abstractmethod
from dataclasses import dataclass
import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm.auto import trange


class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold


class CausalWeightsTrainer(nn.Module):

    def __init__(self,
                 model_weights: torch.Tensor,
                 momentum: bool,
                 lr: float,
                 prune_threshold: float,
                 l1_regularization_coeff: float,
                 device: Union[str, torch.device] = best_device()):
        super().__init__()
        self.momentum = momentum
        self.lr = lr
        self.prune_threshold = prune_threshold
        self.device = device
        self.l1_regularization_coeff = l1_regularization_coeff
        self.flattened_dims = np.prod(model_weights.size(), dtype=int)
        self.weights_dim_multiplier = 1
        if self.momentum:
            self.weights_dim_multiplier = 3
        self.flattened_dims *= self.weights_dim_multiplier
        self.layer = nn.Linear(
            self.flattened_dims, 1, bias=False, device=self.device)
        self.optimizer = optim.Adam(self.layer.parameters(), lr=self.lr)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layer(X)

    def get_l1_loss(self) -> torch.Tensor:
        return self.l1_regularization_coeff * torch.norm(
            self.layer.weight, p=1)

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        self.layer.train()
        Y_hat = torch.squeeze(self.forward(X), dim=1)
        Y = torch.flatten(Y)
        loss = F.mse_loss(Y_hat, Y) + self.get_l1_loss()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        with torch.no_grad():
            self._prune_small_weights()

    def get_non_zero_weights(
            self, prune_threshold: Union[None, float] = None) -> torch.Tensor:
        if prune_threshold is None:
            prune_threshold = self.prune_threshold
        weights = torch.flatten(self.layer.weight.detach().clone())
        if self.momentum:
            weights = weights.view(self.weights_dim_multiplier, -1)
        weights_mask = torch.atleast_2d(torch.abs(weights) <= prune_threshold)
        weights_mask = torch.all(weights_mask, dim=0)
        return torch.where(weights_mask, 0, 1)

    def _prune_small_weights(self):
        params_to_prune = [(self.layer, 'weight')]
        prune.global_unstructured(
            params_to_prune,
            pruning_method=ThresholdPruning,
            threshold=self.prune_threshold)
        for module, name in params_to_prune:
            prune.remove(module, name)


class ParamDataset(Dataset):

    def __init__(
            self, param_base_dir: str, loss_base_dir: str, momentum: bool,
            device: Union[str, torch.device] = best_device()):
        self.param_base_dir = param_base_dir
        self.loss_base_dir = loss_base_dir
        self.momentum = momentum
        self.device = device
        self.num_items = min(len(os.listdir(self.param_base_dir)),
                             len(os.listdir(self.loss_base_dir)))

    def __len__(self) -> int:
        return self.num_items - 2 if self.momentum else self.num_items - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        delta_param = self.get_delta_param(idx)
        delta_loss = self.get_delta_loss(idx)
        return delta_param, delta_loss

    def get_delta_param_vanilla(self, idx: int) -> torch.Tensor:
        delta_param = self.get_delta(self.param_base_dir, idx)
        delta_param = torch.square(delta_param)
        return delta_param

    def get_delta_param_momentum(self, idx: int) -> torch.Tensor:
        delta_param_t = self.get_delta(self.param_base_dir, idx)
        delta_param_t_sq = torch.square(delta_param_t)
        delta_param_t_plus_1 = self.get_delta(
            self.param_base_dir, idx + 1)
        delta_param_t_plus_1_sq = torch.square(delta_param_t_plus_1)
        delta_param_t_t_plus_1 = delta_param_t * delta_param_t_plus_1
        return torch.cat(
            (delta_param_t_sq, delta_param_t_plus_1_sq,
             delta_param_t_t_plus_1))

    def get_delta_param(self, idx: int) -> torch.Tensor:
        if self.momentum:
            return self.get_delta_param_momentum(idx)
        else:
            return self.get_delta_param_vanilla(idx)

    def get_delta_loss(self, idx: int) -> torch.Tensor:
        if self.momentum:
            idx += 1
        return self.get_delta(self.loss_base_dir, idx)

    def get_delta(self, dir: str, idx: int) -> torch.Tensor:
        first_file_path = os.path.join(dir, f'ckpt.{idx}')
        first_tensor = torch.load(first_file_path).to(self.device)
        second_file_path = os.path.join(dir, f'ckpt.{idx + 1}')
        second_tensor = torch.load(second_file_path).to(self.device)
        return second_tensor - first_tensor


@dataclass
class SGDPrunerConfig(PrunerConfig):
    momentum: bool
    pruner_lr: float
    prune_threshold: float
    l1_regularization_coeff: float
    causal_weights_num_epochs: int


class SGDPruner(Pruner):

    _SUPPORTED_MODULES = [
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
    ]

    @staticmethod
    def is_module_supported(module: nn.Module) -> bool:
        for supported_module in SGDPruner._SUPPORTED_MODULES:
            if isinstance(module, supported_module):
                return True
        return False

    def __init__(self, config: SGDPrunerConfig):
        super().__init__(config)
        self.counter = 0
        self.momentum = config.momentum
        self.causal_weights_num_epochs = config.causal_weights_num_epochs
        self.causal_weights_trainers = nn.ModuleDict()
        for param in self.params:
            module = self.modules_dict[param]
            self.causal_weights_trainers[param] = CausalWeightsTrainer(
                module.weight, config.momentum, config.pruner_lr,
                config.prune_threshold, config.l1_regularization_coeff,
                self.device)

    def provide_loss(self, loss: torch.Tensor) -> None:
        torch.save(loss, self._get_checkpoint_path('loss'))
        for param in self.params:
            module = self.modules_dict[param]
            torch.save(torch.flatten(module.weight.detach().clone()),
                       self._get_checkpoint_path(param))
        self.counter += 1

    def train_pruning_weights(self) -> None:
        for param, param_dir in self.param_checkpoint_dirs.items():
            self._train_pruning_weights_for_param(param, param_dir)

    def compute_masks(
            self, pruning_threshold: Union[float, None] = None) -> None:
        self.train_pruning_weights()
        for module_name, module in self.modules_dict.items():
            mask = self._get_mask(module_name, pruning_threshold)
            prune.custom_from_mask(module, 'weight', mask)

    def _get_mask(
            self, name: str,
            pruning_threshold: Union[float, None] = None) -> torch.Tensor:
        trainer = self.causal_weights_trainers[name]
        mask = trainer.get_non_zero_weights(pruning_threshold)
        mask = mask.reshape_as(self.modules_dict[name].weight)
        return mask

    def _get_checkpoint_path(self, param_name: str) -> str:
        if param_name == 'loss':
            path = self.loss_checkpoint_dir
        else:
            path = self.param_checkpoint_dirs[param_name]
        return os.path.join(path, f'{self.iteration}', f'ckpt.{self.counter}')

    def _train_pruning_weights_for_param(
            self, param: str, param_dir: str) -> None:
        param_dir = os.path.join(param_dir, f'{self.iteration}')
        loss_dir = os.path.join(self.loss_checkpoint_dir, f'{self.iteration}')
        dataset = ParamDataset(param_dir, loss_dir,
                               self.momentum, device=self.device)
        delta_params, delta_losses = zip(
            *[dataset[idx] for idx in range(len(dataset))])
        delta_params = torch.stack(delta_params)
        delta_losses = torch.stack(delta_losses)
        for _ in trange(self.causal_weights_num_epochs, leave=False):
            self.causal_weights_trainers[param].fit(
                delta_params, delta_losses)
