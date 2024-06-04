from causalpruner.base import Pruner, best_device

from abc import abstractmethod
import os
import shutil
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class CausalWeightsTrainer(nn.Module):

    def __init__(self, model_weights: torch.Tensor,
                 lr: float,
                 prune_threshold: float,
                 l1_regularization_coeff: float,
                 device: Union[str, torch.device] = best_device()):
        super().__init__()
        self.lr = lr
        self.prune_threshold = prune_threshold
        self.device = device
        self.l1_regularization_coeff = l1_regularization_coeff
        self.flattened_dims = np.prod(model_weights.size(), dtype=int)

    @abstractmethod
    def get_l1_loss(self) -> torch.Tensor:
        raise NotImplementedError(
            "CausalWeightsTrainer is an abstract class.")

    @abstractmethod
    def get_non_zero_weights(
            self, prune_threshold: Union[None, float] = None) -> torch.Tensor:
        raise NotImplementedError(
            "CausalWeightsTrainer is an abstract class.")


class CausalWeightsTrainerVanilla(CausalWeightsTrainer):

    def __init__(self, model_weights: torch.Tensor,
                 lr: float,
                 prune_threshold: float,
                 l1_regularization_coeff: float,
                 device: Union[str, torch.device] = best_device()):
        super().__init__(model_weights, lr, prune_threshold,
                         l1_regularization_coeff, device=device)
        self.layer = nn.Linear(
            self.flattened_dims, 1, bias=False, device=self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

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

    def get_non_zero_weights(
            self, prune_threshold: Union[None, float] = None) -> torch.Tensor:
        if prune_threshold is None:
            prune_threshold = self.prune_threshold
        weights = torch.flatten(self.layer.weight.detach().clone())
        weights_mask = torch.abs(weights) <= prune_threshold
        return torch.where(weights_mask, 0, 1)


class CausalWeightsTrainerMomentum(CausalWeightsTrainer):

    def __init__(self, model_weights: torch.Tensor,
                 lr: float,
                 prune_threshold: float,
                 l1_regularization_coeff: float,
                 device: Union[str, torch.device] = best_device()):
        super().__init__(model_weights, lr, prune_threshold,
                         l1_regularization_coeff, device=device)

        self.layer1 = nn.Linear(
            self.flattened_dims, 1, bias=False, device=self.device)
        self.layer2 = nn.Linear(
            self.flattened_dims, 1, bias=False, device=self.device)
        self.layer3 = nn.Linear(
            self.flattened_dims, 1, bias=False, device=self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, delta_weights_k_0_sq: torch.Tensor,
                delta_weights_k_1_sq: torch.Tensor,
                delta_weights_k_0_k_1: torch.Tensor) -> torch.Tensor:
        return self.layer1(delta_weights_k_0_sq) + self.layer2(
            delta_weights_k_1_sq) + self.layer3(delta_weights_k_0_k_1)

    def get_l1_loss(self) -> torch.Tensor:
        return self.l1_regularization_coeff * (torch.norm(
            self.layer1.weight, p=1) +
            torch.norm(
            self.layer2.weight, p=1) +
            torch.norm(
            self.layer3.weight, p=1))

    def fit(
            self, X: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            Y: torch.Tensor):
        self.layer1.train()
        self.layer2.train()
        self.layer3.train()
        Y_hat = torch.squeeze(self.forward(X[0], X[1], X[2]), dim=1)
        Y = torch.flatten(Y)
        loss = F.mse_loss(Y_hat, Y) + self.get_l1_loss()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_non_zero_weights(
            self, prune_threshold: Union[None, float] = None) -> torch.Tensor:
        if prune_threshold is None:
            prune_threshold = self.prune_threshold
        weights1 = torch.flatten(self.layer1.weight.detach().clone())
        weights1_mask = torch.abs(weights1) <= prune_threshold
        weights2 = torch.flatten(self.layer2.weight.detach().clone())
        weights2_mask = torch.abs(weights2) <= prune_threshold
        weights3 = torch.flatten(self.layer3.weight.detach().clone())
        weights3_mask = torch.abs(weights3) <= prune_threshold
        return torch.where(
            weights1_mask & weights2_mask & weights3_mask, 0, 1)


class ParamDataset(Dataset):

    def __init__(
            self, param_base_dir: str, loss_base_dir: str, momentum: bool):
        self.param_base_dir = param_base_dir
        self.loss_base_dir = loss_base_dir
        self.momentum = momentum
        self.num_items = min(len(os.listdir(self.param_base_dir)),
                             len(os.listdir(self.loss_base_dir)))

    def __len__(self) -> int:
        return self.num_items - 2 if self.momentum else self.num_items - 1

    def __getitem__(self, idx: int) -> tuple[Union[torch.Tensor,
                                                   tuple[torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]:
        delta_param = self.get_delta_param(idx)
        delta_loss = self.get_delta_loss(idx)
        return delta_param, delta_loss

    def get_delta_param_vanilla(self, idx: int) -> torch.Tensor:
        delta_param = self.get_delta(self.param_base_dir, idx)
        delta_param = torch.square(delta_param)
        return delta_param

    def get_delta_param_momentum(self, idx: int) -> tuple[torch.Tensor,
                                                          torch.Tensor, torch.Tensor]:
        delta_param_t = self.get_delta(self.param_base_dir, idx)
        delta_param_t_sq = torch.square(delta_param_t)
        delta_param_t_plus_1 = self.get_delta(
            self.param_base_dir, idx + 1)
        delta_param_t_plus_1_sq = torch.square(delta_param_t_plus_1)
        delta_param_t_t_plus_1 = delta_param_t * delta_param_t_plus_1
        return (
            delta_param_t_sq, delta_param_t_plus_1_sq, delta_param_t_t_plus_1)

    def get_delta_param(self, idx: int) -> Union[torch.Tensor,
                                                 tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.momentum:
            return self.get_delta_param_momentum(idx)
        return self.get_delta_param_vanilla(idx)

    def get_delta_loss(self, idx: int) -> torch.Tensor:
        if self.momentum:
            idx += 1
        return self.get_delta(self.loss_base_dir, idx)

    def get_delta(self, dir: str, idx: int) -> torch.Tensor:
        first_file_path = os.path.join(dir, f'ckpt.{idx}')
        first_tensor = torch.load(first_file_path)
        second_file_path = os.path.join(dir, f'ckpt.{idx + 1}')
        second_tensor = torch.load(second_file_path)
        return second_tensor - first_tensor


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

    def __init__(
            self, model: nn.Module, checkpoint_dir: str,
            momentum: bool, pruner_lr: float, prune_threshold: float,
            l1_regularization_coeff: float,
            causal_weights_batch_size: int,
            causal_weights_num_epochs: int,
            start_clean: bool,
            device: Union[str, torch.device] = best_device()):
        super().__init__(model, device)

        self.momentum = momentum
        self.iteration = -1
        self.counter = 0
        self.causal_weights_batch_size = causal_weights_batch_size
        self.causal_weights_num_epochs = causal_weights_num_epochs

        trainer = CausalWeightsTrainerVanilla
        if self.momentum:
            trainer = CausalWeightsTrainerMomentum
        self.causal_weights_trainers = nn.ModuleDict()
        for param in self.params:
            module = self.modules_dict[param]
            self.causal_weights_trainers[param] = trainer(
                module.weight, pruner_lr, prune_threshold,
                l1_regularization_coeff, self.device)

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

    def start_pruning(self) -> None:
        for param in self.params:
            param_dir = os.path.join(
                self.param_checkpoint_dirs[param], 'initial')
            os.makedirs(param_dir, exist_ok=True)
            module = self.modules_dict[param]
            torch.save(torch.flatten(module.weight.detach().clone()),
                       os.path.join(param_dir, 'ckpt.initial'))

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

    def _get_mask(
            self, name: str,
            pruning_threshold: Union[float, None] = None) -> torch.Tensor:
        trainer = self.causal_weights_trainers[name]
        mask_this_iteration = trainer.get_non_zero_weights(pruning_threshold)
        mask_this_iteration = torch.reshape(
            mask_this_iteration, self.modules_dict[name].weight.size())
        module = self.modules_dict[name]
        if hasattr(module, 'weight_mask'):
            mask = getattr(module, 'weight_mask')
            mask_this_iteration = torch.logical_and(mask, mask_this_iteration)
        return mask_this_iteration

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
        dataset = ParamDataset(param_dir, loss_dir, self.momentum)
        dataloader = DataLoader(
            dataset,
            batch_size=self.causal_weights_batch_size,
        )
        for delta_params, delta_losses in dataloader:
            for _ in range(self.causal_weights_num_epochs):
                self.causal_weights_trainers[param].fit(
                    delta_params, delta_losses)


def get_sgd_pruner(
        model: nn.Module,
        checkpoint_dir: str,
        momentum: bool = False,
        *,
        pruner_lr: float = 1e-3,
        prune_threshold: float = 5e-6,
        l1_regularization_coeff: float = 1e-5,
        causal_weights_batch_size: int = 16,
        causal_weights_num_epochs: int = 10,
        start_clean: bool = True,
        device=best_device()) -> SGDPruner:
    assert checkpoint_dir != ''
    return SGDPruner(
        model, checkpoint_dir, momentum, pruner_lr, prune_threshold,
        l1_regularization_coeff, causal_weights_batch_size,
        causal_weights_num_epochs, start_clean, device)
