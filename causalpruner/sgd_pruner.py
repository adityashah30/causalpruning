from causalpruner.base import Pruner, best_device

from abc import abstractmethod
import os
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
                 prune_threshold: float,
                 l1_regularization_coeff: float,
                 device: Union[str, torch.device] = best_device()):
        super().__init__()
        self.prune_threshold = prune_threshold
        self.device = device
        self.l1_regularization_coeff = l1_regularization_coeff
        self.flattened_dims = np.prod(model_weights.size(), dtype=int)

    @abstractmethod
    def get_l1_loss(self) -> torch.Tensor:
        raise NotImplementedError(
            "CausalWeightsTrainer is an abstract class.")

    @abstractmethod
    def get_non_zero_weights(self) -> torch.Tensor:
        raise NotImplementedError(
            "CausalWeightsTrainer is an abstract class.")


class CausalWeightsTrainerVanilla(CausalWeightsTrainer):

    def __init__(self, model_weights: torch.Tensor,
                 prune_threshold: float,
                 l1_regularization_coeff: float,
                 device: Union[str, torch.device] = best_device()):
        super().__init__(model_weights, prune_threshold,
                         l1_regularization_coeff, device=device)
        self.layer = nn.Linear(
            self.flattened_dims, 1, bias=False, device=self.device)
        self.optimizer = optim.Adam(self.parameters())

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

    def get_non_zero_weights(self) -> torch.Tensor:
        weights = torch.flatten(self.layer.weight.detach().clone())
        weights_mask = torch.abs(weights) <= self.prune_threshold
        return torch.where(weights_mask, 0, 1)


class CausalWeightsTrainerMomentum(CausalWeightsTrainer):

    def __init__(self, model_weights: torch.Tensor,
                 prune_threshold: float,
                 l1_regularization_coeff: float,
                 device: Union[str, torch.device] = best_device()):
        super().__init__(model_weights, prune_threshold,
                         l1_regularization_coeff, device=device)

        self.layer1 = nn.Linear(
            self.flattened_dims, 1, bias=False, device=self.device)
        self.layer2 = nn.Linear(
            self.flattened_dims, 1, bias=False, device=self.device)
        self.layer3 = nn.Linear(
            self.flattened_dims, 1, bias=False, device=self.device)
        self.optimizer = optim.Adam(self.parameters())

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

    def get_non_zero_weights(self) -> torch.Tensor:
        weights1 = torch.flatten(self.layer1.weight.detach().clone())
        weights1_mask = torch.abs(weights1) <= self.prune_threshold
        weights2 = torch.flatten(self.layer1.weight.detach().clone())
        weights2_mask = torch.abs(weights2) <= self.prune_threshold
        weights3 = torch.flatten(self.layer1.weight.detach().clone())
        weights3_mask = torch.abs(weights3) <= self.prune_threshold
        return torch.where(
            weights1_mask & weights2_mask & weights3_mask, 0, 1)


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

    def __init__(self, model: nn.Module,
                 momentum: bool,
                 prune_threshold: float,
                 l1_regularization_coeff: float,
                 device: Union[str, torch.device] = best_device()):
        super().__init__(model, device)

        self.momentum = momentum
        self.prune_threshold = prune_threshold
        self.counter = 0

        trainer = CausalWeightsTrainerVanilla
        if self.momentum:
            trainer = CausalWeightsTrainerMomentum

        self.causal_weights_trainers = nn.ModuleDict()
        for param_name, param in self.params_dict.items():
            self.causal_weights_trainers[param_name] = trainer(
                param, self.prune_threshold, l1_regularization_coeff,
                self.device)

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
        return trainer.get_non_zero_weights()

    def compute_masks(self) -> None:
        for module_name, module in self.modules_dict.items():
            mask = self.get_mask(module_name)
            mask = torch.reshape(
                mask, self.params_dict[module_name].size())
            prune.custom_from_mask(module, 'weight', mask)


class OnlineSGDPruner(CausalPruner):

    def __init__(self, model: nn.Module,
                 momentum: bool,
                 prune_threshold: float,
                 l1_regularization_coeff: float,
                 num_epochs_batched: int,
                 causal_weights_num_epochs: int,
                 device: Union[str, torch.device] = best_device()):
        super().__init__(
            model, momentum, prune_threshold, l1_regularization_coeff, device)
        self.num_epochs_batched = num_epochs_batched
        self.causal_weights_num_epochs = causal_weights_num_epochs
        self.weights = dict()
        device = torch.device('cpu')
        for param_name, param in self.params_dict.items():
            device = param.device
            param_size = np.prod(param.size(), dtype=int)
            self.weights[param_name] = torch.zeros(
                (self.num_epochs_batched, param_size), device=device)
        self.losses = torch.zeros(
            self.num_epochs_batched, device=device)

    def provide_loss(self, loss: torch.Tensor) -> None:
        self.losses[self.counter] = loss.detach().clone()
        for param_name in self.params_dict:
            self.weights[param_name][
                self.counter] = self.get_flattened_weight(param_name)
        self.counter += 1
        if self.counter % self.num_epochs_batched == 0:
            self.train_pruning_weights()

    def get_delta_params_vanilla(
            self, params: torch.Tensor) -> torch.Tensor:
        delta_params = torch.diff(params, dim=0)
        delta_params = torch.square(delta_params)
        return delta_params

    def get_delta_params_momentum(self, params: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_params = torch.diff(params, dim=0)
        delta_param_t = delta_params[:-1]
        delta_param_t_sq = torch.square(delta_param_t)
        delta_param_t_plus_1 = delta_params[1:]
        delta_param_t_plus_1_sq = torch.square(delta_param_t_plus_1)
        delta_param_t_t_plus_1 = delta_param_t * delta_param_t_plus_1
        return (delta_param_t_sq,
                delta_param_t_plus_1_sq,
                delta_param_t_t_plus_1)

    def get_delta_params(self, params) -> Union[torch.Tensor,
                                                tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.momentum:
            return self.get_delta_params_momentum(params)
        return self.get_delta_params_vanilla(params)

    def get_delta_losses(self, losses: torch.Tensor) -> torch.Tensor:
        delta_losses = torch.diff(losses)
        if self.momentum:
            delta_losses = delta_losses[1:]
        return delta_losses

    def train_pruning_weights(self) -> None:
        self.counter = 0
        delta_losses = self.get_delta_losses(self.losses)
        for param_name in self.params_dict:
            params = self.weights[param_name]
            delta_params = self.get_delta_params(params)
            trainer = self.causal_weights_trainers[param_name]
            for _ in range(self.causal_weights_num_epochs):
                trainer.fit(delta_params, delta_losses)


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


class CheckpointSGDPruner(CausalPruner):

    def __init__(
            self, model: nn.Module, checkpoint_dir: str,
            momentum: bool, prune_threshold: float,
            l1_regularization_coeff: float,
            causal_weights_batch_size: int,
            causal_weights_num_epochs: int,
            device: Union[str, torch.device] = best_device()):
        super().__init__(
            model, momentum, prune_threshold,
            l1_regularization_coeff, device)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.causal_weights_batch_size = causal_weights_batch_size
        self.causal_weights_num_epochs = causal_weights_num_epochs
        self.loss_checkpoint_dir = os.path.join(
            self.checkpoint_dir, 'loss')
        os.makedirs(self.loss_checkpoint_dir, exist_ok=True)
        self.param_checkpoint_dirs = dict()
        for param_name in self.params_dict:
            self.param_checkpoint_dirs[param_name] = os.path.join(
                self.checkpoint_dir, param_name)
            os.makedirs(self.param_checkpoint_dirs[param_name],
                        exist_ok=True)

    def provide_loss(self, loss: torch.Tensor) -> None:
        torch.save(loss, self.get_checkpoint_path('loss'))
        for param_name, param in self.params_dict.items():
            torch.save(torch.flatten(param.detach().clone()),
                       self.get_checkpoint_path(param_name))
        self.counter += 1

    def get_checkpoint_path(self, param_name: str) -> str:
        if param_name == 'loss':
            path = self.loss_checkpoint_dir
        else:
            path = self.param_checkpoint_dirs[param_name]
        return os.path.join(path, f'ckpt.{self.counter}')

    def compute_masks(self):
        self.train_pruning_weights()
        super().compute_masks()

    def train_pruning_weights(self) -> None:
        for param, param_dir in self.param_checkpoint_dirs.items():
            self._train_pruning_weights_for_param(param, param_dir)

    def _train_pruning_weights_for_param(
            self, param: str, param_dir: str) -> None:
        dataset = ParamDataset(
            param_dir, self.loss_checkpoint_dir,
            self.momentum)
        dataloader = DataLoader(
            dataset,
            batch_size=self.causal_weights_batch_size,
        )
        for delta_params, delta_losses in dataloader:
            for _ in range(self.causal_weights_num_epochs):
                self.causal_weights_trainers[param].fit(
                    delta_params, delta_losses)


def get_sgd_pruner(
        model: nn.Module, *,
        momentum: bool = False,
        prune_threshold: float = 1e-3,
        l1_regularization_coeff: float = 1e-5,
        online: bool = True,
        num_epochs_batched: int = 16,
        checkpoint_dir: str = '',
        causal_weights_batch_size: int = 16,
        causal_weights_num_epochs: int = 10,
        device=best_device()) -> CausalPruner:
    if online:
        return OnlineSGDPruner(
            model, momentum, prune_threshold, l1_regularization_coeff,
            num_epochs_batched, causal_weights_num_epochs, device)
    else:
        assert checkpoint_dir != ''
        return CheckpointSGDPruner(
            model, checkpoint_dir, momentum, prune_threshold,
            l1_regularization_coeff, causal_weights_batch_size,
            causal_weights_num_epochs, device)
