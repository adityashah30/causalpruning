from causalpruner.base import CausalPruner, best_device

import os
import shutil
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SGDPruner(CausalPruner):

    def __init__(self, model: nn.Module, *,
                 momentum: bool = False, prune_threshold: float = 1e-3,
                 device: Union[str, torch.device] = best_device()):
        super().__init__(model, prune_threshold=prune_threshold, device=device)
        self.momentum = momentum


class OnlineSGDPruner(SGDPruner):

    def __init__(self, model: nn.Module, *, momentum: bool = False,
                 prune_threshold: float = 1e-3, num_epochs_batched: int = 16,
                 causal_weights_num_epochs: int = 10,
                 device: Union[str, torch.device] = best_device()):
        super().__init__(model, momentum=momentum,
                         prune_threshold=prune_threshold, device=device)
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
                self.counter] = self.get_flattened_weight(param_name)
        self.counter += 1
        if self.counter % self.num_epochs_batched == 0:
            self.train_pruning_weights()

    def get_delta_params(
            self, params: torch.Tensor) -> torch.Tensor:
        delta_params = torch.diff(params, dim=0)
        if self.momentum:
            delta_param_t_plus_1 = delta_params[1:]
            delta_param_t = delta_params[:-1]
            delta_params = torch.square(delta_param_t_plus_1) + torch.square(
                delta_param_t) + delta_param_t_plus_1 * delta_param_t
        else:
            delta_params = torch.square(delta_params)
        return delta_params

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
            self, param_base_dir: str, loss_base_dir: str, *,
            momentum: bool = False):
        self.param_base_dir = param_base_dir
        self.loss_base_dir = loss_base_dir
        self.momentum = momentum
        self.num_items = min(len(os.listdir(self.param_base_dir)),
                             len(os.listdir(self.loss_base_dir)))

    def __len__(self) -> int:
        return self.num_items - 2 if self.momentum else self.num_items - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        delta_param = self.get_delta_param(idx)
        delta_loss = self.get_delta_loss(idx)
        return delta_param, delta_loss

    def get_delta_param(self, idx: int) -> torch.Tensor:
        if self.momentum:
            delta_param_t_plus_1 = self.get_delta(self.param_base_dir, idx + 1)
            delta_param_t = self.get_delta(self.param_base_dir, idx)
            delta_param = torch.square(
                delta_param_t_plus_1) + torch.square(delta_param_t) + delta_param_t * delta_param_t_plus_1
        else:
            delta_param = self.get_delta(self.param_base_dir, idx)
            delta_param = torch.square(delta_param)
        return delta_param

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


class CheckpointSGDPruner(SGDPruner):

    def __init__(self, model: nn.Module, checkpoint_dir: str, *,
                 momentum: bool = False, prune_threshold: float = 1e-3,
                 causal_weights_batch_size: int = 16,
                 causal_weights_num_epochs: int = 10,
                 device: Union[str, torch.device] = best_device()):
        super().__init__(model, momentum=momentum,
                         prune_threshold=prune_threshold, device=device)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.causal_weights_batch_size = causal_weights_batch_size
        self.causal_weights_num_epochs = causal_weights_num_epochs
        self.loss_checkpoint_dir = os.path.join(self.checkpoint_dir, 'loss')
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
            param_dir, self.loss_checkpoint_dir, momentum=self.momentum)
        dataloader = DataLoader(
            dataset,
            batch_size=self.causal_weights_batch_size,
        )
        for delta_params, delta_losses in dataloader:
            for _ in range(self.causal_weights_num_epochs):
                self.causal_weights_trainers[param].fit(
                    delta_params, delta_losses)


def get_sgd_pruner(
        model: nn.Module, *, mometum: bool = False, online: bool = True,
        checkpoint_dir: str = '') -> Optional[SGDPruner]:
    if online:
        return OnlineSGDPruner(model, momentum=mometum)
    else:
        assert checkpoint_dir != ''
        return CheckpointSGDPruner(model, checkpoint_dir, momentum=mometum)
