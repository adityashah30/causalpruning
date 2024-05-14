from causalpruner import base

import os
import shutil
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader


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

    def train_pruning_weights(self) -> None:
        self.counter = 0
        delta_losses = torch.diff(self.losses)
        for param_name in self.params_dict:
            weights = self.weights[param_name]
            delta_weights = torch.diff(weights, dim=0)
            delta_weights_squared = torch.pow(delta_weights, 2)
            trainer = self.causal_weights_trainers[param_name]
            for _ in range(self.causal_weights_num_epochs):
                trainer.fit(delta_weights_squared, delta_losses)


class ParamDataset(Dataset):

    def __init__(self, param_base_dir: str, loss_base_dir: str, transform=None):
        self.param_base_dir = param_base_dir
        self.loss_base_dir = loss_base_dir
        self.num_items = min(len(os.listdir(self.param_base_dir)),
                             len(os.listdir(self.loss_base_dir)))
        self.transform = transform

    def __len__(self) -> int:
        return self.num_items - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        param_delta = self.get_delta(self.param_base_dir, idx)
        loss_delta = self.get_delta(self.loss_base_dir, idx)
        if self.transform:
            param_delta = self.transform(param_delta)
        return param_delta, loss_delta

    def get_delta(self, dir: str, idx: int) -> torch.Tensor:
        first_file_path = os.path.join(dir, f'ckpt.{idx}')
        first_tensor = torch.flatten(torch.load(first_file_path))
        second_file_path = os.path.join(dir, f'ckpt.{idx + 1}')
        second_tensor = torch.flatten(torch.load(second_file_path))
        return first_tensor - second_tensor


class CheckpointSGDPruner(SGDPruner):

    def __init__(self, model: nn.Module, checkpoint_dir: str,
                 *, start_clean: bool = True,
                 end_clean: bool = False,
                 prune_threshold: float = 1e-3,
                 causal_weights_batch_size: int = 16,
                 causal_weights_num_epochs: int = 10):
        super().__init__(model, prune_threshold)
        self.checkpoint_dir = checkpoint_dir
        self.end_clean = end_clean
        if start_clean:
            self._remove_checkpoint_dir()
        self._ensure_checkpoint_dir_exists()
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

    def __del__(self):
        if self.end_clean:
            self._remove_checkpoint_dir()

    def _remove_checkpoint_dir(self):
        shutil.rmtree(self.checkpoint_dir)

    def _ensure_checkpoint_dir_exists(self):
        if os.path.isdir(self.checkpoint_dir):
            return
        os.makedirs(self.checkpoint_dir)

    def provide_loss(self, loss: torch.Tensor) -> None:
        torch.save(loss, self.get_checkpoint_path('loss'))
        for param_name, param in self.params_dict.items():
            torch.save(param, self.get_checkpoint_path(param_name))
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
            transform=lambda d: torch.pow(d, 2))
        dataloader = DataLoader(
            dataset,
            batch_size=self.causal_weights_batch_size,
            shuffle=True
        )
        for _ in range(self.causal_weights_num_epochs):
            for param_delta, loss_delta in dataloader:
                self.causal_weights_trainers[param].fit(
                    param_delta, loss_delta)


def get_sgd_pruner(
        model: nn.Module, *, online: bool = True,
        checkpoint_dir: str = '') -> Optional[SGDPruner]:
    if online:
        return OnlineSGDPruner(model)
    else:
        assert checkpoint_dir != ''
        return CheckpointSGDPruner(model, checkpoint_dir)
