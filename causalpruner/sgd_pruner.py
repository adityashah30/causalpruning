from causalpruner.base import Pruner, PrunerConfig, best_device
from causalpruner.lasso_optimizer import LassoSGD

from dataclasses import dataclass
import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm, trange


class CausalWeightsTrainer(nn.Module):

    def __init__(self,
                 model_weights: torch.Tensor,
                 momentum: bool,
                 init_lr: float,
                 l1_regularization_coeff: float,
                 num_iter: int,
                 tol: float = 1e-4,
                 num_iter_no_change: int = 5,
                 device: Union[str, torch.device] = best_device()):
        super().__init__()
        self.momentum = momentum
        self.device = device
        self.l1_regularization_coeff = l1_regularization_coeff
        self.num_iter = num_iter
        self.tol = tol
        self.num_iter_no_change = num_iter_no_change
        self.flattened_dims = np.prod(model_weights.size(), dtype=int)
        self.weights_dim_multiplier = 1
        if self.momentum:
            self.weights_dim_multiplier = 3
        self.flattened_dims *= self.weights_dim_multiplier
        self.layer = nn.Linear(
            self.flattened_dims, 1, bias=False, device=self.device)
        nn.init.zeros_(self.layer.weight)
        self.optimizer = LassoSGD(self.layer.parameters(
        ), init_lr=init_lr, alpha=l1_regularization_coeff)

    @torch.no_grad
    def reset(self):
        self.optimizer.reset()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layer(X)

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> int:
        self.layer.train()
        X = X.to(self.device)
        Y = Y.to(self.device)
        best_loss = np.inf
        iter_no_change = 0
        for index in trange(self.num_iter, leave=False):
            self.optimizer.zero_grad()
            Y_hat = torch.squeeze(self.forward(X), dim=1)
            Y = torch.flatten(Y)
            loss = F.mse_loss(Y_hat, Y)
            loss.backward()
            self.optimizer.step()
            loss = loss.item()
            if loss > (best_loss - self.tol):
                iter_no_change += 1
            else:
                iter_no_change = 0
            if loss < best_loss:
                best_loss = loss
            if iter_no_change >= self.num_iter_no_change:
                return index
        return self.num_iter

    @torch.no_grad
    def get_non_zero_weights(self) -> torch.Tensor:
        mask = self.layer.weight.clone()
        if self.momentum:
            mask = mask.view(self.weights_dim_multiplier, -1)
        mask = torch.all(mask == 0, dim=0)
        return torch.where(mask, 0, 1)


class ParamDataset(Dataset):

    def __init__(
            self, param_base_dir: str, loss_base_dir: str, momentum: bool):
        self.param_base_dir = param_base_dir
        self.loss_base_dir = loss_base_dir
        self.momentum = momentum
        self.num_items = min(len(os.listdir(self.param_base_dir)),
                             len(os.listdir(self.loss_base_dir)))
        self.cache = dict()
        self._compute_mean_and_std()

    @torch.no_grad
    def _compute_mean_and_std(self):
        param, loss = self.get_item(0)
        param_total = param
        loss_total = loss
        param_sq_total = torch.square(param)
        loss_sq_total = torch.square(loss)
        num_items = len(self)
        for idx in range(1, num_items):
            param, loss = self.get_item(idx)
            param_sq, loss_sq = torch.square(param), torch.square(loss)
            param_total += param
            loss_total += loss
            param_sq_total += param_sq
            loss_sq_total += loss_sq
        self.param_mean = (param_total / num_items)
        self.loss_mean = (loss_total / num_items)
        self.param_std = torch.sqrt(
            param_sq_total / num_items - torch.square(self.param_mean))
        self.loss_std = torch.sqrt(
            loss_sq_total / num_items - torch.square(self.loss_mean))

    @torch.no_grad
    def __len__(self) -> int:
        return self.num_items - 2 if self.momentum else self.num_items - 1

    @torch.no_grad
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.cache.get(idx)
        if item is None:
            param, loss = self.get_item(idx)
            param = ((param - self.param_mean) /
                     (self.param_std + 1e-6))
            loss = ((loss - self.loss_mean) /
                    (self.loss_std + 1e-6))
            item = (param, loss)
            self.cache[idx] = item
        return item

    @torch.no_grad
    def get_item(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        delta_param = self.get_delta_param(idx)
        delta_loss = self.get_delta_loss(idx)
        return delta_param, delta_loss

    @torch.no_grad
    def get_delta_param(self, idx: int) -> torch.Tensor:
        if self.momentum:
            return self.get_delta_param_momentum(idx)
        else:
            return self.get_delta_param_vanilla(idx)

    @torch.no_grad
    def get_delta_loss(self, idx: int) -> torch.Tensor:
        if self.momentum:
            idx += 1
        return self.get_delta(self.loss_base_dir, idx)

    @torch.no_grad
    def get_delta_param_vanilla(self, idx: int) -> torch.Tensor:
        delta_param = self.get_delta(self.param_base_dir, idx)
        delta_param = torch.square(delta_param)
        return delta_param

    @torch.no_grad
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

    @torch.no_grad
    def get_delta(self, dir: str, idx: int) -> torch.Tensor:
        first_file_path = os.path.join(dir, f'ckpt.{idx}')
        first_tensor = torch.load(first_file_path)
        second_file_path = os.path.join(dir, f'ckpt.{idx + 1}')
        second_tensor = torch.load(second_file_path)
        return second_tensor - first_tensor


@dataclass
class SGDPrunerConfig(PrunerConfig):
    momentum: bool
    pruner_init_lr: float
    l1_regularization_coeff: float
    causal_weights_num_epochs: int
    causal_weights_batch_size: int


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
        self.causal_weights_batch_size = config.causal_weights_batch_size
        self.causal_weights_trainers = nn.ModuleDict()
        for param in self.params:
            module = self.modules_dict[param]
            self.causal_weights_trainers[param] = CausalWeightsTrainer(
                module.weight, self.momentum, config.pruner_init_lr,
                config.l1_regularization_coeff,
                config.causal_weights_num_epochs, device=self.device)

    @torch.no_grad
    def provide_loss(self, loss: torch.Tensor) -> None:
        loss = loss.detach().clone().cpu()
        torch.save(loss, self._get_checkpoint_path('loss'))
        for param in self.params:
            module = self.modules_dict[param]
            weight = module.weight.detach().clone().cpu()
            torch.save(torch.flatten(weight),
                       self._get_checkpoint_path(param))
        self.counter += 1

    def train_pruning_weights(self) -> None:
        for param, param_dir in self.param_checkpoint_dirs.items():
            self._train_pruning_weights_for_param(param, param_dir)

    def compute_masks(self) -> None:
        self.train_pruning_weights()
        for module_name, module in self.modules_dict.items():
            mask = self._get_mask(module_name)
            prune.custom_from_mask(module, 'weight', mask)

    @torch.no_grad
    def _get_mask(self, name: str) -> torch.Tensor:
        trainer = self.causal_weights_trainers[name]
        mask = trainer.get_non_zero_weights()
        mask = mask.reshape_as(self.modules_dict[name].weight)
        return mask

    @torch.no_grad
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
                               self.momentum)
        dataloader = DataLoader(
            dataset, batch_size=self.causal_weights_batch_size)
        self.causal_weights_trainers[param].reset()
        for index, (delta_params, delta_losses) in enumerate(dataloader):
            num_steps = self.causal_weights_trainers[param].fit(
                delta_params, delta_losses)
            tqdm.write(
                f'{param}/{index} pruning converged in {num_steps} steps')
        torch.cuda.empty_cache()
