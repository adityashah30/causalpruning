from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
from sklearn.linear_model import SGDRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from causalpruner.lasso_optimizer import LassoSGD


@dataclass
class CausalWeightsTrainerConfig:
    init_lr: float
    momentum: bool
    l1_regularization_coeff: float
    max_iter: int
    loss_tol: float
    num_iter_no_change: int
    param_name: str = ''
    backend: Literal['sklearn', 'torch'] = 'torch'


class CausalWeightsTrainer(ABC):

    def __init__(self,
                 config: CausalWeightsTrainerConfig,
                 device: Union[str, torch.device]):
        self.param_name = config.param_name
        self.init_lr = config.init_lr
        self.momentum = config.momentum
        self.l1_regularization_coeff = config.l1_regularization_coeff
        self.max_iter = config.max_iter
        self.loss_tol = config.loss_tol
        self.num_iter_no_change = config.num_iter_no_change
        self.device = device
        self.weights_dim_multiplier = 1
        if self.momentum:
            self.weights_dim_multiplier = 3

    def supports_batch_training(self) -> bool:
        return True

    @abstractmethod
    def reset(self):
        raise NotImplementedError('Use the sklearn or pytorch version')

    @abstractmethod
    def fit(self, dataloader: DataLoader) -> int:
        raise NotImplementedError('Use the sklearn or pytorch version')

    @abstractmethod
    def get_non_zero_weights(self) -> torch.Tensor:
        raise NotImplementedError('Use the sklearn or pytorch version')


class CausalWeightsTrainerSklearn(CausalWeightsTrainer):

    def __init__(self,
                 config: CausalWeightsTrainerConfig,
                 device: Union[str, torch.device]):
        super().__init__(config, device)

    def reset(self):
        self.trainer = SGDRegressor(
            loss='squared_error',
            penalty='l1',
            alpha=self.l1_regularization_coeff,
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=self.loss_tol,
            n_iter_no_change=self.num_iter_no_change,
            shuffle=True)

    def supports_batch_training(self) -> bool:
        return False

    def fit(self, dataloader: DataLoader) -> int:
        X, Y = next(iter(dataloader))
        X = X.detach().cpu().numpy()
        Y = Y.detach().cpu().numpy()
        self.trainer.fit(X, Y)
        return self.trainer.n_iter_

    def get_non_zero_weights(self) -> torch.Tensor:
        mask = np.copy(self.trainer.coef_)
        if self.momentum:
            mask = mask.reshape((self.weights_dim_multiplier, -1))
        mask = np.atleast_2d(mask)
        mask = np.all(mask == 0, axis=0)
        mask = np.where(mask, 0, 1)
        return torch.tensor(mask, device=self.device)


class CausalWeightsTrainerTorch(CausalWeightsTrainer):

    def __init__(self,
                 config: CausalWeightsTrainerConfig,
                 device: Union[str, torch.device],
                 model_weights: torch.Tensor):
        super().__init__(config, device)
        self.flattened_dims = np.prod(model_weights.size(), dtype=int)
        self.flattened_dims *= self.weights_dim_multiplier
        self.layer = nn.Linear(
            self.flattened_dims, 1, bias=False, device=self.device)
        self.optimizer = LassoSGD(
            self.layer.parameters(),
            init_lr=self.init_lr, alpha=self.l1_regularization_coeff)

    @torch.no_grad
    def reset(self):
        nn.init.zeros_(self.layer.weight)
        self.optimizer.reset()

    def fit(self, dataloader: DataLoader) -> int:
        self.layer.train()
        best_loss = np.inf
        iter_no_change = 0
        for iter in trange(
                self.max_iter, leave=False, desc=f'Prune weight fitting'):
            sumloss = 0.0
            pbar_prune = tqdm(dataloader, leave=False)
            for idx, (X, Y) in enumerate(pbar_prune):
                pbar_prune.set_description(f'Pruning {self.param_name}/{idx}')
                X = X.to(device=self.device, non_blocking=True)
                Y = Y.to(device=self.device, non_blocking=True)
                num_items = X.shape[0]
                for idx in range(num_items):
                    output = self.layer(X[idx])
                    label = Y[idx].view(-1)
                    loss = 0.5 * F.mse_loss(output, label, reduction='sum')
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    sumloss += loss.item()
            if sumloss > (best_loss - self.loss_tol):
                iter_no_change += 1
            else:
                iter_no_change = 0
            if sumloss < best_loss:
                best_loss = loss
            if iter_no_change >= self.num_iter_no_change:
                return iter + 1
        return self.max_iter

    @torch.no_grad
    def get_non_zero_weights(self) -> torch.Tensor:
        mask = self.layer.weight
        if self.momentum:
            mask = mask.view(self.weights_dim_multiplier, -1)
        mask = torch.all(mask == 0, dim=0)
        return torch.where(mask, 0, 1)


def get_causal_weights_trainer(
        config: CausalWeightsTrainerConfig,
        device: Union[str, torch.device], *args) -> CausalWeightsTrainer:
    if config.backend == 'sklearn':
        return CausalWeightsTrainerSklearn(config, device)
    elif config.backend == 'torch':
        return CausalWeightsTrainerTorch(config, device, *args)
    raise NotImplementedError('Unsupported backed for CausalWeightsTrainer')
