from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
from sklearn.linear_model import SGDRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

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
    backend: Literal['sklearn', 'torch'] = 'sklearn'


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

    @abstractmethod
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> int:
        raise NotImplementedError('Use the sklearn or pytorch version')

    @abstractmethod
    def get_non_zero_weights(self) -> torch.Tensor:
        raise NotImplementedError('Use the sklearn or pytorch version')


class CausalWeightsTrainerSklearn(CausalWeightsTrainer):

    def __init__(self,
                 config: CausalWeightsTrainerConfig,
                 device: Union[str, torch.device]):
        super().__init__(config, device)

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> int:
        X = X.detach().cpu().numpy()
        Y = Y.detach().cpu().numpy()
        self.trainer = SGDRegressor(
            loss='squared_error',
            penalty='l1',
            alpha=self.l1_regularization_coeff,
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=self.loss_tol,
            n_iter_no_change=self.num_iter_no_change,
            shuffle=True)
        self.trainer.fit(X, Y)
        return self.trainer.n_iter_

    def get_non_zero_weights(self) -> torch.Tensor:
        mask = np.copy(self.trainer.coef_)
        if self.momentum:
            mask = mask.reshape((self.weights_dim_multiplier, -1))
        mask = np.atleast_2d(mask)
        mask = np.all(np.abs(mask) <= self.l1_regularization_coeff, axis=0)
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
        nn.init.zeros_(self.layer.weight)
        self.optimizer = LassoSGD(
            self.layer.parameters(),
            init_lr=self.init_lr, alpha=self.l1_regularization_coeff)

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> int:
        self.optimizer.reset()
        self.layer.train()
        X = X.to(self.device)
        Y = Y.to(self.device)
        num_items = X.shape[0]
        best_loss = np.inf
        iter_no_change = 0
        pbar_train = tqdm(total=self.max_iter * num_items, leave=False)
        pbar_train.set_description(f'Pruning {self.param_name}')
        for iter in range(self.max_iter):
            sumloss = 0.0
            indices = torch.randperm(num_items)
            for idx in indices:
                pbar_train.update(1)
                self.optimizer.zero_grad()
                output = self.layer(X[idx])
                label = Y[idx].view(-1)
                loss = 0.5 * F.mse_loss(output, label, reduction='sum')
                loss.backward()
                self.optimizer.step()
                sumloss += loss.item()
            if loss > (best_loss - self.loss_tol):
                iter_no_change += 1
            else:
                iter_no_change = 0
            if loss < best_loss:
                best_loss = loss
            if iter_no_change >= self.num_iter_no_change:
                return iter + 1
        return self.max_iter

    @torch.no_grad
    def get_non_zero_weights(self) -> torch.Tensor:
        mask = self.layer.weight
        if self.momentum:
            mask = mask.view(self.weights_dim_multiplier, -1)
        mask = torch.all(torch.abs(mask) <=
                         self.l1_regularization_coeff, dim=0)
        return torch.where(mask, 0, 1)


def get_causal_weights_trainer(
        config: CausalWeightsTrainerConfig, 
        device: Union[str, torch.device],*args) -> CausalWeightsTrainer:
    if config.backend == 'sklearn':
        return CausalWeightsTrainerSklearn(config, device)
    elif config.backend == 'torch':
        return CausalWeightsTrainerTorch(config, device, *args)
    raise NotImplementedError('Unsupported backed for CausalWeightsTrainer')