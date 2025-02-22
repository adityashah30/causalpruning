from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from lightning.fabric import Fabric
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
    fabric: Fabric
    init_lr: float
    momentum: bool
    l1_regularization_coeff: float
    max_iter: int
    loss_tol: float
    num_iter_no_change: int
    backend: Literal['sklearn', 'torch'] = 'torch'


class CausalWeightsTrainer(ABC):

    def __init__(self,
                 config: CausalWeightsTrainerConfig):
        self.init_lr = config.init_lr
        self.momentum = config.momentum
        self.l1_regularization_coeff = config.l1_regularization_coeff
        self.max_iter = config.max_iter
        self.loss_tol = config.loss_tol
        self.num_iter_no_change = config.num_iter_no_change
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
                 config: CausalWeightsTrainerConfig):
        super().__init__(config)

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
        Y = np.atleast_2d(Y.detach().cpu().numpy())
        self.trainer.fit(X, Y)
        return self.trainer.n_iter_

    @torch.no_grad()
    def get_non_zero_weights(self) -> torch.Tensor:
        mask = np.copy(self.trainer.coef_)
        if self.momentum:
            mask = mask.reshape((self.weights_dim_multiplier, -1))
        mask = np.atleast_2d(mask)
        mask = np.all(mask == 0, axis=0)
        mask = np.where(mask, 0, 1)
        return torch.tensor(mask)


class CausalWeightsTrainerTorch(CausalWeightsTrainer):

    def __init__(self,
                 config: CausalWeightsTrainerConfig,
                 num_params: int):
        super().__init__(config)
        self.fabric = config.fabric
        self.num_params = num_params
        self.num_params *= self.weights_dim_multiplier
        self.layer = nn.Linear(self.num_params, 1, bias=False)
        self.optimizer = LassoSGD(
            self.layer.parameters(),
            init_lr=self.init_lr, alpha=self.l1_regularization_coeff)
        self.layer, self.optimizer = self.fabric.setup(
            self.layer, self.optimizer)

    def supports_batch_training(self) -> bool:
        return True

    @torch.no_grad()
    def reset(self):
        nn.init.zeros_(self.layer.weight)
        self.optimizer.reset()

    def fit(self, dataloader: DataLoader) -> int:
        dataloader = self.fabric.setup_dataloaders(dataloader)
        self.layer.train()
        best_loss = np.inf
        iter_no_change = 0
        for iter in trange(
                self.max_iter, leave=False, desc=f'Prune weight fitting'):
            sumloss = 0.0
            numlossitems = 0
            pbar_prune = tqdm(dataloader, leave=False)
            for idx, (X, Y) in enumerate(pbar_prune):
                pbar_prune.set_description(f'Pruning weights/{idx}')
                num_items = X.shape[0]
                for idx in range(num_items):
                    output = self.layer(X[idx])
                    label = Y[idx].view(-1)
                    loss = 0.5 * F.mse_loss(output, label, reduction='sum')
                    self.fabric.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    sumloss += loss.item()
                    numlossitems += 1
            total_loss = torch.tensor([sumloss])
            total_loss = self.fabric.all_reduce(total_loss, reduce_op='sum')
            num_loss = torch.tensor([numlossitems])
            num_loss = self.fabric.all_reduce(num_loss, reduce_op='sum')
            loss = total_loss.item() / num_loss.item()
            if loss > (best_loss - self.loss_tol):
                iter_no_change += 1
            else:
                iter_no_change = 0
            if loss < best_loss:
                best_loss = loss
            if iter_no_change >= self.num_iter_no_change:
                return iter + 1
        return self.max_iter

    @torch.no_grad()
    def get_non_zero_weights(self) -> torch.Tensor:
        mask = self.layer.weight
        if self.momentum:
            mask = mask.view(self.weights_dim_multiplier, -1)
        mask = torch.all(mask == 0, dim=0)
        return torch.where(mask, 0, 1)


def get_causal_weights_trainer(
        config: CausalWeightsTrainerConfig, *args) -> CausalWeightsTrainer:
    if config.backend == 'sklearn':
        return CausalWeightsTrainerSklearn(config)
    elif config.backend == 'torch':
        return CausalWeightsTrainerTorch(config, *args)
    raise NotImplementedError('Unsupported backed for CausalWeightsTrainer')
