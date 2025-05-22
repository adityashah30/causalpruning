from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from lightning.fabric import Fabric
import numpy as np
from sklearn.linear_model import SGDRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

# from causalpruner.lasso_optimizer import LassoSGD


@dataclass
class CausalWeightsTrainerConfig:
    fabric: Fabric
    init_lr: float
    l1_regularization_coeff: float
    prune_amount: float
    max_iter: int
    loss_tol: float
    num_iter_no_change: int
    initialization: Literal["zeros", "xavier_normal"] = "zeros"
    backend: Literal["sklearn", "torch"] = "torch"


class CausalWeightsTrainer(ABC):
    def __init__(self, config: CausalWeightsTrainerConfig):
        self.init_lr = config.init_lr
        self.l1_regularization_coeff = config.l1_regularization_coeff
        self.prune_amount = config.prune_amount
        self.max_iter = config.max_iter
        self.loss_tol = config.loss_tol
        self.num_iter_no_change = config.num_iter_no_change

    def supports_batch_training(self) -> bool:
        return True

    @abstractmethod
    def fit(self, dataloader: DataLoader) -> int:
        raise NotImplementedError("Use the sklearn or pytorch version")

    @abstractmethod
    def get_non_zero_weights(self) -> torch.Tensor:
        raise NotImplementedError("Use the sklearn or pytorch version")


class CausalWeightsTrainerSklearn(CausalWeightsTrainer):
    def __init__(self, config: CausalWeightsTrainerConfig):
        super().__init__(config)
        self.trainer = SGDRegressor(
            loss="squared_error",
            penalty="l1",
            alpha=self.l1_regularization_coeff,
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=self.loss_tol,
            n_iter_no_change=self.num_iter_no_change,
            shuffle=True,
        )

    def supports_batch_training(self) -> bool:
        return False

    def fit(self, dataloader: DataLoader) -> int:
        X, Y = next(iter(dataloader))
        X = X.cpu().numpy()
        Y = np.ravel(Y.cpu().numpy())
        self.trainer.fit(X, Y)
        return self.trainer.n_iter_

    @torch.no_grad()
    def get_non_zero_weights(self) -> torch.Tensor:
        mask = np.copy(self.trainer.coef_)
        mask = np.atleast_2d(mask)
        mask = np.all(mask == 0, axis=0)
        mask = np.where(mask, 0, 1)
        return torch.tensor(mask)


class CausalWeightsTrainerTorch(CausalWeightsTrainer):
    def __init__(
        self,
        config: CausalWeightsTrainerConfig,
        num_params: int,
        initial_mask: torch.Tensor,
    ):
        super().__init__(config)
        self.fabric = config.fabric
        self.num_params = num_params
        self.layer = nn.Linear(self.num_params, 1, bias=False)
        initialization = config.initialization.lower()
        if initialization == "zeros":
            nn.init.zeros_(self.layer.weight)
        elif initialization == "xavier_normal":
            nn.init.xavier_normal_(self.layer.weight)
        mask = initial_mask.view_as(self.layer.weight)
        prune.custom_from_mask(self.layer, "weight", mask)
        # self.l1_regularization_coeff = 1.0 / num_params
        self.l2_regularization_coeff = self.l1_regularization_coeff
        self.optimizer = optim.SGD(
            self.layer.parameters(),
            lr=self.init_lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )
        self.layer, self.optimizer = self.fabric.setup(
            self.layer, self.optimizer)

    def supports_batch_training(self) -> bool:
        return True

    def _l1_penalty(self) -> float:
        return torch.norm(self.layer.weight, p=1)

    def _l2_penalty(self) -> float:
        return torch.norm(self.layer.weight, p=2)

    def fit(self, dataloader: DataLoader) -> int:
        dataloader = self.fabric.setup_dataloaders(dataloader)
        self.layer.train()
        best_loss = np.inf
        iter_no_change = 0

        conv_iter = self.max_iter
        for iter in trange(
            self.max_iter, leave=False, desc="Prune weight fitting", dynamic_ncols=True
        ):
            sumloss = 0.0
            numlossitems = 0
            for X, Y in tqdm(dataloader, leave=False, dynamic_ncols=True):
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.layer(X)
                Y = Y.view(outputs.size())
                loss = (
                    F.huber_loss(outputs, Y, reduction="mean", delta=1e-3)
                    + self.l1_regularization_coeff * self._l1_penalty()
                    + self.l2_regularization_coeff * self._l2_penalty()
                )
                self.fabric.backward(loss)
                self.optimizer.step()
                sumloss += loss.item()
                numlossitems += 1
            total_loss = torch.tensor([sumloss])
            total_loss = self.fabric.all_reduce(total_loss, reduce_op="sum")
            num_loss = torch.tensor([numlossitems])
            num_loss = self.fabric.all_reduce(num_loss, reduce_op="sum")
            num_items = num_loss.item()
            loss = total_loss.item() / num_items
            tqdm.write(
                f"Pruning iter: {iter + 1}; "
                + f"loss: {loss:.4e}; best_loss: {best_loss:.4e}"
            )
            if loss > (best_loss - self.loss_tol):
                iter_no_change += 1
            else:
                iter_no_change = 0
            if loss < best_loss:
                best_loss = loss
                best_model_state = deepcopy(self.layer.state_dict())
            if iter_no_change >= self.num_iter_no_change:
                conv_iter = iter + 1
                break
        self.layer.load_state_dict(best_model_state)
        prune.l1_unstructured(
            self.layer.module, name="weight", amount=self.prune_amount
        )
        return conv_iter

    @torch.no_grad()
    def get_non_zero_weights(self) -> torch.Tensor:
        return torch.flatten(self.layer.weight_mask)


def get_causal_weights_trainer(
    config: CausalWeightsTrainerConfig, *args
) -> CausalWeightsTrainer:
    if config.backend == "sklearn":
        return CausalWeightsTrainerSklearn(config)
    elif config.backend == "torch":
        return CausalWeightsTrainerTorch(config, *args)
    raise NotImplementedError("Unsupported backed for CausalWeightsTrainer")
