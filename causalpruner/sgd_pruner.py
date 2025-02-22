import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import shutil
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from causalpruner.base import Pruner, PrunerConfig
from causalpruner.causal_weights_trainer import (
    CausalWeightsTrainerConfig,
    get_causal_weights_trainer,
)


class ParamDataset(Dataset):

    def __init__(
            self,
            weights_base_dir: str,
            loss_base_dir: str,
            momentum: bool):
        self.weights_base_dir = weights_base_dir
        self.loss_base_dir = loss_base_dir
        self.momentum = momentum
        self.num_items = min(len(os.listdir(self.weights_base_dir)),
                             len(os.listdir(self.loss_base_dir)))

    @torch.no_grad()
    def __len__(self) -> int:
        return self.num_items - 2 if self.momentum else self.num_items - 1

    @torch.no_grad()
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        delta_weights = self.get_delta_weights(idx)
        delta_loss = self.get_delta_loss(idx)
        return delta_weights, delta_loss

    @torch.no_grad()
    def get_delta_weights(self, idx: int) -> torch.Tensor:
        if self.momentum:
            return self.get_delta_weights_momentum(idx)
        else:
            return self.get_delta_weights_vanilla(idx)

    @torch.no_grad()
    def get_delta_loss(self, idx: int) -> torch.Tensor:
        if self.momentum:
            idx += 1
        return self.get_delta(self.loss_base_dir, idx)

    @torch.no_grad()
    def get_delta_weights_vanilla(self, idx: int) -> torch.Tensor:
        delta_weights = self.get_delta(self.weights_base_dir, idx)
        delta_weights = torch.square(delta_weights)
        return delta_weights

    @torch.no_grad()
    def get_delta_weights_momentum(self, idx: int) -> torch.Tensor:
        delta_weights_t = self.get_delta(self.weights_base_dir, idx)
        delta_weights_t_sq = torch.square(delta_weights_t)
        delta_weights_t_plus_1 = self.get_delta(
            self.weights_base_dir, idx + 1)
        delta_weights_t_plus_1_sq = torch.square(delta_weights_t_plus_1)
        delta_weights_t_t_plus_1 = delta_weights_t * delta_weights_t_plus_1
        return torch.cat(
            (delta_weights_t_sq, delta_weights_t_plus_1_sq,
             delta_weights_t_t_plus_1))

    @torch.no_grad()
    def get_delta(self, dir: str, idx: int) -> torch.Tensor:
        first_file_path = os.path.join(dir, f'ckpt.{idx}')
        first_tensor = torch.load(first_file_path)
        second_file_path = os.path.join(dir, f'ckpt.{idx + 1}')
        second_tensor = torch.load(second_file_path)
        return second_tensor - first_tensor


@dataclass
class SGDPrunerConfig(PrunerConfig):
    batch_size: int
    num_dataloader_workers: int
    multiprocess_checkpoint_writer: bool
    delete_checkpoint_dir_after_training: bool
    trainer_config: CausalWeightsTrainerConfig


class SGDPruner(Pruner):

    def __init__(self, config: SGDPrunerConfig):
        super().__init__(config)
        self.counter = 0
        self.multiprocess_checkpoint_writer = config.multiprocess_checkpoint_writer
        if self.multiprocess_checkpoint_writer:
            self.checkpointer = ThreadPoolExecutor()
            self.checkpoint_futures = []
        num_params = 0
        self.params_to_dims = dict()
        for param in self.params:
            self.params_to_dims[param] = np.prod(
                self.modules_dict[param].weight.size(), dtype=int)
            num_params += self.params_to_dims[param]
        self.trainer_config = config.trainer_config
        self.trainer = get_causal_weights_trainer(
            self.trainer_config, num_params)

    @torch.no_grad()
    def start_iteration(self):
        super().start_iteration()
        self.checkpoint_futures = []

    @torch.no_grad()
    def provide_loss(self, loss: float) -> None:
        if not self.fabric.is_global_zero:
            return
        self.write_tensor(torch.tensor(
            loss), self._get_checkpoint_path('loss'))
        get_tensor = lambda param: torch.flatten(
            self.modules_dict[param].weight.detach().to(
                device='cpu', non_blocking=True))
        weights = torch.cat(tuple(map(get_tensor, self.params)))
        self.write_tensor(weights, self._get_checkpoint_path('weights'))
        self.counter += 1

    @torch.no_grad()
    def write_tensor(self, tensor: torch.Tensor, path: str):
        tensor = torch.flatten(tensor)
        if not self.multiprocess_checkpoint_writer:
            torch.save(tensor, path)
            return
        # Use multiprocessing to write checkpoint
        while psutil.virtual_memory().percent >= 99.5:
            time.sleep(0.1)  # 100ms
        future = self.checkpointer.submit(torch.save, tensor, path)
        self.checkpoint_futures.append(future)

    def compute_masks(self):
        self.train_pruning_weights()
        with torch.no_grad():
            masks = self.get_masks()
            for module_name, module in self.modules_dict.items():
                prune.custom_from_mask(module, 'weight', masks[module_name])

    def train_pruning_weights(self) -> None:
        if self.multiprocess_checkpoint_writer and self.fabric.is_global_zero:
            concurrent.futures.wait(self.checkpoint_futures)
            del self.checkpoint_futures
            self.checkpoint_futures = []
        self.fabric.barrier()
        weights_dir = os.path.join(self.weights_checkpoint_dir, f'{self.iteration}')
        loss_dir = os.path.join(self.loss_checkpoint_dir, f'{self.iteration}')
        dataset = ParamDataset(weights_dir, loss_dir, 
                               self.trainer_config.momentum)
        self.trainer.reset()
        batch_size = self.config.batch_size
        if batch_size < 0 or not self.trainer.supports_batch_training():
            batch_size = len(dataset)
        num_workers = self.config.num_dataloader_workers
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0)
        num_iters = self.trainer.fit(dataloader)
        if num_iters == self.trainer_config.max_iter:
            tqdm.write(f'Pruning failed to converge in ' +
                       f'{num_iters} steps. Consider increasing ' +
                       '--causal_weights_num_epochs')
        else:
            tqdm.write(f'Pruning converged in {num_iters} steps')
        if self.fabric.is_global_zero:
            self._delete_checkpoint_dir(weights_dir)
            self._delete_checkpoint_dir(loss_dir)
        self.fabric.barrier()

    def _delete_checkpoint_dir(self, dirpath: str):
        if not self.config.delete_checkpoint_dir_after_training:
            return
        shutil.rmtree(dirpath)

    @torch.no_grad()
    def get_masks(self) -> dict[str, torch.Tensor]:
        mask = self.trainer.get_non_zero_weights()
        masks = dict()
        start_index, end_index = 0, 0
        for param in self.params:
            end_index += self.params_to_dims[param]
            weight = self.modules_dict[param].weight
            masks[param] = mask[start_index:end_index].reshape_as(
                weight).to(weight.device, non_blocking=True)
            start_index = end_index
        return masks

    def _get_checkpoint_path(self, param_name: str) -> str:
        if param_name == 'loss':
            path = self.loss_checkpoint_dir
        elif param_name == 'weights':
            path = self.weights_checkpoint_dir
        return os.path.join(path, f'{self.iteration}', f'ckpt.{self.counter}')
