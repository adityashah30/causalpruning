import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import glob
import os
import shutil
import time
from typing import Callable, Optional

import numpy as np
import psutil
import torch
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from causalpruner.base import Pruner, PrunerConfig
from causalpruner.causal_weights_trainer import (
    CausalWeightsTrainerConfig,
    get_causal_weights_trainer,
)


_ZSTATS_PATTERN = 'zstats.pth'
_SENTINEL = 1e-10


@dataclass
class ZStats:
    mean: torch.Tensor
    std: torch.Tensor


class ParamDataset(Dataset):

    def __init__(
            self,
            weights_base_dir: str,
            loss_base_dir: str,
            use_zscaling: bool):
        self.weights_base_dir = weights_base_dir
        self.loss_base_dir = loss_base_dir
        file_pattern = 'ckpt.*'
        self.num_items = min(len(glob.glob(file_pattern, root_dir=self.weights_base_dir)),
                             len(glob.glob(file_pattern, root_dir=self.loss_base_dir)))
        self.use_zscaling = use_zscaling
        self.weights_zstats = self._load_zstats(self.weights_base_dir)
        self.num_dimensions = self.weights_zstats.mean.numel()

    @torch.no_grad()
    def _load_zstats(self, base_dir: str) -> ZStats:
        zstats_dict = torch.load(os.path.join(base_dir, _ZSTATS_PATTERN))
        return ZStats(
            mean=zstats_dict['mean'],
            std=zstats_dict['std'],
        )

    @torch.no_grad()
    def __len__(self) -> int:
        return self.num_items

    @torch.no_grad()
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        delta_weights = self.get_delta_param(
            self.weights_base_dir, idx, self.weights_zstats)
        delta_loss = self.get_delta_param(
            self.loss_base_dir, idx, None)
        return delta_weights, delta_loss

    @torch.no_grad()
    def get_delta_param(self,
                        dir: str,
                        idx: int,
                        zstats: Optional[ZStats] = None) -> torch.Tensor:
        param = self.get_tensor(dir, idx)
        if zstats is not None and self.use_zscaling:
            param = (param - zstats.mean) / zstats.std
            param /= (self.num_dimensions * self.num_items)
        return param

    @torch.no_grad()
    def get_tensor(self, dir: str, idx: int) -> torch.Tensor:
        file_path = os.path.join(dir, f'ckpt.{idx}')
        return torch.load(file_path)


class ZStatsComputer:

    def __init__(self):
        self.sum_x = None
        self.sum_x_squared = None
        self.num_items = 0
        self.mean_ = None
        self.std_ = None

    @torch.no_grad()
    def add(self, x: torch.Tensor):
        if self.sum_x is None:
            self.sum_x = torch.zeros_like(x)
            self.sum_x_squared = torch.zeros_like(x)
        self.sum_x += x
        self.sum_x_squared += torch.square(x)
        self.num_items += 1

    @property
    @torch.no_grad()
    def mean(self) -> torch.Tensor:
        if self.mean_ is None:
            self.mean_ = self.sum_x / self.num_items
        return self.mean_

    @property
    @torch.no_grad()
    def std(self) -> torch.Tensor:
        if self.std_ is None:
            variance = (self.sum_x_squared / self.num_items) - \
                torch.square(self.mean)
            std_dev = torch.sqrt(variance)
            abs_std_dev = torch.abs(std_dev)
            non_zero_abs_std_dev = abs_std_dev > 0
            if not torch.any(non_zero_abs_std_dev):
                min_std_dev = _SENTINEL
            else:
                min_std_dev = torch.min(abs_std_dev[non_zero_abs_std_dev])
            std_dev[std_dev == 0] = min_std_dev
            self.std_ = std_dev
        return self.std_


class DeltaComputer:

    def __init__(self,
                 transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = torch.nn.Identity()):
        self.first_tensor = None
        self.second_tensor = None
        self.transform = transform
        self.zstats_computer = ZStatsComputer()

    @torch.no_grad()
    def add(self, weight: torch.Tensor) -> Optional[torch.Tensor]:
        if self.first_tensor is None:
            self.first_tensor = self.transform(weight)
            return None
        self.second_tensor = self.transform(weight)
        delta = self.second_tensor - self.first_tensor
        self.zstats_computer.add(delta)
        self.first_tensor = self.second_tensor
        return delta

    @property
    def mean(self) -> torch.Tensor:
        return self.zstats_computer.mean

    @property
    def std(self) -> torch.Tensor:
        return self.zstats_computer.std


@dataclass
class SGDPrunerConfig(PrunerConfig):
    batch_size: int
    num_dataloader_workers: int
    multiprocess_checkpoint_writer: bool
    delete_checkpoint_dir_after_training: bool
    use_zscaling: bool
    trainer_config: CausalWeightsTrainerConfig


class SGDPruner(Pruner):

    def __init__(self, config: SGDPrunerConfig):
        super().__init__(config)
        self.multiprocess_checkpoint_writer = config.multiprocess_checkpoint_writer
        if self.multiprocess_checkpoint_writer:
            self.checkpointer = ThreadPoolExecutor()
            self.checkpoint_futures = []
        self.use_zscaling = config.use_zscaling
        self.num_params = 0
        self.params_to_dims = dict()
        for param in self.params:
            self.params_to_dims[param] = np.prod(
                self.modules_dict[param].weight.size(), dtype=int)
            self.num_params += self.params_to_dims[param]
        self.trainer_config = config.trainer_config

    @torch.no_grad()
    def start_iteration(self):
        super().start_iteration()
        self.weights_dir = os.path.join(
            self.weights_checkpoint_dir, f'{self.iteration}')
        self.delta_weights_computer = DeltaComputer(transform=torch.square)
        self.loss_dir = os.path.join(
            self.loss_checkpoint_dir, f'{self.iteration}')
        self.delta_loss_computer = DeltaComputer()
        self.checkpoint_futures = []
        self.counter = -1

    @torch.no_grad()
    def provide_loss(self, loss: float) -> None:
        if not self.fabric.is_global_zero:
            return

        loss = torch.tensor(loss)
        delta_loss = self.delta_loss_computer.add(loss)
        if delta_loss is not None:
            self.write_tensor(
                delta_loss, self._get_checkpoint_path(self.loss_dir))

        def get_tensor(param): return torch.flatten(
            self.modules_dict[param].weight.detach().cpu())

        weights = torch.cat(tuple(map(get_tensor, self.params)))
        delta_weights = self.delta_weights_computer.add(weights)
        if delta_weights is not None:
            self.write_tensor(
                delta_weights, self._get_checkpoint_path(self.weights_dir))

        self.counter += 1

    @torch.no_grad()
    def write_tensor(self, tensor: torch.Tensor, path: str):
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
            self._write_zscaling_params()
        self.fabric.barrier()

        self.trainer = get_causal_weights_trainer(
            self.trainer_config, self.num_params)

        dataset = ParamDataset(
            self.weights_dir, self.loss_dir, self.use_zscaling)
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
            persistent_workers=True)

        num_iters = self.trainer.fit(dataloader)
        if num_iters == self.trainer_config.max_iter:
            tqdm.write(f'Pruning failed to converge in ' +
                       f'{num_iters} steps. Consider increasing ' +
                       '--causal_weights_num_epochs')
        else:
            tqdm.write(f'Pruning converged in {num_iters} steps')
        if self.fabric.is_global_zero:
            self._delete_checkpoint_dir(self.weights_dir)
            self._delete_checkpoint_dir(self.loss_dir)
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

    def _get_checkpoint_path(self, checkpoint_dir: str) -> str:
        return os.path.join(checkpoint_dir, f'ckpt.{self.counter}')

    def _write_zscaling_params(self):
        self._write_zscaling_params_from_computer(self.delta_loss_computer,
                                                  self.loss_dir)
        self._write_zscaling_params_from_computer(self.delta_weights_computer,
                                                  self.weights_dir)

    def _write_zscaling_params_from_computer(self, computer: ZStatsComputer, dir_path: str):
        dir_path = os.path.join(dir_path, _ZSTATS_PATTERN)
        torch.save({
            'mean': computer.mean,
            'std': computer.std,
        }, dir_path)
