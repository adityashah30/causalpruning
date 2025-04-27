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


class ParamDataset(Dataset):
    def __init__(self, weights_base_dir: str, loss_base_dir: str):
        self.weights_base_dir = weights_base_dir
        self.loss_base_dir = loss_base_dir
        file_pattern = "ckpt.*"
        self.num_items = min(
            len(glob.glob(file_pattern, root_dir=self.weights_base_dir)),
            len(glob.glob(file_pattern, root_dir=self.loss_base_dir)),
        )

    @torch.no_grad()
    def __len__(self) -> int:
        return self.num_items

    @torch.no_grad()
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        delta_weights = self.get_delta_param(self.weights_base_dir, idx)
        delta_loss = self.get_delta_param(self.loss_base_dir, idx)
        return delta_weights, delta_loss

    @torch.no_grad()
    def get_delta_param(self, dir: str, idx: int) -> torch.Tensor:
        file_path = os.path.join(dir, f"ckpt.{idx}")
        return torch.load(file_path)


class DeltaComputer:
    def __init__(
        self,
        transform: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = torch.nn.Identity(),
    ):
        self.first_tensor = None
        self.second_tensor = None
        self.transform = transform

    @torch.no_grad()
    def add_first(self, weight: torch.Tensor):
        self.first_tensor = self.transform(weight)

    @torch.no_grad()
    def add_second(self, weight: torch.Tensor):
        self.second_tensor = self.transform(weight)

    @torch.no_grad()
    def get_delta(self) -> Optional[torch.Tensor]:
        if self.first_tensor is None or self.second_tensor is None:
            return None
        delta = self.second_tensor - self.first_tensor
        return delta


@dataclass
class SGDPrunerConfig(PrunerConfig):
    batch_size: int
    num_dataloader_workers: int
    pin_memory: bool
    threaded_checkpoint_writer: bool
    delete_checkpoint_dir_after_training: bool
    trainer_config: CausalWeightsTrainerConfig


class SGDPruner(Pruner):
    def __init__(self, config: SGDPrunerConfig):
        super().__init__(config)
        self.threaded_checkpoint_writer = config.threaded_checkpoint_writer
        if self.threaded_checkpoint_writer:
            self.checkpointer = ThreadPoolExecutor()
            self.checkpoint_futures = []
        self.num_params = 0
        self.params_to_dims = dict()
        for param in self.params:
            self.params_to_dims[param] = np.prod(
                self.modules_dict[param].weight.size(), dtype=int
            )
            self.num_params += self.params_to_dims[param]
        self.trainer_config = config.trainer_config

    @torch.no_grad()
    def start_iteration(self):
        super().start_iteration()
        self.weights_dir = os.path.join(
            self.weights_checkpoint_dir, f"{self.iteration}"
        )
        self.delta_weights_computer = DeltaComputer(transform=torch.square)
        self.loss_dir = os.path.join(
            self.loss_checkpoint_dir, f"{self.iteration}")
        self.delta_loss_computer = DeltaComputer()
        self.checkpoint_futures = []
        self.counter = 0

    @torch.no_grad()
    def get_flattened_weight(self) -> torch.Tensor:
        def get_tensor(param):
            return torch.flatten(self.modules_dict[param].weight.detach().cpu())

        return torch.cat(tuple(map(get_tensor, self.params)))

    @torch.no_grad()
    def get_flattened_mask(self) -> torch.Tensor:
        def get_mask(param):
            param_module = self.modules_dict[param]
            if not hasattr(param_module, "weight_mask"):
                mask = torch.ones_like(param_module.weight)
            else:
                mask = param_module.weight_mask
            return torch.flatten(mask.detach().cpu())

        return torch.cat(tuple(map(get_mask, self.params)))

    @torch.no_grad()
    def provide_loss_before_step(self, loss: float) -> None:
        if not self.fabric.is_global_zero:
            return

        loss = torch.tensor(loss)
        self.delta_loss_computer.add_first(loss)
        self.delta_weights_computer.add_first(self.get_flattened_weight())

    @torch.no_grad()
    def provide_loss_after_step(self, loss: float) -> None:
        if not self.fabric.is_global_zero:
            return

        loss = torch.tensor(loss)
        self.delta_loss_computer.add_second(loss)
        self.delta_weights_computer.add_second(self.get_flattened_weight())

        delta_loss = self.delta_loss_computer.get_delta()
        if delta_loss is not None:
            self.write_tensor(
                delta_loss, self._get_checkpoint_path(self.loss_dir))

        delta_weights = self.delta_weights_computer.get_delta()
        if delta_weights is not None:
            self.write_tensor(
                delta_weights, self._get_checkpoint_path(self.weights_dir)
            )

        self.counter += 1

    @torch.no_grad()
    def write_tensor(self, tensor: torch.Tensor, path: str):
        if not self.threaded_checkpoint_writer:
            torch.save(tensor, path)
            return
        # Use threading to write checkpoint
        while psutil.virtual_memory().percent >= 99.5:
            time.sleep(0.1)  # 100ms
        future = self.checkpointer.submit(torch.save, tensor, path)
        self.checkpoint_futures.append(future)

    def compute_masks(self):
        self.train_pruning_weights()
        with torch.no_grad():
            masks = self.get_masks()
            for module_name, module in self.modules_dict.items():
                prune.custom_from_mask(module, "weight", masks[module_name])

    def train_pruning_weights(self) -> None:
        if self.threaded_checkpoint_writer and self.fabric.is_global_zero:
            concurrent.futures.wait(self.checkpoint_futures)
            del self.checkpoint_futures
            self.checkpoint_futures = []
        self.fabric.barrier()

        self.trainer = get_causal_weights_trainer(
            self.trainer_config, self.num_params, self.get_flattened_mask()
        )

        dataset = ParamDataset(self.weights_dir, self.loss_dir)

        batch_size = self.config.batch_size
        if batch_size < 0 or not self.trainer.supports_batch_training():
            batch_size = len(dataset)
        num_workers = self.config.num_dataloader_workers
        pin_memory = self.config.pin_memory

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        num_iters = self.trainer.fit(dataloader)
        if num_iters == self.trainer_config.max_iter:
            tqdm.write(
                "Pruning failed to converge in "
                + f"{num_iters} steps. Consider increasing "
                + "--causal_weights_num_epochs"
            )
        else:
            tqdm.write(f"Pruning converged in {num_iters} steps")
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
            masks[param] = (
                mask[start_index:end_index]
                .reshape_as(weight)
                .to(weight.device, non_blocking=True)
            )
            start_index = end_index
        return masks

    def _get_checkpoint_path(self, checkpoint_dir: str) -> str:
        return os.path.join(checkpoint_dir, f"ckpt.{self.counter}")
