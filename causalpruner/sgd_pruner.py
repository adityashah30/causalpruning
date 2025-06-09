import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass
from functools import partial
import gc
import glob
import os
import shutil
import time
from typing import Callable, Optional

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from causalpruner.average import AverageMeter
from causalpruner.base import Pruner, PrunerConfig
from causalpruner.causal_weights_trainer import (
    CausalWeightsTrainerConfig,
    get_causal_weights_trainer,
)

_ZSTATS_PATTERN = "zstats.pth"


@dataclass
class ZStats:
    num_params: int
    mean: torch.Tensor
    std: torch.Tensor
    global_mean: torch.Tensor
    global_std: torch.Tensor


class ParamDataset(Dataset):
    def __init__(
        self,
        weights_base_dir: str,
        loss_base_dir: str,
        train_lr: float,
    ):
        self.weights_base_dir = weights_base_dir
        self.loss_base_dir = loss_base_dir
        file_pattern = "ckpt.*"
        self.num_items = min(
            len(glob.glob(file_pattern, root_dir=self.weights_base_dir)),
            len(glob.glob(file_pattern, root_dir=self.loss_base_dir)),
        )
        self.lr_scaling_factor = train_lr * train_lr
        self.loss_zstats = self._load_zstats(self.loss_base_dir)

    @torch.no_grad()
    def _load_zstats(self, base_dir: str) -> ZStats:
        zstats_dict = torch.load(os.path.join(base_dir, _ZSTATS_PATTERN))
        zstats = ZStats(
            num_params=zstats_dict["num_params"],
            global_mean=zstats_dict["global_mean"],
            global_std=zstats_dict["global_std"],
            mean=zstats_dict["mean"],
            std=zstats_dict["std"],
        )
        return zstats

    @torch.no_grad()
    def __len__(self) -> int:
        return self.num_items

    @torch.no_grad()
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        delta_weights = self.get_delta_param(self.weights_base_dir, idx)
        delta_weights /= self.lr_scaling_factor
        delta_loss = self.get_delta_param(self.loss_base_dir, idx)
        delta_loss /= self.loss_zstats.mean
        return delta_weights, delta_loss

    @torch.no_grad()
    def get_delta_param(
        self, dir: str, idx: int, zstats: Optional[ZStats] = None
    ) -> torch.Tensor:
        file_path = os.path.join(dir, f"ckpt.{idx}")
        val = torch.load(file_path)
        val = torch.nan_to_num(val, nan=0, posinf=0, neginf=0)
        return val


class ZStatsComputer:
    def __init__(self):
        self.sum_x = None
        self.sum_x_squared = None
        self.num_items = 0
        self.num_params = 0
        self.mean_ = None
        self.std_ = None
        self.global_sum_x = 0.0
        self.global_sum_x_squared = 0.0
        self.global_num_items = 0
        self.global_mean_ = None
        self.global_std_ = None

    @torch.no_grad()
    def to(self, device: torch.device) -> "ZStatsComputer":
        def move_tensor(tensor):
            if tensor is not None and tensor.device != device:
                return tensor.to(device)
            else:
                return tensor

        self.sum_x = move_tensor(self.sum_x)
        self.sum_x_squared = move_tensor(self.sum_x_squared)
        self.mean_ = move_tensor(self.mean_)
        self.std_ = move_tensor(self.std_)
        return self

    @torch.no_grad()
    def add(self, x: torch.Tensor):
        if self.sum_x is None:
            self.num_params = torch.numel(x)
            self.sum_x = torch.zeros_like(x)
            self.sum_x_squared = torch.zeros_like(x)
        x_squared = torch.square(x)
        self.sum_x += x
        self.sum_x_squared += x_squared
        self.num_items += 1
        self.global_sum_x += torch.sum(x)
        self.global_sum_x_squared += torch.sum(x_squared)
        self.global_num_items += torch.count_nonzero(x)

    @property
    @torch.no_grad()
    def mean(self) -> torch.Tensor:
        if self.mean_ is None:
            self.mean_ = self.sum_x / self.num_items
        return self.mean_

    @property
    @torch.no_grad()
    def global_mean(self) -> torch.Tensor:
        if self.global_mean_ is None:
            self.global_mean_ = self.global_sum_x / self.global_num_items
        return self.global_mean_

    @property
    @torch.no_grad()
    def std(self) -> torch.Tensor:
        if self.std_ is None:
            variance = (self.sum_x_squared / self.num_items) - torch.square(self.mean)
            std_dev = torch.sqrt(variance)
            self.std_ = std_dev
        return self.std_

    @property
    @torch.no_grad()
    def global_std(self) -> torch.Tensor:
        if self.global_std_ is None:
            variance = (
                self.global_sum_x_squared / self.global_num_items
            ) - torch.square(self.global_mean)
            std_dev = torch.sqrt(variance)
            self.global_std_ = std_dev
        return self.global_std_


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
        self.zstats_computer = ZStatsComputer()

    @torch.no_grad()
    def add_first(self, weight: torch.Tensor):
        self.first_tensor = weight

    @torch.no_grad()
    def add_second(self, weight: torch.Tensor):
        self.second_tensor = weight

    @torch.no_grad()
    def get_delta(self) -> Optional[torch.Tensor]:
        if self.first_tensor is None or self.second_tensor is None:
            return None
        delta = self.second_tensor - self.first_tensor
        result = self.transform(delta)
        self.zstats_computer.add(result)
        return result


@dataclass
class SGDPrunerConfig(PrunerConfig):
    prune_dataloader: DataLoader
    prune_optimizer_lr: float
    num_prune_iterations: int
    num_prune_epochs: int
    threaded_checkpoint_writer: bool
    delete_checkpoint_dir_after_training: bool
    trainer_config: CausalWeightsTrainerConfig
    num_batches_in_epoch: int = -1
    loss_fn: Callable = partial(F.cross_entropy, label_smoothing=0.1)


class SGDPruner(Pruner):
    def __init__(self, config: SGDPrunerConfig):
        super().__init__(config)

        self.prune_dataloader = self.fabric.setup_dataloaders(config.prune_dataloader)

        self.loss_checkpoint_dir = os.path.join(self.checkpoint_dir, "loss")
        self.weights_checkpoint_dir = os.path.join(self.checkpoint_dir, "weights")

        # Setup directories on the global_rank = 0
        if self.fabric.is_global_zero:
            if config.start_clean and os.path.exists(self.checkpoint_dir):
                shutil.rmtree(self.checkpoint_dir)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.loss_checkpoint_dir, exist_ok=True)
            os.makedirs(self.weights_checkpoint_dir, exist_ok=True)
            self.threaded_checkpoint_writer = config.threaded_checkpoint_writer
            if self.threaded_checkpoint_writer:
                self.checkpointer = ThreadPoolExecutor()
                self.checkpoint_futures = []
        self.fabric.barrier()

        self.num_params = 0
        self.params_to_dims = dict()
        for param in self.params:
            self.params_to_dims[param] = np.prod(
                self.modules_dict[param].weight.size(), dtype=int
            )
            self.num_params += self.params_to_dims[param]
        self.trainer_config = config.trainer_config

        self.prune_optimizer = optim.SGD(
            config.model.parameters(), lr=config.prune_optimizer_lr
        )
        self.prune_optimizer_lr = config.prune_optimizer_lr
        self.prune_optimizer = self.fabric.setup_optimizers(self.prune_optimizer)

        self.verbose = config.verbose

    def run_prune_iteration(self) -> None:
        super().run_prune_iteration()
        self.start_iteration()
        config = self.config
        num_batches_in_epoch = config.num_batches_in_epoch
        prune_pbar = tqdm(
            range(config.num_prune_epochs),
            leave=False,
            desc="Pruning",
            dynamic_ncols=True,
        )
        for epoch in prune_pbar:
            config.model.train()
            loss_avg = AverageMeter(self.fabric)
            epoch_pbar = tqdm(
                self.prune_dataloader,
                leave=False,
                desc=f"Prune epoch: {epoch}",
                dynamic_ncols=True,
            )
            batch_counter = 0
            for inputs, labels in epoch_pbar:
                self.prune_optimizer.zero_grad(set_to_none=True)
                outputs = config.model(inputs)
                # Compute loss
                loss = config.loss_fn(outputs, labels)
                self.provide_loss_before_step(loss)
                # Take a gradient step
                self.fabric.backward(loss)
                loss_avg.update(loss)
                self.prune_optimizer.step()
                # Compute loss again
                with torch.no_grad():
                    outputs = config.model(inputs)
                    loss = config.loss_fn(outputs, labels)
                    self.provide_loss_after_step(loss)
                if num_batches_in_epoch > 0 and batch_counter >= num_batches_in_epoch:
                    break
                batch_counter += 1
            epoch_pbar.close()
            loss = loss_avg.mean()
            iter_str = f"{self.iteration}/{config.num_prune_iterations}"
            epoch_str = f"{epoch + 1}/{config.num_prune_epochs}"
            prune_pbar.set_description(
                f"Prune: Iteration {iter_str}; "
                + f"Epoch: {epoch_str}; "
                + f"Loss/Train: {loss:.4f}"
            )
        prune_pbar.close()
        # Shutdown prune_dataloader's worker until next iteration to save resources.
        del self.prune_dataloader._iterator
        self.prune_dataloader._iterator = None
        self.compute_masks()
        self.reset_weights()
        self.reset_params()

    @torch.no_grad()
    def start_iteration(self):
        if self.fabric.is_global_zero:
            iteration_name = f"{self.iteration}"
            loss_dir = os.path.join(self.loss_checkpoint_dir, iteration_name)
            os.makedirs(loss_dir, exist_ok=True)
            weights_dir = os.path.join(self.weights_checkpoint_dir, iteration_name)
            os.makedirs(weights_dir, exist_ok=True)
            self.weights_dir = os.path.join(
                self.weights_checkpoint_dir, f"{self.iteration}"
            )
            self.delta_weights_computer = DeltaComputer(transform=torch.square)
            self.loss_dir = os.path.join(self.loss_checkpoint_dir, f"{self.iteration}")
            self.delta_loss_computer = DeltaComputer()
            self.checkpoint_futures = []
            self.init_model_state = copy.deepcopy(self.config.model.state_dict())
        self.fabric.barrier()

    @torch.no_grad()
    def get_flattened_weight(self) -> torch.Tensor:
        weights = []
        for param in self.params:
            weight = self.modules_dict[param].weight.detach().clone()
            weights.append(weight)
        return torch.cat(tuple(map(torch.flatten, weights)))

    @torch.no_grad()
    def get_flattened_mask(self) -> torch.Tensor:
        def get_mask(param):
            param_module = self.modules_dict[param]
            if not hasattr(param_module, "weight_mask"):
                mask = torch.ones_like(param_module.weight)
            else:
                mask = param_module.weight_mask.detach()
            return torch.flatten(mask)

        return torch.cat(tuple(map(get_mask, self.params)))

    @torch.no_grad()
    def provide_loss_before_step(self, loss: torch.tensor) -> None:
        if self.fabric.is_global_zero:
            torch.cuda.synchronize()
            self.delta_loss_computer.add_first(loss)
            self.delta_weights_computer.add_first(self.get_flattened_weight())
            torch.cuda.synchronize()
        self.fabric.barrier()

    @torch.no_grad()
    def provide_loss_after_step(self, loss: torch.tensor) -> None:
        if self.fabric.is_global_zero:
            torch.cuda.synchronize()
            self.delta_loss_computer.add_second(loss)
            self.delta_weights_computer.add_second(self.get_flattened_weight())

            delta_loss = self.delta_loss_computer.get_delta().to("cpu")
            if delta_loss is not None:
                self.write_tensor(delta_loss, self._get_checkpoint_path(self.loss_dir))

            delta_weights = self.delta_weights_computer.get_delta().to("cpu")
            if delta_weights is not None:
                self.write_tensor(
                    delta_weights, self._get_checkpoint_path(self.weights_dir)
                )
            torch.cuda.synchronize()
        self.fabric.barrier()
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
        self.config.model.load_state_dict(self.init_model_state)
        with torch.no_grad():
            masks = self.get_masks()
            for module_name, module in self.modules_dict.items():
                prune.custom_from_mask(module, "weight", masks[module_name])
        del self.trainer
        self.trainer = None
        torch.cuda.empty_cache()
        gc.collect()

    def train_pruning_weights(self) -> None:
        if self.threaded_checkpoint_writer and self.fabric.is_global_zero:
            concurrent.futures.wait(self.checkpoint_futures)
            del self.checkpoint_futures
            self.checkpoint_futures = []
            self._write_zscaling_params()
        self.fabric.barrier()

        torch.cuda.empty_cache()
        gc.collect()

        self.trainer = get_causal_weights_trainer(
            self.trainer_config,
            self.num_params,
            self.get_flattened_mask(),
            self.iteration + 1,
            self.config.num_prune_iterations,
            self.verbose,
        )

        dataset = ParamDataset(
            self.weights_dir,
            self.loss_dir,
            self.prune_optimizer_lr,
        )

        batch_size = self.trainer_config.batch_size
        if batch_size < 0 or not self.trainer.supports_batch_training():
            batch_size = len(dataset)
        pin_memory = self.trainer_config.pin_memory
        num_workers = self.trainer_config.num_dataloader_workers

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
                mask[start_index:end_index].to(weight.device).reshape_as(weight)
            )
            start_index = end_index
        return masks

    def _get_checkpoint_path(self, checkpoint_dir: str) -> str:
        return os.path.join(checkpoint_dir, f"ckpt.{self.counter}")

    def _write_zscaling_params(self):
        self._write_zscaling_params_from_computer(
            self.delta_loss_computer, self.loss_dir
        )
        self._write_zscaling_params_from_computer(
            self.delta_weights_computer, self.weights_dir
        )

    def _write_zscaling_params_from_computer(
        self, computer: DeltaComputer, dir_path: str
    ):
        dir_path = os.path.join(dir_path, _ZSTATS_PATTERN)
        zstats_computer = computer.zstats_computer.to("cpu")
        torch.save(
            {
                "num_params": zstats_computer.num_params,
                "mean": zstats_computer.mean,
                "std": zstats_computer.std,
                "global_mean": zstats_computer.global_mean,
                "global_std": zstats_computer.global_std,
            },
            dir_path,
        )
