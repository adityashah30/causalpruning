import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import dataclass
import os
import shutil
import time

import psutil
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from causalpruner.base import Pruner, PrunerConfig, best_device
from causalpruner.causal_weights_trainer import (
    CausalWeightsTrainerConfig,
    get_causal_weights_trainer,
)


class ParamDataset(Dataset):

    def __init__(
            self,
            param_base_dir: str,
            loss_base_dir: str,
            momentum: bool,
            preload: bool = True):
        self.param_base_dir = param_base_dir
        self.loss_base_dir = loss_base_dir
        self.momentum = momentum
        self.num_items = min(len(os.listdir(self.param_base_dir)),
                             len(os.listdir(self.loss_base_dir)))
        self.preload = preload
        if preload:
            self.preload_data()

    @torch.no_grad
    def preload_data(self):
        params_list, loss_list = [], []
        for index in range(len(self)):
            param, loss = self.get_item(index)
            params_list.append(param)
            loss_list.append(loss)
        self.preloaded_params = torch.stack(params_list)
        self.preloaded_losses = torch.stack(loss_list)

    @torch.no_grad
    def __len__(self) -> int:
        return self.num_items - 2 if self.momentum else self.num_items - 1

    @torch.no_grad
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.preload:
            param = self.preloaded_params[idx]
            loss = self.preloaded_losses[idx]
        else:
            param, loss = self.get_item(idx)
        return (param, loss)

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
    preload: bool
    batch_size: int
    delete_checkpoint_dir_after_training: bool
    trainer_config: CausalWeightsTrainerConfig


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
        self.trainer_config = config.trainer_config
        self.causal_weights_trainers = dict()
        self.checkpointer = ProcessPoolExecutor(
            max_workers=min(psutil.cpu_count() - 2, 6))
        self.checkpoint_futures = []
        for param_name in self.params:
            trainer_config_copy = copy.deepcopy(self.trainer_config)
            trainer_config_copy.param_name = param_name
            module = self.modules_dict[param_name]
            trainer = get_causal_weights_trainer(
                trainer_config_copy, self.device, module.weight)
            self.causal_weights_trainers[param_name] = trainer

    @torch.no_grad
    def start_iteration(self):
        super().start_iteration()
        self.checkpoint_futures = []

    @torch.no_grad
    def provide_loss(self, loss: torch.Tensor) -> None:
        loss = loss.detach().cpu()
        self.write_tensor(loss, self._get_checkpoint_path('loss'))
        for param in self.params:
            module = self.modules_dict[param]
            weight = module.weight.detach().cpu()
            weight = torch.flatten(weight)
            self.write_tensor(weight, self._get_checkpoint_path(param))
        self.counter += 1

    @torch.no_grad
    def write_tensor(self, tensor: torch.Tensor, path: str):
        while psutil.virtual_memory().percent >= 95:
            time.sleep(0.1)  # 100ms
        future = self.checkpointer.submit(torch.save, tensor, path)
        self.checkpoint_futures.append(future)

    def compute_masks(self):
        self.train_pruning_weights()
        for module_name, module in self.modules_dict.items():
            mask = self.get_mask(module_name)
            prune.custom_from_mask(module, 'weight', mask)

    def train_pruning_weights(self) -> None:
        params = self.param_checkpoint_dirs.items()
        concurrent.futures.wait(self.checkpoint_futures)
        pbar_pruning = tqdm(total=len(params), leave=False)
        for param, param_dir in params:
            pbar_pruning.set_description(param)
            pbar_pruning.update(1)
            self._train_pruning_weights_for_param(param, param_dir)
        self._delete_checkpoint_dirs()

    def _delete_checkpoint_dirs(self):
        if not self.config.delete_checkpoint_dir_after_training:
            return
        loss_dir = os.path.join(
            self.loss_checkpoint_dir, f'{self.iteration}')
        shutil.rmtree(loss_dir)
        for param_dir in self.param_checkpoint_dirs.values():
            param_dir = os.path.join(param_dir, f'{self.iteration}')
            shutil.rmtree(param_dir)

    @torch.no_grad
    def get_mask(self, name: str) -> torch.Tensor:
        trainer = self.causal_weights_trainers[name]
        mask = trainer.get_non_zero_weights()
        mask = mask.reshape_as(self.modules_dict[name].weight)
        return mask

    @ torch.no_grad
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
                               self.trainer_config.momentum,
                               preload=self.config.preload)
        trainer = self.causal_weights_trainers[param]
        trainer.reset()
        batch_size = self.config.batch_size
        if batch_size < 0 or not trainer.supports_batch_training():
            batch_size = len(dataset)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, pin_memory=True)
        pbar_prune = tqdm(dataloader, leave=False)
        for idx, (delta_params, delta_losses) in enumerate(pbar_prune):
            pbar_prune.set_description(f'Pruning {param}/{idx}')
            num_iters = trainer.fit(delta_params, delta_losses)
        if num_iters == self.trainer_config.max_iter:
            tqdm.write(f'{param} pruning failed to converge in ' +
                       f'{num_iters} steps. Consider increasing ' +
                       '--causal_weights_num_epochs')
        else:
            tqdm.write(f'{param} pruning converged in {num_iters} steps')
