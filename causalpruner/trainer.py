from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import get_sgd_pruner, best_device


@dataclass
class DataConfig:
    train_dataset: Dataset
    test_dataset: Dataset
    batch_size: int = 8192
    num_workers: int = 2
    shuffle: bool = True
    pin_memory: bool = True


@dataclass
class EpochConfig:
    num_pre_prune_epochs: int = 10
    num_prune_iterations: int = 10
    num_prune_epochs: int = 10
    num_post_prune_epochs: int = 100


@dataclass
class TrainerConfig:
    model: nn.Module
    optimizer: optim.Optimizer
    data_config: DataConfig
    epoch_config: EpochConfig
    tensorboard_dir: str
    loss_fn: Callable = F.cross_entropy
    device: Union[str, torch.device] = best_device()


@dataclass
class PrunerConfig:
    pruner: str
    checkpoint_dir: str
    start_clean: bool


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer(ABC):

    def __init__(self, config: TrainerConfig):
        self.config = config
        # Shortcuts for easy access
        self.data_config = config.data_config
        self.epoch_config = config.epoch_config
        self.total_epochs = (self.epoch_config.num_pre_prune_epochs
                             + self.epoch_config.num_prune_iterations *
                             self.epoch_config.num_prune_epochs
                             + self.epoch_config.num_post_prune_epochs)
        self.device = config.device
        self.pbar = tqdm(total=self.total_epochs)
        self.global_step = -1
        self.writer = SummaryWriter(config.tensorboard_dir)
        self._make_dataloaders()

    def __del__(self):
        self.pbar.close()
        self.writer.close()

    def _make_dataloaders(self):
        data_config = self.data_config
        self.trainloader = DataLoader(
            data_config.train_dataset, batch_size=data_config.batch_size,
            shuffle=data_config.shuffle, pin_memory=data_config.pin_memory,
            num_workers=data_config.num_workers,
            persistent_workers=data_config.num_workers > 0)
        self.testloader = DataLoader(
            data_config.test_dataset, batch_size=data_config.batch_size,
            shuffle=data_config.shuffle, pin_memory=data_config.pin_memory,
            num_workers=data_config.num_workers,
            persistent_workers=data_config.num_workers > 0)

    def run(self):
        self._run_pre_prune()
        self._run_prune()
        self._run_post_prune()
        self.compute_prune_stats()

    def _run_pre_prune(self):
        config = self.config
        epoch_config = self.epoch_config
        for epoch in range(epoch_config.num_pre_prune_epochs):
            self.global_step += 1
            self.pbar.update(1)
            config.model.train()
            loss_avg = AverageMeter()
            for data in self.trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                config.optimizer.zero_grad()
                outputs = config.model(inputs)
                loss = config.loss_fn(outputs, labels)
                loss.backward()
                config.optimizer.step()
                loss_avg.update(loss.item(), inputs.size(0))
            self.writer.add_scalar(
                'Loss/train', loss_avg.avg, self.global_step)
            accuracy = self.eval_model()
            self.pbar.set_description(
                f'Pre-Prune: Epoch {epoch}/{epoch_config.num_pre_prune_epochs}' + f'; Loss/Train: {loss_avg.avg:.4f}'
                + f'; Accuracy/Test: {accuracy:.4f}')

    @abstractmethod
    def _run_prune(self):
        raise NotImplementedError(
            'Trainer is an abstract class. Use an appropriate subclass.')

    def _run_post_prune(self):
        config = self.config
        epoch_config = self.epoch_config
        for epoch in range(epoch_config.num_post_prune_epochs):
            self.global_step += 1
            self.pbar.update(1)
            config.model.train()
            loss_avg = AverageMeter()
            for data in self.trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                config.optimizer.zero_grad()
                outputs = config.model(inputs)
                loss = config.loss_fn(outputs, labels)
                loss.backward()
                config.optimizer.step()
                loss_avg.update(loss.item(), inputs.size(0))
            self.writer.add_scalar(
                'Loss/train', loss_avg.avg, self.global_step)
            accuracy = self.eval_model()
            self.pbar.set_description(
                f'Post-Prune: '
                + f'Epoch {epoch}/{epoch_config.num_post_prune_epochs}' +
                f'; Loss/Train: {loss_avg.avg:.4f}' +
                f'; Accuracy/Test: {accuracy:.4f}')

    def eval_model(self) -> float:
        model = self.config.model
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                total += labels.size(0)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        self.writer.add_scalar('Accuracy/Test', accuracy, self.global_step)
        return accuracy

    def compute_prune_stats(self):
        for (name, param) in self.config.model.named_buffers():
            name = name.rstrip('.weight_mask')
            non_zero = torch.count_nonzero(param)
            total = torch.count_nonzero(torch.ones_like(param))
            pruned = total - non_zero
            frac = 100 * pruned / total
            print(f'Name: {name}; Total: {total}; '
                  f'non-zero: {non_zero}; pruned: {pruned}; '
                  f'percent: {frac:.4f}%')


@dataclass
class SGDPrunerConfig(PrunerConfig):
    momentum: bool
    pruner_lr: float = 1e-3
    prune_threshold: float = 5e-6
    l1_regularization_coeff: float = 1e-5
    causal_weights_num_epochs: int = 500


class SGDPrunerTrainer(Trainer):

    def __init__(self, config: TrainerConfig, pruner_config: SGDPrunerConfig):
        super().__init__(config)
        self.pruner_config = pruner_config
        self.pruner = get_sgd_pruner(
            self.config.model,
            pruner_config.checkpoint_dir,
            pruner_config.momentum,
            pruner_lr=pruner_config.pruner_lr,
            prune_threshold=pruner_config.prune_threshold,
            l1_regularization_coeff=pruner_config.l1_regularization_coeff,
            causal_weights_num_epochs=pruner_config.causal_weights_num_epochs,
            start_clean=pruner_config.start_clean,
        )

    def _run_prune(self):
        config = self.config
        epoch_config = self.epoch_config
        self.pruner.start_pruning()
        for iteration in range(epoch_config.num_prune_iterations):
            self.pruner.start_iteration()
            for epoch in range(epoch_config.num_prune_epochs):
                self.global_step += 1
                self.pbar.update(1)
                config.model.train()
                loss_avg = AverageMeter()
                for data in self.trainloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(
                        self.device), labels.to(
                        self.device)
                    config.optimizer.zero_grad()
                    outputs = config.model(inputs)
                    loss = config.loss_fn(outputs, labels)
                    self.pruner.provide_loss(loss)
                    loss.backward()
                    config.optimizer.step()
                    loss_avg.update(loss.item(), inputs.size(0))
                self.writer.add_scalar(
                    'Loss/train', loss_avg.avg, self.global_step)
                accuracy = self.eval_model()
                iter_str = f'{iteration}/{epoch_config.num_prune_iterations}'
                epoch_str = f'{epoch}/{epoch_config.num_prune_epochs}'
                self.pbar.set_description(
                    f'Prune: Iteration {iter_str}; Epoch: {epoch_str}'
                    + f'; Loss/Train: {loss_avg.avg:.4f}'
                    + f'; Accuracy/Test: {accuracy:.4f}')
            self.pruner.compute_masks()
            self.pruner.reset_weights()
            self.compute_prune_stats()
