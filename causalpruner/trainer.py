from dataclasses import dataclass
import os
from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from . import Pruner, best_device


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
    post_prune_optimizer: optim.Optimizer
    data_config: DataConfig
    epoch_config: EpochConfig
    tensorboard_dir: str
    checkpoint_dir: str
    loss_fn: Callable = F.cross_entropy
    device: Union[str, torch.device] = best_device()


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


class Trainer:

    def __init__(self, config: TrainerConfig, pruner: Pruner):
        self.config = config
        self.pruner = pruner
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
        print(f'Pruning method: {self.pruner}')
        self._run_pre_prune()
        self._run_prune()
        self._checkpoint_model('prune')
        self._run_post_prune()
        self._checkpoint_model('post_prune')
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
                f'Pre-Prune: Epoch {epoch+1}/{epoch_config.num_pre_prune_epochs}' + f'; Loss/Train: {loss_avg.avg:.4f}'
                + f'; Accuracy/Test: {accuracy:.4f}')

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
                iter_str = f'{iteration+1}/{epoch_config.num_prune_iterations}'
                epoch_str = f'{epoch+1}/{epoch_config.num_prune_epochs}'
                self.pbar.set_description(
                    f'Prune: Iteration {iter_str}; Epoch: {epoch_str}'
                    + f'; Loss/Train: {loss_avg.avg:.4f}'
                    + f'; Accuracy/Test: {accuracy:.4f}')
            self.pruner.compute_masks()
            self.pruner.reset_weights()
            self.compute_prune_stats()

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
                config.post_prune_optimizer.zero_grad()
                outputs = config.model(inputs)
                loss = config.loss_fn(outputs, labels)
                loss.backward()
                config.post_prune_optimizer.step()
                loss_avg.update(loss.item(), inputs.size(0))
            self.writer.add_scalar(
                'Loss/train', loss_avg.avg, self.global_step)
            accuracy = self.eval_model()
            self.pbar.set_description(
                f'Post-Prune: '
                + f'Epoch {epoch+1}/{epoch_config.num_post_prune_epochs}' +
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
        tqdm.write('\n======================================================\n')
        tqdm.write(f'Global Step: {self.global_step}')
        for (name, param) in self.config.model.named_buffers():
            name = name.rstrip('.weight_mask')
            non_zero = torch.count_nonzero(param)
            total = torch.count_nonzero(torch.ones_like(param))
            pruned = total - non_zero
            percent = 100 * pruned / total
            tqdm.write(f'Name: {name}; Total: {total}; '
                       f'non-zero: {non_zero}; pruned: {pruned}; '
                       f'percent: {percent:.2f}%')
            self.writer.add_scalar(f'{name}/pruned', pruned, self.global_step)
            self.writer.add_scalar(
                f'{name}/pruned_percent', percent, self.global_step)
        tqdm.write('\n======================================================\n')

    def _checkpoint_model(self, id: str):
        fname = os.path.join(self.config.checkpoint_dir, f'model.{id}.ckpt')
        torch.save(self.config.model.state_dict(), fname)
