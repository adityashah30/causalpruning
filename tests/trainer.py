from dataclasses import dataclass
import os
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
from tqdm.auto import tqdm

from causalpruner import Pruner, best_device


@dataclass
class DataConfig:
    train_dataset: Dataset
    test_dataset: Dataset
    num_classes: int
    batch_size: int
    num_workers: int
    shuffle: bool


@dataclass
class EpochConfig:
    num_pre_prune_epochs: int
    num_prune_iterations: int
    num_prune_epochs: int
    num_train_epochs: int
    # Number of steps to run the dataloader. Note that we run over the entire
    # dataset by default -- which will happen for any value < 0.
    # Use a positive value to limit iterating to a specific number of batches.
    num_batches_in_epoch: int = -1
    # Take a grad every `grad_step_num_batches`. Setting a value > 1 will cause
    # gradient accumulation.
    grad_step_num_batches: int = 1
    # The frequency with which the tqdm progress bar is updated. Set to a
    # larger value for a fast iteration -- else tqdm update will be the
    # bottleneck.
    tqdm_update_frequency: int = 1


@dataclass
class TrainerConfig:
    hparams: dict[str, str]
    model: nn.Module
    prune_optimizer: optim.Optimizer
    train_optimizer: optim.Optimizer
    train_convergence_loss_tolerance: float
    train_loss_num_epochs_no_change: int
    data_config: DataConfig
    epoch_config: EpochConfig
    tensorboard_dir: str
    checkpoint_dir: str
    loss_fn: Callable = F.cross_entropy
    verbose: bool = True
    train_only: bool = False
    model_to_load_for_training: str = 'prune.final'
    model_to_save_after_training: str = 'trained'
    device: Union[str, torch.device] = best_device()


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@dataclass
class EvalMetrics:
    accuracy: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor
    f1_score: torch.Tensor


def write_scalars(
        writer: SummaryWriter, tag: str, val: torch.Tensor, global_step: int):
    for idx, value in enumerate(val):
        writer.add_scalar(f'{tag}/class_{idx}',  value, global_step)


class Trainer:

    def __init__(self, config: TrainerConfig, pruner: Optional[Pruner] = None):
        self.config = config
        self.pruner = pruner
        # Shortcuts for easy access
        self.data_config = config.data_config
        self.epoch_config = config.epoch_config
        self.total_epochs = self.epoch_config.num_train_epochs
        if not config.train_only:
            self.total_epochs += (self.epoch_config.num_pre_prune_epochs
                                  + self.epoch_config.num_prune_iterations *
                                  self.epoch_config.num_prune_epochs)
        self.device = config.device
        self.pbar = tqdm(total=self.total_epochs)
        self.global_step = -1
        self.writer = SummaryWriter(config.tensorboard_dir)
        self.writer.add_hparams(config.hparams, {})
        self._make_dataloaders()
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def __del__(self):
        self.pbar.close()
        self.writer.close()

    def _make_dataloaders(self):
        data_config = self.data_config
        self.trainloader = DataLoader(
            data_config.train_dataset, batch_size=data_config.batch_size,
            shuffle=data_config.shuffle, pin_memory=True,
            num_workers=data_config.num_workers,
            persistent_workers=data_config.num_workers > 0)
        self.testloader = DataLoader(
            data_config.test_dataset, batch_size=data_config.batch_size,
            shuffle=False, pin_memory=True,
            num_workers=data_config.num_workers,
            persistent_workers=data_config.num_workers > 0)

    def _should_prune(self) -> bool:
        return not self.config.train_only and self.pruner is not None

    def run(self):
        if self._should_prune():
            tqdm.write(f'Pruning method: {self.pruner}')
            self._run_pre_prune()
            self._run_prune()
        self._run_training()
        if self._should_prune():
            self.pruner.apply_masks()
        self._checkpoint_model(self.config.model_to_save_after_training)
        self.get_all_eval_metrics()

    def _run_pre_prune(self):
        config = self.config
        epoch_config = self.epoch_config
        num_batches_in_epoch = epoch_config.num_batches_in_epoch
        grad_step_num_batches = epoch_config.grad_step_num_batches
        tqdm_update_frequency = epoch_config.tqdm_update_frequency
        for epoch in range(epoch_config.num_pre_prune_epochs):
            self.global_step += 1
            self.pbar.update(1)
            config.model.train()
            loss_avg = AverageMeter()
            pbar = tqdm(
                self.trainloader, leave=False, desc=f'Pre-prune epoch: {epoch}')
            for batch_counter, data in enumerate(pbar):
                inputs, labels = data
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = config.model(inputs)
                loss = config.loss_fn(outputs, labels)
                loss.backward()
                if batch_counter % grad_step_num_batches == 0:
                    config.prune_optimizer.step()
                    config.prune_optimizer.zero_grad(set_to_none=True)
                loss_avg.update(loss.item())
                if (batch_counter + 1) % tqdm_update_frequency == 0:
                    pbar.update(tqdm_update_frequency)
                if (num_batches_in_epoch > 0 and
                        batch_counter >= num_batches_in_epoch):
                    break
            pbar.close()
            self.writer.add_scalar(
                'Loss/train', loss_avg.avg, self.global_step)
            accuracy = self.eval_model()
            self.pbar.set_description(
                f'Pre-Prune: Epoch {epoch+1}/{epoch_config.num_pre_prune_epochs}; ' +
                f'Loss/Train: {loss_avg.avg:.4f}; ' +
                f'Accuracy/Test: {accuracy:.4f}')

    def _run_prune(self):
        epoch_config = self.epoch_config
        self.pruner.start_pruning()
        for iteration in range(epoch_config.num_prune_iterations):
            self._run_prune_iteration(iteration)
            self._checkpoint_model(f'prune.{iteration}')
        self._checkpoint_model('prune.final')

    def _run_prune_iteration(self, iteration):
        config = self.config
        epoch_config = self.epoch_config
        num_batches_in_epoch = epoch_config.num_batches_in_epoch
        grad_step_num_batches = epoch_config.grad_step_num_batches
        tqdm_update_frequency = epoch_config.tqdm_update_frequency
        self.pruner.start_iteration()
        for epoch in range(epoch_config.num_prune_epochs):
            self.global_step += 1
            self.pbar.update(1)
            config.model.train()
            loss_avg = AverageMeter()
            grad_step_loss_avg = AverageMeter()
            pbar = tqdm(
                self.trainloader, leave=False, desc=f'Prune epoch: {epoch}')
            for batch_counter, data in enumerate(pbar):
                inputs, labels = data
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = config.model(inputs)
                loss = config.loss_fn(outputs, labels)
                loss.backward()
                grad_step_loss_avg.update(loss.item())
                loss_avg.update(loss.item())
                if batch_counter % grad_step_num_batches == 0:
                    self.pruner.provide_loss(grad_step_loss_avg.avg)
                    config.prune_optimizer.step()
                    config.prune_optimizer.zero_grad(set_to_none=True)
                    grad_step_loss_avg.reset()
                if (batch_counter + 1) % tqdm_update_frequency == 0:
                    pbar.update(tqdm_update_frequency)
                if (num_batches_in_epoch > 0 and
                        batch_counter >= num_batches_in_epoch):
                    break
            pbar.close()
            self.writer.add_scalar(
                'Loss/train', loss_avg.avg, self.global_step)
            accuracy = np.nan
            if self.pruner.config.eval_after_epoch:
                accuracy = self.eval_model()
            iter_str = f'{iteration+1}/{epoch_config.num_prune_iterations}'
            epoch_str = f'{epoch+1}/{epoch_config.num_prune_epochs}'
            self.pbar.set_description(
                f'Prune: Iteration {iter_str}; ' +
                f'Epoch: {epoch_str}; ' +
                f'Loss/Train: {loss_avg.avg:.4f}; ' +
                f'Accuracy/Test: {accuracy:.4f}')
        self.pruner.compute_masks()
        self.compute_prune_stats()
        self.pruner.reset_weights()

    def _run_training(self):
        self._load_model(self.config.model_to_load_for_training)
        config = self.config
        epoch_config = self.epoch_config
        num_batches_in_epoch = epoch_config.num_batches_in_epoch
        grad_step_num_batches = epoch_config.grad_step_num_batches
        tqdm_update_frequency = epoch_config.tqdm_update_frequency
        best_loss = np.inf
        iter_no_change = 0
        for epoch in range(epoch_config.num_train_epochs):
            self.global_step += 1
            self.pbar.update(1)
            config.model.train()
            loss_avg = AverageMeter()
            pbar = tqdm(
                self.trainloader, leave=False, desc=f'Train epoch: {epoch}')
            for batch_counter, data in enumerate(pbar):
                inputs, labels = data
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = config.model(inputs)
                loss = config.loss_fn(outputs, labels)
                loss.backward()
                if batch_counter % grad_step_num_batches == 0:
                    config.train_optimizer.step()
                    config.train_optimizer.zero_grad(set_to_none=True)
                loss_avg.update(loss.item())
                if (batch_counter + 1) % tqdm_update_frequency == 0:
                    pbar.update(tqdm_update_frequency)
                if (num_batches_in_epoch > 0 and
                        batch_counter >= num_batches_in_epoch):
                    break
            pbar.close()
            loss = loss_avg.avg
            self.writer.add_scalar('Loss/train', loss, self.global_step)
            accuracy = self.eval_model()
            self.pbar.set_description(
                f'Training: ' +
                f'Epoch {epoch+1}/{epoch_config.num_train_epochs}; ' +
                f'Loss/Train: {loss:.4f}; ' +
                f'Best Loss/Train: {best_loss:.4f}; ' +
                f'Accuracy/Test: {accuracy:.4f}')
            if loss > (best_loss - config.train_convergence_loss_tolerance):
                iter_no_change += 1
            else:
                iter_no_change = 0
            if loss < best_loss:
                best_loss = loss
            if iter_no_change >= config.train_loss_num_epochs_no_change:
                tqdm.write(f'Model converged in {epoch+1} epochs')
                break

    @torch.no_grad
    def eval_model(self) -> float:
        model = self.config.model
        model.eval()
        accuracy = MulticlassAccuracy().to(self.device)
        for data in tqdm(self.testloader, leave=False, desc='Eval'):
            inputs, labels = data
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            outputs = model(inputs)
            accuracy.update(outputs, labels)
        accuracy = accuracy.compute().item()
        self.writer.add_scalar('Accuracy/Test', accuracy, self.global_step)
        return accuracy

    @torch.no_grad
    def get_all_eval_metrics(self):
        model = self.config.model
        model.eval()
        accuracy = MulticlassAccuracy(
            num_classes=self.data_config.num_classes,
            average=None).to(self.device)
        precision = MulticlassPrecision(
            num_classes=self.data_config.num_classes,
            average=None).to(self.device)
        recall = MulticlassRecall(
            num_classes=self.data_config.num_classes,
            average=None).to(self.device)
        f1_score = MulticlassF1Score(
            num_classes=self.data_config.num_classes,
            average=None).to(self.device)
        for data in self.testloader:
            inputs, labels = data
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            outputs = model(inputs)
            accuracy.update(outputs, labels)
            precision.update(outputs, labels)
            recall.update(outputs, labels)
            f1_score.update(outputs, labels)
        eval_metrics = EvalMetrics(
            accuracy=accuracy.compute(),
            precision=precision.compute(),
            recall=recall.compute(),
            f1_score=f1_score.compute())
        self._print_eval_metrics(eval_metrics)

    def _print_eval_metrics(self, eval_metrics: EvalMetrics):
        tqdm.write('\n======================================================\n')
        tqdm.write('Final eval metrics:')
        tqdm.write(f'Accuracy: {eval_metrics.accuracy}')
        tqdm.write(f'Precision: {eval_metrics.precision}')
        tqdm.write(f'Recall: {eval_metrics.recall}')
        tqdm.write(f'F1 Score: {eval_metrics.f1_score}')
        tqdm.write('\n======================================================\n')
        write_scalars(self.writer,
                      "Final/Accuracy",
                      eval_metrics.accuracy,
                      self.global_step)
        write_scalars(self.writer,
                      "Final/Precision",
                      eval_metrics.precision,
                      self.global_step)
        write_scalars(self.writer,
                      "Final/Recall",
                      eval_metrics.recall,
                      self.global_step)
        write_scalars(self.writer,
                      "Final/F1Score",
                      eval_metrics.f1_score,
                      self.global_step)

    @torch.no_grad
    def compute_prune_stats(self):
        if not self.config.verbose:
            return
        tqdm.write('\n======================================================\n')
        tqdm.write(f'Global Step: {self.global_step + 1}')
        all_params_total = 0
        all_params_pruned = 0
        for (name, param) in self.config.model.named_buffers():
            if '.weight_mask' not in name:
                continue
            name = name.rstrip('.weight_mask')
            non_zero = torch.count_nonzero(param)
            total = param.numel()
            all_params_total += total
            pruned = total - non_zero
            all_params_pruned += pruned
            percent = 100 * pruned / total
            tqdm.write(f'Name: {name}; Total: {total}; '
                       f'non-zero: {non_zero}; pruned: {pruned}; '
                       f'percent: {percent:.2f}%')
            self.writer.add_scalar(f'{name}/pruned', pruned, self.global_step)
            self.writer.add_scalar(
                f'{name}/pruned_percent', percent, self.global_step)
        all_params_non_zero = all_params_total - all_params_pruned
        all_params_percent = 100 * all_params_pruned / \
            (all_params_total + 1e-6)
        tqdm.write(f'Name: All; Total: {all_params_total}; '
                   f'non-zero: {all_params_non_zero}; ' +
                   f'pruned: {all_params_pruned}; '
                   f'percent: {all_params_percent:.2f}%')
        self.writer.add_scalar(
            f'all/pruned', all_params_pruned, self.global_step)
        self.writer.add_scalar(
            f'all/pruned_percent', all_params_percent, self.global_step)
        tqdm.write('\n======================================================\n')

    def _checkpoint_model(self, id: str):
        fname = os.path.join(self.config.checkpoint_dir, f'model.{id}.ckpt')
        torch.save(self.config.model.state_dict(), fname)

    def _load_model(self, id: str):
        fname = os.path.join(self.config.checkpoint_dir, f'model.{id}.ckpt')
        if not os.path.exists(fname):
            tqdm.write(f'Model not found at {fname}')
            return
        if self.pruner is not None:
            self.pruner.apply_identity_masks()
        tqdm.write(f'Model loaded from {fname}')
        self.config.model.load_state_dict(torch.load(fname))
