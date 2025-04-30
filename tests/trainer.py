import copy
from dataclasses import dataclass
import os
from typing import Callable, Optional, Union

from lightning.fabric import Fabric
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from tqdm.auto import tqdm

from causalpruner import Pruner
from test_utils import (
    AverageMeter,
    EvalMetrics,
    MetricsComputer,
)


@dataclass
class DataConfig:
    train_dataset: Dataset
    test_dataset: Dataset
    num_classes: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    shuffle: bool


@dataclass
class EpochConfig:
    # Number of epochs to train the model before starting pruning. This ensures
    # that the model starts from a decent state where we can start to indentify
    # causal relationship between loss and params. This param is not required
    # for model with pretrained weights.
    num_pre_prune_epochs: int
    # Number of prune iterations to run. Every prune iteration involves two
    # steps:
    # 1. Training -- controlled by `num_train_epochs_while_pruning`. Helps learn
    #                new weights after the last pruning iteration.
    # 2. Pruning -- controlled by `num_prune_epochs`. Runs training and logs
    #               loss and param updates that are then fed to the Pruner.
    num_prune_iterations: int
    # Number of pure training epochs before pruning. This helps train the model
    # for longer without logging loss and param updates. This helps for large
    # models where storing all loss and param updates might be too expensive,
    # but we need to train the model between pruning steps to ensure good
    # performance
    num_train_epochs_before_pruning: int
    # Number of training + pruning epochs. The model is trained in these epochs
    # and the loss and param updates are stored to disk for pruning.
    num_prune_epochs: int
    # Number of training epochs once pruning is complete.
    num_train_epochs: int
    # Number of steps to run the dataloader. Note that we run over the entire
    # dataset by default -- which will happen for any value < 0.
    # Use a positive value to limit iterating to a specific number of batches.
    num_batches_in_epoch: int = -1
    # The frequency with which the tqdm progress bar is updated. Set to a
    # larger value for a fast iteration -- else tqdm update will be the
    # bottleneck.
    tqdm_update_frequency: int = 1


@dataclass
class LRRangeFinderConfig:
    enable: bool
    min_lr: float
    max_lr: float
    num_steps: int
    ewa_alpha: float = 0.98


@dataclass
class TrainerConfig:
    hparams: dict[str, str]
    fabric: Fabric
    model: nn.Module
    prune_optimizer: optim.Optimizer
    train_optimizer: optim.Optimizer
    lrrt_config: LRRangeFinderConfig
    early_stopping: bool
    train_convergence_loss_tolerance: float
    train_loss_num_epochs_no_change: int
    data_config: DataConfig
    epoch_config: EpochConfig
    tensorboard_dir: str
    checkpoint_dir: str
    use_one_cycle_lr_scheduler: bool = False
    loss_fn: Callable = F.cross_entropy
    verbose: bool = True
    train_only: bool = False
    model_to_load_for_training: str = "prune.final"
    model_to_save_after_training: str = "trained"


def set_optimizer_lr(optimizer: optim.Optimizer, new_lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def get_optimizer_lr(optimizer: optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


class Trainer:
    def __init__(self, config: TrainerConfig, pruner: Optional[Pruner] = None):
        self.config = config
        self.fabric = config.fabric
        self.pruner = pruner
        # Shortcuts for easy access
        self.data_config = config.data_config
        self.epoch_config = config.epoch_config
        self.total_epochs = self.epoch_config.num_train_epochs
        self.lr_scheduler = None
        if not config.train_only:
            self.total_epochs += (
                self.epoch_config.num_pre_prune_epochs
                + self.epoch_config.num_prune_iterations
                * (
                    self.epoch_config.num_train_epochs_before_pruning
                    + self.epoch_config.num_prune_epochs
                )
            )
        self.device = self.fabric.device
        self.pbar = tqdm(total=self.total_epochs, dynamic_ncols=True)
        self.global_step = -1
        self.writer = SummaryWriter(config.tensorboard_dir)
        self.writer.add_hparams(config.hparams, {})
        self._make_dataloaders()
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.data_config.num_classes
        ).to(self.device)
        self.metrics_computer = MetricsComputer(self.data_config.num_classes).to(
            self.device
        )
        self.train_optimizer_init_state = copy.deepcopy(
            self.config.train_optimizer.state_dict()
        )

    def __del__(self):
        self.pbar.close()
        self.writer.close()

    def add_scalar(self, name: str, scalar: Union[float, torch.Tensor], step: int):
        if not self.fabric.is_global_zero:
            return
        self.writer.add_scalar(name, scalar, step)

    def add_scalars(self, name: str, val: torch.Tensor, step: int):
        if not self.fabric.is_global_zero:
            return
        for idx, value in enumerate(val):
            self.writer.add_scalar(f"{name}/class_{idx}", value, step)

    def _make_dataloaders(self):
        data_config = self.data_config
        self.trainloader = DataLoader(
            data_config.train_dataset,
            batch_size=data_config.batch_size,
            shuffle=data_config.shuffle,
            pin_memory=data_config.pin_memory,
            num_workers=data_config.num_workers,
            persistent_workers=data_config.num_workers > 0,
        )
        self.testloader = DataLoader(
            data_config.test_dataset,
            batch_size=data_config.batch_size,
            shuffle=False,
            pin_memory=data_config.pin_memory,
            num_workers=data_config.num_workers,
            persistent_workers=data_config.num_workers > 0,
        )
        self.trainloader, self.testloader = self.fabric.setup_dataloaders(
            self.trainloader, self.testloader
        )

    def _should_prune(self) -> bool:
        return not self.config.train_only and self.pruner is not None

    def run(self):
        if self._should_prune():
            tqdm.write(f"Pruning method: {self.pruner}")
            self._run_pre_prune()
            self._run_prune()
        self._run_training()
        self.get_all_eval_metrics()

    def _run_pre_prune(self):
        config = self.config
        epoch_config = self.epoch_config
        num_batches_in_epoch = epoch_config.num_batches_in_epoch
        tqdm_update_frequency = epoch_config.tqdm_update_frequency
        for epoch in range(epoch_config.num_pre_prune_epochs):
            self.global_step += 1
            self.pbar.update(1)
            config.model.train()
            loss_avg = AverageMeter()
            pbar = tqdm(
                self.trainloader,
                leave=False,
                desc=f"Pre-prune epoch: {epoch}",
                dynamic_ncols=True,
            )
            for batch_counter, data in enumerate(pbar):
                inputs, labels = data
                outputs = config.model(inputs)
                loss = config.loss_fn(outputs, labels)
                self.fabric.backward(loss)
                config.train_optimizer.step()
                config.train_optimizer.zero_grad(set_to_none=True)
                loss_avg.update(loss.item())
                if (batch_counter + 1) % tqdm_update_frequency == 0:
                    pbar.update(tqdm_update_frequency)
                if num_batches_in_epoch > 0 and batch_counter >= num_batches_in_epoch:
                    break
            pbar.close()
            self.add_scalar("Loss/train", loss_avg.avg, self.global_step)
            accuracy = self.eval_model()
            self.pbar.set_description(
                f"Pre-Prune: Epoch {epoch + 1}/{epoch_config.num_pre_prune_epochs}; "
                + f"Loss/Train: {loss_avg.avg:.4f}; "
                + f"Accuracy/Test: {accuracy:.4f}"
            )

    def _run_prune(self):
        epoch_config = self.epoch_config
        self.pruner.start_pruning()
        for iteration in range(epoch_config.num_prune_iterations):
            self._train_model_before_prune(iteration)
            self._run_prune_iteration(iteration)
            self._checkpoint_model(f"prune.{iteration}")
        self._checkpoint_model("prune.final")

    def _train_model_before_prune(self, iteration):
        config = self.config
        epoch_config = self.epoch_config
        num_batches_in_epoch = epoch_config.num_batches_in_epoch
        tqdm_update_frequency = epoch_config.tqdm_update_frequency
        self.config.train_optimizer.load_state_dict(self.train_optimizer_init_state)
        for epoch in range(epoch_config.num_train_epochs_before_pruning):
            self.global_step += 1
            self.pbar.update(1)
            config.model.train()
            loss_avg = AverageMeter()
            pbar = tqdm(
                self.trainloader,
                leave=False,
                desc=f"Train before pruning epoch: {epoch}",
                dynamic_ncols=True,
            )
            for batch_counter, data in enumerate(pbar):
                inputs, labels = data
                outputs = config.model(inputs)
                loss = config.loss_fn(outputs, labels)
                self.fabric.backward(loss)
                config.train_optimizer.step()
                config.train_optimizer.zero_grad(set_to_none=True)
                loss_avg.update(loss.item())
                if (batch_counter + 1) % tqdm_update_frequency == 0:
                    pbar.update(tqdm_update_frequency)
                if num_batches_in_epoch > 0 and batch_counter >= num_batches_in_epoch:
                    break
            pbar.close()
            loss = loss_avg.avg
            self.add_scalar("Loss/train", loss, self.global_step)
            accuracy = self.eval_model()
            iter_str = f"{iteration + 1}/{epoch_config.num_prune_iterations}"
            epoch_str = (
                f"{epoch + 1}/" + f"{epoch_config.num_train_epochs_before_pruning}"
            )
            self.pbar.set_description(
                f"Train before Prune: Iteration {iter_str}; "
                + f"Epoch: {epoch_str}; "
                + f"Loss/Train: {loss_avg.avg:.4f}; "
                + f"Accuracy/Test: {accuracy:.4f}"
            )

    def _run_prune_iteration(self, iteration):
        config = self.config
        epoch_config = self.epoch_config
        num_batches_in_epoch = epoch_config.num_batches_in_epoch
        tqdm_update_frequency = epoch_config.tqdm_update_frequency
        self.pruner.start_iteration()
        for epoch in range(epoch_config.num_prune_epochs):
            self.global_step += 1
            self.pbar.update(1)
            config.model.train()
            loss_avg = AverageMeter()
            pbar = tqdm(
                self.trainloader,
                leave=False,
                desc=f"Prune epoch: {epoch}",
                dynamic_ncols=True,
            )
            for batch_counter, data in enumerate(pbar):
                inputs, labels = data
                outputs = config.model(inputs)
                # Compute loss
                loss = config.loss_fn(outputs, labels)
                self.pruner.provide_loss_before_step(loss.item())
                # Take a gradient step
                self.fabric.backward(loss)
                loss_avg.update(loss.item())
                config.prune_optimizer.step()
                config.prune_optimizer.zero_grad(set_to_none=True)
                # Compute loss again
                outputs = config.model(inputs)
                loss = config.loss_fn(outputs, labels)
                self.pruner.provide_loss_after_step(loss.item())
                if (batch_counter + 1) % tqdm_update_frequency == 0:
                    pbar.update(tqdm_update_frequency)
                if num_batches_in_epoch > 0 and batch_counter >= num_batches_in_epoch:
                    break
            pbar.close()
            self.add_scalar("Loss/train", loss_avg.avg, self.global_step)
            accuracy = np.nan
            if self.pruner.config.eval_after_epoch:
                accuracy = self.eval_model()
            iter_str = f"{iteration + 1}/{epoch_config.num_prune_iterations}"
            epoch_str = f"{epoch + 1}/{epoch_config.num_prune_epochs}"
            self.pbar.set_description(
                f"Prune: Iteration {iter_str}; "
                + f"Epoch: {epoch_str}; "
                + f"Loss/Train: {loss_avg.avg:.4f}; "
                + f"Accuracy/Test: {accuracy:.4f}"
            )
        del self.trainloader._iterator
        self.trainloader._iterator = None
        self.pruner.compute_masks()
        self.compute_prune_stats()
        self.pruner.reset_weights()

    def _calculate_total_steps(self):
        epoch_config = self.epoch_config
        data_config = self.data_config

        num_items = len(data_config.train_dataset)
        batch_size = data_config.batch_size
        num_epochs = epoch_config.num_train_epochs
        num_batches_in_epoch = epoch_config.num_batches_in_epoch

        num_batches = (num_items + batch_size - 1) // batch_size
        if num_batches_in_epoch <= 0:
            num_batches_in_epoch = num_batches

        total_steps = num_epochs * num_batches_in_epoch
        return total_steps

    def _run_lrrt(self):
        lrrt_config = self.config.lrrt_config
        model = self.config.model
        optimizer = self.config.train_optimizer
        loss_fn = self.config.loss_fn
        num_steps = lrrt_config.num_steps
        max_lr = lrrt_config.max_lr
        min_lr = lrrt_config.min_lr
        mult = (max_lr / min_lr) ** (1 / num_steps)
        lr = min_lr
        ewa_alpha = lrrt_config.ewa_alpha

        init_model_state = copy.deepcopy(model.state_dict())
        init_optimizer_state = copy.deepcopy(optimizer.state_dict())

        avg_loss = 0.0
        batch_num = 0
        best_loss = np.inf
        losses = []
        lrs = []
        pbar = tqdm(
            range(num_steps), desc="Running LRRT", leave=False, dynamic_ncols=True
        )
        while batch_num < num_steps:
            for data in self.trainloader:
                batch_num += 1
                pbar.update(1)
                if batch_num >= num_steps:
                    break
                set_optimizer_lr(optimizer, lr)
                inputs, labels = data
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                avg_loss = ewa_alpha * avg_loss + (1 - ewa_alpha) * loss.item()
                smoothed_loss = avg_loss / (1 - ewa_alpha**batch_num)
                if smoothed_loss > 4 * best_loss:
                    break
                if smoothed_loss < best_loss:
                    best_loss = smoothed_loss
                losses.append(smoothed_loss)
                lrs.append(lr)
                self.fabric.backward(loss)
                optimizer.step()
                lr *= mult

        np.savez(
            os.path.join(self.config.checkpoint_dir, "lrrt.npz"), lrs=lrs, losses=losses
        )

        losses = losses[10:-5]
        lrs = lrs[10:-5]

        model.load_state_dict(init_model_state)
        optimizer.load_state_dict(init_optimizer_state)

        best_lr = lrs[np.argmin(losses)]

        return best_lr

    def create_one_cycle_lr_scheduler(self, max_lr: float):
        total_steps = self._calculate_total_steps()
        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
            self.config.train_optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
        )

    def _run_training(self):
        self._load_model(self.config.model_to_load_for_training)
        self.config.train_optimizer.load_state_dict(self.train_optimizer_init_state)
        config = self.config
        if config.lrrt_config.enable:
            max_lr = self._run_lrrt()
        else:
            max_lr = get_optimizer_lr(config.train_optimizer)
        min_lr = 0.1 * max_lr
        if config.use_one_cycle_lr_scheduler:
            set_optimizer_lr(config.train_optimizer, min_lr)
            self.create_one_cycle_lr_scheduler(max_lr)
        else:
            set_optimizer_lr(config.train_optimizer, max_lr)
        print(f"Setting learning rate: {get_optimizer_lr(config.train_optimizer):.1e}")
        epoch_config = self.epoch_config
        num_batches_in_epoch = epoch_config.num_batches_in_epoch
        tqdm_update_frequency = epoch_config.tqdm_update_frequency
        best_loss = np.inf
        best_accuracy = -np.inf
        iter_no_change = 0
        for epoch in range(epoch_config.num_train_epochs):
            self.global_step += 1
            self.pbar.update(1)
            config.model.train()
            loss_avg = AverageMeter()
            pbar = tqdm(
                self.trainloader,
                leave=False,
                desc=f"Train epoch: {epoch}",
                dynamic_ncols=True,
            )
            for batch_counter, data in enumerate(pbar):
                inputs, labels = data
                outputs = config.model(inputs)
                loss = config.loss_fn(outputs, labels)
                self.fabric.backward(loss)
                config.train_optimizer.step()
                config.train_optimizer.zero_grad(set_to_none=True)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                loss_avg.update(loss.item())
                if (batch_counter + 1) % tqdm_update_frequency == 0:
                    pbar.update(tqdm_update_frequency)
                if num_batches_in_epoch > 0 and batch_counter >= num_batches_in_epoch:
                    break
            pbar.close()
            total_loss = torch.tensor([loss_avg.sum])
            total_loss = self.fabric.all_reduce(total_loss, reduce_op="sum")
            num_loss = torch.tensor([loss_avg.count])
            num_loss = self.fabric.all_reduce(num_loss, reduce_op="sum")
            loss = total_loss.item() / num_loss.item()
            self.add_scalar("Loss/train", loss, self.global_step)
            accuracy = self.eval_model()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self._checkpoint_model(self.config.model_to_save_after_training)
            if loss > (best_loss - config.train_convergence_loss_tolerance):
                iter_no_change += 1
            else:
                iter_no_change = 0
            if loss < best_loss:
                best_loss = loss
            self.pbar.set_description(
                f"Training: "
                + f"Epoch {epoch + 1}/{epoch_config.num_train_epochs}; "
                + f"Loss/Train: {loss:.4f}; "
                + f"Best Loss/Train: {best_loss:.4f}; "
                + f"Accuracy/Test: {accuracy:.4f}; "
                + f"Best Accuracy/Test: {best_accuracy:.4f}"
            )
            if (
                config.early_stopping
                and iter_no_change >= config.train_loss_num_epochs_no_change
            ):
                tqdm.write(f"Model converged in {epoch + 1} epochs")
                break

    @torch.no_grad
    def eval_model(self) -> float:
        model = self.config.model
        model.eval()
        self.val_accuracy.reset()
        for data in tqdm(self.testloader, leave=False, desc="Eval", dynamic_ncols=True):
            inputs, labels = data
            outputs = model(inputs)
            self.val_accuracy(outputs, labels)
        accuracy = self.val_accuracy.compute()
        self.add_scalar("Accuracy/Test", accuracy, self.global_step)
        return accuracy

    @torch.no_grad
    def get_all_eval_metrics(self):
        model = self.config.model
        model.eval()
        for data in tqdm(
            self.testloader, leave=False, desc="Eval Stats", dynamic_ncols=True
        ):
            inputs, labels = data
            outputs = model(inputs)
            self.metrics_computer.add(outputs, labels)
        eval_metrics = self.metrics_computer.compute()
        self._print_eval_metrics(eval_metrics)

    def _print_eval_metrics(self, eval_metrics: EvalMetrics):
        if not self.fabric.is_global_zero:
            return
        tqdm.write("\n======================================================\n")
        tqdm.write("Final eval metrics:")
        tqdm.write(f"Accuracy: {eval_metrics.accuracy}")
        tqdm.write(f"Precision: {eval_metrics.precision}")
        tqdm.write(f"Recall: {eval_metrics.recall}")
        tqdm.write(f"F1 Score: {eval_metrics.f1_score}")
        tqdm.write("\n======================================================\n")
        self.add_scalars("Final/Accuracy", eval_metrics.accuracy, self.global_step)
        self.add_scalars("Final/Precision", eval_metrics.precision, self.global_step)
        self.add_scalars("Final/Recall", eval_metrics.recall, self.global_step)
        self.add_scalars("Final/F1Score", eval_metrics.f1_score, self.global_step)

    @torch.no_grad
    def compute_prune_stats(self):
        if not self.config.verbose:
            return
        if self.fabric.global_rank != 0:
            return
        tqdm.write("\n======================================================\n")
        tqdm.write(f"Global Step: {self.global_step + 1}")
        all_params_total = 0
        all_params_pruned = 0
        for name, param in self.config.model.named_buffers():
            if ".weight_mask" not in name:
                continue
            name = name.rstrip(".weight_mask")
            non_zero = torch.count_nonzero(param)
            total = param.numel()
            all_params_total += total
            pruned = total - non_zero
            all_params_pruned += pruned
            percent = 100 * pruned / total
            tqdm.write(
                f"Name: {name}; Total: {total}; "
                f"non-zero: {non_zero}; pruned: {pruned}; "
                f"percent: {percent:.2f}%"
            )
            self.add_scalar(f"{name}/pruned", pruned, self.global_step)
            self.add_scalar(f"{name}/pruned_percent", percent, self.global_step)
        all_params_non_zero = all_params_total - all_params_pruned
        all_params_percent = 100 * all_params_pruned / (all_params_total + 1e-6)
        tqdm.write(
            f"Name: All; Total: {all_params_total}; "
            f"non-zero: {all_params_non_zero}; " + f"pruned: {all_params_pruned}; "
            f"percent: {all_params_percent:.2f}%"
        )
        self.add_scalar(f"all/pruned", all_params_pruned, self.global_step)
        self.add_scalar(f"all/pruned_percent", all_params_percent, self.global_step)
        tqdm.write("\n======================================================\n")

    def _checkpoint_model(self, id: str):
        fname = os.path.join(self.config.checkpoint_dir, f"model.{id}.ckpt")
        self.fabric.save(fname, {"model": self.config.model})

    def _load_model(self, id: str):
        fname = os.path.join(self.config.checkpoint_dir, f"model.{id}.ckpt")
        if not os.path.exists(fname):
            tqdm.write(f"Model not found at {fname}")
            return
        if self.pruner is not None:
            self.pruner.apply_identity_masks()
        tqdm.write(f"Model loaded from {fname}")
        self.fabric.load(fname, {"model": self.config.model})
