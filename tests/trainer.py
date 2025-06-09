import copy
from dataclasses import dataclass
from functools import partial
import os
from typing import Any, Callable, Optional, Union

from lightning.fabric import Fabric
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, default_collate
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from tqdm.auto import tqdm

from causalpruner import Pruner
from causalpruner.average import AverageMeter
from causalpruner.lrrt import (
    set_optimizer_lr,
)
from lr_schedulers import (
    LrSchedulerConfig,
    create_lr_scheduler,
    wrap_lr_scheduler,
)
from test_utils import (
    EvalMetrics,
    MetricsComputer,
)


@dataclass
class DataConfig:
    train_dataset: Dataset
    test_dataset: Dataset
    num_classes: int
    batch_size: int
    batch_size_while_pruning: int
    num_workers: int
    pin_memory: bool
    shuffle: bool
    collate_fn: Optional[Callable[[Any], Any]] = None


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
    # Number of steps to run the dataloader. Note that we run over the entire
    # dataset by default -- which will happen for any value < 0.
    # Use a positive value to limit iterating to a specific number of batches.
    num_batches_in_epoch_while_pruning: int = -1
    # The frequency with which the tqdm progress bar is updated. Set to a
    # larger value for a fast iteration -- else tqdm update will be the
    # bottleneck.
    tqdm_update_frequency: int = 1


@dataclass
class TrainerConfig:
    hparams: dict[str, str]
    fabric: Fabric
    model: nn.Module
    train_optimizer: optim.Optimizer
    data_config: DataConfig
    epoch_config: EpochConfig
    tensorboard_dir: str
    checkpoint_dir: str
    lr_scheduler_config: LrSchedulerConfig
    loss_fn: Callable = partial(F.cross_entropy, label_smoothing=0.1)
    verbose: bool = True
    train_only: bool = False
    model_to_load_for_training: str = "prune.final"
    model_to_save_after_training: str = "trained"


class Trainer:
    def __init__(self, config: TrainerConfig, pruner: Optional[Pruner] = None):
        self.config = config
        self.fabric = config.fabric
        self.pruner = pruner
        # Shortcuts for easy access
        self.data_config = config.data_config
        self.epoch_config = config.epoch_config
        self.total_epochs = self.epoch_config.num_train_epochs
        self.lr_scheduler_config = config.lr_scheduler_config
        if not config.train_only:
            self.total_epochs += (
                self.epoch_config.num_pre_prune_epochs
                + self.epoch_config.num_prune_iterations
                * (
                    self.epoch_config.num_train_epochs_before_pruning
                    + self.epoch_config.num_prune_epochs
                )
                - self.epoch_config.num_train_epochs_before_pruning
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
        collate_fn = default_collate
        if data_config.collate_fn is not None:

            def collate_fn(batch):
                return data_config.collate_fn(*default_collate(batch))

        self.trainloader = DataLoader(
            data_config.train_dataset,
            batch_size=data_config.batch_size,
            shuffle=data_config.shuffle,
            pin_memory=data_config.pin_memory,
            num_workers=data_config.num_workers,
            persistent_workers=data_config.num_workers > 0,
            collate_fn=collate_fn,
        )
        self.testloader = DataLoader(
            data_config.test_dataset,
            batch_size=data_config.batch_size,
            shuffle=False,
            pin_memory=data_config.pin_memory,
            num_workers=data_config.num_workers // 2,
            persistent_workers=data_config.num_workers > 0,
        )
        (self.trainloader, self.testloader) = self.fabric.setup_dataloaders(
            self.trainloader, self.testloader
        )

    def _should_prune(self) -> bool:
        return not self.config.train_only and self.pruner is not None

    def run(self):
        epoch_config = self.config.epoch_config
        if self._should_prune():
            tqdm.write(f"Pruning method: {self.pruner}")
            self._run_training(
                epoch_config.num_pre_prune_epochs, "Training before Pruning"
            )
            self._run_prune()
        self._run_training(
            epoch_config.num_train_epochs,
            "Training after Pruning",
            self.config.model_to_load_for_training,
            self.config.model_to_save_after_training,
        )
        self.get_all_eval_metrics()

    def _run_prune(self):
        epoch_config = self.epoch_config
        self.pruner.start_pruning()
        for iteration in range(epoch_config.num_prune_iterations):
            if iteration > 0:
                self._run_training(
                    epoch_config.num_train_epochs_before_pruning,
                    f"Training while Pruning {iteration}",
                )
            self.pruner.run_prune_iteration()
            self.pbar.update(epoch_config.num_prune_epochs)
            self.compute_prune_stats()
            self._checkpoint_model(f"prune.{iteration}")
        self._checkpoint_model("prune.final")

    def _run_training(
        self,
        num_epochs: int,
        desc: str,
        model_path_to_load: Optional[str] = None,
        model_path_to_save: Optional[str] = None,
    ):
        if num_epochs <= 0:
            return
        if model_path_to_load is not None:
            self._load_model(model_path_to_load)
        config = self.config
        epoch_config = config.epoch_config
        num_batches_in_epoch = epoch_config.num_batches_in_epoch
        tqdm_update_frequency = epoch_config.tqdm_update_frequency
        best_loss = np.inf
        best_accuracy = -np.inf
        lr_scheduler = None
        lr_scheduler_config = copy.deepcopy(self.lr_scheduler_config)
        lr_scheduler_config.num_epochs = num_epochs
        lr_scheduler_config.num_batches = len(self.trainloader)
        if lr_scheduler_config.name != "":
            set_optimizer_lr(config.train_optimizer, lr_scheduler_config.train_lr)
            lr_scheduler = create_lr_scheduler(
                lr_scheduler_config,
                config.train_optimizer,
            )
            lr_scheduler = wrap_lr_scheduler(lr_scheduler)
        for epoch in range(num_epochs):
            self.global_step += 1
            self.pbar.update(1)
            config.model.train()
            loss_avg = AverageMeter(self.fabric)
            pbar = tqdm(
                self.trainloader,
                leave=False,
                desc=f"{desc}: {epoch + 1}",
                dynamic_ncols=True,
            )
            batch_counter = 0
            for inputs, labels in pbar:
                config.train_optimizer.zero_grad(set_to_none=True)
                outputs = config.model(inputs)
                loss = config.loss_fn(outputs, labels)
                self.fabric.backward(loss)
                config.train_optimizer.step()
                loss_avg.update(loss)
                if lr_scheduler is not None:
                    lr_scheduler.step_after_batch()
                if (batch_counter + 1) % tqdm_update_frequency == 0:
                    pbar.update(tqdm_update_frequency)
                if num_batches_in_epoch > 0 and batch_counter >= num_batches_in_epoch:
                    break
                batch_counter += 1
            pbar.close()
            if lr_scheduler is not None:
                lr_scheduler.step_after_epoch()
            loss = loss_avg.mean()
            self.add_scalar("Loss/train", loss, self.global_step)
            accuracy = self.eval_model()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                if model_path_to_save is not None:
                    self._checkpoint_model(model_path_to_save)
            if loss < best_loss:
                best_loss = loss
            self.pbar.set_description(
                f"{desc}: "
                + f"{epoch + 1}/{num_epochs}; "
                + f"Loss/Train: {loss:.4f}; "
                + f"Best Loss/Train: {best_loss:.4f}; "
                + f"Accuracy/Test: {accuracy:.4f}; "
                + f"Best Accuracy/Test: {best_accuracy:.4f}"
            )

    @torch.no_grad()
    def eval_model(self) -> float:
        accuracy = np.nan
        if self.fabric.is_global_zero:
            model = self.config.model
            model.eval()
            self.val_accuracy.reset()
            for data in tqdm(
                self.testloader, leave=False, desc="Eval", dynamic_ncols=True
            ):
                inputs, labels = data
                outputs = model(inputs)
                self.val_accuracy(outputs, labels)
            accuracy = self.val_accuracy.compute()
            self.add_scalar("Accuracy/Test", accuracy, self.global_step)
        self.fabric.barrier()
        return accuracy

    @torch.no_grad()
    def get_all_eval_metrics(self):
        if not self.fabric.is_global_zero:
            return
        model = self.config.model
        model.eval()
        self.val_accuracy.reset()
        self.metrics_computer.reset()
        for data in tqdm(
            self.testloader, leave=False, desc="Eval Stats", dynamic_ncols=True
        ):
            inputs, labels = data
            outputs = model(inputs)
            self.metrics_computer.add(outputs, labels)
            self.val_accuracy(outputs, labels)
        eval_metrics = self.metrics_computer.compute()
        final_accuracy = self.val_accuracy.compute()
        self._print_eval_metrics(final_accuracy, eval_metrics)

    def _print_eval_metrics(self, final_accuracy: float, eval_metrics: EvalMetrics):
        if not self.fabric.is_global_zero:
            return
        tqdm.write("\n======================================================\n")
        tqdm.write("Final eval metrics:")
        tqdm.write(f"Final Accuracy: {final_accuracy:.4f}")
        tqdm.write(f"Accuracy: {eval_metrics.accuracy}")
        tqdm.write(f"Precision: {eval_metrics.precision}")
        tqdm.write(f"Recall: {eval_metrics.recall}")
        tqdm.write(f"F1 Score: {eval_metrics.f1_score}")
        tqdm.write("\n======================================================\n")
        self.add_scalars("Final/Accuracy", eval_metrics.accuracy, self.global_step)
        self.add_scalars("Final/Precision", eval_metrics.precision, self.global_step)
        self.add_scalars("Final/Recall", eval_metrics.recall, self.global_step)
        self.add_scalars("Final/F1Score", eval_metrics.f1_score, self.global_step)

    @torch.no_grad()
    def compute_prune_stats(self):
        if not self.config.verbose:
            return
        if not self.fabric.is_global_zero:
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
