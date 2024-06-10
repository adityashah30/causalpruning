from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from causalpruner import AverageMeter, PrunerConfig, Pruner
from causalpruner import Trainer, TrainerConfig
from causalpruner import best_device


class MagPruner(Pruner):
    def __init__(
            self, model: nn.Module, checkpoint_dir: str, amount: float,
            start_clean: bool,
            device: Union[str, torch.device] = best_device()):
        super().__init__(model, checkpoint_dir, start_clean, device)
        self.amount = amount

    def compute_masks(self) -> None:
        params_to_prune = []
        for param in self.params:
            params_to_prune.append((self.modules_dict[param], 'weight'))
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.amount,
        )


@dataclass
class MagPrunerConfig(PrunerConfig):
    prune_amount: float = 0.4


class MagPrunerTrainer(Trainer):

    def __init__(self, config: TrainerConfig, pruner_config: MagPrunerConfig):
        super().__init__(config)
        self.pruner_config = pruner_config
        self.pruner = MagPruner(config.model, pruner_config.checkpoint_dir,
                                pruner_config.prune_amount,
                                start_clean=pruner_config.start_clean,
                                device=config.device)

    def _run_prune(self):
        config = self.config
        epoch_config = self.epoch_config
        for iteration in range(epoch_config.num_prune_iterations):
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
