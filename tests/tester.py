from dataclasses import dataclass
from typing import Any

from .context import causalpruner

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


@dataclass
class PruningTesterParams:
    train_dataset: data.Dataset
    test_dataset: data.Dataset
    model: nn.Module
    optimizer: optim.Optimizer
    pruner: causalpruner.CausalPruner
    log_dir: str
    model_checkpoint_dir: str
    warmup_epochs: int
    pruning_epochs: int
    post_pruning_epochs: int


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


class PruningTester:

    def __init__(self, params: PruningTesterParams):
        self.params = params
        self.loss_avg = AverageMeter()

    def run(self):
        # Set up logging and checkpointing

        # First run the warm up phase
        self._run_warm_up()

    def _run_warm_up(self):
        for _ in range(self.params.warmup_epochs):
            model.train()
            for _, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                pbar_train.update(1)
                loss_train.update(loss.item(), inputs.size(0))
        writer.add_scalar("Loss/train", loss_train.avg, epoch)
