from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from tqdm.auto import tqdm

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    OneCycleLR,
)


@dataclass
class LrSchedulerConfig:
    name: str
    train_lr: float
    max_train_lr: float
    num_epochs: int
    num_batches: int


class LRSchedulerType(Enum):
    APPLIED_AFTER_EPOCH = 1
    APPLIED_AFTER_BATCH = 2


class CausalPrunerLRScheduler(LRScheduler):
    def __init__(self, lr_scheduler: LRScheduler, lr_scheduler_type: LRSchedulerType):
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_type = lr_scheduler_type

    def step(self):
        raise NotImplementedError("Use either step_after_batch or step_after_epoch")

    def step_after_batch(self):
        if self.lr_scheduler_type == LRSchedulerType.APPLIED_AFTER_BATCH:
            self.lr_scheduler.step()

    def step_after_epoch(self):
        if self.lr_scheduler_type == LRSchedulerType.APPLIED_AFTER_EPOCH:
            self.lr_scheduler.step()
        # tqdm.write(f"Setting learning rate to: {self.lr_scheduler.get_last_lr()}")

    def get_lr(self):
        return self.lr_scheduler.get_lr()

    def load_state_dict(self, state_dict: dict[Any, Any]):
        self.lr_scheduler.load_state_dict(state_dict)

    def state_dict(self):
        return self.lr_scheduler.state_dict()


def wrap_lr_scheduler(lr_scheduler: LRScheduler) -> Optional[CausalPrunerLRScheduler]:
    if isinstance(lr_scheduler, OneCycleLR):
        return CausalPrunerLRScheduler(
            lr_scheduler, LRSchedulerType.APPLIED_AFTER_BATCH
        )
    elif isinstance(lr_scheduler, CosineAnnealingLR):
        return CausalPrunerLRScheduler(
            lr_scheduler, LRSchedulerType.APPLIED_AFTER_EPOCH
        )
    raise NotImplementedError("LRScheduler not supported")


def create_lr_scheduler(
    config: LrSchedulerConfig,
    optimizer: Optimizer,
) -> LRScheduler:
    lr_scheduler = config.name.lower()
    if lr_scheduler == "onecycle":
        max_lr = config.max_train_lr
        train_lr = config.train_lr
        total_steps = config.num_epochs * config.num_batches
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=max_lr / train_lr,
            final_div_factor=10**4,
        )
        return lr_scheduler
    elif lr_scheduler == "cosineannealing":
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=config.num_epochs, eta_min=1e-3
        )
        return lr_scheduler
    return None
