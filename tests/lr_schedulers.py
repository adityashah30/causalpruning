from enum import Enum
from typing import Any, Optional

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    OneCycleLR,
)


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
