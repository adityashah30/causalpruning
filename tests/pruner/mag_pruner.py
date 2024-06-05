from typing import Union

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from ..context import causalpruner


class MagPruner(causalpruner.Pruner):
    def __init__(self, model: nn.Module, amount: float = 0.4,
                 device: Union[str, torch.device] = causalpruner.best_device()):
        super().__init__(model, device)
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
