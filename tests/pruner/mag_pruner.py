from causalpruner import Pruner, best_device

from typing import Union

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class MagPruner(Pruner):
    def __init__(self, model: nn.Module,
                 amount: float = 0.4,
                 device: Union[str, torch.device] = best_device()):
        super().__init__(model, device)
        self.amount = amount

    def compute_masks(self) -> None:
        parameters_to_prune = []
        for param in self.params_dict.values():
            parameters_to_prune.append(param)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.amount,
        )
