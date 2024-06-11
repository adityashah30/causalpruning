from dataclasses import dataclass

import torch.nn.utils.prune as prune

from causalpruner import PrunerConfig, Pruner


MagPrunerConfig = PrunerConfig


class MagPruner(Pruner):
    def __init__(self, config: MagPrunerConfig):
        super().__init__(config)

    def compute_masks(self) -> None:
        params_to_prune = []
        for param in self.params:
            params_to_prune.append((self.modules_dict[param], 'weight'))
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.config.amount,
        )
