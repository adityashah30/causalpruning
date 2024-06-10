from dataclasses import dataclass

import torch.nn.utils.prune as prune

from causalpruner import PrunerConfig, Pruner


@dataclass
class MagPrunerConfig(PrunerConfig):
    prune_amount: float = 0.4


class MagPruner(Pruner):
    def __init__(self, config: MagPrunerConfig):
        super().__init__(config)
        self.prune_amount = config.prune_amount

    def compute_masks(self) -> None:
        params_to_prune = []
        for param in self.params:
            params_to_prune.append((self.modules_dict[param], 'weight'))
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.prune_amount,
        )
