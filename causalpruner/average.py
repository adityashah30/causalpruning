from lightning import Fabric
import torch


class AverageMeter:
    def __init__(self, fabric: Fabric):
        self.fabric = fabric
        self.device = fabric.device
        self.reset()

    def reset(self):
        self.sum = torch.tensor([0.0], device=self.device)
        self.count = torch.tensor([0], device=self.device)

    def update(self, val: torch.Tensor):
        self.sum += val
        self.count += 1

    def mean(self) -> float:
        total_loss = self.fabric.all_reduce(self.sum, reduce_op="sum")
        num_loss = self.fabric.all_reduce(self.count, reduce_op="sum")
        mean = total_loss / num_loss
        return mean.item()
