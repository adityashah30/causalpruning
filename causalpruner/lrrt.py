import copy
from lightning import fabric
import numpy as np
import os
from typing import Callable, Optional
from tqdm.auto import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def get_optimizer_lr(optimizer: optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def set_optimizer_lr(optimizer: optim.Optimizer, new_lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def run_lrrt(
    fabric: fabric.Fabric,
    model: nn.Module,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    loss_fn: Callable,
    num_steps: int,
    max_lr: float,
    min_lr: float,
    ewa_alpha: float = 0.98,
    checkpoint_dir: Optional[str] = None,
):
    mult = (max_lr / min_lr) ** (1 / num_steps)
    lr = min_lr

    init_model_state = copy.deepcopy(model.state_dict())
    init_optimizer_state = copy.deepcopy(optimizer.state_dict())

    avg_loss = 0.0
    batch_num = 0
    best_loss = np.inf
    losses = []
    lrs = []
    pbar = tqdm(range(num_steps), desc="Running LRRT",
                leave=False, dynamic_ncols=True)

    while batch_num < num_steps:
        for data in dataloader:
            batch_num += 1
            pbar.update(1)
            if batch_num >= num_steps:
                break
            set_optimizer_lr(optimizer, lr)
            inputs, labels = data
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            labels = labels.view(outputs.size())
            loss = loss_fn(outputs, labels)
            avg_loss = ewa_alpha * avg_loss + (1 - ewa_alpha) * loss.item()
            smoothed_loss = avg_loss / (1 - ewa_alpha**batch_num)
            if smoothed_loss > 4 * best_loss:
                break
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            losses.append(smoothed_loss)
            lrs.append(lr)
            fabric.backward(loss)
            optimizer.step()
            lr *= mult

    if checkpoint_dir is not None:
        np.savez(os.path.join(checkpoint_dir, "lrrt.npz"),
                 lrs=lrs, losses=losses)

    losses = losses[10:-5]
    lrs = lrs[10:-5]

    model.load_state_dict(init_model_state)
    optimizer.load_state_dict(init_optimizer_state)

    best_lr = lrs[np.argmin(losses)]

    return best_lr


def create_one_cycle_lr_scheduler(
    optimizer: optim.Optimizer, start_lr: float, max_lr: float, total_steps: int
) -> optim.lr_scheduler.OneCycleLR:
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.1,
        div_factor=max_lr / start_lr,
        anneal_strategy="cos",
    )
