from typing import Callable, Optional

import numpy as np
import torch
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
)


class LassoSGD(Optimizer):
    """
    This implements the truncated gradient approach by
    [Tsuruoka, Y., Tsujii, J., and Ananiadou, S., 2009].
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.1,
        time_decay: float = 0.1,
        alpha: float = 1e-8,
    ):
        defaults = dict(lr=lr, time_decay=time_decay, alpha=alpha)
        super().__init__(params, defaults)
        self.timestep = 0
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                param_state = self.state[p]
                param_state["u"] = 0.0
                if "q" not in param_state:
                    param_state["q"] = p.detach().clone()
                param_state["q"].fill_(0.0)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()
        self.timestep += 1
        for group in self.param_groups:
            init_lr = group["lr"]
            time_decay = group["time_decay"]
            alpha = group["alpha"]
            lr = init_lr / np.power(self.timestep, time_decay)
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                d_p = torch.clamp(d_p, -1e12, 1e12)
                p.add_(d_p, alpha=-lr)
                param_state = self.state[p]
                u = param_state["u"]
                u += lr * alpha
                param_state["u"] = u
                q = param_state["q"]
                q = q.to(p.device)
                z = p.detach().clone()
                a = z > 0
                p[a] = torch.clamp(p[a] - (u + q[a]), min=0.0)
                a = z < 0
                p[a] = torch.clamp(p[a] + (u - q[a]), max=0.0)
                q.add_(p - z)
                param_state["q"] = q
        return loss
