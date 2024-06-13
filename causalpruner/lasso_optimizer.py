from typing import Callable, Optional

import numpy as np
import torch
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
)


class LassoSGD(Optimizer):

    '''
    This implements the truncated gradient approach by
    [Tsuruoka, Y., Tsujii, J., and Ananiadou, S., 2009].
    '''

    def __init__(self, params: ParamsT, init_lr: float = 1e-3,
                 time_decay: float = 0.25, alpha: float = 1e-7):
        defaults = dict(
            init_lr=init_lr, time_decay=time_decay, alpha=alpha)
        super().__init__(params, defaults)
        self.reset()

    @torch.no_grad
    def reset(self):
        self.timestep = 0
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                param_state = self.state[p]
                data = p.data
                param_state['q'] = torch.zeros_like(
                    data, memory_format=torch.preserve_format)
                param_state['u'] = 0.0

    @torch.no_grad
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()
        self.timestep += 1
        for group in self.param_groups:
            init_lr = group['init_lr']
            time_decay = group['time_decay']
            alpha = group['alpha']
            lr = init_lr / np.power(self.timestep, time_decay)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p = torch.clamp(d_p, -1e12, 1e12)
                half_update = -d_p * lr
                param_state = self.state[p]
                u = param_state['u']
                u += lr * alpha
                param_state['u'] = u
                w = p.data.add_(half_update)
                z = p.data.detach().clone()
                zeros = torch.zeros_like(
                    z, memory_format=torch.preserve_format)
                q = param_state['q']
                a = w > 0
                w[a] = torch.maximum(zeros[a], w[a] - (u + q[a]))
                a = w < 0
                w[a] = torch.minimum(zeros[a], w[a] + (u - q[a]))
                p.data = w
                q.add_(w - z)
                param_state['q'] = q
        return loss
