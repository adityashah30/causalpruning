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
                if 'q' in param_state:
                    del param_state['q']
                    torch.cuda.empty_cache()
                param_state['q'] = torch.zeros_like(
                    p, memory_format=torch.preserve_format)
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
                d_p = p.grad
                d_p = torch.clamp(d_p, -1e12, 1e12)
                p.add_(d_p, alpha=-lr)
                z = p.detach().clone()
                zeros = torch.zeros_like(
                    z, memory_format=torch.preserve_format)
                param_state = self.state[p]
                u = param_state['u']
                u += lr * alpha
                param_state['u'] = u
                q = param_state['q']
                a = z > 0
                p[a] = torch.maximum(zeros[a], p[a] - (u + q[a]))
                a = z < 0
                p[a] = torch.minimum(zeros[a], p[a] + (u - q[a]))
                q.add_(p - z)
                param_state['q'] = q
        return loss
