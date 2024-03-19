import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional


class SophiaOptim(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04, weight_decay=1e-1,
                 *, maximize: bool = False, capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, maximize=maximize,
                        capturable=capturable)
        super(SophiaOptim, self).__init__(params, defaults)