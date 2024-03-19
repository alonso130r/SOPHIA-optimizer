import torch
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow weights step size: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")

        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_counter = 0

        # initialize lookahead state
        self.slow_weights = [[p.clone().detach() for p in group['params']] for group in optimizer.param_groups]

    @torch.no_grad
    def step(self, closure=None):
        # update fast weights
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter % self.k == 0:
            # update slow weights + sync fast weights
            for group, slow_group in zip(self.optimizer.param_groups, self.slow_weights):
                for p, slow_p in zip(group['params'], slow_group):
                    if p.grad is None:
                        continue

                    slow_p.data.add_(self.alpha, p.data - slow_p.data)

        return loss

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        state = {
            'optimizer': self.optimizer.state_dict(),
            'slow_weights': self.slow_weights,
            'step_counter': self.step_counter,
            'alpha': self.alpha,
            'k': self.k
        }
        return state

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.slow_weights = state_dict['slow_weights']
        self.step_counter = state_dict['step_counter']
        self.alpha = state_dict['alpha']
        self.k = state_dict['k']