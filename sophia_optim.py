import torch
from torch.optim.optimizer import Optimizer


def hutchinson_hessian_diag_estimate(model, loss):
    """
    Approximates the diagonal of the Hessian matrix using Hutchinson's method
    This function should be called right after the backward pass on the loss

    :param model:The neural network model
    :param loss: The loss for which the Hessian is computed
    :return: A dictionary mapping parameter tensors to their corresponding Hessian diagonal estimate
    """
    hessian_diag = {}
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        # reset gradients to zero for backpass 2
        p.grad.zero_()

    # random rademacher dist for hessian-vector products
    v = [torch.randint_like(p, high=2, device=p.device).float().mul_(2).sub_(1) for p in model.parameters() if p.requires_grad]

    # compute hessian-vector product
    grad_params = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    Hv = torch.autograd.grad(grad_params, params, grad_outputs=v, allow_unused=True)

    for i, (param, hvp) in zip(params, Hv):
        if hvp is not None:
            # diag of hessian is element-wise product of v, Hv
            hessian_diag[param] = v[i].mul(hvp).detach()
        else:
            # handle no-grad
            hessian_diag[param] = torch.zeros_like(param)

    return hessian_diag


class SophiaOptim(Optimizer):
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, rho=0.1, weight_decay=0.01, ema_decay=0.999):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        self.model = model
        defaults = dict(lr=lr, betas=betas, eps=eps, rho=rho, weight_decay=weight_decay)
        super(SophiaOptim, self).__init__(model.parameters(), defaults)

        # activate ema decay
        self.ema_decay = ema_decay
        self.ema_weights = {name: param.clone().detach() for name, param in model.named_parameters()}

    @torch.no_grad()
    def step(self, closure=None) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # backwards pass on loss to get gradients
        self.model.zero_grad()
        loss.backward()

        # compute hessian diag estimates using hutchinson
        hessian_diag_estimates = hutchinson_hessian_diag_estimate(self.model, loss)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # decoupled weight decay
                if group['weight_decay'] != 0:
                    # apply directly to parameter
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # throw error for sparse gradients
                if grad.is_sparse:
                    raise RuntimeError("Sophia does not support sparse gradients")

                state = self.state[p]

                # state initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_hessian_diag'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, exp_hessian_diag = state['exp_avg'], state['exp_avg_sq'], state['exp_hessian_diag']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # update biased first moment estimate
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                # update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # update hessian diagonal approx
                if p in hessian_diag_estimates:
                    exp_hessian_diag.mul_(beta2).add_(1 - beta2, hessian_diag_estimates[p])

                # bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_hessian_diag = exp_hessian_diag / bias_correction2

                # precondition gradients with hessian info
                denom = corrected_exp_hessian_diag.sqrt().add_(group['eps'])
                step_direction = corrected_exp_avg / denom

                # clipping precondition grads
                step_direction.clamp_(-group['rho'], group['rho'])

                # apply update w/ weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                p.data.add_(step_direction, alpha=-group['lr'])

                # apply adaptive lr
                adaptive_lr = group['lr'] / (corrected_exp_hessian_diag.sqrt().add_(
                    group['eps']))
                p.data.add_(step_direction, alpha=-adaptive_lr)

        # apply ema
        for name, param in self.model.named_parameters():
            if name in self.ema_weights:
                ema_param = self.ema_weights[name]
                ema_param.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

        return loss

    # ema weight functions
    def apply_ema_weights(self):
        for name, param in self.model.named_parameters():
            if name in self.ema_weights:
                param.data.copy_(self.ema_weights[name].data)

    def restore_original_weights(self, original_weights):
        for name, param in self.model.named_parameters():
            if name in original_weights:
                param.data.copy_(original_weights[name].data)
