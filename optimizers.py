import torch.optim as optim

class FedAvgOptimizer(optim.SGD):
    """Standard SGD—clients do full local updates, FedAvg uses utils.average_models."""
    pass  # inherit behavior

class FedProxOptimizer(optim.SGD):
    """SGD with proximal term µ∥w - w_global∥²."""
    def __init__(self, params, lr, mu):
        super().__init__(params, lr=lr)
        self.mu = mu

    def step(self, closure=None, global_params=None):
        loss = super().step(closure)
        if global_params is None:
            return loss
        for p, g in zip(self.param_groups[0]['params'], global_params.parameters()):
            p.data.add_(-self.mu * self.param_groups[0]['lr'], p.data - g.data)
        return loss

class FedDANEOptimizer(optim.SGD):
    """DANE: prox + gradient correction by global gradient snapshot."""
    def __init__(self, params, lr, mu):
        super().__init__(params, lr=lr)
        self.mu = mu

    def step(self, closure=None, global_params=None, global_gradients=None):
        loss = super().step(closure)
        if global_params is None or global_gradients is None:
            return loss
        for p, g, gg in zip(self.param_groups[0]['params'], global_params.parameters(), global_gradients):
            # corrected gradient: local + global - old_local + µ (w - w_global)
            corrected = p.grad.data + gg.data + self.mu * (p.data - g.data)
            p.data.add_(-self.param_groups[0]['lr'], corrected)
        return loss

class FedSGDOptimizer(optim.Optimizer):
    """Apply only averaged global gradient each round (no local epochs)."""
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None, global_gradients=None):
        loss = None
        if closure is not None:
            loss = closure()
        if global_gradients is None:
            return loss
        for group in self.param_groups:
            for p, g in zip(group['params'], global_gradients):
                p.data.add_(-group['lr'], g.data)
        return loss
