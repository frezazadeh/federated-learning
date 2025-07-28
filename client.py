import torch
import torch.nn.functional as F

class Client:
    """Encapsulates a single client’s local training."""
    def __init__(self, model, train_loader, config, device):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.device       = device
        Optim = {
            "fedavg":  optimizers.FedAvgOptimizer,
            "fedprox": optimizers.FedProxOptimizer,
            "feddane": optimizers.FedDANEOptimizer,
            "fedsgd":  optimizers.FedSGDOptimizer,
        }[config.algorithm]
        # pass mu for prox/DANE
        self.optimizer = Optim(
            model.parameters(),
            lr=config.lr,
            mu=getattr(config, "mu", None)
        )

    def train(self, global_model=None, global_grads=None):
        """Run local epochs or a single pass for FedSGD."""
        self.model.train()
        for _ in range(1 if self.optimizer.__class__.__name__=="FedSGDOptimizer" else self.config.local_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss   = F.nll_loss(output, target)
                loss.backward()
                # pass global params / grads if needed
                self.optimizer.step(
                    closure=None,
                    global_params=global_model,
                    global_gradients=global_grads
                )
        return self.model
