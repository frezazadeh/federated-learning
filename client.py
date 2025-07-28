from tqdm import trange, tqdm
import torch
import torch.nn.functional as F
from optimizers import (
    FedAvgOptimizer,
    FedProxOptimizer,
    FedDANEOptimizer,
    FedSGDOptimizer,
)

class Client:
    """Encapsulates a single client’s local training logic, with progress bars."""
    def __init__(self, model, train_loader, config, device, client_id):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.config       = config
        self.device       = device
        self.client_id    = client_id

        # Map algorithm name to optimizer class
        optimizer_map = {
            "fedavg":  FedAvgOptimizer,
            "fedprox": FedProxOptimizer,
            "feddane": FedDANEOptimizer,
            "fedsgd":  FedSGDOptimizer,
        }
        optimizer_cls = optimizer_map[config.algorithm]

        # Instantiate optimizer (pass mu for prox/DANE)
        if config.algorithm in ("fedprox", "feddane"):
            self.optimizer = optimizer_cls(
                self.model.parameters(),
                lr=config.lr,
                mu=config.mu
            )
        else:
            self.optimizer = optimizer_cls(
                self.model.parameters(),
                lr=config.lr
            )

    def train(self, global_model=None, global_grads=None):
        """Perform local training, showing epoch and batch progress."""
        self.model.train()
        epochs = 1 if self.config.algorithm == "fedsgd" else self.config.local_epochs

        for epoch in trange(1, epochs + 1,
                            desc=f"Client {self.client_id} Epoch",
                            leave=False):
            # iterate over batches with tqdm
            for batch_idx, (data, target) in enumerate(
                    tqdm(self.train_loader,
                         desc=f"Client {self.client_id} Batches",
                         leave=False,
                         unit="batch")):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()

                # call step with correct signature
                if self.config.algorithm == "fedavg":
                    self.optimizer.step()
                elif self.config.algorithm == "fedprox":
                    self.optimizer.step(global_params=global_model)
                elif self.config.algorithm == "feddane":
                    self.optimizer.step(
                        global_params=global_model,
                        global_gradients=global_grads
                    )
                elif self.config.algorithm == "fedsgd":
                    self.optimizer.step(global_gradients=global_grads)

        return self.model
