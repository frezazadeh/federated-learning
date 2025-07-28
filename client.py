import torch
import torch.nn.functional as F
from optimizers import FedAvgOptimizer, FedProxOptimizer, FedDANEOptimizer, FedSGDOptimizer

class Client:
    """Encapsulates a single client’s local training logic."""
    def __init__(self, model, train_loader, config, device):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.config       = config
        self.device       = device

        # Map algorithm name to optimizer class
        optimizer_map = {
            "fedavg":  FedAvgOptimizer,
            "fedprox": FedProxOptimizer,
            "feddane": FedDANEOptimizer,
            "fedsgd":  FedSGDOptimizer,
        }
        optimizer_cls = optimizer_map[config.algorithm]

        # Instantiate optimizer with/without mu
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
        """Perform local training on client's data."""
        self.model.train()
        # FedSGD does only one pass; others run local_epochs
        epochs = 1 if self.config.algorithm == "fedsgd" else self.config.local_epochs

        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()

                # Call the correct step signature for each algorithm
                if self.config.algorithm == "fedavg":
                    self.optimizer.step()
                elif self.config.algorithm == "fedprox":
                    self.optimizer.step(global_params=global_model)
                elif self.config.algorithm == "feddane":
                    self.optimizer.step(global_params=global_model,
                                        global_gradients=global_grads)
                elif self.config.algorithm == "fedsgd":
                    self.optimizer.step(global_gradients=global_grads)

        return self.model# client.py

import torch
import torch.nn.functional as F
from optimizers import FedAvgOptimizer, FedProxOptimizer, FedDANEOptimizer, FedSGDOptimizer

class Client:
    """Encapsulates a single client’s local training logic."""
    def __init__(self, model, train_loader, config, device):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.config       = config
        self.device       = device

        # Map algorithm name to optimizer class
        optimizer_map = {
            "fedavg":  FedAvgOptimizer,
            "fedprox": FedProxOptimizer,
            "feddane": FedDANEOptimizer,
            "fedsgd":  FedSGDOptimizer,
        }
        optimizer_cls = optimizer_map[config.algorithm]

        # Instantiate optimizer with/without mu
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
        """Perform local training on client's data."""
        self.model.train()
        # FedSGD does only one pass; others run local_epochs
        epochs = 1 if self.config.algorithm == "fedsgd" else self.config.local_epochs

        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()

                # Call the correct step signature for each algorithm
                if self.config.algorithm == "fedavg":
                    self.optimizer.step()
                elif self.config.algorithm == "fedprox":
                    self.optimizer.step(global_params=global_model)
                elif self.config.algorithm == "feddane":
                    self.optimizer.step(global_params=global_model,
                                        global_gradients=global_grads)
                elif self.config.algorithm == "fedsgd":
                    self.optimizer.step(global_gradients=global_grads)

        return self.model
