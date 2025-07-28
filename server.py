import random
import torch
import logging

class Server:
    """Orchestrates federated rounds for any algorithm."""
    def __init__(self, config, train_loaders, test_loader, device):
        from model import CNNModel
        from client import Client
        from utils import average_models, average_gradients

        self.config      = config
        self.device      = device
        self.test_loader = test_loader
        self.global_model= CNNModel().to(device)
        self.clients     = [
            Client(CNNModel(), loader, config, device)
            for loader in train_loaders
        ]
        self.avg_models  = average_models
        self.avg_grads   = average_gradients
        logging.basicConfig(level=logging.INFO)

    def evaluate(self):
        self.global_model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                pred = self.global_model(data).argmax(1)
                correct += (pred == target).sum().item()
                total   += target.size(0)
        acc = 100 * correct / total
        logging.info(f"Global Accuracy: {acc:.2f}%")

    def run(self):
        torch.manual_seed(self.config.seed)
        for rnd in range(1, self.config.num_rounds + 1):
            # select & drop
            m        = max(1, int(self.config.frac * len(self.clients)))
            selected = random.sample(self.clients, m)
            active   = random.sample(selected, max(1, int((1-self.config.drop_rate)*m)))

            # for gradient schemes, compute global grads first
            global_grads = None
            if self.config.algorithm in ("fedsgd", "feddane"):
                temp_models = [c.model for c in selected]
                self.global_model = self.avg_grads(self.global_model, temp_models)
                global_grads = [p.grad for p in self.global_model.parameters()]

            # local updates
            local_models = [
                c.train(global_model=self.global_model, global_grads=global_grads)
                for c in active
            ]

            # aggregate for FedAvg, FedProx, FedDANE
            if self.config.algorithm in ("fedavg", "fedprox", "feddane"):
                self.global_model = self.avg_models(self.global_model, local_models)

            logging.info(f"Round {rnd} complete.")
            self.evaluate()
