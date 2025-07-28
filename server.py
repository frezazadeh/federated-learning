import random
import torch
import logging
from tqdm import trange, tqdm

class Server:
    """Orchestrates federated rounds for any algorithm, with progress bars."""
    def __init__(self, config, train_loaders, test_loader, device):
        from model import CNNModel
        from client import Client
        from utils import average_models, average_gradients

        self.config      = config
        self.device      = device
        self.test_loader = test_loader
        self.global_model= CNNModel().to(device)

        # create Client objects with IDs
        self.clients     = [
            Client(CNNModel(), loader, config, device, client_id=i)
            for i, loader in enumerate(train_loaders, start=1)
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

        # outer loop with a progress bar over rounds
        for rnd in trange(1, self.config.num_rounds + 1,
                          desc="Federated Rounds", unit="round"):
            # select & drop
            m        = max(1, int(self.config.frac * len(self.clients)))
            selected = random.sample(self.clients, m)
            active   = random.sample(selected, max(1, int((1-self.config.drop_rate)*m)))

            # For gradient-based methods, compute averaged grads first
            global_grads = None
            if self.config.algorithm in ("fedsgd", "feddane"):
                temp_models = [c.model for c in selected]
                self.global_model = self.avg_grads(self.global_model, temp_models)
                global_grads = [p.grad for p in self.global_model.parameters()]

            # train each active client with progress bar
            local_models = []
            for client in tqdm(active,
                               desc=f"Round {rnd} Clients",
                               leave=False,
                               unit="client"):
                local_models.append(client.train(
                    global_model=self.global_model,
                    global_grads=global_grads
                ))

            # aggregate for FedAvg, FedProx, FedDANE
            if self.config.algorithm in ("fedavg", "fedprox", "feddane"):
                self.global_model = self.avg_models(self.global_model, local_models)

            logging.info(f"⭑ Round {rnd} complete.")
            self.evaluate()
