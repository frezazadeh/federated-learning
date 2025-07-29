import copy
import torch
import torch.nn.functional as F
from model import MNIST_CNN

class Server:
    def __init__(self, config, train_loaders, test_loader, device):
        self.config = config
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.device = device

        # Initialize global model
        self.global_model = MNIST_CNN().to(self.device)

    def select_clients(self):
        num_selected = max(1, int(self.config.frac * len(self.train_loaders)))
        return list(range(num_selected))  # simple: first frac fraction

    def train_clients(self, selected_clients):
        local_models = []
        for idx in selected_clients:
            local_model = copy.deepcopy(self.global_model)
            local_model.train()
            optimizer = torch.optim.SGD(local_model.parameters(), lr=self.config.local_lr)

            for _ in range(self.config.local_epochs):
                for data, target in self.train_loaders[idx]:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()

            local_models.append(local_model)
        return local_models

    def avg_grads(self, global_model, local_models):
        # Average model parameters (FedAvg)
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            stacked = torch.stack([lm.state_dict()[key].float() for lm in local_models], dim=0)
            global_dict[key] = stacked.mean(dim=0)
        global_model.load_state_dict(global_dict)
        return global_model

    def evaluate(self):
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        acc = 100.0 * correct / total
        return acc

    def run(self):
        for r in range(1, self.config.num_rounds + 1):
            selected = self.select_clients()
            local_models = self.train_clients(selected)
            self.global_model = self.avg_grads(self.global_model, local_models)
            acc = self.evaluate()
            print(f"Round {r:3d} | Global Test Accuracy: {acc:.2f}%")
