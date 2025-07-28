import torch
import numpy as np
from copy import deepcopy
from config import cfg
import torch.nn.functional as F

class Server:
    def __init__(self, model_template, clients, test_loader):
        self.clients = clients
        self.test_loader = test_loader
        self.global_model = model_template().to(cfg.DEVICE)

    def select_clients(self):
        """Select a random subset of clients for the current round."""
        return np.random.choice(self.clients, cfg.CLIENTS_PER_ROUND, replace=False)

    def aggregate_weights(self, client_weights):
        """Aggregate client weights to create the new global model (FedAvg)."""
        avg_weights = deepcopy(client_weights[0])
        for key in avg_weights.keys():
            # Sum the weights from all clients
            for i in range(1, len(client_weights)):
                avg_weights[key] += client_weights[i][key]
            # Average the weights
            avg_weights[key] = torch.div(avg_weights[key], len(client_weights))
        return avg_weights

    def evaluate(self):
        """Evaluate the global model on the test dataset."""
        self.global_model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
                output = self.global_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        print(f'Test Set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
