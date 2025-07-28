import torch
import torch.optim as optim
import torch.nn.functional as F
from config import cfg

class Client:
    def __init__(self, model_template, data_loader):
        self.model = model_template().to(cfg.DEVICE)
        self.data_loader = data_loader
        self.optimizer = optim.SGD(self.model.parameters(), lr=cfg.LEARNING_RATE)

    def set_weights(self, state_dict):
        """Load weights from the server's global model."""
        self.model.load_state_dict(state_dict)

    def train(self, global_model_state=None):
        """Train the model on local data for a number of epochs."""
        self.model.train()
        for _ in range(cfg.LOCAL_EPOCHS):
            for data, target in self.data_loader:
                data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                
                # FedProx modification: add proximal term
                if global_model_state:
                    prox_term = 0.0
                    for local_param, global_param in zip(self.model.parameters(), global_model_state.values()):
                        prox_term += torch.sum(torch.pow(local_param - global_param.to(cfg.DEVICE), 2))
                    loss += (cfg.MU / 2) * prox_term

                loss.backward()
                self.optimizer.step()
        
        return self.model.state_dict()
