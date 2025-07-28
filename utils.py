import copy
import torch

def average_models(global_model, client_models):
    """FedAvg: average parameters across clients."""
    new_state = copy.deepcopy(global_model.state_dict())
    for k in new_state:
        stacked = torch.stack([cm.state_dict()[k] for cm in client_models], dim=0)
        new_state[k] = stacked.mean(dim=0)
    global_model.load_state_dict(new_state)
    return global_model

def average_gradients(global_model, client_models):
    """Average gradients from clients for FedSGD/FedDANE."""
    avg_grads = {}
    for name, _ in global_model.named_parameters():
        stacked = torch.stack([cm.state_dict()[name] for cm in client_models], dim=0)
        avg_grads[name] = stacked.mean(dim=0)
    # Apply averaged gradients to global model
    for name, param in global_model.named_parameters():
        param.grad = avg_grads[name]
    return global_model
