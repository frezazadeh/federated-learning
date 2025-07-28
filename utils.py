import torch

def average_models(global_model, local_models):
    """
    Averages the parameters of a list of local models into the global model.

    Args:
        global_model: the central model to update (torch.nn.Module)
        local_models: list of torch.nn.Module instances from clients

    Returns:
        The global_model with its parameters set to the element-wise mean
        of the clients' parameters.
    """
    # Grab state dicts and stack/average each tensor
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        # collect this parameter from each client
        params = [local.state_dict()[key].float() for local in local_models]
        stacked = torch.stack(params, dim=0)
        global_dict[key] = stacked.mean(dim=0)
    global_model.load_state_dict(global_dict)
    return global_model

def average_gradients(local_gradients):
    """
    Averages client gradients into a single gradient dict.

    Args:
        local_gradients: list of dicts mapping parameter names to
                         torch.Tensor gradients (as returned by
                         model.named_parameters())

    Returns:
        A dict of averaged gradients.
    """
    # assume at least one client
    avg_grads = {}
    # iterate over each param name
    for key in local_gradients[0].keys():
        # collect this gradient from each client, cast to float
        grads = [grads_dict[key].float() for grads_dict in local_gradients]
        stacked = torch.stack(grads, dim=0)
        avg_grads[key] = stacked.mean(dim=0)
    return avg_grads
