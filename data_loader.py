import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def load_mnist_partitions(config):
    # Transformations for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Partition dataset equally among clients
    num_clients = config.num_clients
    data_per_client = len(train_dataset) // num_clients

    train_loaders = []
    for i in range(num_clients):
        start = i * data_per_client
        end = start + data_per_client if i < num_clients - 1 else len(train_dataset)
        subset = Subset(train_dataset, list(range(start, end)))
        loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True)
        train_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loaders, test_loader
