import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def load_mnist_partitions(config):
    """Download MNIST, normalize, and split IID across clients."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(".", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(".", train=False, download=True, transform=transform)

    # Shuffle and chunk indices
    indices = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(config.seed))
    shards  = indices.chunk(config.num_clients)

    train_loaders = [
        DataLoader(Subset(train_ds, shard), batch_size=config.batch_size, shuffle=True)
        for shard in shards
    ]
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    return train_loaders, test_loader
