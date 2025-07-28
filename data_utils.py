import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from config import cfg

def get_data_loaders():
    """Downloads data and returns train/test loaders and client data splits."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    if cfg.DATA_DISTRIBUTION == 'iid':
        # Shuffle and split indices for IID
        num_items = int(len(train_dataset) / cfg.TOTAL_CLIENTS)
        all_indices = np.random.permutation(len(train_dataset))
        client_indices = [all_indices[i*num_items:(i+1)*num_items] for i in range(cfg.TOTAL_CLIENTS)]
    else: # Non-IID
        # Sort data by labels and create shards, then distribute shards
        num_shards, num_imgs = 200, 300 # 200 shards of 300 images each
        idx_shard = [i for i in range(num_shards)]
        client_indices = [[] for _ in range(cfg.TOTAL_CLIENTS)]
        idxs = np.arange(num_shards * num_imgs)
        labels = train_dataset.targets.numpy()
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]
        
        # Distribute shards to clients
        for i in range(cfg.TOTAL_CLIENTS):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                client_indices[i].extend(idxs[rand*num_imgs:(rand+1)*num_imgs])
                
    client_loaders = [
        DataLoader(Subset(train_dataset, indices), batch_size=cfg.BATCH_SIZE, shuffle=True)
        for indices in client_indices
    ]
    
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE * 2)
    
    return client_loaders, test_loader
