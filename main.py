import os
import argparse
import torch
from server import Server
from data import get_loaders
from config import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--algo', type=str, default='fedavg',
                        help='Federated learning algorithm to use')
    cli = parser.parse_args()

    # load configuration
    config = Config(cli.config)

    # prepare data loaders
    train_loaders, test_loader = get_loaders(config)

    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize server
    server = Server(config, train_loaders, test_loader, device)

    # run federated rounds
    for round_idx in range(config.rounds):
        server.round(round_idx)

    # ensure save directory exists
    save_path = config.save_path.format(algorithm=cli.algo)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # save the trained global model
    torch.save(server.global_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()
