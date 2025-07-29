import os
import argparse
import torch
from config import FLConfig
from data_loader import load_mnist_partitions
from server import Server

def parse_args():
    p = argparse.ArgumentParser(description="Federated Learning CLI")
    p.add_argument("--algo", type=str, default="fedavg",
                   choices=["fedavg", "feddane", "fedprox", "fedsgd"]);
    return p.parse_args()


def main():
    cli = parse_args()
    config = FLConfig(algorithm=cli.algo)
    device = torch.device("cuda" if (config.use_cuda and torch.cuda.is_available()) else "cpu")

    # Load per-client train loaders and a central test loader
    train_loaders, test_loader = load_mnist_partitions(config)

    # Initialize and run federated server
    server = Server(config, train_loaders, test_loader, device)
    server.run()

    # Ensure save directory exists
    save_path = config.save_path.format(algorithm=cli.algo)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save global model
    torch.save(server.global_model.state_dict(), save_path)
    print(f"Saved global model to {save_path}")


if __name__ == "__main__":
    main()
