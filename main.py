import argparse
import torch
from config import FLConfig
from data_loader import load_mnist_partitions
from server import Server

def parse_args():
    p = argparse.ArgumentParser(description="Federated Learning CLI")
    p.add_argument("--algo", type=str, default="fedavg",
                   choices=["fedavg","feddane","fedprox","fedsgd"])
    return p.parse_args()

def main():
    cli = parse_args()
    config = FLConfig(algorithm=cli.algo)
    device = torch.device("cuda" if (config.use_cuda and torch.cuda.is_available()) else "cpu")
    train_ldrs, test_ldr = load_mnist_partitions(config)
    server = Server(config, train_ldrs, test_ldr, device)
    server.run()
    torch.save(server.global_model.state_dict(), config.save_path.format(algorithm=cli.algo))

if __name__ == "__main__":
    main()
