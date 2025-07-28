# config.py
from dataclasses import dataclass

@dataclass
class FLConfig:
    """Federated learning experiment settings."""
    algorithm: str = "fedavg"  # fedavg | feddane | fedprox | fedsgd
    num_clients: int = 10
    frac: float = 0.9         # fraction of clients per round
    drop_rate: float = 0.1    # simulate stragglers
    num_rounds: int = 5
    local_epochs: int = 5
    batch_size: int = 64
    lr: float = 0.01
    mu: float = 0.1           # FedProx & FedDANE proximal term
    seed: int = 42
    use_cuda: bool = False
    save_path: str = "models/{algorithm}.pt"
