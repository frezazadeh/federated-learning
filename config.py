class FLConfig:
    def __init__(self, algorithm: str = "fedavg", num_clients: int = 3):
        # Federated learning settings
        self.algorithm = algorithm
        self.num_clients = num_clients
        self.num_rounds = 5            # Total communication rounds
        self.local_epochs = 1          # Local epochs per client
        self.frac = 1.0                # Fraction of clients each round
        self.local_lr = 0.01           # Learning rate for local updates
        self.global_lr = 1.0           # (Not used in FedAvg) global learning rate
        self.batch_size = 32           # Batch size for training
        self.use_cuda = True           # Use GPU if available

        # Where to save the global model
        self.save_path = "models/global_{algorithm}.pth"
