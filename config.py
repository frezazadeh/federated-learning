import torch

class Config:
    def __init__(self):
        self.ROUNDS = 5
        self.CLIENTS_PER_ROUND = 7
        self.TOTAL_CLIENTS = 50
        self.LOCAL_EPOCHS = 3
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.01
        self.MU = 0.1  # For FedProx
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DATASET = 'FashionMNIST'
        # Set to 'non-iid' to simulate a more realistic scenario
        self.DATA_DISTRIBUTION = 'iid' 

# Instantiate the configuration
cfg = Config()
