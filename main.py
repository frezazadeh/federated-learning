from config import cfg
from models import CNNModel
from data_utils import get_data_loaders
from client import Client
from server import Server
from copy import deepcopy

def run_fedavg():
    print("--- Running FedAvg Simulation ---")
    client_loaders, test_loader = get_data_loaders()
    clients = [Client(CNNModel, loader) for loader in client_loaders]
    server = Server(CNNModel, clients, test_loader)
    
    for round_num in range(cfg.ROUNDS):
        print(f"\n--- Round {round_num + 1}/{cfg.ROUNDS} ---")
        selected_clients = server.select_clients()
        client_weights = []
        
        # Distribute model and train
        for client in selected_clients:
            client.set_weights(server.global_model.state_dict())
            weights = client.train() # FedAvg doesn't need global state
            client_weights.append(deepcopy(weights))
            
        # Aggregate weights
        global_weights = server.aggregate_weights(client_weights)
        server.global_model.load_state_dict(global_weights)
        
        # Evaluate
        server.evaluate()

if __name__ == "__main__":
    run_fedavg()
