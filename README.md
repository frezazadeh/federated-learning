# 🏛️ A Professional Federated Learning Framework

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![PyYAML](https://img.shields.io/badge/PyYAML-6.0-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

This repository provides an advanced, object-oriented, and extensible framework for implementing and experimenting with Federated Learning (FL) algorithms in PyTorch. It is designed with professional software engineering principles to serve as a robust foundation for research and development.

## ✨ Core Design Principles

This framework is built upon several key design patterns to ensure scalability and maintainability:

1.  **Object-Oriented Structure**: The entire process is managed by `Server` and `Client` objects, encapsulating state and logic for clear, modular code.

2.  **Strategy Design Pattern**: FL algorithms (e.g., FedAvg, FedProx) are implemented as interchangeable "Aggregation Strategies." You can add new algorithms without modifying the core server workflow, promoting clean separation of concerns.

3.  **Configuration-Driven Experiments**: All hyperparameters and settings are managed through `YAML` configuration files. This decouples the experiment setup from the source code, allowing for easy, reproducible runs.

4.  **Centralized Logging**: Uses Python's native `logging` module for structured and informative console output.

## 📂 Framework Structure

The framework is organized for maximum clarity and extensibility:


```
federated-learning-tutorial/
├── main.py             # Main script to run the FL simulation
├── config.py           # All hyperparameters and settings
├── models.py           # CNN model definitions
├── data_utils.py       # Data loading and distribution logic
├── client.py           # Defines the client's behavior
├── server.py           # Defines the server's orchestration logic
├── requirements.txt    # Project dependencies
└── README.md           
```


## 🛠️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/frezazadeh/federated-learning.git
    cd federated-learning
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Running an Experiment

Experiments are run by passing a configuration file to `main.py`.

1.  **Configure your experiment:**
    Modify `configs/fed_prox_config.yaml` to set your desired parameters (e.g., change the `strategy` to `fedavg`, adjust the learning rate, or change the number of rounds).

2.  **Execute the run:**
    ```bash
    python main.py --config_path configs/fed_prox_config.yaml
    ```

The framework will handle the rest: setting up the server, clients, and strategy, and running the full federated training process.

## Extending the Framework

Adding a new algorithm is simple:

1.  Create a new file in `src/strategies/`, for example `my_new_strategy.py`.
2.  Inside, create a class that inherits from `BaseAggregationStrategy` and implement the `aggregate` method.
3.  Update the `strategy` field in your `YAML` config file to the name of your new strategy.

That's it! The framework is designed for this kind of easy extension.
