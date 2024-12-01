import flwr as fl
from strategy import SaveModelStrategy

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=30),
    strategy=SaveModelStrategy())


