import flwr as fl
from typing import Union, Optional
from collections import OrderedDict
from tensorflow import keras
from keras import layers
import numpy as np

net = keras.Sequential([
    # Conv Layer 1
    layers.Conv1D(64, kernel_size=3, padding="same", activation="relu", input_shape=(1, 50)),
    layers.BatchNormalization(),
    
    # Conv Layer 2
    layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
    layers.BatchNormalization(),

    # Reshape for RNN input (batch_size, timesteps, features)
    layers.Reshape((1, -1)),

    # LSTM Layer
    layers.LSTM(64, return_sequences=False),

    # Fully Connected Layers
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ):

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated_ndarrays to disk
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"weights/final-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics