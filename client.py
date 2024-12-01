import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix,classification_report
from keras.utils import to_categorical
from keras import layers
from utils import load_data
import flwr as fl

model = keras.Sequential([
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

model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer='adam')

(X_train,y_train),(X_test,y_test) = load_data()
print(y_train[86])
y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)
print(y_train[86])

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self,config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=5, batch_size=32)
        return model.get_weights(), len(X_train), {}
    
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}

if __name__ == "__main__":
    # model.fit(X_train, y_train, epochs=1, batch_size=32)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
