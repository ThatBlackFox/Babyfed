import numpy as np
import gradio as gr
from tensorflow import keras
import tensorflow as tf
from keras import layers
from utils import features_extractor

def load_weights_from_npz(model: keras.Model, filepath: str):
    """
    Load weights into a TensorFlow model from a .npz file.

    Args:
        model (keras.Model): The TensorFlow model to load weights into.
        filepath (str): Path to the .npz file containing weights.
    """
    # Load the saved weights as a list of numpy arrays
    aggregated_weights = np.load(filepath)
    weights_list = [aggregated_weights[f"arr_{i}"] for i in range(len(aggregated_weights))]

    # Set the weights to the model
    model.set_weights(weights_list)
    print(f"Model weights loaded from {filepath}")

# Example usage:
# Restore weights after training round 1
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

load_weights_from_npz(net, "weights/final-weights.npz")

def find_cry(audio):
    class_indexes = {
        0: "Probably asphyxia",
        1: "Probably deaf",
        2: "Hungry",
        3: 'Normal'
    }
    data = features_extractor(audio)
    output = net.predict(data.reshape(1,1,50))[0]
    print(output)
    output_1 = int(tf.argmax(output))
    return f"The baby cry says its: {class_indexes[output_1]} with confidence of {float(output[output_1]*100):.3}%"

input_audio = gr.Audio(
    sources=["microphone","upload"],
    streaming=False,
    waveform_options=gr.WaveformOptions(
        waveform_color="#01C6FF",
        waveform_progress_color="#0066B4",
        skip_length=2,
        show_controls=False,
    ),
    type='filepath'
)
demo = gr.Interface(
    fn=find_cry,
    inputs=input_audio,
    outputs=gr.components.Text()
)

if __name__ == "__main__":
    demo.launch(share=True)