import numpy as np
from nnue.neural_network import NeuralNetwork
import json
import struct
from paths import here

def save_model(neural_network: NeuralNetwork, filename: str) -> None:
    """
    saves a model after training to models/ as a .npz (concise and efficient file format for saving models)
    """
    filepath = here("models", filename)
    weights_biases = {}
    weights = neural_network.get_weights()
    biases = neural_network.get_biases()
    for i, (W, b) in enumerate(zip(weights, biases)):
        weights_biases[f"layer_{i}_W"] = W
        weights_biases[f"layer_{i}_b"] = b
    np.savez(filepath, **weights_biases)
    print(f"Model saved to {filepath}")

def load_model(neural_network: NeuralNetwork, filename: str) -> None:
    """
    loads a saved model as a .npz from models/ using a specified name
    """
    filepath = here("models", filename)
    data = np.load(filepath)
    num_trainable = len(neural_network.get_trainable_layers())
    weights = [data[f"layer_{i}_W"] for i in range(num_trainable)]
    biases = [data[f"layer_{i}_b"] for i in range(num_trainable)]
    neural_network.set_weights(weights)
    neural_network.set_biases(biases)
    print("Model successfully loaded")

def save_results(results: dict, filename: str = "results.json") -> None:
    """
    saves a model's results as a .json to results/
    default filename is results.json
    """
    filepath = here("results", filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")

def export_model_for_c(neural_network: NeuralNetwork, filename: str) -> None:
    """
    export model as raw binary (.bin) for C
    
    format:
    - [int32] num_weight_layers (number of weight matrices)
    - [int32 x (num_weight_layers + 1)] layer_dims (e.g., 768, 128, 1)
    - for each weight layer:
        - [float32 x (in_features * out_features)] weights (row-major)
        - [float32 x out_features] biases
    """

    filepath = here("models", filename)
    weights = neural_network.get_weights()
    biases = neural_network.get_biases()
    layer_dims = neural_network.layer_dims

    with open(filepath, "wb") as f:
        # write header
        # struct.pack() converts python int to 4 raw bytes (int32)
        # i is the format string (4 byte int)
        # <i for little endian formatting
        # 2nd param is the value
        f.write(struct.pack("<i", len(weights)))
        for dim in layer_dims:
            f.write(struct.pack("<i", dim))

        # write weights and biases
        for W, b in zip(weights, biases):
            # ensure weights and biases are in float32 and contiguous
            W_f32 = np.ascontiguousarray(W, dtype=np.float32)
            b_f32 = np.ascontiguousarray(b, dtype=np.float32)

            # tobytes() converts to raw bytes (like struct.pack() but for larger quantities like arrays)
            f.write(W_f32.tobytes())
            f.write(b_f32.tobytes())

    print(f"Model exported for C to {filepath}")
    print(f"  Layers: {" -> ".join(map(str, layer_dims))}")
    print(f"  Total params: {sum(W.size + b.size for W, b in zip(weights, biases)):,}")