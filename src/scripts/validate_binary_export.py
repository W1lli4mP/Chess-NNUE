import chess

from nnue.neural_network import NeuralNetwork
from nnue.model_io import load_model
from nnue.inference import load_binary_model, dense_from_fen, forward_binary_dense

from config import MODEL_NAME, LAYER_DIMS, L2_LAMBDA, MOMENTUM
from paths import here

"""
validate whether the exported .bin model matches the saved Python
.npz model

for several static FENs, this compares NeuralNetwork.forward()
against manually loaded binary weights + numpy forward pass
"""

BIN_MODEL_NAME = MODEL_NAME.replace(".npz", ".bin")

def main():
    nn = NeuralNetwork(LAYER_DIMS, L2_LAMBDA, MOMENTUM)
    load_model(nn, MODEL_NAME)

    dims, weights, biases = load_binary_model(here("models", BIN_MODEL_NAME))

    print("binary dims:", dims)
    print("expected dims:", LAYER_DIMS)

    positions = [
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rn1qkbnr/ppp2ppp/3p4/4p3/4P1b1/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 4",
        "8/8/8/8/8/8/4Q3/K6k w - - 0 1",
        "8/8/8/8/8/8/4Q3/K6k b - - 0 1",
        "8/8/8/8/8/8/4q3/K6k w - - 0 1",
        "8/8/8/8/8/8/4q3/K6k b - - 0 1",
    ]

    for fen in positions:
        x = dense_from_fen(fen)

        torch_or_numpy_out = nn.forward(x)
        binary_out = forward_binary_dense(x, weights, biases)

        a = float(torch_or_numpy_out[0, 0])
        b = float(binary_out[0, 0])
        diff = abs(a - b)

        print()
        print(fen)
        print("saved model output:", a)
        print("binary output:     ", b)
        print("difference:        ", diff)

        if diff > 1e-5:
            raise AssertionError(f"Binary mismatch on {fen}: {diff}")

    print()
    print("Binary export validation passed.")


if __name__ == "__main__":
    main()