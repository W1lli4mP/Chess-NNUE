from nnue.neural_network import NeuralNetwork
from nnue.model_io import load_model, export_model_for_c
from config import LAYER_DIMS, L2_LAMBDA, MOMENTUM, MODEL_EXPORT_NAME

# create NN with same architecture and load weights
# TODO: could add a get hyperparams helper from a JSON result file, separates concerns

# guard so imports dont export the model unintentionally
def main():
    nn = NeuralNetwork(LAYER_DIMS, L2_LAMBDA, MOMENTUM)
    load_model(nn, MODEL_EXPORT_NAME)

    # export model from .npz to .bin
    export_model_for_c(nn, MODEL_EXPORT_NAME.replace(".npz", ".bin"))

if __name__ == "__main__":
    main()