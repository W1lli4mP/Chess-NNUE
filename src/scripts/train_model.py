import numpy as np

from nnue.neural_network import NeuralNetwork
from nnue.data import get_data
from nnue.train import train_loop, get_results
from nnue.model_io import save_model, save_results, export_model_for_c
from config import MODEL_NAME, MODEL_RESULTS_NAME, LAYER_DIMS, L2_LAMBDA, MOMENTUM, TRAINING_SEED

def main():
    np.random.seed(TRAINING_SEED)

    # initialise NN
    nn = NeuralNetwork(LAYER_DIMS, L2_LAMBDA, MOMENTUM)

    # load data shards
    train_X, train_y, test_X, test_y = get_data()

    # train NN
    print("STARTING")
    training_history = train_loop(nn, train_X, train_y, test_X, test_y)

    # save results
    results = get_results(training_history)
    save_results(results, MODEL_RESULTS_NAME)

    # save python model
    save_model(nn, MODEL_NAME)
    
    # export to binary for C engine
    export_model_for_c(nn, MODEL_NAME.replace(".npz", ".bin"))

if __name__ == "__main__":
    main()