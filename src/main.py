import numpy as np
from neural_network import NeuralNetwork
from data import get_data
from train import train_loop, evaluate, get_results
from utils import save_model, load_model, save_results
from config import MODEL_NAME, MODEL_RESULTS_NAME, LAYER_DIMS, L2_LAMBDA, MOMENTUM

def main():
    # # INITIALISE NN
    nn = NeuralNetwork(LAYER_DIMS, L2_LAMBDA, MOMENTUM)

    # EXTRACTING THE DATA
    train_X, train_y, test_X, test_y = get_data()

    # # start training NN
    print("STARTING")
    training_history = train_loop(nn, train_X, train_y, test_X, test_y)
    results = get_results(training_history)
    save_results(results, MODEL_RESULTS_NAME)

    # save model
    save_model(nn, MODEL_NAME)

main()