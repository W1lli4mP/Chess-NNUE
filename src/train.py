import numpy as np
from neural_network import NeuralNetwork, loss_function
from data import DataLoader, sparse_to_dense_batch
from config import LEARNING_RATE, BATCH_SIZE, HIDDEN_UNITS, EPOCHS, MODEL_NAME, L2_LAMBDA, DROPOUT_RATE, MOMENTUM

def train_epoch(neural_network: NeuralNetwork, train_loader: DataLoader) -> float:
    """
    train for one epoch using batched data loader
    return average loss
    """
    epoch_loss = 0.0
    n_samples = 0

    for batch_X, batch_y in train_loader:
        batch_size = batch_X.shape[0]

        # forward propagation
        a = neural_network.forward(batch_X)

        # calculate loss
        loss_data = loss_function(a, batch_y)
        loss_total = neural_network.calculate_total_loss(loss_data)
        epoch_loss += loss_total
        n_samples += batch_size

        # backward propagation (includes gradient descent)
        neural_network.backward(batch_y, batch_size)
    
    return epoch_loss / n_samples

def evaluate(neural_network: NeuralNetwork, test_loader: DataLoader) -> float:
    """
    evaluate model on test set and return mean absolute error
    """
    total_error = 0.0
    n_samples = 0

    for batch_X, batch_y in test_loader:
        preds = neural_network.forward(batch_X, training=False)
        total_error += np.sum(np.abs(preds - batch_y))
        n_samples += len(batch_y)

    return total_error / n_samples

def train_loop(neural_network: NeuralNetwork, train_features: np.ndarray, train_y: np.ndarray, test_features: np.ndarray, test_y: np.ndarray) -> list:
    """
    training loop with evaluation
    features are sparse (N, 32), converted to dense per batch
    return list of dicts containing the training results/history per epoch
    """
    train_loader = DataLoader(train_features, train_y, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_features, test_y, BATCH_SIZE, shuffle=False)
    
    training_history = []
    for epoch in range(1, EPOCHS + 1):
        # calculate loss and MAE
        loss = train_epoch(neural_network, train_loader)
        mae = evaluate(neural_network, test_loader)
        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {loss:.6f}  test_mae: {mae:.4f}")

        # record into list as object
        training_history.append({
            "epoch": epoch,
            "train_loss": loss,
            "test_mae": mae
        })
    return training_history

def get_results(training_history: list) -> dict:
    """
    construct results dict from training metrics and returns it
    """
    final_mae = training_history[-1]["test_mae"]
    final_loss = training_history[-1]["train_loss"]

    return {
        "model_name": MODEL_NAME,
        "final_mae": final_mae,
        "final_loss": final_loss,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "hidden_units": HIDDEN_UNITS,
            "epochs_trained": EPOCHS,
            "l2_lambda": L2_LAMBDA,
            "dropout_rate": DROPOUT_RATE,
            "momentum": MOMENTUM
        },
        "training_history": training_history
    }