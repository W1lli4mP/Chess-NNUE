import numpy as np
from neural_network import NeuralNetwork, loss_function
from config import LEARNING_RATE, BATCH_SIZE, HIDDEN_UNITS, EPOCHS, MODEL_NAME

# actual training loop for a batch
def batch_loop(neural_network: NeuralNetwork, n_train: int, train_X_shuffled: np.ndarray, train_y_shuffled: np.ndarray) -> float:
    """
    train all batches in an epoch (forward -> loss -> backward -> gradient descent) and return epoch loss
    """
    epoch_loss = 0.0
    for i in range(0, n_train, BATCH_SIZE):
        batch_X = train_X_shuffled[i: i + BATCH_SIZE]
        batch_y = train_y_shuffled[i: i + BATCH_SIZE]

        batch_size = batch_X.shape[0]

        # forward propagation -> calculate loss -> backward propagation -> gradient descent
        # forward propagation
        a = neural_network.forward(batch_X)

        # calculate loss
        loss_data = loss_function(a, batch_y)
        loss_total = neural_network.calculate_total_loss(loss_data)
        epoch_loss += loss_total

        # backward propagation (includes gradient descent)
        neural_network.backward(batch_y, batch_size)
    
    # return new epoch loss
    return epoch_loss

def train_epoch(neural_network: NeuralNetwork, train_X: np.ndarray, train_y: np.ndarray) -> float:
    """
    train for one epoch and return the loss
    """
    n_train = train_X.shape[0]
    perm = np.random.permutation(n_train)
    train_X_shuffled = train_X[perm]
    train_y_shuffled = train_y[perm]

    epoch_loss = batch_loop(neural_network, n_train, train_X_shuffled, train_y_shuffled)
    return epoch_loss / n_train

def evaluate(neural_network: NeuralNetwork, test_X: np.ndarray, test_y: np.ndarray) -> float:
    """
    test the model and return the mean absolute error
    """
    test_a = neural_network.forward(test_X, training=False) # don't apply dropout during testing
    mae = np.mean(np.abs(test_a.flatten() - test_y))
    return mae

def train_loop(neural_network: NeuralNetwork, train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray) -> list:
    """
    actual training/epoch loop with evaluation
    return list of dicts containing the training results/history per epoch
    """
    training_history = []
    for epoch in range(1, EPOCHS + 1):
        # calculate loss and MAE
        loss = train_epoch(neural_network, train_X, train_y)
        mae = evaluate(neural_network, test_X, test_y)
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
            "epochs_trained": EPOCHS
        },
        "training_history": training_history
    }