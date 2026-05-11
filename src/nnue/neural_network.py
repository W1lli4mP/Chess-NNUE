import numpy as np
from config import LEARNING_RATE, DROPOUT_RATE

# activation functions
def ReLU(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def ReLU_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)

def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def tanh_derivative(z: np.ndarray) -> np.ndarray:
    return 1 - tanh(z) ** 2

# loss functions
def mean_squared_error(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum((y_hat - y) ** 2)

def mean_squared_error_derivative(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2 * (y_hat - y)

# activation and loss configuration
def hidden_activation_function(z: np.ndarray) -> np.ndarray:
    return ReLU(z)

def hidden_activation_function_derivative(z: np.ndarray) -> np.ndarray:
    return ReLU_derivative(z)

def output_activation_function(z: np.ndarray) -> np.ndarray:
    return tanh(z)

def output_activation_function_derivative(z: np.ndarray) -> np.ndarray:
    return tanh_derivative(z)

def loss_function(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    return mean_squared_error(a, y)

def loss_function_derivative(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    return mean_squared_error_derivative(a, y)

# neural network and layer classes
class NeuralNetwork:
    def __init__(self, layer_dims: list[int], l2_lambda: float = 0.0, momentum: float = 0.9) -> None:
        # encapsulate the NN's dimensions to each object
        self.layer_dims = layer_dims

        # initialise lambda for calculating the total loss
        self.l2_lambda = l2_lambda
        
        # layer dims is a list compiled of the number of nodes in each layer (e.g. 784, 128, 10 means 784 inputs, 128 hidden, 10 outputs)
        self.layers = []
        for i in range(len(layer_dims) - 1):
            in_features = layer_dims[i]
            out_features = layer_dims[i + 1]

            weights = np.random.normal(size=(in_features, out_features), loc=0, scale=0.01) # mean = 0, standard deviation = 0.01 (small)
            biases = np.zeros(out_features)

            # if output layer, initialise it
            if i == len(layer_dims) - 2:
                self.layers.append(OutputLayer(biases, weights, l2_lambda, momentum))
            else:
                self.layers.append(Layer(biases, weights, l2_lambda, momentum))
                self.layers.append(Dropout(DROPOUT_RATE))
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.activations = [x]
        for layer in self.layers:
            # filter for dropout layers due to mismatched args
            if isinstance(layer, Dropout):
                x = layer.forward(x, training)
            else:
                x = layer.forward(x)
            self.activations.append(x)
        return x
    
    def backward(self, y: np.ndarray, batch_size: int) -> None:
        # do backpropagation + gradient descent on output layer first
        output_layer = self.layers[-1]

        ## retrieve the output layer's activation (called y_hat instead of a)
        y_hat = self.activations[-1]

        # get previous layer's activation
        a_prev = self.activations[-2]

        # get upstream gradient
        dLda = output_layer.backward(y_hat, a_prev, y, batch_size)

        # traverse HIDDEN layers in reverse order for backpropagation
        # NOTE: directions are relative to the forward direction despite traversal order
        for i in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[i]

            # filter for backdrop layers - they only require dLda 
            if isinstance(current_layer, Dropout):
                dLda = current_layer.backward(dLda)
            else:
                # retrieve activation from the previous layer's activations
                a_prev = self.activations[i]
                a = self.activations[i + 1]
                
                # backpropagate to find upstream gradient (pre-gradient descent) for preceding layer and apply gradient descent
                dLda = current_layer.backward(a, a_prev, dLda, batch_size)

    def calculate_total_loss(self, loss_data: float) -> float:
        l2_penalty = 0.0
        # total loss is defined as the loss data (original loss) with added decay/2 * sum of squared weights (penalty)
        # L_total = L_data + lambda/2 * sum(W^2)
        for layer in self.layers:
            # skip dropout layers since they have no weights
            if isinstance(layer, Dropout):
                continue
            l2_penalty += np.sum(layer.W ** 2)
        l2_penalty *= (self.l2_lambda / 2)
        return loss_data + l2_penalty
    
    # useful helpers for combating dropout layer dependencies
    def get_trainable_layers(self) -> list:
        return [layer for layer in self.layers if hasattr(layer, "W")]
    
    def get_weights(self) -> list:
        return [layer.W for layer in self.get_trainable_layers()]
    
    def get_biases(self) -> list:
        return [layer.b for layer in self.get_trainable_layers()]

    def set_weights(self, weights: np.ndarray) -> None:
        for layer, W in zip(self.get_trainable_layers(), weights):
            layer.W = W
    
    def set_biases(self, biases: np.ndarray) -> None:
        for layer, b in zip(self.get_trainable_layers(), biases):
            layer.b = b

class Layer:
    def __init__(self, biases: np.ndarray, weights: np.ndarray, l2_lambda: float = 0.0, momentum: float = 0.9) -> None:
        self.b = biases
        self.W = weights
        self.l2_lambda = l2_lambda
        self.momentum = momentum
        self.v_W = 0.0
        self.v_b = 0.0

    def activation_function(self, z: np.ndarray) -> np.ndarray:
        return hidden_activation_function(z)

    def activation_function_derivative(self, z: np.ndarray) -> np.ndarray:
        return ReLU_derivative(z)

    def get_preactivation(self, x: np.ndarray) -> np.ndarray:
        # calculate weighted sum x1w1 + x2w2 + ... + xnwn + b
        z = x @ self.W + self.b # @ means multiply for matrix products
        return z

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = self.get_preactivation(x)
        a = self.activation_function(z)
        return a
    
    def backward(self, a: np.ndarray, x: np.ndarray, dLda: np.ndarray, batch_size: int) -> np.ndarray:
        ## compute delta
        # dLdz = dLda * dadz
        z = self.get_preactivation(x)
        delta = dLda * self.activation_function_derivative(z)

        ## calculate upstream gradient for previous layer (pre-gradient descent)
        # dLda_prev = dLda * dadz * W^T
        dLda_prev = delta @ self.W.T

        ## calculate components for gradient descent
        # per-sample bias gradient (dLdb)
        # dLdb = delta * dzdb
        # dLdb = delta * 1
        dLdb = delta

        # per-sample weight gradient (dLdW)
        # dLdW = dLdz * dzdW
        # dLdW = delta * x
        dLdW = x.T @ delta # other way round due to shapes

        ## calculate mini-batch gradient
        # bias component
        dLdb_batch = np.sum(dLdb, axis=0) / batch_size

        # weight component
        dLdW_batch = dLdW / batch_size

        ## apply mini-batch gradient descent
        self.gradient_descent(dLdW_batch, dLdb_batch)

        ## backpropagate upstream gradient to preceding layer
        return dLda_prev
    
    def gradient_descent(self, dLdW: np.ndarray, dLdb: np.ndarray) -> None:
        # compute velocities
        self.v_W = self.momentum * self.v_W - LEARNING_RATE * (dLdW + self.l2_lambda * self.W) # added regularisation for weight (called weight decay)
        self.v_b = self.momentum * self.v_b - LEARNING_RATE * dLdb

        self.W += self.v_W
        self.b += self.v_b

class OutputLayer(Layer):
    def __init__(self, biases: np.ndarray, weights: np.ndarray, l2_lambda: float = 0.0, momentum: float = 0.9) -> None:
        super().__init__(biases, weights, l2_lambda, momentum)
    
    def activation_function(self, z: np.ndarray) -> np.ndarray:
        return output_activation_function(z)

    def activation_function_derivative(self, z: np.ndarray) -> np.ndarray:
        return output_activation_function_derivative(z)

    def get_preactivation(self, x: np.ndarray) -> np.ndarray:
        return super().get_preactivation(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return super().forward(x)

    def backward(self, a: np.ndarray, x: np.ndarray, y: np.ndarray, batch_size: int) -> np.ndarray:
        ## compute delta (dLdz) for MSE + tanh output
        # dLdz = dLda * dadz = 2(a - y) * tanh(z)
        z = self.get_preactivation(x)
        delta = loss_function_derivative(a, y) * self.activation_function_derivative(z)

        ## calculate upstream gradient for previous layer (pre-gradient descent)
        # dLda_prev = dLda * dadz * W^T
        # dLda_prev = delta * W^T
        dLda_prev = delta @ self.W.T

        ## calculate components for gradient descent
        # per-sample bias gradient (dLdb)
        # dLdb = dLdz * dzdb
        # dLdb = delta * 1
        dLdb = delta

        # per-sample weight gradient (dLdW)
        # dLdW = dLdz * dzdW
        # dLdW = delta * x
        dLdW = x.T @ delta # other way round due to shapes

        ## calculate mini-batch gradient
        # bias component
        dLdb_batch = np.sum(dLdb, axis=0) / batch_size

        # weight component
        dLdW_batch = dLdW / batch_size

        ## apply mini-batch gradient descent
        self.gradient_descent(dLdW_batch, dLdb_batch)

        ## backpropagate upstream gradient to preceding layer
        return dLda_prev

    def gradient_descent(self, dLdW: np.ndarray, dLdb: np.ndarray) -> None:
        super().gradient_descent(dLdW, dLdb)

class Dropout:
    def __init__(self, dropout_rate: float = 0.5) -> None:
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, a: np.ndarray, training: bool = True) -> np.ndarray:
        # don't attempt to apply dropout out of training or when dropout rate is 0 (no effect)
        if not training or self.dropout_rate == 0:
            return a
        
        # inverted dropout formula
        # use bernoulli distribution (binomial distribution but when n = 1)
        self.mask = np.random.binomial(1, 1 - self.dropout_rate, a.shape)
        return a * self.mask / (1 - self.dropout_rate)

    def backward(self, dLda: np.ndarray) -> np.ndarray:
        # if there is not mask then dropout was not applied (e.g. when dropout is 0)
        if self.mask is None:
            return dLda

        return dLda * self.mask / (1 - self.dropout_rate)