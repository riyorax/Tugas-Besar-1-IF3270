import numpy as np
from dense_layer import DenseLayer
from loss_functions import LossFunction
# from loss_function import LossFunction

# ANN Class
class NeuralNetwork:
    def __init__(self, loss_function_option):
        self.layers = []
        self.loss_functions = {
            'mse': (LossFunction.mse, LossFunction.mse_derivative),
            'log_loss': (LossFunction.binary_cross_entropy, LossFunction.binary_cross_entropy_derivative)
        }
        # self.loss_function = self.loss_functions['mse']

        loss_pair = self.loss_functions[loss_function_option]
        self.loss_function = loss_pair[0]
        self.loss_derivative = loss_pair[1]
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def compile(self, loss):
        self.loss = loss
        if loss not in self.loss_functions:
            raise ValueError("Loss function not supported.")
        loss_pair = self.loss_functions[loss]
        self.loss_function = loss_pair[0]
        self.loss_derivative = loss_pair[1]
    
    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, output):
        for layer in reversed(self.layers):
            output = layer.backward(output, self.learning_rate)
    
    def init_optimizer(self):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.optimizer = self.optimizer
    
    def train(self, X, Y, epochs = 10000, optimizer = "gradient_descent", learning_rate = 0.1, batch_size = 50, softmax_logloss=False, isOne_hot=False):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.init_optimizer()
        n_batches = int(np.ceil(X.shape[0] / batch_size))
        y = Y.copy()
        if isOne_hot:
            Y = one_hot(Y)

        for epoch in range(epochs):
            total_error = 0
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, X.shape[0])
                
                X_batch = X[start:end]
                y_batch = Y[start:end]

                # Forward pass for the entire batch
                output = self.forward(X_batch)
                
                # Calculate batch loss and gradients
                error = self.loss_function(y_batch, output)
                if softmax_logloss:
                    grad = (output - y_batch) / batch_size
                else:
                    grad = self.loss_derivative(y_batch, output)
                
                # Backward pass
                self.backward(grad)
                
                total_error += error
            total_error /= n_batches
            if (epoch % 10 == 0):
                output = self.forward(X)
                pred = get_predictions(output)
                print(f'Epoch {epoch}, Loss: {total_error}')
                if (isOne_hot):
                    print(f'Epoch {epoch}, accuracy: {get_accuracy(pred, y)}')
    
    def predict(self, X):
        return self.forward(X)

def print_parameters(layer, num_elements=5):
    print(f"First {num_elements} weights: {layer.weights.flatten()[:num_elements]}")
    print(f"First {num_elements} biases: {layer.biases.flatten()[:num_elements]}")

def get_predictions(A2):
    return np.argmax(A2, axis=1)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y