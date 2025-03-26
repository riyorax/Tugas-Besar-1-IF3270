import numpy as np
from dense_layer import DenseLayer
from loss_functions import LossFunction
import time
# from loss_function import LossFunction

# ANN Class
class NeuralNetwork:
    def __init__(self, loss_function_option):
        self.layers = []
        self.loss_functions = {
            'mse': (LossFunction.mse, LossFunction.mse_derivative),
            'bce': (LossFunction.binary_cross_entropy, LossFunction.binary_cross_entropy_derivative),
            'cce':(LossFunction.categorical_cross_entropy, LossFunction.categorical_cross_entropy_derivative)
        }

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
    
    def _progress_bar(self, current, total, bar_length=50, training_loss=None, val_loss=None):
      progress = min(1.0, current / total)
      arrow = '=' * int(round(progress * bar_length) - 1) + '>'
      spaces = ' ' * (bar_length - len(arrow))
      
      if training_loss is not None and val_loss is not None:
           print(f'\r[{arrow + spaces}] {int(progress * 100)}% - loss: {training_loss:.4f} - val_loss: {val_loss:.4f}', end='')
      else:
          print(f'\r[{arrow + spaces}] {int(progress * 100)}%', end='')
      
      if current == total:
          print()
    
    def train(self, X, Y, epochs = 10000, optimizer = "gradient_descent", learning_rate = 0.1, batch_size = 50, softmax_logloss=False, isOne_hot=False, verbose=1, validation_data=None):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.init_optimizer()
        n_batches = int(np.ceil(X.shape[0] / batch_size))
        y = Y.copy()
        if isOne_hot:
            Y = one_hot(Y)
            
        history = {
            'loss': [],
            'val_loss': []
        }
        
        has_validation = validation_data is not None
        
        if has_validation:
            X_val, Y_val = validation_data
            if isOne_hot:
                Y_val = one_hot(Y_val)

        for epoch in range(epochs):
            epoch_start_time = time.time()
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
                if verbose == 1:
                    self._progress_bar(batch_idx + 1, n_batches)
            epoch_time = time.time() - epoch_start_time
            train_loss = total_error / n_batches
            history['loss'].append(train_loss)
            
            val_loss = None
            if has_validation:
                val_output = self.forward(X_val)
                val_loss = self.loss_function(Y_val, val_output)
                history['val_loss'].append(val_loss)
                
            if verbose == 1:
                if has_validation:
                    print(f'\rEpoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}')
                else:
                    print(f'\rEpoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {train_loss:.4f}')
                    
            elif verbose > 1 and (epoch % 10 == 0):
                output = self.forward(X)
                pred = get_predictions(output)
                print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}')
                if isOne_hot:
                    print(f'Epoch {epoch+1}/{epochs}, accuracy: {get_accuracy(pred, y):.4f}')
            
        return history

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