import numpy as np
from dense_layer import DenseLayer
from loss_functions import LossFunction
from value import Value

# from loss_function import LossFunction


# ANN Class
class NeuralNetwork:
    def __init__(self, loss_function_option):
        self.layers = []
        self.loss_functions = {
            "mse": self.mse,
            "log_loss": self.binary_cross_entropy,
            "categorical_cross_entropy": self.categorical_cross_entropy,
        }
        self.loss_function = self.loss_functions[loss_function_option]

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input_data):
        if not isinstance(input_data, Value):
            x = Value(input_data)
        else:
            x = input_data

        for layer in self.layers:
            x = layer.forward(x)
        return x

    def mse(self, y_true, y_pred):
        if not isinstance(y_true, Value):
            y_true = Value(y_true)

        if len(y_true.data.shape) == 1 and len(y_pred.data.shape) == 2:
            num_classes = y_pred.data.shape[1]
            batch_size = y_true.data.shape[0]

            one_hot = np.zeros((batch_size, num_classes))
            for i in range(batch_size):
                if 0 <= y_true.data[i] < num_classes:
                    one_hot[i, int(y_true.data[i])] = 1

            y_true = Value(one_hot)

        diff = y_pred - y_true
        squared = diff * diff
        return squared.mean()

    def binary_cross_entropy(self, y_true, y_pred):
        if not isinstance(y_true, Value):
            y_true = Value(y_true)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-7
        clipped_pred = y_pred.clip(epsilon, 1 - epsilon)
        loss = -(y_true * clipped_pred.log() + (1 - y_true) * (1 - clipped_pred).log())
        return loss.mean()

    def categorical_cross_entropy(self, y_true, y_pred):
        if not isinstance(y_true, Value):
            y_true = Value(y_true)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-7
        clipped_pred = y_pred.clip(epsilon, 1 - epsilon)
        loss = -(y_true * clipped_pred.log()).sum(axis=1)
        return loss.mean()

    def train(
        self,
        X,
        Y,
        epochs=1000,
        learning_rate=0.01,
        batch_size=32,
        verbose=1,
        validation_data=None,
        isOne_hot=False,
        softmax_logloss=False,
    ):
        import time
        import numpy as np

        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        y_orig = Y.copy()

        if isOne_hot:
            Y = self._one_hot(Y)

        history = {"loss": [], "val_loss": []}

        has_validation = validation_data is not None

        if has_validation:
            X_val, Y_val = validation_data
            if isOne_hot:
                Y_val = self._one_hot(Y_val)

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0

            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)

                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                # Forward pass
                Y_pred = self.forward(X_batch)

                # Compute loss
                loss = self.loss_function(Y_batch, Y_pred)

                # Backward pass - compute gradients
                loss.backward()

                # Update weights using gradients
                self._update_parameters(learning_rate)

                total_loss += loss.data

            avg_loss = total_loss / n_batches
            history["loss"].append(avg_loss)

            val_loss = None
            if has_validation:
                val_output = self.forward(X_val)
                val_loss = self.loss_function(Y_val, val_output).data
                history["val_loss"].append(val_loss)

            epoch_time = time.time() - epoch_start_time

            if verbose == 1:
                if has_validation:
                    print(
                        f"\rEpoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {avg_loss:.4f} - val_loss: {val_loss:.4f}"
                    )
                else:
                    print(
                        f"\rEpoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {avg_loss:.4f}"
                    )
            elif verbose > 1 and ((epoch + 1) % 10 == 0 or epoch == 0):
                output = self.forward(X)
                pred = self._get_predictions(output.data)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
                if isOne_hot:
                    accuracy = self._get_accuracy(pred, y_orig)
                    print(f"Epoch {epoch+1}/{epochs}, accuracy: {accuracy:.4f}")

        return history

    def _update_parameters(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, "weights") and hasattr(layer, "biases"):
                layer.weights.data -= learning_rate * layer.weights.grad
                layer.biases.data -= learning_rate * layer.biases.grad

                layer.weights.grad = np.zeros_like(layer.weights.data)
                layer.biases.grad = np.zeros_like(layer.biases.data)

    def predict(self, X):
        return self.forward(X).data


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
