import numpy as np
from dense_layer import DenseLayer
from loss_functions import LossFunction
from value import Value
import time
import matplotlib.pyplot as plt


# ANN Class
class NeuralNetwork:
    def __init__(self, loss_function_option):
        self.layers = []
        self.loss_functions = {
            "mse": LossFunction.mse,
            "binary_cross_entropy": LossFunction.binary_cross_entropy,
            "categorical_cross_entropy": LossFunction.categorical_cross_entropy,
        }
        self.loss_function = self.loss_functions[loss_function_option]
        self.loss = loss_function_option
        self.last_gradients = []
        self.current_history = None

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

    def _progress_bar(
        self, current, total, bar_length=50, training_loss=None, val_loss=None
    ):
        progress = min(1.0, current / total)
        arrow = "=" * int(round(progress * bar_length) - 1) + ">"
        spaces = " " * (bar_length - len(arrow))

        if training_loss is not None and val_loss is not None:
            print(
                f"\r[{arrow + spaces}] {int(progress * 100)}% - loss: {training_loss:.4f} - val_loss: {val_loss:.4f}",
                end="",
            )
        else:
            print(f"\r[{arrow + spaces}] {int(progress * 100)}%", end="")

        if current == total:
            print()

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
    ):

        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        y_orig = Y.copy()

        if isOne_hot:
            Y = one_hot(Y) if not hasattr(self, "_one_hot") else self._one_hot(Y)

        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

        has_validation = validation_data is not None

        if has_validation:
            X_val, Y_val = validation_data
            y_val_orig = Y_val.copy()
            if isOne_hot:
                Y_val = (
                    one_hot(Y_val)
                    if not hasattr(self, "_one_hot")
                    else self._one_hot(Y_val)
                )

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

                if verbose == 1:
                    self._progress_bar(batch_idx + 1, n_batches)

            avg_loss = total_loss / n_batches
            history["loss"].append(avg_loss)

            train_output = self.forward(X)
            train_pred = (
                np.argmax(train_output.data, axis=1)
                if train_output.data.shape[1] > 1
                else (train_output.data > 0.5).astype(int)
            )
            train_accuracy = np.sum(train_pred == y_orig) / y_orig.size
            history["accuracy"].append(train_accuracy)

            val_loss = None
            val_accuracy = None
            if has_validation:
                val_output = self.forward(X_val)
                val_loss = self.loss_function(Y_val, val_output).data
                history["val_loss"].append(val_loss)

                val_pred = (
                    np.argmax(val_output.data, axis=1)
                    if val_output.data.shape[1] > 1
                    else (val_output.data > 0.5).astype(int)
                )
                val_accuracy = np.sum(val_pred == y_val_orig) / y_val_orig.size
                history["val_accuracy"].append(val_accuracy)

            epoch_time = time.time() - epoch_start_time

            if verbose == 1:
                if has_validation:
                    print(
                        f"\rEpoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {avg_loss:.4f} - accuracy: {train_accuracy:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}"
                    )
                else:
                    print(
                        f"\rEpoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {avg_loss:.4f} - accuracy: {train_accuracy:.4f}"
                    )
            elif verbose > 1 and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(
                    f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}"
                )
        
        self.current_history = history
        return history

    def _update_parameters(self, learning_rate):
        # Make sure last_gradient size is as loong as needed
        if len(self.last_gradients) < len(self.layers):
            for _ in range(len(self.layers) - len(self.last_gradients)):
                self.last_gradients.append(None)

        for i, layer in enumerate(self.layers):
            if hasattr(layer, "weights") and hasattr(layer, "biases"):
                self.last_gradients[i] = layer.weights.grad.copy()

                layer.weights.data -= learning_rate * layer.weights.grad
                layer.biases.data -= learning_rate * layer.biases.grad

                layer.weights.grad = np.zeros_like(layer.weights.data)
                layer.biases.grad = np.zeros_like(layer.biases.data)

    def predict(self, X):
        return self.forward(X).data

    def plot_weight_distribution(self, layer_indices=None):
        # Find layers with weights
        if layer_indices is None:
            layer_indices = [
                i for i, layer in enumerate(self.layers) if hasattr(layer, "weights")
            ]

        num_layers = len(layer_indices)
        if num_layers == 0:
            print("No layers with weights found.")
            return

        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
        if num_layers == 1:
            axes = [axes]

        # Create subplot for each layer
        for i, layer_idx in enumerate(layer_indices):
            if layer_idx >= len(self.layers) or not hasattr(
                self.layers[layer_idx], "weights"
            ):
                print(f"Layer {layer_idx} does not exist or doesn't have weights.")
                continue

            weights = self.layers[layer_idx].weights.data.flatten()

            axes[i].hist(weights, bins=50, alpha=0.7)
            axes[i].set_title(f"Layer {layer_idx} Weight Distribution")
            axes[i].set_xlabel("Weight Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_gradient_distribution(self, layer_indices=None, log_scale=True):
        if not hasattr(self, "last_gradients"):
            print(
                "No gradients have been stored yet. Run at least one training step first."
            )
            return

        # Find all layers with weights
        if layer_indices is None:
            layer_indices = [
                i for i, layer in enumerate(self.layers) if hasattr(layer, "weights")
            ]

        num_layers = len(layer_indices)
        if num_layers == 0:
            print("No layers with weights found.")
            return

        # Create subplot grid
        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
        if num_layers == 1:
            axes = [axes]

        for i, layer_idx in enumerate(layer_indices):
            if layer_idx >= len(self.layers) or not hasattr(
                self.layers[layer_idx], "weights"
            ):
                print(f"Layer {layer_idx} does not exist or doesn't have weights.")
                continue

            gradients = self.last_gradients[layer_idx].flatten()

            n, bins, patches = axes[i].hist(gradients, bins=50, alpha=0.7)

            # Set log if needed
            if log_scale:
                axes[i].set_yscale("log")

            axes[i].set_title(f"Layer {layer_idx} Gradient Distribution")
            axes[i].set_xlabel("Gradient Value")
            axes[i].set_ylabel("Frequency (log scale)" if log_scale else "Frequency")

            axes[i].ticklabel_format(axis="x", style="sci", scilimits=(-4, 4))

            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    
    def plot_training(self):
        if self.current_history is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            ax1.plot(self.current_history["loss"], label="Train Loss")
            if "val_loss" in self.current_history and self.current_history["val_loss"]:
                ax1.plot(self.current_history["val_loss"], label="Validation Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Loss per Epoch")
            ax1.legend()
            ax1.grid(True)

            # Accuracy plot
            ax2.plot(self.current_history["accuracy"], label="Train Accuracy")
            if "val_accuracy" in self.current_history and self.current_history["val_accuracy"]:
                ax2.plot(self.current_history["val_accuracy"], label="Validation Accuracy")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.set_title("Accuracy per Epoch")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.show()


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