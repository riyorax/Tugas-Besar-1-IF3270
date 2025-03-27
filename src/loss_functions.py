import numpy as np
from value import Value


class LossFunction:
    @staticmethod
    def mse(y_true, y_pred):
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

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        if not isinstance(y_true, Value):
            y_true = Value(y_true)

        # Add small epsilon to avoid log(0)
        epsilon = 1e-7
        clipped_pred = y_pred.clip(epsilon, 1 - epsilon)
        
        # Use multiplication by -1 instead of negation
        term1 = y_true * clipped_pred.log()
        term2 = (1 - y_true) * (1 - clipped_pred).log()
        loss = (term1 + term2) * (-1)  # Multiply by -1 instead of using negation
        
        return loss.mean()

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        if not isinstance(y_true, Value):
            y_true = Value(y_true)

        # Add small epsilon to avoid log(0)
        epsilon = 1e-7
        clipped_pred = y_pred.clip(epsilon, 1 - epsilon)
        
        # Use multiplication by -1 instead of negation
        log_probs = clipped_pred.log()
        weighted_log_probs = y_true * log_probs
        summed = weighted_log_probs.sum(axis=1)
        loss = summed * (-1)
        
        return loss.mean()


    # Wont be used with autodiff
    @staticmethod
    def mse_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred, epsilon=1e-15):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

    @staticmethod
    def categorical_cross_entropy_derivative(y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]
