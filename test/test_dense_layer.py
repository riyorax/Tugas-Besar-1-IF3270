import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
import numpy as np
from dense_layer import DenseLayer
from activations import ReLu, Tanh, Sigmoid, Linear

class DummyActivation:
    def forward(self, x):
        return x
    def backward(self, grad):
        return grad


class TestDenseLayer(unittest.TestCase):

    def setUp(self):
        self.input_data = np.random.randn(4, 3)
        self.output_grad = np.ones((4, 5))

    def test_forward_with_activation(self):
        activations = [ReLu(), Sigmoid(), Tanh(), Linear()]
        for act in activations:
            with self.subTest(activation=act.__class__.__name__):
                layer = DenseLayer(output_size=5, activation=act)
                output = layer.forward(self.input_data)
                self.assertEqual(output.shape, (4, 5))
                
    def test_backward_output_shape(self):
        layer = DenseLayer(output_size=5, init="random", bias_init="Zero", activation=DummyActivation())
        _ = layer.forward(self.input_data)
        grad_input = layer.backward(self.output_grad, learning_rate=0.01)
        self.assertEqual(grad_input.shape, self.input_data.shape, "Backward gradient shape mismatch")

    def test_weights_and_biases_init(self):
        layer = DenseLayer(output_size=5, init="One", bias_init="One")
        _ = layer.forward(self.input_data)
        self.assertTrue(np.all(layer.weights == 1), "Weights not initialized to ones")
        self.assertTrue(np.all(layer.biases == 1), "Biases not initialized to ones")


if __name__ == '__main__':
    unittest.main()
