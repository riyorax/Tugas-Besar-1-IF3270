
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import numpy as np
from activations import ReLu, Sigmoid
from dense_layer import DenseLayer
from ann import NeuralNetwork


X = np.array([
    [0.1, 0.2, 0.3],
    [0.9, 0.8, 0.7],
    [0.2, 0.2, 0.3],
    [0.8, 0.7, 0.6],
])

Y = np.array([0, 1, 0, 1])

model = NeuralNetwork('mse')

model.add_layer(DenseLayer(output_size=4, activation=ReLu(), init="Xavier"))
model.add_layer(DenseLayer(output_size=2, activation=Sigmoid(), init="Xavier"))

model.train(
    X,
    Y,
    epochs=100,
    batch_size=2,
    learning_rate=0.05,
    optimizer="gradient_descent",
    isOne_hot=True
)