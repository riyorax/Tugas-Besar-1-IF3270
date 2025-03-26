import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
from dense_layer import DenseLayer
from activations import ReLu, Sigmoid
from loss_functions import LossFunction

X = np.array([[0.1, 0.2, 0.3]])
y_true = np.array([[1, 0]])

layer1 = DenseLayer(output_size=4, activation=ReLu(), init="Xavier")
layer2 = DenseLayer(output_size=2, activation=Sigmoid(), init="Xavier")


out1 = layer1.forward(X)
out2 = layer2.forward(out1)
print(out2)

loss = LossFunction.mse(y_true, out2)
print(loss)
grad = LossFunction.mse_derivative(y_true, out2)
print(grad)

# 
grad2 = layer2.backward(grad, learning_rate=0.01)
layer1.backward(grad2, learning_rate=0.01)