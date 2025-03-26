import numpy as np


class Activation:
    def __init__(self, activation_function):
        self.activation_function = activation_function

    def __call__(self, x):
        return self.activation_function(x)

    def forward(self, x):
        return self.activation_function(x)


def tanh_activation(x):
    return x.tanh()


def sigmoid_activation(x):
    return x.sigmoid()


def relu_activation(x):
    return x.relu()


def linear_activation(x):
    return x.linear()


def softmax_activation(x):
    return x.softmax()


tanh = Activation(tanh_activation)
sigmoid = Activation(sigmoid_activation)
relu = Activation(relu_activation)
linear = Activation(linear_activation)
softmax = Activation(softmax_activation)
