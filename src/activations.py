class Activation:
    def __init__(self, activation_function, name):
        self.activation_function = activation_function
        self.name = name

    def __call__(self, x):
        return self.activation_function(x)

    def forward(self, x):
        return self.activation_function(x)
    
    def get_name(self):
        return self.name


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


tanh = Activation(tanh_activation, 'tanh')
sigmoid = Activation(sigmoid_activation, 'sigmoid')
relu = Activation(relu_activation, 'relu')
linear = Activation(linear_activation, 'linear')
softmax = Activation(softmax_activation, 'softmax')
