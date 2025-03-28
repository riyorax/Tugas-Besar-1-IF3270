import numpy as np
from value import Value


class DenseLayer:
    def __init__(
        self,
        output_size,
        activation=None,
        init="random",
        bias_init="Zero",
        mean=0.0,
        var=1.0,
        lower_bound=-0.5,
        upper_bound=0.5,
        seed=None,
        reg_type=None,
        reg_param=0.0,
    ):
        self.output_size = output_size
        self.activation = activation

        self.init = init
        self.bias_init = bias_init
        self.mean = mean
        self.var = var
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.seed = seed

        self.reg_type = reg_type
        self.reg_param = reg_param

        self.weights = None
        self.biases = None

    def _initialize_weights(self, input_size):
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.init == "random":
            weights_data = np.random.randn(input_size, self.output_size) - 0.5
        elif self.init == "Xavier":
            weights_data = np.random.randn(input_size, self.output_size) * np.sqrt(
                1.0 / input_size
            )
        elif self.init == "He":
            weights_data = np.random.randn(input_size, self.output_size) * np.sqrt(
                2.0 / input_size
            )
        elif self.init == "Zero":
            weights_data = np.zeros((input_size, self.output_size))
        elif self.init == "One":
            weights_data = np.ones((input_size, self.output_size))
        elif self.init == "uniform":
            weights_data = np.random.uniform(
                low=self.lower_bound,
                high=self.upper_bound,
                size=(input_size, self.output_size),
            )
        elif self.init == "normal":
            weights_data = np.random.normal(
                loc=self.mean,
                scale=np.sqrt(self.var),
                size=(input_size, self.output_size),
            )
        else:
            raise ValueError(f"Invalid weight initialization method: {self.init}")
        self.weights = Value(weights_data, label="weights")

    def _initialize_bias(self, input_size):
        if self.seed is not None:
          np.random.seed(self.seed + 1)
        if self.bias_init == "random":
            bias_data = np.random.randn(1, self.output_size) - 0.5
        elif self.bias_init == "Xavier":
            bias_data = np.random.randn(1, self.output_size) * np.sqrt(1.0 / input_size)
        elif self.bias_init == "He":
            bias_data = np.random.randn(1, self.output_size) * np.sqrt(2.0 / input_size)
        elif self.bias_init == "Zero":
            bias_data = np.zeros((1, self.output_size))
        elif self.bias_init == "One":
            bias_data = np.ones((1, self.output_size))
        elif self.bias_init == "uniform":
            bias_data = np.random.uniform(
                low=self.lower_bound, high=self.upper_bound, size=(1, self.output_size)
            )
        elif self.bias_init == "normal":
            bias_data = np.random.normal(
                loc=self.mean, scale=np.sqrt(self.var), size=(1, self.output_size)
            )
        else:
            raise ValueError(f"Unknown bias initialization method: {self.bias_init}")

        self.biases = Value(bias_data, label="biases")

    def forward(self, input_value):
        if not isinstance(input_value, Value):
            input_value = Value(input_value, label="input")

        self.input = input_value

        if self.weights is None or self.biases is None:
            self.input_size = input_value.data.shape[1]
            self._initialize_weights(self.input_size)
            self._initialize_bias(self.input_size)

        linear_output = self.input.matmul(self.weights) + self.biases

        if self.activation:
            return self.activation(linear_output)
        else:
            return linear_output
