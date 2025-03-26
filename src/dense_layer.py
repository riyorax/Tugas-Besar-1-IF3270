import numpy as np

class DenseLayer:
    def __init__(self, output_size, activation=None, init="random", bias_init="Zero", mean=0.0, var=1.0, lower_bound=-0.5, upper_bound=0.5, seed=None, reg_type=None, reg_param=0.0, optimizer="gradient_descent", t=1, beta_1=0.9, beta_2=0.999, epsilon=1e-8):

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

        self.optimizer = optimizer
        self.t = t
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0

        self.weights = None
        self.biases = None
        self.grad_weights = None
        self.grad_biases = None

    def _initialize_weights(self, input_size):
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.init == "random":
            self.weights = np.random.randn(input_size, self.output_size) - 0.5
        elif self.init == "Xavier":
            self.weights = np.random.randn(input_size, self.output_size) * np.sqrt(1. / input_size)
        elif self.init == "He":
            self.weights = np.random.randn(input_size, self.output_size) * np.sqrt(2. / input_size)
        elif self.init == "Zero":
            self.weights = np.zeros((input_size, self.output_size))
        elif self.init == "One":
            self.weights = np.ones((input_size, self.output_size))
        elif self.init == "uniform":
            self.weights = np.random.uniform(
                low=self.lower_bound,
                high=self.upper_bound,
                size=(input_size, self.output_size)
            )
        elif self.init == "normal":
            self.weights = np.random.normal(
                loc=self.mean,
                scale=np.sqrt(self.var),
                size=(input_size, self.output_size)
            )
        else:
            raise ValueError(f"Invalid weight initialization method: {self.init}")

    def _initialize_bias(self, input_size):
      if self.bias_init == "random":
        self.biases = np.random.randn(1, self.output_size) - 0.5
      elif self.bias_init == "Xavier":
          self.biases = np.random.randn(1, self.output_size) * np.sqrt(1. / input_size)
      elif self.bias_init == "He":
          self.biases = np.random.randn(1, self.output_size) * np.sqrt(2. / input_size)
      elif self.bias_init == "Zero":
          self.biases = np.zeros((1, self.output_size))
      elif self.bias_init == "One":
          self.biases = np.ones((1, self.output_size))
      elif self.bias_init == "uniform":
          self.biases = np.random.uniform(
              low=self.lower_bound,
              high=self.upper_bound,
              size=(1, self.output_size)
          )
      elif self.bias_init == "normal":
          self.biases = np.random.normal(
              loc=self.mean,
              scale=np.sqrt(self.var),
              size=(1, self.output_size)
          )
      else:
          raise ValueError(f"Unknown bias initialization method: {self.bias_init}")


    def calculate_update(self, dw, db, learning_rate):
        self.m_dw = self.beta_1 * self.m_dw + (1 - self.beta_1) * dw
        self.v_dw = self.beta_2 * self.v_dw + (1 - self.beta_2) * (dw ** 2)

        self.m_db = self.beta_1 * self.m_db + (1 - self.beta_1) * db
        self.v_db = self.beta_2 * self.v_db + (1 - self.beta_2) * (db ** 2)

        m_dw_corr = self.m_dw / (1 - self.beta_1 ** self.t)
        v_dw_corr = self.v_dw / (1 - self.beta_2 ** self.t)
        m_db_corr = self.m_db / (1 - self.beta_1 ** self.t)
        v_db_corr = self.v_db / (1 - self.beta_2 ** self.t)

        update_w = learning_rate * m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon)
        update_b = learning_rate * m_db_corr / (np.sqrt(v_db_corr) + self.epsilon)

        return update_w, update_b

    def forward(self, input):
        if len(input.shape) > 2:
            input = np.reshape(input, (input.shape[0], input.shape[1]))

        if self.weights is None:
            self.input_size = input.shape[1]
            self._initialize_weights(self.input_size)

        self.input = input
        self.linear_output = np.dot(self.input, self.weights) + self.biases

        if self.activation:
            return self.activation.forward(self.linear_output)
        else:
            return self.linear_output

    def backward(self, output_gradient, learning_rate):
        if self.activation:
            output_gradient = self.activation.backward(output_gradient)

        m = self.input.shape[0]
        weight_gradient = np.dot(self.input.T, output_gradient) / m
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True) / m

        self.grad_weights = weight_gradient
        self.grad_biases = bias_gradient

        if self.reg_type == 'l1':
            weight_gradient += self.reg_param * np.sign(self.weights)
        elif self.reg_type == 'l2':
            weight_gradient += self.reg_param * self.weights

        if self.optimizer == "adam":
            update_weights, update_biases = self.calculate_update(weight_gradient, bias_gradient, learning_rate)
        elif self.optimizer == "gradient_descent":
            update_weights = learning_rate * weight_gradient
            update_biases = learning_rate * bias_gradient
        else:
            raise ValueError("Unknown optimizer")

        self.weights -= update_weights
        self.biases -= update_biases

        return np.dot(output_gradient, self.weights.T)
