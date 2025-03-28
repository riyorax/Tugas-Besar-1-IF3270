import numpy as np


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data, dtype=float)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data=shape{self.data.shape}, {self.data})"

    def __add__(self, other):
        other_data = other.data if isinstance(other, Value) else np.array(other)
        out = Value(
            self.data + other_data,
            (self, other) if isinstance(other, Value) else (self,),
            "+",
        )
        def _backward():
            grad_shape = np.broadcast(
                self.data, other_data if isinstance(other, Value) else other
            ).shape

            if self.grad.shape != grad_shape:
                axes_to_sum = tuple(
                    i
                    for i, (a, b) in enumerate(zip(self.grad.shape, grad_shape))
                    if a != b
                )
                sum_grad = np.sum(out.grad, axis=axes_to_sum, keepdims=True)
                self.grad += sum_grad
            else:
                self.grad += out.grad

            if isinstance(other, Value):
                if other.grad.shape != grad_shape:
                    axes_to_sum = tuple(
                        i
                        for i, (a, b) in enumerate(zip(other.grad.shape, grad_shape))
                        if a != b
                    )
                    sum_grad = np.sum(out.grad, axis=axes_to_sum, keepdims=True)
                    other.grad += sum_grad
                else:
                    other.grad += out.grad

        out._backward = _backward
        return out
      
    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        t = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Value(t, (self,), "mean")

        def _backward():
            grad_broadcasted = np.ones_like(self.data) * out.grad / n
            self.grad += grad_broadcasted

        out._backward = _backward
        return out
      
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other_data = other.data if isinstance(other, Value) else np.array(other)
        out = Value(
            self.data - other_data,
            (self, other) if isinstance(other, Value) else (self,),
            "-",
        )

        def _backward():
            self.grad += out.grad
            if isinstance(other, Value):
                other.grad -= out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other):
        other_data = other if not isinstance(other, Value) else other.data
        out = Value(
            other_data - self.data,
            (self,) if not isinstance(other, Value) else (other, self),
            "-",
        )

        def _backward():
            self.grad -= out.grad
            if isinstance(other, Value):
                other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other_data = other.data if isinstance(other, Value) else np.array(other)
        out = Value(
            self.data * other_data,
            (self, other) if isinstance(other, Value) else (self,),
            "*",
        )

        def _backward():
            self.grad += other_data * out.grad
            if isinstance(other, Value):
                other.grad += self.data * out.grad

        out._backward = _backward
        return out
      

    def __neg__(self):
        out = Value(-self.data, (self,), "-")
        
        def _backward():
            self.grad -= out.grad
            
        out._backward = _backward
        return out
        
    def __rmul__(self, other):
        return self.__mul__(other)

    def matmul(self, other):
        out = Value(np.matmul(self.data, other.data), (self, other), "@")

        def _backward():
            self.grad += np.matmul(out.grad, other.data.T)
            other.grad += np.matmul(self.data.T, out.grad)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        t = 1 / (1 + np.exp(-self.data))
        out = Value(t, (self,), "sigmoid")

        def _backward():
            self.grad += (t * (1 - t)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        t = np.maximum(0, self.data)
        out = Value(t, (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def linear(self):
        out = Value(self.data, (self,), "linear")

        def _backward():
            self.grad += out.grad

        out._backward = _backward
        return out

    def softmax(self):
        exp_x = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
        t = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        out = Value(t, (self,), "softmax")

        def _backward():
            n = self.data.shape[0]
            
            for i in range(n):
                softmax_i = t[i].reshape(-1, 1)
                grad_i = out.grad[i].reshape(-1, 1)
                
                jacobian = np.diagflat(softmax_i) - np.dot(softmax_i, softmax_i.T)
                
                self.grad[i] += np.dot(jacobian, grad_i).flatten()

        out._backward = _backward
        return out
      
    def log(self):
        t = np.log(self.data)
        out = Value(t, (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        t = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Value(t, (self,), "sum")
        
        def _backward():
            if axis is not None and not keepdims:
                expanded_grad = out.grad
                
                axes = [axis] if not isinstance(axis, tuple) else list(axis)
                axes.sort(reverse=True)
                
                for ax in axes:
                    expanded_grad = np.expand_dims(expanded_grad, axis=ax)
                    
                self.grad += expanded_grad
            else:
                self.grad += out.grad
        
        out._backward = _backward
        return out

    def clip(self, min_val, max_val):
        clipped_data = np.clip(self.data, min_val, max_val)
        out = Value(clipped_data, (self,), "clip")

        def _backward():
            grad_mask = (self.data >= min_val) & (self.data <= max_val)
            self.grad += grad_mask * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
