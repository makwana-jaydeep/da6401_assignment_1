"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from ann.activations import ACT_FN, ACT_GRAD


class Layer:
    def __init__(self, in_dim, out_dim, activation, weight_init):
        self.activation = activation
        self._init_weights(in_dim, out_dim, weight_init)
        # gradients exposed after backward
        self.grad_W = None
        self.grad_b = None
        # cache for backpropgation
        self._input = None
        self._z = None

    def _init_weights(self, in_dim, out_dim, method):
        if method == "xavier":
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            self.W = np.random.uniform(-limit, limit, (in_dim, out_dim))

        else:  # random
            self.W = np.random.randn(in_dim, out_dim) * 0.01
        self.b = np.zeros((1, out_dim))

    def forward(self, x):
        self._input = x
        self._z = x @ self.W + self.b
        if self.activation is None:
            return self._z
        return ACT_FN[self.activation](self._z)

    def backward(self, delta):
        """delta: the gradient flowing back from the next layer, measured with respect to this layers 
        output after activation. It already includes the averaging over the batch from the loss,
        so you dont need to divide again."""

        if self.activation is not None:
            delta = delta * ACT_GRAD[self.activation](self._z)
        self.grad_W = self._input.T @ delta
        self.grad_b = np.sum(delta, axis=0, keepdims=True)
        return delta @ self.W.T