import numpy as np

class DeltaBarDeltaOptimizer:
    def __init__(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.previous_gradient = np.zeros_like(self.learning_rate)
        self.momentum_buffer = np.zeros_like(self.learning_rate)

    def update(self, gradient, t):
        delta = gradient - self.previous_gradient

        self.momentum_buffer = self.momentum(t) * self.momentum_buffer + delta

        new_learning_rate = self.learning_rate * (1 + np.linalg.norm(self.momentum_buffer))
        self.learning_rate = max(new_learning_rate, 1e-6)  # Prevent learning rate from becoming too small

        self.previous_gradient = gradient

        return self.learning_rate
