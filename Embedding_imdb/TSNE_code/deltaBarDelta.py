import numpy as np
   
class DeltaBarDeltaOptimizer:
    def __init__(self, learning_rate, momentum=0.7, decay_factor=0.1, increase_factor=5, epsilon=1e-4):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay_factor = decay_factor
        self.increase_factor = increase_factor
        self.epsilon = epsilon
        self.prev_delta_bar = None

    def update(self, gradient):
        if self.prev_delta_bar is None:
            self.prev_delta_bar = np.zeros_like(gradient)

        sign = np.sign(gradient * self.prev_delta_bar)

        # Learning rate update
        positive_sign_indices = np.where(sign > 0)
        negative_sign_indices = np.where(sign < 0)

        # Increase learning rate for positive signs
        self.learning_rate[positive_sign_indices] += self.increase_factor

        # Decrease learning rate for negative signs
        self.learning_rate[negative_sign_indices] -= self.decay_factor * self.learning_rate[negative_sign_indices]

        # Ensure learning rate doesn't become too small
        self.learning_rate = np.maximum(self.learning_rate, self.epsilon)

        # compute delta_bar(t)
        self.prev_delta_bar = (1 - self.momentum) * gradient + self.momentum * self.prev_delta_bar

        return self.learning_rate