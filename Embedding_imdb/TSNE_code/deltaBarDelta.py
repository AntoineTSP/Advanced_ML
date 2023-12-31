import numpy as np
   
class DeltaBarDeltaOptimizer:
    """Create an optimizer that follows the delta-bar-delta method.

    Parameters
    ----------
    learning_rate : float
        Current value of the learning rate.
    momentum : float, optional
        Momentum determines how much influence the weighted average of previous
        gradients has on the current value of the learning rate, by default 0.7.
    decay_factor : float, optional
        Multiplicative constant to apply to learning rates that need to be slowed
        down, by default 0.1.
    increase_factor : int, optional
        Increment to apply to learning rates that need to be accelerated, by
        default 5.
    epsilon : float, optional
        Minimum threshold for the learning rate, by default 1e-4.

    Attributes
    ----------
    prev_delta_bar : numpy.ndarray
        Previous value of delta_bar used for momentum calculations.

    Methods
    -------
    update(gradient)
        Update the learning rate based on the provided gradient.

    Examples
    --------
    >>> optimizer = DeltaBarDeltaOptimizer(learning_rate=50)
    >>> for epoch in range(num_epochs):
    ...     gradient, _ = self.gradient_descent(t, Y, P, Q, t_student_q_distances)
            learning_rate = optimizer.update(gradient)
            Y_t_plus_1 = self.Y - learning_rate * gradient + momentum * (self.Y - previous_Y) 
            # momentum here corresponds to another momentum, associated to the TSNE instance
    """
        
    def __init__(self, learning_rate, momentum=0.7, decay_factor=0.1, increase_factor=5, epsilon=1e-4):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay_factor = decay_factor
        self.increase_factor = increase_factor
        self.epsilon = epsilon
        self.prev_delta_bar = None

    def update(self, gradient):
        """Update the learning rate based on the provided gradient.

        Parameters
        ----------
        gradient : numpy.ndarray, shape (n_samples, n_components)
            The gradient of the KL divergence wrt low-dimensional space datapoints

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_components)
            The updated learning rates for each component
        """
        
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