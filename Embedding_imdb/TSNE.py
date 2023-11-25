import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

# local imports
from joint_probabilities import compute_joint_probabilities

class TSNE():
    def __init__(self, n_components, perplexity, early_exaggeration=4, iterations_with_early_exaggeration=50, learning_rate=None, metric="euclidean", n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.iterations_with_early_exaggeration = iterations_with_early_exaggeration
        self.learning_rate = learning_rate
        self.metric = metric
        self.n_iter = max(250, n_iter)

    def compute_gradient(self, Y, P, Q, t_student_q_distances):
        gradient = np.zeros((P.shape[0], self.n_components))
        for i in range(P.shape[1]):
            gradient[i] = 4 * np.sum(((P[i, :] - Q[i, :]) * t_student_q_distances[i, :])[:, np.newaxis] * (Y[i] - Y), axis=0)
            
        return gradient

    def fit_transform(self, X, verbose=True):
        squared_distances = pairwise_distances(X, metric=self.metric, squared=True)
        P = compute_joint_probabilities(squared_distances, self.perplexity, n_steps=100, ENTROPY_TOLERANCE=1e-5, display_tqdm=verbose)

        # initial solution samples from a multivariate centered normal
        Y = np.random.multivariate_normal(np.zeros(self.n_components), 1e-4 * np.eye(self.n_components), size=X.shape[0])
        previous_Y = np.zeros(Y.shape) # will be used to store value of Y(t-1)

        if verbose:
            print("Starting gradient descent loop...")

        if self.learning_rate is None:
            self.learning_rate = max(X.shape[0] / self.early_exaggeration / 4, 50)

        range_ = tqdm(range(self.n_iter)) if verbose else range(self.n_iter)
        for t in range_:
            q_distances = pairwise_distances(Y, metric=self.metric, squared=True)
            # compute low-dimensional affinities q_{ij}
            t_student_q_distances = (1.0 + q_distances) ** (-1)
            Q = t_student_q_distances / np.sum(t_student_q_distances, axis=1, keepdims=True)

            # gradient descent
            if t < self.iterations_with_early_exaggeration:
                gradient = self.compute_gradient(Y, self.early_exaggeration*P, Q, t_student_q_distances)
            else:
                gradient = self.compute_gradient(Y, P, Q, t_student_q_distances)
                
            momentum = 0.5 if t < 250 else 0.8
            Y_t_plus_1 = Y + self.learning_rate * gradient + momentum * (Y - previous_Y)

            # update Ys
            previous_Y = Y
            Y = Y_t_plus_1

        return Y

            
        