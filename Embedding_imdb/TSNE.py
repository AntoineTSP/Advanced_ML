import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

# local imports
from joint_probabilities import compute_joint_probabilities

class TSNE():
    def __init__(self, n_components, perplexity, metric="euclidean", n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.metric = metric
        self.n_iter = max(250, n_iter)

    def compute_gradient(self, Y, P, Q, t_student_q_distances):
        gradient = np.zeros((P.shape[0], self.n_components))
        for i in range(P.shape[1]):
            gradient[i] = 4 * np.sum(((P[i, :] - Q[i, :]) * t_student_q_distances[i, :])[:, np.newaxis] * (Y[i] - Y), axis=0)
            
        return gradient

    def fit_transform(self, X, verbose=False):
        squared_distances = pairwise_distances(X, metric=self.metric, squared=True)
        P = compute_joint_probabilities(squared_distances, self.perplexity, n_steps=100, ENTROPY_TOLERANCE=1e-5, display_tqdm=verbose)

        # initial solution samples from a multivariate centered normal
        Y = np.random.multivariate_normal(np.zeros(self.n_components), 1e-4 * np.eye(self.n_components), size=n_samples)

        if verbose:
            print("Starting gradient descent loop...")

        q_distances = pairwise_distances(Y, metric=metric, squared=True)
        # compute low-dimensional affinities q_{ij}
        t_student_q_distances = (1.0 + q_distances) ** (-1)
        Q = t_student_q_distances / np.sum(t_student_q_distances, axis=1, keepdims=True)

        # compute gradient
        