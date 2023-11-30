
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm
from numba import njit, float64, prange
import time

# local imports
from TSNE_code.joint_probabilities import compute_joint_probabilities
from TSNE_code.deltaBarDelta import DeltaBarDeltaOptimizer

def _compute_squared_distances(X, metric):
    return pairwise_distances(X, metric=metric, squared=True)

# @njit
# def _compute_gradient(Y, P, Q, t_student_q_distances, n_components):
#     gradient_f = np.zeros((P.shape[0], n_components))
#     for i in range(P.shape[1]):
#         gradient_f[i] = 4 * np.sum(((P[i, :] - Q[i, :]) * t_student_q_distances[i, :])[:, np.newaxis] * (Y[i] - Y), axis=0)

#     return gradient_f

@njit(parallel=True)
def _compute_gradient(Y, P, Q, t_student_q_distances, n_components):
    grad = np.zeros((P.shape[0], n_components))
    for i in prange(P.shape[0]):
        sum_ = np.zeros(n_components)
        for j in range(P.shape[1]):
            sum_ += (P[i, j] - Q[i, j]) * (Y[i] - Y[j]) * t_student_q_distances[i, j]
        grad[i] = 4 * sum_
    return grad

@njit
def _compute_t_student_distances(q_distances):
    # compute low-dimensional affinities q_{ij}
    t_student_q_distances = (1.0 + q_distances) ** (-1)
    Q = t_student_q_distances / (np.sum(t_student_q_distances) - t_student_q_distances.shape[0]) # we remove the shape because the sum we divide by should not take into account 
                                                                                                         # elements on the diagonal of t_student_q_distances, all equal to 1
    return Q, t_student_q_distances

def _check_convergence(gradient, min_grad_norm):
    grad_norm = np.linalg.norm(gradient)
    return abs(grad_norm) < min_grad_norm

def _momentum(t):
    return 0.5 if t < 250 else 0.8


class TSNE():
    def __init__(self, n_components, perplexity, early_exaggeration=4, iterations_with_early_exaggeration=50, learning_rate=None, metric="euclidean", 
                 n_iter=1000, n_steps_probas=100, min_grad_norm=1e-5, interval_convergence_check=50, momentum_rule=_momentum, adaptive_learning_rate=True):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.iterations_with_early_exaggeration = iterations_with_early_exaggeration
        self.learning_rate = learning_rate
        self.metric = metric
        self.n_iter = max(250, n_iter)
        self.n_steps_probas = n_steps_probas
        self.min_grad_norm = min_grad_norm
        self.interval_convergence_check = interval_convergence_check
        self.momentum_rule = momentum_rule
        self.adaptive_learning_rate = adaptive_learning_rate

    def fit_transform(self, X, verbose=0):
        squared_distances = _compute_squared_distances(X, self.metric)
        P = compute_joint_probabilities(squared_distances, self.perplexity, n_steps=self.n_steps_probas, ENTROPY_TOLERANCE=1e-5, display_tqdm=(verbose>=2))

        # initial solution samples from a multivariate centered normal
        Y = np.random.multivariate_normal(np.zeros(self.n_components), 1e-4 * np.eye(self.n_components), size=X.shape[0])
        previous_Y = np.zeros(Y.shape) # will be used to store value of Y(t-1)

        if verbose >= 1:
            print("Starting gradient descent loop...")

        if self.learning_rate is None:
            self.learning_rate = max(X.shape[0] / self.early_exaggeration / 4, 50)

        optimizer = DeltaBarDeltaOptimizer(learning_rate=self.learning_rate, momentum=lambda t: 0.9)

        range_ = tqdm(range(self.n_iter)) if verbose >= 1 else range(self.n_iter)
        for t in range_:

            q_distances = pairwise_distances(Y, metric=self.metric, squared=True, n_jobs=-1)
            Q, t_student_q_distances = _compute_t_student_distances(q_distances)

            # gradient descent
            if t < self.iterations_with_early_exaggeration:
                gradient = _compute_gradient(Y, self.early_exaggeration*P, Q, t_student_q_distances, self.n_components)
            else:
                gradient = _compute_gradient(Y, P, Q, t_student_q_distances, self.n_components)
            learning_rate = optimizer.update(gradient, t) if self.adaptive_learning_rate else self.learning_rate

            if t % self.interval_convergence_check == 0:
                if _check_convergence(gradient, self.min_grad_norm):
                    if verbose >= 1:
                        print(f"Algorithm has converged at step {t}")
                    return Y

            momentum = self.momentum_rule(t)
            Y_t_plus_1 = Y - learning_rate * gradient + momentum * (Y - previous_Y)

            # update Ys
            previous_Y = Y
            Y = Y_t_plus_1

        return Y

            
        