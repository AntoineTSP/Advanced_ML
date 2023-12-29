import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm
from numba import njit, float64, prange
import time
from IPython import display
import matplotlib.pyplot as plt

# local imports
from TSNE_code.joint_probabilities import compute_joint_probabilities
from TSNE_code.deltaBarDelta import DeltaBarDeltaOptimizer


EPSILON = np.finfo(float).eps


def _compute_squared_distances(X, metric):
    return pairwise_distances(X, metric=metric, squared=True, n_jobs=-1)

@njit
def _compute_kl_divergence(P, Q):
    return np.sum(P * np.log(np.maximum(P, EPSILON) / Q))

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
                 n_iter=1000, n_steps_probas=100, min_grad_norm=1e-5, interval_convergence_check=50, momentum_rule=_momentum, adaptive_learning_rate=True, patience=None):
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
        self.patience = patience
        
        self.kl_divergence = []
        self.Y = None
        
    def gradient_descent(self, t, Y, P, Q, t_student_q_distances):
        if t < self.iterations_with_early_exaggeration:
            gradient = _compute_gradient(Y, self.early_exaggeration*P, Q, t_student_q_distances, self.n_components)
        else:
            gradient = _compute_gradient(Y, P, Q, t_student_q_distances, self.n_components)

        convergence = False
        if t % self.interval_convergence_check == 0:
            convvergence = _check_convergence(gradient, self.min_grad_norm)
            
        return gradient, convergence
    
    # def dynamically_plot_kl_divergence(self, lower_bound=None, lower_bound_label='', figsize=(7, 4)):
    #     plt.figure(figsize=figsize)
    #     plt.cla()
    #     label = 'Custom' if not self.adaptive_learning_rate else 'Custom + adaptive lr'
    #     plt.plot(self.kl_divergence, label=label, c='black')
    #     if lower_bound is not None:
    #         plt.plot([0, len(self.kl_divergence)], [lower_bound, lower_bound], label=lower_bound_label)
    #     plt.grid('on')
    #     plt.legend()
    #     plt.title("KL divergence through gradient descent")
    #     display.display(plt.gcf())
    #     display.clear_output(wait=True)
    #     plt.close()
    
    def dynamically_plot_kl_divergence(self, t, bar_length=50, lower_bound=None, lower_bound_label='', figsize=(7, 6)):
        fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot KL divergence on the upper subplot
        label = 'Custom' if not self.adaptive_learning_rate else 'Custom + adaptive lr'
        ax[0].plot(self.kl_divergence, label=label, c='black')      
        if lower_bound is not None:
            ax[0].plot([0, len(self.kl_divergence)], [lower_bound, lower_bound], label=lower_bound_label)
        ax[0].grid('on')
        ax[0].legend()
        ax[0].set_title("KL divergence through gradient descent")

        # Progress bar in the lower subplot
        ax[1].axis('off')
        progress_value = t / self.n_iter
        bar = "[" + "#" * int(bar_length * progress_value) + "-" * (bar_length - int(bar_length * progress_value)) + "]"
        ax[1].text(0.5, 0.5, f"Progress: {round(100*progress_value, 2)}% {bar}", ha='center', va='center', fontsize=12)

        display.display(fig)
        display.clear_output(wait=True)
        plt.close()
        

    def fit_transform(self, X, reference_kl_divergence=None, title=None, verbose=0):
        squared_distances = _compute_squared_distances(X, self.metric)
        P = compute_joint_probabilities(squared_distances, self.perplexity, n_steps=self.n_steps_probas, ENTROPY_TOLERANCE=1e-5, display_tqdm=(verbose>=2))

        # initial solution samples from a multivariate centered normal
        if self.Y is None:
            self.Y = np.random.multivariate_normal(np.zeros(self.n_components), 1e-4 * np.eye(self.n_components), size=X.shape[0])
        previous_Y = self.Y # will be used to store value of Y(t-1)

        if verbose >= 1:
            print("Starting gradient descent loop...")

        if self.learning_rate is None:
            self.learning_rate = max(X.shape[0] / self.early_exaggeration / 4, 50)

        optimizer = DeltaBarDeltaOptimizer(learning_rate=np.zeros_like(self.Y) + self.learning_rate)
        
        if self.patience is not None:
            best_error = np.finfo(float).max
            step_best_error = 0

        range_ = tqdm(range(self.n_iter)) if verbose in [1, 2] else range(self.n_iter)
        for t in range_:

            q_distances = pairwise_distances(self.Y, metric=self.metric, squared=True, n_jobs=-1)
            Q, t_student_q_distances = _compute_t_student_distances(q_distances)
            
            # gradient descent
            gradient, convergence_check = self.gradient_descent(t, self.Y, P, Q, t_student_q_distances)
            if convergence_check:
                if verbose >= 1:
                    print(f"Algorithm has converged at step {t}")
                return self.Y
                
            if self.patience is not None:
                kl_divergence = _compute_kl_divergence(P, Q)
                self.kl_divergence.append(kl_divergence)
                
                if kl_divergence <= best_error:
                    best_error = kl_divergence
                    step_best_error = t
                    if verbose >= 3 and (t % 20 == 0 or t == self.n_iter-1):
                        title = title if title is not None else 'reference KL divergence'
                        self.dynamically_plot_kl_divergence(t, bar_length=50, lower_bound=reference_kl_divergence, lower_bound_label=title)
                        
                elif t - step_best_error > self.patience:
                    if verbose >= 1:
                        print(f"Algorithm has stopped at step {t}")
                    return self.Y

            learning_rate = optimizer.update(gradient) if self.adaptive_learning_rate else self.learning_rate
            momentum = self.momentum_rule(t)
            Y_t_plus_1 = self.Y - learning_rate * gradient + momentum * (self.Y - previous_Y)

            # update Ys
            previous_Y = self.Y
            self.Y = Y_t_plus_1

        return self.Y

            
        