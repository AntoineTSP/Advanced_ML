import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm
from numba import njit, prange # allows us to easily parallelize and optimize our code by compiling it
from IPython import display
import matplotlib.pyplot as plt

# local imports
from TSNE_code.joint_probabilities import compute_joint_probabilities
from TSNE_code.deltaBarDelta import DeltaBarDeltaOptimizer


EPSILON = np.finfo(float).eps
VERBOSE_NONE = 0
VERBOSE_PROGRESS_BAR = 1
VERBOSE_BINARY_SEARCH = 2
VERBOSE_DYNAMIC_PLOTS = 3


def _compute_squared_distances(X, metric):
    """Compute the squared pairwise distances.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        Elements to compute the distances on
    metric : str
        Metric to compute distances with

    Returns
    -------
    numpy.ndarray, shape (n_samples, n_samples)
        The matrix of squared pairwise_distances
    """
    
    return pairwise_distances(X, metric=metric, squared=True, n_jobs=-1)

@njit
def _compute_kl_divergence(P, Q):
    """Compute the Kullback-Leibler divergence.

    Parameters
    ----------
    P : numpy.ndarray, shape (n_samples, n_features)
        High-dimensional space joint probabilities
    Q : numpy.ndarray, shape (n_samples, n_components)
        Low-dimensional space joint probabilities

    Returns
    -------
    float
        Kullback-Leilbler divergence 
    """
    
    return np.sum(P * np.log(np.maximum(P, EPSILON) / Q))

@njit(parallel=True)
def _compute_gradient(Y, P, Q, t_student_q_distances, n_components):
    """Compute the gradient of the KL divergence wrt the projection on the low-dimensional space.

    Parameters
    ----------
    Y : numpy.ndarray, shape (n_samples, n_components)
        Projection of original data on low-dimensional space
    P : numpy.ndarray, shape (n_samples, n_features)
        High-dimensional space joint probabilities
    Q : numpy.ndarray, shape (n_samples, n_components)
        Low-dimensional space joint probabilities
    t_student_q_distances : numpy.ndarray, shape (n_samples, n_samples)
        Array containing (1 / ||y_i - y_j||²), (y_i) elements of Y
    n_components : int
        Dimension of low-dimensional space

    Returns
    -------
    numpy.ndarray, shape (n_samples, n_components)
        Gradient of KL divergence wrt Y
    """
    
    grad = np.zeros((P.shape[0], n_components))
    for i in prange(P.shape[0]):
        sum_ = np.zeros(n_components)
        for j in range(P.shape[1]):
            sum_ += (P[i, j] - Q[i, j]) * (Y[i] - Y[j]) * t_student_q_distances[i, j]
        grad[i] = 4 * sum_
    return grad

@njit
def _compute_t_student_distances(q_distances):
    """Compute low-dimensional affinities q_{ij}.

    Parameters
    ----------
    q_distances : numpy.ndarray, shape (n_samples, n_samples)
        Squared pairwise distances of Y, the projection on the low-dimensional space 

    Returns
    -------
    Q : numpy.ndarray, shape (n_samples, n_samples)
        Matrix containing the low-dimensional affinities q_{ij}
    t_student_q_distances : numpy.ndarray, shape (n_samples, n_samples)
        Array containing (1 / ||y_i - y_j||²), (y_i) elements of Y
    """
    
    t_student_q_distances = (1.0 + q_distances) ** (-1)
    Q = t_student_q_distances / (np.sum(t_student_q_distances) - t_student_q_distances.shape[0]) # we remove the shape because the sum we divide by should not take into account 
                                                                                                         # elements on the diagonal of t_student_q_distances, all equal to 1
    return Q, t_student_q_distances

def _check_convergence(gradient, min_grad_norm):
    """Check whether the norm of the gradient is lower than a fixed quantity"""
    grad_norm = np.linalg.norm(gradient)
    return abs(grad_norm) < min_grad_norm

def _momentum(t):
    """Default momentum rule"""
    return 0.5 if t < 250 else 0.8


class TSNE():
    """Create an instance of the class TSNE

        Parameters
        ----------
        n_components : int 
            Dimension of the low-dimensional space to project data onto
        perplexity : float
            The perplexity parameter balances the trade-off between preserving local
            and global structure in the data. It is a smoothing parameter that
            influences the effective number of nearest neighbors each point has during
            the algorithm's optimization process. A higher perplexity value may
            capture more global structure, but it can also lead to more focus on
            global patterns at the expense of local details. Conversely, a lower
            perplexity may result in more emphasis on local details but might miss
            overarching patterns. Typical values range from 5 to 50, and the choice
            depends on the characteristics of the data.
        early_exaggeration : int, optional
            Controls the initial spacing between clusters in the t-SNE visualization, by default 4
        iterations_with_early_exaggeration : int, optional
            Number of iterations where ealy exaggeration is applied, by default 50
        learning_rate : float, optional
            Initial value of the learning rate used for the gradient descent, by default None
        metric : str, optional
            Metric to use to compute distances, by default "euclidean"
        n_iter : int, optional
            Number of iterations for the gradient descent, by default 1000
        n_steps_probas : int, optional
            Number of steps allowed for the binary search to find the value of the variance of the Gaussian
            to compute the matrix of joint probabilities on the high-dimensional space P, by default 100
        min_grad_norm : _type_, optional
            If the norm of the gradient falls below this value, the gradient descent is stopped early, by default 1e-5
        interval_convergence_check : int, optional
            Convergence will be checked every 'interval_convergence_check' iterations. A smaller number guarantees that potential
            convegence will be caught more quickly, but demands more computation time, by default 50
        momentum_rule : method, optional
            The rule to apply for the momentum, must be a method taking t, the current gradient descent step, as single parameter. 
            If momentum is constant, define a function which returns said constant, by default -> 0.5 if t < 250 else 0.8
        adaptive_learning_rate : bool, optional
            If True, the delta-bar-delta method is applied to adaptively adjust the learning rate during the gradient descent optimization process. 
            This adaptive learning rate can help improve convergence and performance, especially in cases where the scale of gradients varies 
            across dimensions, by default True
        patience : int, optional
            Determines the number of consecutive epochs with no improvement of the KL divergence before the training process is stopped. 
            If None, the gradient descent will run for n_iter steps, whether there are improvements or not, by default None
            
        Methods
        -------
        fit_transform(X, reference_kl_divergence=None, title=None, verbose=0)
            Fit X into a low-dimensional space.
            
        Examples
        --------
        >>> from TSNE_code.TSNE_utils import TSNE
        >>> import numpy as np
        >>> X = np.random.rand(100, 50)  # Replace with your actual data
        >>> reference_kl_divergence = 42.0  # Replace with your reference value
        >>> tsne = TSNE(n_components=2, perplexity=30, adaptive_learning_rate=True, patience=50)
        >>> low_dim_representation = tsne.fit_transform(X, reference_kl_divergence=reference_kl_divergence, title='Reference', verbose=3)
        >>> # Plot the low-dimensional representation or perform further analysis
        """
        
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
        """Run one iteration of the gradient descent.

        Parameters
        ----------
        t : int
            Current step of the gradient descent
        Y : numpy.ndarray, shape (n_samples, n_components)
            Projection of original data on low-dimensional space
        P : numpy.ndarray, shape (n_samples, n_features)
            High-dimensional space joint probabilities
        Q : numpy.ndarray, shape (n_samples, n_components)
            Low-dimensional space joint probabilities
        t_student_q_distances : numpy.ndarray, shape (n_samples, n_samples)
            Array containing (1 / ||y_i - y_j||²), (y_i) elements of Y
            

        Returns
        -------
        gradient : numpy.ndarray, shape (n_samples, n_components)
            Gradient computed at step t
        convergence : bool
            Whether the algorithm has converged yet
        """
        
        if t < self.iterations_with_early_exaggeration:
            gradient = _compute_gradient(Y, self.early_exaggeration*P, Q, t_student_q_distances, self.n_components)
        else:
            gradient = _compute_gradient(Y, P, Q, t_student_q_distances, self.n_components)

        convergence = False
        if t % self.interval_convergence_check == 0:
            convvergence = _check_convergence(gradient, self.min_grad_norm)
            
        return gradient, convergence

    
    def dynamically_plot_kl_divergence(self, t, bar_length=50, lower_bound=None, lower_bound_label='', figsize=(7, 6)):
        """Plot the evolution of the KL divergence up to step t.

        Parameters
        ----------
        t : int
            Current step of the gradient descent
        bar_length : int, optional
            Length of the progress bar to display underneath the figure, by default 50
        lower_bound : float, optional
            Lower bound to display on the figure.
            Typically the final KL divergence of another instance of t-SNE we want to compare the current instance with, by default None
        lower_bound_label : str, optional
            Label to give as a legend to the lower bound if one is provided, by default ''
        figsize : tuple, optional
            Size of the figure to display, by default (7, 6)
        """
        
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
        """Fit X into a low-dimensional space.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Original data on the high-dimensional space
        reference_kl_divergence : float, optional
            Typically the final KL divergence of another instance of t-SNE we want to compare the current instance with, by default None
        title : _type_, optional
            Label to give as a legend to reference_kl_divergence, if one is provided, on the dynamic plots, by default None
        verbose : int, optional
            Verbosity level: 0 -> no verbosity
                             1 -> progress bar for the gradient descent
                             2 -> verbosity 1 + progress bars for the binary search for the computation of conditional probabilities
                             3 -> verbosity 2 + dynamic plot of the evolution of the KL divergence during gradient descent
            , by default 0

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_components)
            Projected data onto the low-dimensional space
        """
        
        squared_distances = _compute_squared_distances(X, self.metric)
        P = compute_joint_probabilities(squared_distances, self.perplexity, n_steps=self.n_steps_probas, ENTROPY_TOLERANCE=1e-5, display_tqdm=(verbose>=VERBOSE_BINARY_SEARCH))

        # initial solution samples from a multivariate centered normal
        if self.Y is None:
            self.Y = np.random.multivariate_normal(np.zeros(self.n_components), 1e-4 * np.eye(self.n_components), size=X.shape[0])
        previous_Y = self.Y # will be used to store value of Y(t-1)

        if verbose >= VERBOSE_PROGRESS_BAR :
            print("Starting gradient descent loop...")

        if self.learning_rate is None:
            self.learning_rate = max(X.shape[0] / self.early_exaggeration / 4, 50)

        optimizer = DeltaBarDeltaOptimizer(learning_rate=np.zeros_like(self.Y) + self.learning_rate)
        
        if self.patience is not None:
            best_error = np.finfo(float).max
            step_best_error = 0

        range_ = tqdm(range(self.n_iter)) if verbose in [VERBOSE_PROGRESS_BAR , VERBOSE_BINARY_SEARCH ] else range(self.n_iter)
        for t in range_:

            q_distances = pairwise_distances(self.Y, metric=self.metric, squared=True, n_jobs=-1)
            Q, t_student_q_distances = _compute_t_student_distances(q_distances)
            
            # gradient descent
            gradient, convergence_check = self.gradient_descent(t, self.Y, P, Q, t_student_q_distances)
            if convergence_check:
                if verbose >= VERBOSE_PROGRESS_BAR:
                    print(f"Algorithm has converged at step {t}")
                return self.Y
                
            if self.patience is not None:
                kl_divergence = _compute_kl_divergence(P, Q)
                self.kl_divergence.append(kl_divergence)
                
                if kl_divergence <= best_error or t < self.iterations_with_early_exaggeration:
                    best_error = kl_divergence
                    step_best_error = t
                    if verbose >= VERBOSE_DYNAMIC_PLOTS  and (t % 20 == 0 or t == self.n_iter-1):
                        title = title if title is not None else 'reference KL divergence'
                        self.dynamically_plot_kl_divergence(t, bar_length=50, lower_bound=reference_kl_divergence, lower_bound_label=title)
                        
                elif t - step_best_error > self.patience:
                    if verbose >= VERBOSE_PROGRESS_BAR:
                        print(f"Algorithm has stopped at step {t}")
                    return self.Y

            learning_rate = optimizer.update(gradient) if self.adaptive_learning_rate else self.learning_rate
            momentum = self.momentum_rule(t)
            Y_t_plus_1 = self.Y - learning_rate * gradient + momentum * (self.Y - previous_Y)

            # update Ys
            previous_Y = self.Y
            self.Y = Y_t_plus_1

        return self.Y
 