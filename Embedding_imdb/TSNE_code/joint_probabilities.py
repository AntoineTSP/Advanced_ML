import numpy as np
from math import log2, exp, log
from tqdm import tqdm
from numba import njit

EPSILON = 1e-8

def log2_with_mask(arr):
    """Apply log2 to each non-null elements of a positive array.

    Parameters
    ----------
    arr : numpy.ndarray
        Array of positive or null elements

    Returns
    -------
    numpy.ndarray
        Non-null elements of passed array were applied log2 to, 0s were left untouched
    """
    
    result = np.zeros_like(arr, dtype=np.float64)
    mask = (arr != 0)
    result[mask] = np.log2(arr[mask])
    return result

@njit
def binary_search_perplexity_i(i, sqdistances, P, desired_entropy, n_samples, n_neighbors, n_steps, ENTROPY_TOLERANCE):    
    """
    Binary search to find conditional probabilities to all other datapoints, for a given datapoint.

    Parameters
    ----------
    i : int
        Index of the data point.
    sqdistances : numpy.ndarray, shape (n_samples, n_neighbors)
        Squared pairwise distances between data points.
    P : numpy.ndarray, shape (n_samples, n_neighbors)
        Matrix to store conditional probabilities.
    desired_entropy : float
        Desired Shannon entropy.
    n_samples : int
        Number of data points.
    n_neighbors : int
        Number of neighbors to compute distances with.
    n_steps : int
        Number of steps for the binary search.
    ENTROPY_TOLERANCE : float
        Tolerance for the difference between actual and desired entropy.
    """
        
    beta_min = -np.inf
    beta_max = np.inf
    beta = 1.0 # represents 1/(2 * sigma^2)
    
    for l in range(n_steps):
        sum_Pi = 0.0
        for j in range(n_neighbors):
            if j != i:
                P[i, j] = exp(-sqdistances[i, j] * beta)
                sum_Pi += P[i, j] # will be used later on to normalize the p_i|j's

        if sum_Pi == 0.0:
            sum_Pi = EPSILON # avoid dividing by 0

        P[i, :] /= sum_Pi
        sum_disti_Pi = np.sum(np.multiply(sqdistances[i, :], P[i, :]))

        entropy = log(sum_Pi) + beta * sum_disti_Pi # beta * sum_disti_Pi acts as a regluarization term
        entropy_diff = entropy - desired_entropy

        if abs(entropy_diff) <= ENTROPY_TOLERANCE:
            break

        if entropy_diff > 0.0: # entropy is too high, we need to increase beta i.e decrease sigma
            beta_min = beta
            if beta_max == np.inf:
                beta *= 2.0
            else:
                beta = (beta + beta_max) / 2.0
        else:
            beta_max = beta
            if beta_min == -np.inf:
                beta /= 2.0
            else:
                beta = (beta + beta_min) / 2.0

def binary_search_perplexity(sqdistances, perplexity, n_steps=100, ENTROPY_TOLERANCE=1e-5, display_tqdm=False):
    """
    Compute conditional probabilities using binary search for a given perplexity.

    Parameters
    ----------
    sqdistances : numpy.ndarray, shape (n_samples, n_neighbors)
        Squared pairwise distances between data points.
    perplexity : float
        Perplexity value.
    n_steps : int, optional
        Number of steps for the binary search, by default 100.
    ENTROPY_TOLERANCE : float, optional
        Tolerance for the difference between actual and desired entropy, by default 1e-5.
    display_tqdm : bool, optional
        If True, a progress bar will be displayed to show progress of rows of P computed, by default False.

    Returns
    -------
    numpy.ndarray, shape (n_samples, n_neighbors)
        Matrix of conditional probabilities.
    """
    
    n_samples, n_neighbors = sqdistances.shape
    desired_entropy = log(perplexity)

    P = np.zeros((n_samples, n_neighbors), dtype=np.float64)

    if display_tqdm:
        print("Computing binary search for conditional probabilities...")
        
    range_ = tqdm(range(n_samples)) if display_tqdm else range(n_samples)
    for i in range_:
        binary_search_perplexity_i(i, sqdistances, P, desired_entropy, n_samples, n_neighbors, n_steps, ENTROPY_TOLERANCE)

    return P

def compute_joint_probabilities(sqdistances, perplexity, n_steps=100, ENTROPY_TOLERANCE=1e-5, display_tqdm=False):
    n_samples, n_neighbors = sqdistances.shape
    conditional_probas = binary_search_perplexity(sqdistances, perplexity, n_steps=n_steps, ENTROPY_TOLERANCE=ENTROPY_TOLERANCE, display_tqdm=display_tqdm)
    
    return (conditional_probas + conditional_probas.T) / (2 * n_samples)