import numpy as np
from math import log2, exp, log
from tqdm import tqdm
from numba import njit
from numba import types as nb

EPSILON = 1e-8

def log2_with_mask(arr):
    result = np.zeros_like(arr, dtype=np.float64)
    mask = (arr != 0)
    result[mask] = np.log2(arr[mask])
    return result

#@njit(nb.float64(nb.float64), nogil=True)
def numba_log2(x):
    return np.log2(x)

#@njit
def binary_search_perplexity_i(sqdistances, P, n_samples, n_neighbors, n_steps, ENTROPY_TOLERANCE):
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

        shannon_entropy = -np.sum(P[i, :] * log2_with_mask(P[i, :]))
        entropy = shannon_entropy + beta * sum_disti_Pi # beta * sum_disti_Pi acts as a regluarization term
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

#@njit
def binary_search_perplexity(sqdistances, perplexity, n_steps=100, ENTROPY_TOLERANCE=1e-5, display_tqdm=False):
    n_samples, n_neighbors = sqdistances.shape
    #desired_entropy = numba_log2(perplexity)
    desired_entropy = log(perplexity)

    P = np.zeros((n_samples, n_neighbors), dtype=np.float64)

    if display_tqdm:
        print("Computing binary search for conditional probabilities...")
        
    range_ = tqdm(range(n_samples)) if display_tqdm else range(n_samples)
    for i in range_:
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

            shannon_entropy = -np.sum(P[i, :] * log2_with_mask(P[i, :]))
            #entropy = shannon_entropy + beta * sum_disti_Pi # beta * sum_disti_Pi acts as a regluarization term
            entropy = log(sum_Pi) + beta * sum_disti_Pi
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

    return P

def compute_joint_probabilities(sqdistances, perplexity, n_steps=100, ENTROPY_TOLERANCE=1e-5, display_tqdm=False):
    n_samples, n_neighbors = sqdistances.shape
    conditional_probas = binary_search_perplexity(sqdistances, perplexity, n_steps=n_steps, ENTROPY_TOLERANCE=ENTROPY_TOLERANCE, display_tqdm=display_tqdm)
    
    return (conditional_probas + conditional_probas.T) / (2 * n_samples)