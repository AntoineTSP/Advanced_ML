import numpy as np
import math
from tqdm import tqdm

EPSILON = 1e-8

def binary_search_perplexity(sqdistances, perplexity, n_steps=100, ENTROPY_TOLERANCE=1e-5, display_tqdm=False):
    n_samples, n_neighbors = sqdistances.shape
    desired_entropy = math.log2(perplexity)

    P = np.zeros((n_samples, n_neighbors), dtype=np.float64)

    range_ = tqdm(range(n_samples)) if display_tqdm else range(n_samples)
    for i in range(n_samples):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0 # represents 1/(2 * sigma^2)
        
        for l in range(n_steps):
            sum_Pi = 0.0
            for j in range(n_neighbors):
                if j != i:
                    P[i, j] = math.exp(-sqdistances[i, j] * beta)
                    sum_Pi += P[i, j] # will be used later on to normalier the p_i|j

            if sum_Pi == 0.0:
                sum_Pi = EPSILON # avoid dividing by 0

            P[i, :] /= sum_Pi
            sum_disti_Pi = np.multiply(sqdistances[i, :], P[i, :])

            shannon_entropy = -np.sum(P[i, :] * np.log2(P[i, :]))
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

    return P
