import numpy as np
from typing import Tuple

def generate_heston_paths(S0, v0, mu, kappa, theta, sigma_v, rho, T, n_steps, n_paths, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    means = [0, 0]
    covs = [[1.0, rho], [rho, 1.0]]
    for t in range(1, n_steps + 1):
        Z = np.random.multivariate_normal(means, covs, n_paths)
        Z_s = Z[:, 0]
        Z_v = Z[:, 1]
        v_prev = np.maximum(v[:, t-1], 0)
        S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * Z_s)
        v[:, t] = v[:, t-1] + kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev * dt) * Z_v
        v[:, t] = np.maximum(v[:, t], 0)
    return S, v
