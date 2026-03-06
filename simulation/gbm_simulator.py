import numpy as np
from typing import Optional

def generate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    Z = np.random.standard_normal((n_paths, n_steps))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion
    log_prices = np.cumsum(log_returns, axis=1)
    initial_log_price = np.full((n_paths, 1), np.log(S0))
    log_paths = np.hstack([initial_log_price, initial_log_price + log_prices])
    price_paths = np.exp(log_paths)
    return price_paths
