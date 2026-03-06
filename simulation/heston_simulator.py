import numpy as np
from typing import Optional, Tuple

def generate_heston_paths(
    S0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates joint price and variance paths using the Heston model.
    
    Parameters:
    - S0, v0: Initial spot and variance
    - mu: Asset drift
    - kappa, theta, sigma_v, rho: Heston parameters
    - T, n_steps, n_paths: Simulation parameters
    
    Returns:
    - A tuple (price_paths, variance_paths) each of shape (n_paths, n_steps + 1)
    """
    if seed is not None:
        np.random.seed(seed)
        
    dt = T / n_steps
    
    # Initialize paths
    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    
    # Generate correlated Brownian motions
    # [Z_S, Z_v] with correlation rho
    means = [0, 0]
    covs = [[1.0, rho], [rho, 1.0]]
    
    for t in range(1, n_steps + 1):
        # Generate shocks for all paths at once
        Z = np.random.multivariate_normal(means, covs, n_paths)
        Z_s = Z[:, 0]
        Z_v = Z[:, 1]
        
        # Use simple Euler-Maruyama with full truncation for variance
        # Non-negative variance constraint
        v_prev = np.maximum(v[:, t-1], 0)
        
        # S_{t} = S_{t-1} * exp((mu - 0.5*v_prev)*dt + sqrt(v_prev*dt)*Z_s)
        S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * Z_s)
        
        # v_{t} = v_{t-1} + kappa*(theta - v_prev)*dt + sigma_v*sqrt(v_prev*dt)*Z_v
        v[:, t] = v[:, t-1] + kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev * dt) * Z_v
        
        # Full truncation for variance
        v[:, t] = np.maximum(v[:, t], 0)
        
    return S, v
