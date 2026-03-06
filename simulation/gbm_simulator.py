import numpy as np
from typing import Optional

def generate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates asset price paths using Geometric Brownian Motion (GBM).
    
    Parameters:
    - S0: Initial spot price
    - mu: Expected drift (annualized) - usually the risk-free rate for delta hedging
    - sigma: Volatility (annualized)
    - T: Total time horizon in years
    - n_steps: Number of discrete time steps
    - n_paths: Number of simulated price paths
    - seed: Random seed for reproducibility
    
    Returns:
    - NumPy array of shape (n_paths, n_steps + 1) representing the price paths.
    """
    if seed is not None:
        np.random.seed(seed)
        
    dt = T / n_steps
    
    # Generate standard normal random variables (Z)
    Z = np.random.standard_normal((n_paths, n_steps))
    
    # Pre-calculate the drift and diffusion terms
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    
    # Log returns: ln(S_{t+dt} / S_t) = (mu - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z
    log_returns = drift + diffusion
    
    # Cumulative sum of log returns to get log prices
    log_prices = np.cumsum(log_returns, axis=1)
    
    # Add initial price (in log space)
    initial_log_price = np.full((n_paths, 1), np.log(S0))
    log_paths = np.hstack([initial_log_price, initial_log_price + log_prices])
    
    # Convert back to actual prices
    price_paths = np.exp(log_paths)
    
    return price_paths
