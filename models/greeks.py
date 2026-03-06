import numpy as np
from scipy.stats import norm
from typing import Union, Literal, Dict
from dataclasses import dataclass

@dataclass
class Greeks:
    delta: Union[float, np.ndarray]
    gamma: Union[float, np.ndarray]
    vega: Union[float, np.ndarray]
    theta: Union[float, np.ndarray]
    rho: Union[float, np.ndarray]
    vanna: Union[float, np.ndarray]
    charm: Union[float, np.ndarray]

def calculate_greeks(
    S: Union[float, np.ndarray],
    K: float,
    T: Union[float, np.ndarray],
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call"
) -> Greeks:
    """
    Computes option Greeks using the Black-Scholes-Merton model.
    
    Parameters:
    - S: Current asset price
    - K: Strike price
    - T: Time to expiry in years
    - r: Risk-free rate
    - sigma: Volatility (annualized)
    - option_type: "call" or "put"
    
    Returns:
    - Greeks object containing all first-order and selected second-order Greeks.
    """
    T = np.maximum(T, 1e-10)
    sqrt_T = np.sqrt(T)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    
    # Delta
    if option_type == "call":
        delta = N_d1
    else:
        delta = N_d1 - 1.0
        
    # Gamma (same for Call and Put)
    gamma = n_d1 / (S * sigma * sqrt_T)
    
    # Vega (same for Call and Put)
    # Note: Traditionally Vega is ∂V/∂σ, often quoted per 1% move
    vega = S * n_d1 * sqrt_T
    
    # Theta
    if option_type == "call":
        theta = - (S * n_d1 * sigma) / (2 * sqrt_T) - r * K * np.exp(-r * T) * N_d2
    else:
        theta = - (S * n_d1 * sigma) / (2 * sqrt_T) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        
    # Rho
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * N_d2
    else:
        rho = - K * T * np.exp(-r * T) * norm.cdf(-d2)
        
    # Vanna (dVega/dS or dDelta/dVol)
    vanna = - n_d1 * d2 / sigma
    
    # Charm (dDelta/dT)
    if option_type == "call":
        charm = - n_d1 * (r / (sigma * sqrt_T) - d2 / (2 * T))
    else:
        charm = n_d1 * (r / (sigma * sqrt_T) - d2 / (2 * T))
        
    return Greeks(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        vanna=vanna,
        charm=charm
    )
