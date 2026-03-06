import numpy as np
from scipy.stats import norm
from typing import Union, Literal

def bsm_price(
    S: Union[float, np.ndarray],
    K: float,
    T: Union[float, np.ndarray],
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call"
) -> Union[float, np.ndarray]:
    """
    Computes the Black-Scholes-Merton price for a European option.
    
    Parameters:
    - S: Current asset price (scalar or array)
    - K: Strike price
    - T: Time to expiry in years (scalar or array)
    - r: Risk-free rate
    - sigma: Volatility (annualized)
    - option_type: "call" or "put"
    
    Returns:
    - Option price
    """
    # Handle edge case for T <= 0
    # Use a small epsilon to avoid division by zero
    T = np.maximum(T, 1e-10)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
        
    # If T was originally <= 0, the option value is the intrinsic value
    # We can refine this by checking the original T values if needed,
    # but for simulation purposes, T is usually > 0 until the final step.
    return price
