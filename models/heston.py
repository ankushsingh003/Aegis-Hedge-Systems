import numpy as np
from scipy.integrate import quad
from typing import Union, Literal, Optional

def heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    option_type: Literal["call", "put"] = "call"
) -> float:
    """
    Computes the Heston price for a European option using the Lewis/Lipton formula.
    
    Parameters:
    - S: Initial spot price
    - K: Strike price
    - T: Time to expiry
    - r: Risk-free rate
    - v0: Initial variance
    - kappa: Mean reversion speed of variance
    - theta: Long-run variance
    - sigma_v: Vol of vol (variance vol)
    - rho: Correlation between price and variance shocks
    - option_type: "call" or "put"
    
    Returns:
    - Option price
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    def characteristic_function(u, t, S, r, v0, kappa, theta, sigma_v, rho):
        # Lewis formulation for the characteristic function
        # u is the transform variable
        xi = kappa - rho * sigma_v * 1j * u
        d = np.sqrt(xi**2 + sigma_v**2 * (u**2 + 1j * u))
        g = (xi - d) / (xi + d)
        
        C = (kappa * theta / sigma_v**2) * (
            (xi - d) * t - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g))
        )
        D = (xi - d) / sigma_v**2 * (
            (1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t))
        )
        
        phi = np.exp(C + D * v0 + 1j * u * np.log(S * np.exp(r * t)))
        return phi

    def integrand(u, t, S, K, r, v0, kappa, theta, sigma_v, rho):
        phi = characteristic_function(u - 0.5j, t, S, r, v0, kappa, theta, sigma_v, rho)
        return (np.exp(-1j * u * np.log(K)) * phi / (u**2 + 0.25)).real

    # Integrate from 0 to large enough number (e.g., 100)
    integral, _ = quad(integrand, 0, 100, args=(T, S, K, r, v0, kappa, theta, sigma_v, rho))
    
    # Lipton formula: Price = S - (K * exp(-rT) / pi) * Integral
    price = S - (np.sqrt(S * K) * np.exp(-r * T / 2) / np.pi) * integral
    
    # If put, use put-call parity
    if option_type == "put":
        price = price - S + K * np.exp(-r * T)
        
    return max(price, 0.0)
