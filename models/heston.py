import numpy as np
from scipy.integrate import quad
from typing import Literal

def heston_price(S, K, T, r, v0, kappa, theta, sigma_v, rho, option_type="call"):
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    def characteristic_function(u, t, S, r, v0, kappa, theta, sigma_v, rho):
        xi = kappa - rho * sigma_v * 1j * u
        d = np.sqrt(xi**2 + sigma_v**2 * (u**2 + 1j * u))
        g = (xi - d) / (xi + d)
        C = (kappa * theta / sigma_v**2) * ((xi - d) * t - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g)))
        D = (xi - d) / sigma_v**2 * ((1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t)))
        phi = np.exp(C + D * v0 + 1j * u * np.log(S * np.exp(r * t)))
        return phi

    def integrand(u, t, S, K, r, v0, kappa, theta, sigma_v, rho):
        phi = characteristic_function(u - 0.5j, t, S, r, v0, kappa, theta, sigma_v, rho)
        return (np.exp(-1j * u * np.log(K)) * phi / (u**2 + 0.25)).real

    integral, _ = quad(integrand, 0, 100, args=(T, S, K, r, v0, kappa, theta, sigma_v, rho))
    price = S - (np.sqrt(S * K) * np.exp(-r * T / 2) / np.pi) * integral
    if option_type == "put":
        price = price - S + K * np.exp(-r * T)
    return max(price, 0.0)
