from dataclasses import dataclass
from typing import Literal

@dataclass
class SimulationConfig:
    # Asset parameters
    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0  # Time to expiry in years
    r: float = 0.05
    sigma: float = 0.20
    
    # Simulation parameters
    n_paths: int = 1000
    n_steps: int = 252  # Trading days in a year
    seed: int = 42
    
    # Strategy parameters
    rebalance_freq: Literal["daily", "weekly", "threshold", "gamma_scaled"] = "daily"
    delta_threshold: float = 0.05
    option_type: Literal["call", "put"] = "call"
    model_type: Literal["bsm", "heston"] = "bsm"
    
    # Heston parameters (if model_type == "heston")
    v0: float = 0.04      # Initial variance (sigma^2)
    kappa: float = 2.0    # Mean reversion speed
    theta: float = 0.04   # Long-run variance
    sigma_v: float = 0.3  # Vol of vol
    rho: float = -0.7     # Correlation
    
    # Transaction cost parameters
    cost_model: Literal["proportional", "fixed", "bps"] = "proportional"
    cost_param: float = 0.001  # Cost rate (e.g., 0.1% or 10 bps)
