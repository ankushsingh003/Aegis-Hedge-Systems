from dataclasses import dataclass
from typing import Literal

@dataclass
class SimulationConfig:
    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0
    r: float = 0.05
    sigma: float = 0.20
    n_paths: int = 1000
    n_steps: int = 252
    seed: int = 42
    rebalance_freq: Literal["daily", "weekly", "threshold", "gamma_scaled"] = "daily"
    delta_threshold: float = 0.05
    option_type: Literal["call", "put"] = "call"
    model_type: Literal["bsm", "heston"] = "bsm"
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    sigma_v: float = 0.3
    rho: float = -0.7
    cost_model: Literal["proportional", "fixed", "bps"] = "proportional"
    cost_param: float = 0.001
