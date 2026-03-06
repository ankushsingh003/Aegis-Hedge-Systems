import numpy as np
from typing import Dict, Any

def compute_pnl_statistics(
    simulation_results: Dict[str, np.ndarray],
    initial_premium: np.ndarray,
    config_any: Any
) -> Dict[str, Any]:
    """
    Computes summary statistics for the hedging simulation.
    
    Parameters:
    - simulation_results: Output from run_hedging_simulation
    - initial_premium: Initial option price at t=0
    - config_any: SimulationConfig object
    
    Returns:
    - Dictionary with mean PnL, standard deviation, Sharpe ratio, etc.
    """
    pnl = simulation_results["final_pnl"]
    costs = simulation_results["total_costs"]
    trades = simulation_results["trade_count"]
    
    # annualized stats
    mean_pnl = np.mean(pnl)
    std_pnl = np.std(pnl)
    
    # Sharpe Ratio logic: (Return - Risk-free) / Std
    # Here, 'Return' is the final PnL relative to the premium received or initial capital
    # For a hedge fund, the volatility of the PnL is the primary risk.
    # A common metric is Mean PnL / Std PnL (Information Ratio-like)
    sharpe = mean_pnl / (std_pnl + 1e-8)
    
    # Potential loss (Value at Risk 95%)
    var_95 = np.percentile(pnl, 5)
    
    return {
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "sharpe_ratio": sharpe,
        "var_95": var_95,
        "avg_costs": np.mean(costs),
        "avg_trades": np.mean(trades),
        "max_loss": np.min(pnl),
        "max_gain": np.max(pnl)
    }
