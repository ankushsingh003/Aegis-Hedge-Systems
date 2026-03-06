import numpy as np
from typing import Literal, Dict, Any, List, Optional
from dataclasses import dataclass
from models.bsm import bsm_price
from models.greeks import calculate_greeks
from models.heston import heston_price
from engine.transaction_costs import calculate_transaction_costs
from config.settings import SimulationConfig

@dataclass
class PortfolioState:
    cash: np.ndarray        # Cash balance for each path
    shares: np.ndarray      # Number of shares held for each path (Delta)
    option_value: np.ndarray # Current option price for each path

def run_hedging_simulation(
    paths: np.ndarray,
    config: SimulationConfig,
    variance_paths: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Simulates a dynamic delta-hedging strategy across multiple price paths.
    Supports both BSM (constant vol) and Heston (stochastic vol) models.
    
    Parameters:
    - paths: Asset price paths (n_paths, n_steps + 1)
    - config: SimulationConfig object
    - variance_paths: Optional variance paths for Heston model
    """
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1
    dt = config.T / n_steps
    
    def get_delta_and_price(S_val, v_val, T_rem):
        if config.model_type == "bsm":
            price = bsm_price(S_val, config.K, T_rem, config.r, config.sigma, config.option_type)
            greeks = calculate_greeks(S_val, config.K, T_rem, config.r, config.sigma, config.option_type)
            return greeks.delta, price
        else:
            # Heston model
            # For Heston, we need the variance at this step
            # heston_price doesn't support vectorization easily due to quad integration
            # We compute scalar values in a loop for the first step, then for subsequent
            # However, for performance, we'll implement a vectorized wrapper or loop handle
            deltas = np.zeros(n_paths)
            prices = np.zeros(n_paths)
            eps = S_val * 0.01 # 1% shift for numerical delta
            
            # Note: We loop over paths for Heston since quad is not vectorized
            for i in range(n_paths):
                p_main = heston_price(S_val[i], config.K, T_rem, config.r, v_val[i], 
                                      config.kappa, config.theta, config.sigma_v, config.rho, config.option_type)
                p_plus = heston_price(S_val[i] + eps[i], config.K, T_rem, config.r, v_val[i], 
                                      config.kappa, config.theta, config.sigma_v, config.rho, config.option_type)
                prices[i] = p_main
                deltas[i] = (p_plus - p_main) / eps[i]
            return deltas, prices

    # Initialize portfolio
    S_0 = paths[:, 0]
    v_0 = variance_paths[:, 0] if variance_paths is not None else np.full(n_paths, config.sigma**2)
    
    initial_delta, initial_option_price = get_delta_and_price(S_0, v_0, config.T)
    
    # Initial cash: premium received - cost of buying Delta shares - transaction costs
    initial_shares = initial_delta
    initial_trade_costs = calculate_transaction_costs(initial_shares, S_0, config.cost_model, config.cost_param)
    initial_cash = initial_option_price - (initial_shares * S_0) - initial_trade_costs
    
    # State tracking arrays
    cash = np.copy(initial_cash)
    shares = np.copy(initial_shares)
    
    # To store PnL evolution (optional, for plotting)
    pnl_history = np.zeros((n_paths, n_steps + 1))
    cost_history = np.zeros((n_paths, n_steps + 1))
    trade_count = np.zeros(n_paths)
    
    last_rebalance_delta = np.copy(initial_shares)
    
    # Loop through time steps
    for step in range(1, n_steps + 1):
        t_current = step * dt
        T_remaining = config.T - t_current
        S_t = paths[:, step]
        
        # Grow cash at risk-free rate
        cash *= np.exp(config.r * dt)
        
        # Calculate current Greeks/Delta
        v_t = variance_paths[:, step] if variance_paths is not None else np.full(n_paths, config.sigma**2)
        target_delta, current_price = get_delta_and_price(S_t, v_t, T_remaining)
        
        # Check rebalance condition
        should_rebalance = np.zeros(n_paths, dtype=bool)
        
        if config.rebalance_freq == "daily":
            should_rebalance[:] = True
        elif config.rebalance_freq == "weekly":
            if step % 5 == 0:
                should_rebalance[:] = True
        elif config.rebalance_freq == "threshold":
            should_rebalance = np.abs(target_delta - last_rebalance_delta) > config.delta_threshold
        elif config.rebalance_freq == "gamma_scaled":
            if config.model_type == "bsm":
                greeks = calculate_greeks(S_t, config.K, T_remaining, config.r, config.sigma, config.option_type)
                threshold = 0.01 / (greeks.gamma + 1e-5)
            else:
                threshold = 0.01 # Fallback for Heston
            should_rebalance = np.abs(target_delta - last_rebalance_delta) > threshold
            
        # Perform rebalancing
        delta_trade = np.where(should_rebalance, target_delta - shares, 0.0)
        trade_costs = calculate_transaction_costs(delta_trade, S_t, config.cost_model, config.cost_param)
        
        cash -= (delta_trade * S_t) + trade_costs
        shares += delta_trade
        
        # Update rebalance trackers
        trade_count += should_rebalance.astype(int)
        last_rebalance_delta = np.where(should_rebalance, shares, last_rebalance_delta)
        
        # Track costs
        cost_history[:, step] = trade_costs
        
    # Final Portfolio Value at T
    S_T = paths[:, -1]
    final_option_value = np.zeros(n_paths)
    if config.option_type == "call":
        final_option_value = np.maximum(S_T - config.K, 0.0)
    else:
        final_option_value = np.maximum(config.K - S_T, 0.0)
        
    # PnL = Cash + Shares*Spot - Option_Liability
    # Note: Option_Liability is the value of the option we sold at t=0
    final_pnl = cash + (shares * S_T) - final_option_value
    
    return {
        "final_pnl": final_pnl,
        "total_costs": np.sum(cost_history, axis=1) + initial_trade_costs,
        "trade_count": trade_count + 1, # +1 for initial trade
        "final_shares": shares,
        "final_cash": cash
    }
