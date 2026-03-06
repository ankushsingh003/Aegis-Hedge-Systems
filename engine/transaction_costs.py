import numpy as np
from typing import Literal, Union

def calculate_transaction_costs(
    trade_size: Union[float, np.ndarray],
    spot_price: Union[float, np.ndarray],
    cost_model: Literal["proportional", "fixed", "bps"],
    cost_param: float
) -> Union[float, np.ndarray]:
    """
    Computes transaction costs for a given trade.
    
    Parameters:
    - trade_size: Number of shares traded (Δ trade)
    - spot_price: Current asset price
    - cost_model: Type of cost model
    - cost_param: Parameter for the cost model
    
    Returns:
    - Total transaction cost for the trade.
    """
    abs_trade_size = np.abs(trade_size)
    notional_value = abs_trade_size * spot_price
    
    if cost_model == "proportional":
        # cost_param is the rate (e.g., 0.001 for 0.1%)
        return cost_param * notional_value
    elif cost_model == "fixed":
        # cost_param is the fixed amount per trade
        # Only apply cost if trade_size != 0
        costs = np.zeros_like(trade_size)
        costs[abs_trade_size > 1e-8] = cost_param
        return costs
    elif cost_model == "bps":
        # cost_param is basis points (e.g., 10 for 0.1%)
        return (cost_param / 10000.0) * notional_value
    else:
        raise ValueError(f"Unknown cost model: {cost_model}")
