import numpy as np

def calculate_transaction_costs(trade_size, spot_price, cost_model, cost_param):
    abs_trade_size = np.abs(trade_size)
    notional_value = abs_trade_size * spot_price
    if cost_model == "proportional":
        return cost_param * notional_value
    elif cost_model == "fixed":
        costs = np.zeros_like(trade_size)
        costs[abs_trade_size > 1e-8] = cost_param
        return costs
    elif cost_model == "bps":
        return (cost_param / 10000.0) * notional_value
    else:
        raise ValueError(f"Unknown cost model: {cost_model}")
