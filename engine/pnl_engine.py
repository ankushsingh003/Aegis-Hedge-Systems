import numpy as np

def compute_pnl_statistics(simulation_results, initial_premium, config_any):
    pnl = simulation_results["final_pnl"]
    costs = simulation_results["total_costs"]
    trades = simulation_results["trade_count"]
    mean_pnl = np.mean(pnl)
    std_pnl = np.std(pnl)
    sharpe = mean_pnl / (std_pnl + 1e-8)
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
