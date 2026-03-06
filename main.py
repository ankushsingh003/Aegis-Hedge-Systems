import argparse
import numpy as np
import os
import json
from config.settings import SimulationConfig
from simulation.gbm_simulator import generate_gbm_paths
from engine.hedge_manager import run_hedging_simulation
from engine.pnl_engine import compute_pnl_statistics
from visualisation.plots import generate_all_plots
from models.bsm import bsm_price

def main():
    parser = argparse.ArgumentParser(description="Dynamic Delta-Hedging Simulator")
    
    # Simulation Parameters
    parser.add_argument("--S0", type=float, default=100.0, help="Initial spot price")
    parser.add_argument("--K", type=float, default=100.0, help="Strike price")
    parser.add_argument("--T", type=float, default=1.0, help="Time to expiry (years)")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.20, help="Volatility")
    parser.add_argument("--n_paths", type=int, default=1000, help="Number of simulation paths")
    parser.add_argument("--n_steps", type=int, default=252, help="Steps per path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Strategy Parameters
    parser.add_argument("--rebalance", choices=["daily", "weekly", "threshold", "gamma_scaled"], default="daily", help="Rebalancing strategy")
    parser.add_argument("--delta_threshold", type=float, default=0.05, help="Delta threshold for rebalancing")
    parser.add_argument("--option_type", choices=["call", "put"], default="call", help="Option type")
    
    # Cost Parameters
    parser.add_argument("--cost_model", choices=["proportional", "fixed", "bps"], default="proportional", help="Transaction cost model")
    parser.add_argument("--cost_param", type=float, default=0.001, help="Cost parameter")
    
    # Reporting
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory for saved results")
    
    args = parser.parse_args()
    
    # Init config
    config = SimulationConfig(
        S0=args.S0, K=args.K, T=args.T, r=args.r, sigma=args.sigma,
        n_paths=args.n_paths, n_steps=args.n_steps, seed=args.seed,
        rebalance_freq=args.rebalance, delta_threshold=args.delta_threshold,
        option_type=args.option_type,
        cost_model=args.cost_model, cost_param=args.cost_param
    )
    
    print("--- Starting Dynamic Delta-Hedging Simulation ---")
    print(f"Config: {args.option_type} option | {args.rebalance} rebalancing | {args.cost_model} cost model")
    
    # 1. Generate paths
    paths = generate_gbm_paths(
        S0=config.S0, mu=config.r, sigma=config.sigma, T=config.T,
        n_steps=config.n_steps, n_paths=config.n_paths, seed=config.seed
    )
    
    # 2. Run hedging simulation
    results = run_hedging_simulation(paths, config)
    
    # 3. Compute statistics
    initial_premium = bsm_price(config.S0, config.K, config.T, config.r, config.sigma, config.option_type)
    stats = compute_pnl_statistics(results, initial_premium, config)
    
    # 4. Generate reports
    generate_all_plots(paths, results, stats, args.output_dir)
    
    # Save stats to JSON
    with open(os.path.join(args.output_dir, "simulation_results.json"), "w") as f:
        # Convert NumPy types to standard Python types for JSON serialization
        serializable_stats = {k: float(v) if isinstance(v, (np.float64, np.float32)) else v for k, v in stats.items()}
        json.dump(serializable_stats, f, indent=4)
        
    print("\n--- Simulation Summary ---")
    print(f"Initial Premium:  {initial_premium:.4f}")
    print(f"Mean PnL:         {stats['mean_pnl']:.4f}")
    print(f"Std Dev PnL:      {stats['std_pnl']:.4f}")
    print(f"Sharpe-like Ratio: {stats['sharpe_ratio']:.4f}")
    print(f"VaR 95%:          {stats['var_95']:.4f}")
    print(f"Avg Costs:        {stats['avg_costs']:.4f}")
    print(f"Avg Trades:       {stats['avg_trades']:.1f}")
    print(f"Reports saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
