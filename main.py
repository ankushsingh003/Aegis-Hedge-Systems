import argparse
import numpy as np
import os
import json
from config.settings import SimulationConfig
from simulation.gbm_simulator import generate_gbm_paths
from simulation.heston_simulator import generate_heston_paths
from engine.hedge_manager import run_hedging_simulation
from engine.pnl_engine import compute_pnl_statistics
from visualisation.plots import generate_all_plots
from models.bsm import bsm_price
from models.heston import heston_price
from data.provider import YFinanceProvider

def main():
    parser = argparse.ArgumentParser(description="Aegis Hedge Systems - Advanced Risk Engine")
    
    # Simulation Parameters
    parser.add_argument("--S0", type=float, default=100.0, help="Initial spot price")
    parser.add_argument("--K", type=float, default=100.0, help="Strike price")
    parser.add_argument("--T", type=float, default=1.0, help="Time to expiry (years)")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.20, help="BSM Volatility")
    parser.add_argument("--n_paths", type=int, default=100, help="Number of simulation paths")
    parser.add_argument("--n_steps", type=int, default=252, help="Steps per path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model Selection
    parser.add_argument("--model", choices=["bsm", "heston"], default="bsm", help="Pricing model")
    
    # Heston Parameters
    parser.add_argument("--v0", type=float, default=0.04, help="Initial variance")
    parser.add_argument("--kappa", type=float, default=2.0, help="Reversion speed")
    parser.add_argument("--theta", type=float, default=0.04, help="Long-run variance")
    parser.add_argument("--sigma_v", type=float, default=0.3, help="Vol of vol")
    parser.add_argument("--rho", type=float, default=-0.7, help="Correlation")
    
    # Strategy Parameters
    parser.add_argument("--rebalance", choices=["daily", "weekly", "threshold", "gamma_scaled"], default="daily", help="Rebalancing strategy")
    parser.add_argument("--delta_threshold", type=float, default=0.05, help="Delta threshold")
    parser.add_argument("--option_type", choices=["call", "put"], default="call", help="Option type")
    
    # Cost Parameters
    parser.add_argument("--cost_model", choices=["proportional", "fixed", "bps"], default="proportional", help="Transaction cost model")
    parser.add_argument("--cost_param", type=float, default=0.001, help="Cost parameter")
    
    # Data Integration
    parser.add_argument("--ticker", type=str, help="Live ticker to fetch data for (e.g. SPY, AAPL)")
    
    # Reporting
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory for saved results")
    
    args = parser.parse_args()
    
    # Live Data Handling
    S0 = args.S0
    sigma = args.sigma
    if args.ticker:
        print(f"--- Fetching live data for {args.ticker} ---")
        provider = YFinanceProvider()
        try:
            S0 = provider.get_spot_price(args.ticker)
            sigma = provider.estimate_volatility(args.ticker)
            print(f"Live Spot: {S0:.2f} | Estimated Hist Vol: {sigma:.2%}")
        except Exception as e:
            print(f"Error fetching live data: {e}. Falling back to defaults.")

    # Init config
    config = SimulationConfig(
        S0=S0, K=args.K, T=args.T, r=args.r, sigma=sigma,
        n_paths=args.n_paths, n_steps=args.n_steps, seed=args.seed,
        rebalance_freq=args.rebalance, delta_threshold=args.delta_threshold,
        option_type=args.option_type, model_type=args.model,
        v0=args.v0, kappa=args.kappa, theta=args.theta, sigma_v=args.sigma_v, rho=args.rho,
        cost_model=args.cost_model, cost_param=args.cost_param
    )
    
    print(f"--- Starting {config.model_type.upper()} Hedging Simulation ---")
    
    # 1. Generate paths
    var_paths = None
    if config.model_type == "bsm":
        paths = generate_gbm_paths(
            S0=config.S0, mu=config.r, sigma=config.sigma, T=config.T,
            n_steps=config.n_steps, n_paths=config.n_paths, seed=config.seed
        )
    else:
        paths, var_paths = generate_heston_paths(
            S0=config.S0, v0=config.v0, mu=config.r, kappa=config.kappa,
            theta=config.theta, sigma_v=config.sigma_v, rho=config.rho,
            T=config.T, n_steps=config.n_steps, n_paths=config.n_paths, seed=config.seed
        )
    
    # 2. Run hedging simulation
    results = run_hedging_simulation(paths, config, variance_paths=var_paths)
    
    # 3. Compute statistics
    if config.model_type == "bsm":
        initial_premium = bsm_price(config.S0, config.K, config.T, config.r, config.sigma, config.option_type)
    else:
        initial_premium = heston_price(config.S0, config.K, config.T, config.r, config.v0,
                                       config.kappa, config.theta, config.sigma_v, config.rho, config.option_type)
        
    stats = compute_pnl_statistics(results, initial_premium, config)
    
    # 4. Generate reports
    generate_all_plots(paths, results, stats, args.output_dir)
    
    # Save stats to JSON
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    with open(os.path.join(args.output_dir, "simulation_results.json"), "w") as f:
        serializable_stats = {k: float(v) if isinstance(v, (np.float64, np.float32)) else v for k, v in stats.items()}
        json.dump(serializable_stats, f, indent=4)
        
    print("\n--- Simulation Summary ---")
    print(f"Model:            {config.model_type.upper()}")
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
