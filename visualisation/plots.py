import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any, List
from config.settings import SimulationConfig

def setup_plotting_style():
    """Sets a professional dark theme for plots."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'grid.alpha': 0.2
    })

def plot_gbm_paths(paths: np.ndarray, save_path: str):
    """Plots a subset of GBM paths."""
    n_paths_to_plot = min(paths.shape[0], 100)
    time_steps = np.arange(paths.shape[1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, paths[:n_paths_to_plot, :].T, color='cyan', alpha=0.1)
    
    # Plot percentiles
    p5 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    
    plt.plot(time_steps, p5, color='red', linestyle='--', label='5th Percentile')
    plt.plot(time_steps, p50, color='white', linewidth=2, label='Median')
    plt.plot(time_steps, p95, color='green', linestyle='--', label='95th Percentile')
    
    plt.title("Simulated Geometric Brownian Motion Paths")
    plt.xlabel("Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_pnl_distribution(pnl: np.ndarray, save_path: str):
    """Plots the distribution of final PnL."""
    plt.figure(figsize=(10, 6))
    plt.hist(pnl, bins=50, color='royalblue', edgecolor='white', alpha=0.7, density=True)
    
    # Overlay normal distribution
    mu, std = np.mean(pnl), np.std(pnl)
    x = np.linspace(np.min(pnl), np.max(pnl), 100)
    p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std)**2)
    plt.plot(x, p, 'r', linewidth=2, label=f'Normal Fit (μ={mu:.2f}, σ={std:.2f})')
    
    plt.axvline(mu, color='white', linestyle='-', label='Mean PnL')
    plt.axvline(0, color='yellow', linestyle='--', label='Zero PnL')
    
    plt.title("Hedge PnL Distribution")
    plt.xlabel("Final PnL")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_cost_sensitivity(results_list: List[Dict[str, Any]], param_values: List[float], save_path: str):
    """Plots how PnL varies with transaction costs."""
    means = [r['mean_pnl'] for r in results_list]
    stds = [r['std_pnl'] for r in results_list]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(param_values, means, yerr=stds, fmt='-o', color='gold', ecolor='gray', capsize=5)
    plt.axhline(0, color='white', linestyle='--')
    
    plt.title("PnL Sensitivity to Transaction Costs")
    plt.xlabel("Cost Parameter (bps or rate)")
    plt.ylabel("Mean PnL ± 1 Std Dev")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def generate_all_plots(
    paths: np.ndarray,
    simulation_results: Dict[str, np.ndarray],
    stats: Dict[str, Any],
    report_dir: str
):
    """Master function to generate all required plots."""
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        
    setup_plotting_style()
    
    plot_gbm_paths(paths, os.path.join(report_dir, "fig1_gbm_paths.png"))
    plot_pnl_distribution(simulation_results["final_pnl"], os.path.join(report_dir, "fig3_pnl_distribution.png"))
    
    # Additional plots can be added here
    print(f"All plots saved to {report_dir}")
