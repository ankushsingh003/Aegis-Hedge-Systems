import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any, List, Optional

def setup_plotting_style():
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

def plot_gbm_paths(paths, save_path=None):
    n_paths_to_plot = min(paths.shape[0], 100)
    time_steps = np.arange(paths.shape[1])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_steps, paths[:n_paths_to_plot, :].T, color='cyan', alpha=0.1)
    p5 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    ax.plot(time_steps, p5, color='red', linestyle='--', label='5th Percentile')
    ax.plot(time_steps, p50, color='white', linewidth=2, label='Median')
    ax.plot(time_steps, p95, color='green', linestyle='--', label='95th Percentile')
    ax.set_title("Simulated Price Paths")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    return fig

def plot_pnl_distribution(pnl, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pnl, bins=50, color='royalblue', edgecolor='white', alpha=0.7, density=True)
    mu, std = np.mean(pnl), np.std(pnl)
    x = np.linspace(np.min(pnl), np.max(pnl), 100)
    p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std)**2)
    ax.plot(x, p, 'r', linewidth=2, label=f'Normal Fit (μ={mu:.2f}, σ={std:.2f})')
    ax.axvline(mu, color='white', linestyle='-', label='Mean PnL')
    ax.axvline(0, color='yellow', linestyle='--', label='Zero PnL')
    ax.set_title("Hedge PnL Distribution")
    ax.set_xlabel("Final PnL")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    return fig

def plot_cost_sensitivity(results_list, param_values, save_path):
    means = [r['mean_pnl'] for r in results_list]
    stds = [r['std_pnl'] for r in results_list]
    plt.figure(figsize=(10, 6))
    plt.errorbar(param_values, means, yerr=stds, fmt='-o', color='gold', ecolor='gray', capsize=5)
    plt.axhline(0, color='white', linestyle='--')
    plt.title("PnL Sensitivity to Transaction Costs")
    plt.xlabel("Cost Parameter")
    plt.ylabel("Mean PnL")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def generate_all_plots(paths, simulation_results, stats, report_dir):
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    setup_plotting_style()
    plot_gbm_paths(paths, os.path.join(report_dir, "fig1_gbm_paths.png"))
    plot_pnl_distribution(simulation_results["final_pnl"], os.path.join(report_dir, "fig3_pnl_distribution.png"))
