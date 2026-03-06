import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config.settings import SimulationConfig
from simulation.gbm_simulator import generate_gbm_paths
from simulation.heston_simulator import generate_heston_paths
from engine.hedge_manager import run_hedging_simulation
from engine.pnl_engine import compute_pnl_statistics
from visualisation.plots import plot_gbm_paths, plot_pnl_distribution, setup_plotting_style
from models.bsm import bsm_price
from models.heston import heston_price
from data.provider import YFinanceProvider

# Page Config
st.set_page_config(page_title="Aegis Hedge Systems | Real-Time Dashboard", layout="wide")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4150; }
    .stAlert { background-color: #1e2130; border: 1px solid #3e4150; }
    div[data-testid="stExpander"] { background-color: #1e2130; border: 1px solid #3e4150; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Aegis Hedge Systems")
st.subheader("Real-Time Quantitative Risk Engine")

# Simulation logic (Cached correctly now)
@st.cache_data
def get_sim_results(config_dict: dict):
    # Re-build config from dict to ensure it's hashable and reactive
    config = SimulationConfig(**config_dict)
    
    if config.model_type == "bsm":
        paths = generate_gbm_paths(S0=config.S0, mu=config.r, sigma=config.sigma, T=config.T, n_steps=config.n_steps, n_paths=config.n_paths, seed=config.seed)
        var_paths = None
        initial_premium = bsm_price(config.S0, config.K, config.T, config.r, config.sigma, config.option_type)
    else:
        paths, var_paths = generate_heston_paths(S0=config.S0, v0=config.v0, mu=config.r, kappa=config.kappa, theta=config.theta, sigma_v=config.sigma_v, rho=config.rho, T=config.T, n_steps=config.n_steps, n_paths=config.n_paths, seed=config.seed)
        initial_premium = heston_price(config.S0, config.K, config.T, config.r, config.v0, config.kappa, config.theta, config.sigma_v, config.rho, config.option_type)

    results = run_hedging_simulation(paths, config, variance_paths=var_paths)
    stats = compute_pnl_statistics(results, initial_premium, config)
    return paths, results, stats, initial_premium

@st.cache_data
def fetch_ticker_data(ticker_symbol: str):
    provider = YFinanceProvider()
    spot = provider.get_spot_price(ticker_symbol)
    vol = provider.estimate_volatility(ticker_symbol)
    return spot, vol

# Sidebar - Settings
st.sidebar.header("🕹️ Simulation Control")

ticker_input = st.sidebar.text_input("Live Ticker (e.g. SPY, AAPL)", value="", help="Fetches live spot and hist volatility")

model_type = st.sidebar.selectbox("Pricing Model", ["bsm", "heston"])
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

# Default values
S0_def = 100.0
sigma_def = 0.20

# Update values if ticker is provided
if ticker_input:
    try:
        S0_ticker, sigma_ticker = fetch_ticker_data(ticker_input)
        st.sidebar.success(f"Live {ticker_input}: ${S0_ticker:.2f} | Vol: {sigma_ticker:.2%}")
        S0_def = S0_ticker
        sigma_def = sigma_ticker
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Collapsible sections for parameters
with st.sidebar.expander("📈 Asset Parameters", expanded=True):
    S0_val = st.number_input("Spot Price ($)", value=float(S0_def), format="%.2f")
    K_val = st.number_input("Strike Price ($)", value=float(S0_def), format="%.2f")
    T_val = st.slider("Time to Expiry (Years)", 0.05, 2.0, 1.0)
    r_val = st.slider("Risk-free Rate", 0.0, 0.1, 0.05, step=0.01)
    sigma_val = st.slider("Volatility (σ)", 0.05, 1.0, float(sigma_def), step=0.01)

if model_type == "heston":
    with st.sidebar.expander("🌀 Heston Parameters", expanded=True):
        v0_val = st.number_input("Initial Var (v0)", value=0.04, format="%.4f")
        kappa_val = st.slider("Mean Reversion (κ)", 0.1, 5.0, 2.0)
        theta_val = st.slider("Long-run Var (θ)", 0.01, 0.2, 0.04)
        sigma_v_val = st.slider("Vol of Vol (ξ)", 0.1, 1.0, 0.3)
        rho_val = st.slider("Correlation (ρ)", -1.0, 1.0, -0.7)
else:
    v0_val, kappa_val, theta_val, sigma_v_val, rho_val = 0.04, 2.0, 0.04, 0.3, -0.7

with st.sidebar.expander("🛠️ Strategy & Simulation", expanded=True):
    rebalance_val = st.selectbox("Frequency", ["daily", "weekly", "threshold", "gamma_scaled"])
    cost_model_val = st.selectbox("Cost Model", ["proportional", "fixed", "bps"])
    cost_param_val = st.number_input("Cost Value", value=10.0 if cost_model_val == "bps" else 0.001, format="%.6f")
    n_paths_val = st.number_input("Num Paths", value=100 if model_type == "bsm" else 20, step=10)
    seed_val = st.number_input("Random Seed (Change to shuffle)", value=42)

# Build a dictionary for caching (Streamlit hashes dicts better than objects)
config_dict = {
    "S0": S0_val, "K": K_val, "T": T_val, "r": r_val, "sigma": sigma_val,
    "n_paths": int(n_paths_val), "n_steps": 252,
    "rebalance_freq": rebalance_val, "model_type": model_type,
    "option_type": option_type, "v0": v0_val, "kappa": kappa_val,
    "theta": theta_val, "sigma_v": sigma_v_val, "rho": rho_val,
    "cost_model": cost_model_val, "cost_param": cost_param_val,
    "seed": int(seed_val)
}

# Clear Cache if needed
if st.sidebar.button("🧹 Clear All Cache"):
    st.cache_data.clear()
    st.rerun()

# Run Simulation in REAL TIME (No button required for initial load, but button remains if user wants to force)
# However, the user wants "totally dynamic", so we just call it every render.
# because of st.cache_data, it will be fast if parameters haven't changed.
with st.spinner("Processing..."):
    paths, results, stats, initial_premium = get_sim_results(config_dict)

# 1. Metric Callouts
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean PnL", f"${stats['mean_pnl']:.4f}")
col2.metric("Sharpe-like Ratio", f"{stats['sharpe_ratio']:.2f}")
col3.metric("Avg Trans. Cost", f"${stats['avg_costs']:.4f}")
col4.metric("VaR (95%)", f"${stats['var_95']:.4f}")

# Interpretation Cards
if stats['mean_pnl'] == 0 and initial_premium < 0.01:
    st.warning("⚖️ **Quantitative Alert: Deep OTM Option**  \nYour choice of Strike ($" + str(K_val) + ") is far from Spot ($" + str(S0_val) + "). The Delta is effectively zero.")

# 2. Visuals
setup_plotting_style()
tab1, tab2, tab3 = st.tabs(["📊 Price Paths", "📈 PnL Analysis", "🧠 Quant Interpretation"])

with tab1:
    st.pyplot(plot_gbm_paths(paths))
    st.info("**Real-Time Reactivity:** Change any slider on the left to see the price envelope update instantly.")

with tab2:
    st.pyplot(plot_pnl_distribution(results["final_pnl"]))
    st.warning("**Hedging Slip:** PnL variance represents your residual risk after hedging.")

with tab3:
    st.markdown(f"""
    ### Executive Interpretation
    
    *   **Option Premium**: **${initial_premium:.4f}**
    *   **Hedging Cost**: **${stats['avg_costs']:.4f}** (Annualized slippage)
    *   **Activity**: Total rebalancing trades: **{stats['avg_trades']:.1f}** 
    
    #### Market View:
    {"Volatility is high; prioritize delta-band (threshold) hedging." if sigma_val > 0.4 else "Volatility is stable; daily rebalancing is efficient."}
    """)
