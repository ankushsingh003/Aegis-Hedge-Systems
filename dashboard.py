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
st.set_page_config(page_title="Aegis Hedge Systems | Command Center", layout="wide")

# Premium UI CSS (Glassmorphism & Fixed Header vibes)
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Input Container Styling */
    .control-center {
        background: rgba(23, 28, 36, 0.7);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
    }
    
    /* Metrics Box */
    [data-testid="stMetric"] {
        background: rgba(30, 39, 53, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Metric Labels */
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-weight: 600 !important;
    }
    
    /* Metric Values */
    [data-testid="stMetricValue"] {
        color: #58a6ff !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #8b949e;
    }

    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
    }
    
    /* Sidebar Hide (Optional but cleaner) */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f0f6fc;
        font-weight: 700;
    }
    
    .stHeader {
        background-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Logic Section ---

@st.cache_data
def get_sim_results(config_dict: dict):
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

# --- Application UI ---

# 1. Header & Branding
st.markdown("<h1 style='text-align: center; margin-bottom: 0px;'>🛡️ AEGIS HEDGE SYSTEMS</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; margin-bottom: 30px;'>Next-Gen Derivatives Risk Command Center</p>", unsafe_allow_html=True)

# 2. Top-Level Control Center (Glass Container)
with st.container():
    st.markdown('<div class="control-center">', unsafe_allow_html=True)
    
    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns([1.5, 1, 1, 1])
    
    with row1_col1:
        ticker_input = st.text_input("Live Ticker 🔍", value="", placeholder="e.g. SPY, BTC-USD")
    with row1_col2:
        model_type = st.selectbox("Pricing Engine", ["bsm", "heston"])
    with row1_col3:
        option_type = st.selectbox("Option Class", ["call", "put"])
    with row1_col4:
        seed_val = st.number_input("Chaos Seed 🎲", value=42, step=1)

    # Defaults for reactive inputs
    S0_def, sigma_def = 100.0, 0.20
    if ticker_input:
        try:
            S0_ticker, sigma_ticker = fetch_ticker_data(ticker_input)
            S0_def, sigma_def = S0_ticker, sigma_ticker
            st.toast(f"Synchronized with {ticker_input} market data.", icon="✅")
        except:
            st.error("Invalid Ticker.")

    st.markdown("<hr style='border: 0.1px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    
    row2_col1, row2_col2, row2_col3, row2_col4, row2_col5 = st.columns(5)
    with row2_col1:
        S0_val = st.number_input("Spot ($)", value=float(S0_def), step=1.0)
    with row2_col2:
        K_val = st.number_input("Strike ($)", value=S0_val if ticker_input else 100.0, step=1.0)
    with row2_col3:
        sigma_val = st.slider("Vol (σ)", 0.05, 1.0, float(sigma_def), step=0.01)
    with row2_col4:
        T_val = st.slider("Expiry (Yrs)", 0.05, 2.0, 1.0, step=0.05)
    with row2_col5:
        r_val = st.slider("Rate (%)", 0.00, 0.15, 0.05, step=0.01)

    if model_type == "heston":
        st.markdown("<p style='color: #8b949e; font-size: 0.8rem; margin-top: 10px;'>Advanced Stochastic Parameters</p>", unsafe_allow_html=True)
        h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns(5)
        with h_col1: v0_val = st.number_input("v0", value=0.04, format="%.4f")
        with h_col2: kappa_val = st.slider("Reversion (κ)", 0.1, 5.0, 2.0)
        with h_col3: theta_val = st.slider("Long-run (θ)", 0.01, 0.1, 0.04)
        with h_col4: sigma_v_val = st.slider("Vol-of-Vol (ξ)", 0.1, 1.0, 0.3)
        with h_col5: rho_val = st.slider("Correl (ρ)", -1.0, 1.0, -0.7)
    else:
        v0_val, kappa_val, theta_val, sigma_v_val, rho_val = 0.04, 2.0, 0.04, 0.3, -0.7

    st.markdown("<hr style='border: 0.1px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    
    row3_col1, row3_col2, row3_col3, row3_col4 = st.columns(4)
    with row3_col1:
        rebalance_val = st.selectbox("Hedge Strategy", ["daily", "weekly", "threshold", "gamma_scaled"])
    with row3_col2:
        cost_model_val = st.selectbox("Fees", ["proportional", "bps", "fixed"])
    with row3_col3:
        cost_param_val = st.number_input("Cost Rate", value=0.001, format="%.4f")
    with row3_col4:
        n_paths_val = st.slider("Monte Carlo Paths", 10, 500, 100 if model_type == "bsm" else 30)

    st.markdown('</div>', unsafe_allow_html=True)

# 3. Execution & Results
config_dict = {
    "S0": S0_val, "K": K_val, "T": T_val, "r": r_val, "sigma": sigma_val,
    "n_paths": int(n_paths_val), "n_steps": 252,
    "rebalance_freq": rebalance_val, "model_type": model_type,
    "option_type": option_type, "v0": v0_val, "kappa": kappa_val,
    "theta": theta_val, "sigma_v": sigma_v_val, "rho": rho_val,
    "cost_model": cost_model_val, "cost_param": cost_param_val,
    "seed": int(seed_val)
}

# Real-time computation
with st.spinner("Processing Risk Engine..."):
    paths, results, stats, initial_premium = get_sim_results(config_dict)

# Metrics Grid
m1, m2, m3, m4 = st.columns(4)
m1.metric("PnL EXPECTATION", f"${stats['mean_pnl']:.4f}")
m2.metric("SHARPE (IRIS)", f"{stats['sharpe_ratio']:.2f}")
m3.metric("SLIPPAGE DRAG", f"${stats['avg_costs']:.4f}")
m4.metric("VaR (95%)", f"${stats['var_95']:.4f}")

st.markdown("<br>", unsafe_allow_html=True)

# Tabs for visual deep-dives
setup_plotting_style()
tab_paths, tab_pnl, tab_logic = st.tabs(["� MARKET ENVELOPE", "� HEDGE DISTRIBUTION", "🧠 QUANT INSIGHTS"])

with tab_paths:
    st.pyplot(plot_gbm_paths(paths))
    st.caption("The price envelope illustrates the 95th (Green) and 5th (Red) percentile boundaries across the simulated trading horizon.")

with tab_pnl:
    st.pyplot(plot_pnl_distribution(results["final_pnl"]))
    st.markdown(f"> **Institutional Note**: A wider distribution (Std Dev: ${stats['std_pnl']:.4f}) indicates higher residual risk after hedging.")

with tab_logic:
    st.markdown("### Executive Summary")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown(f"""
        **Option Portfolio**
        - **Asset Class**: {ticker_input if ticker_input else "Synthetic Asset"}
        - **Theoretical Premium**: `${initial_premium:.4f}`
        - **Model Logic**: {model_type.upper()} Calibration
        """)
    with col_l2:
        st.markdown(f"""
        **Hedge Execution**
        - **Frequency**: {rebalance_val.title()}
        - **Total Trades**: {stats['avg_trades']:.1f}
        - **Cost Efficiency**: {'Optimal' if stats['avg_costs'] < initial_premium*0.05 else 'Cost Heavy'}
        """)
    
    if stats['mean_pnl'] == 0 and initial_premium < 0.01:
        st.error("🚨 **Deep OTM Alert**: The current Strike/Spot combination results in zero delta exposure. No hedging required.")

# Footer
st.markdown("<hr style='border: 0.1px solid rgba(255,255,255,0.1); margin-top: 50px;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #484f58; font-size: 0.8rem;'>AEGIS HEDGING ENGINE | BUILT FOR INSTITUTIONAL RESEARCH</p>", unsafe_allow_html=True)
