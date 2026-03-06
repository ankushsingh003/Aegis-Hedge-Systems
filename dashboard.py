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
st.set_page_config(page_title="Aegis Hedge Systems | Quant Dashboard", layout="wide")

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
st.subheader("Institutional Derivatives Risk & Interpretation Engine")

# Cached Simulation logic
@st.cache_data
def get_sim_results(_config: SimulationConfig):
    if _config.model_type == "bsm":
        paths = generate_gbm_paths(S0=_config.S0, mu=_config.r, sigma=_config.sigma, T=_config.T, n_steps=_config.n_steps, n_paths=_config.n_paths, seed=_config.seed)
        var_paths = None
        initial_premium = bsm_price(_config.S0, _config.K, _config.T, _config.r, _config.sigma, _config.option_type)
    else:
        paths, var_paths = generate_heston_paths(S0=_config.S0, v0=_config.v0, mu=_config.r, kappa=_config.kappa, theta=_config.theta, sigma_v=_config.sigma_v, rho=_config.rho, T=_config.T, n_steps=_config.n_steps, n_paths=_config.n_paths, seed=_config.seed)
        initial_premium = heston_price(_config.S0, _config.K, _config.T, _config.r, _config.v0, _config.kappa, _config.theta, _config.sigma_v, _config.rho, _config.option_type)

    results = run_hedging_simulation(paths, _config, variance_paths=var_paths)
    stats = compute_pnl_statistics(results, initial_premium, _config)
    return paths, results, stats, initial_premium

@st.cache_data
def fetch_ticker_data(ticker_symbol: str):
    provider = YFinanceProvider()
    spot = provider.get_spot_price(ticker_symbol)
    vol = provider.estimate_volatility(ticker_symbol)
    return spot, vol

# Sidebar - Settings
st.sidebar.header("🕹️ Simulation Control")

ticker_input = st.sidebar.text_input("Ticker (Optional)", value="", help="Enter a ticker like SPY or AAPL to fetch live data")

model_type = st.sidebar.selectbox("Pricing Model", ["bsm", "heston"])
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

# Collapsible sections for parameters
with st.sidebar.expander("📈 Asset Parameters", expanded=True):
    S0_val = st.number_input("Spot Price ($)", value=100.0)
    K_val = st.number_input("Strike Price ($)", value=100.0)
    T_val = st.slider("Time to Expiry (Years)", 0.1, 2.0, 1.0)
    r_val = st.slider("Risk-free Rate", 0.0, 0.1, 0.05)
    sigma_val = st.slider("Implied Volatility", 0.05, 1.0, 0.20)

# Update values if ticker is provided
if ticker_input:
    try:
        S0_ticker, sigma_ticker = fetch_ticker_data(ticker_input)
        st.sidebar.success(f"Live {ticker_input}: ${S0_ticker:.2f} | Vol: {sigma_ticker:.2%}")
        S0_val = S0_ticker
        sigma_val = sigma_ticker
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

if model_type == "heston":
    with st.sidebar.expander("🌀 Heston Parameters", expanded=True):
        v0_val = st.number_input("Initial Var (v0)", value=0.04)
        kappa_val = st.slider("Mean Reversion (κ)", 0.1, 5.0, 2.0)
        theta_val = st.slider("Long-run Var (θ)", 0.01, 0.2, 0.04)
        sigma_v_val = st.slider("Vol of Vol (ξ)", 0.1, 1.0, 0.3)
        rho_val = st.slider("Correlation (ρ)", -1.0, 1.0, -0.7)
else:
    v0_val, kappa_val, theta_val, sigma_v_val, rho_val = 0.04, 2.0, 0.04, 0.3, -0.7

with st.sidebar.expander("🛠️ Strategy & Simulation", expanded=True):
    rebalance_val = st.selectbox("Frequency", ["daily", "weekly", "threshold", "gamma_scaled"])
    cost_model_val = st.selectbox("Cost Model", ["proportional", "fixed", "bps"])
    cost_param_val = st.number_input("Cost Value", value=10.0 if cost_model_val == "bps" else 0.001, format="%.4f")
    n_paths_val = st.number_input("Num Paths", value=100 if model_type == "bsm" else 20, step=10)
    seed_val = st.number_input("Random Seed", value=42)

# Build Config
config = SimulationConfig(
    S0=S0_val, K=K_val, T=T_val, r=r_val, sigma=sigma_val,
    n_paths=int(n_paths_val), n_steps=252,
    rebalance_freq=rebalance_val, model_type=model_type,
    option_type=option_type,
    v0=v0_val, kappa=kappa_val, theta=theta_val, sigma_v=sigma_v_val, rho=rho_val,
    cost_model=cost_model_val, cost_param=cost_param_val,
    seed=int(seed_val)
)

# Initialize session state for persistent results
if "last_results" not in st.session_state:
    st.session_state.last_results = None

# Run Simulation
if st.sidebar.button("🚀 Run Simulation"):
    with st.spinner("Generating paths and calculating hedges..."):
        st.session_state.last_results = get_sim_results(config)

# Display Results from Session State
if st.session_state.last_results:
    paths, results, stats, initial_premium = st.session_state.last_results

    # 1. Metric Callouts
    col1, col2, col3, col4 = st.columns(4)
    # Highlight 0 metrics to explain deep OTM
    pnl_display = f"${stats['mean_pnl']:.4f}"
    if stats['mean_pnl'] == 0 and initial_premium < 0.01:
        pnl_display = "$0.00 (OTM)"
        
    col1.metric("Mean PnL", pnl_display)
    col2.metric("Sharpe-like Ratio", f"{stats['sharpe_ratio']:.2f}")
    col3.metric("Avg Trans. Cost", f"${stats['avg_costs']:.4f}")
    col4.metric("VaR (95%)", f"${stats['var_95']:.4f}")

    # Interpretation Cards
    if stats['mean_pnl'] == 0 and stats['avg_costs'] == 0:
        st.warning("⚖️ **Quantitative Alert: Deep OTM Option**  \nYour choice of Strike ($" + str(config.K) + ") is far from Spot ($" + str(config.S0) + "). The Delta is ~0, meaning no hedging trades were triggered. This results in zero costs and zero PnL. Try a Strike closer to the Spot or increase Volatility.")

    # 2. Visuals
    setup_plotting_style()
    tab1, tab2, tab3 = st.tabs(["📊 Price Paths", "📈 PnL Analysis", "🧠 Quant Interpretation"])
    
    with tab1:
        st.pyplot(plot_gbm_paths(paths))
        st.info("**Analysis:** The chart displays predicted price envelopes. If you change the **Random Seed**, these paths will shuffle.")

    with tab2:
        st.pyplot(plot_pnl_distribution(results["final_pnl"]))
        st.warning("**Risk Note:** The distribution width ($" + f"{stats['std_pnl']:.2f}" + ") represents your hedging error. Institutional goal is to keep this tight.")

    with tab3:
        st.markdown(f"""
        ### Executive Interpretation
        
        *   **Theoretical Value**: The price of this **{config.option_type}** is **${initial_premium:.4f}**.
        *   **Trading Friction**: Transaction costs caused a drag of **${stats['avg_costs']:.4f}** per path.
        *   **Model Confidence**: Total trades generated: **{stats['avg_trades']:.1f}**. 
        
        #### Strategy Insight:
        {"Your hedging frequency is slightly high for the chosen cost model." if stats['avg_costs'] > initial_premium*0.1 and initial_premium > 0 else "The current rebalancing strategy effectively manages risk vs costs."}
        """)

else:
    st.info("👈 Set your parameters in the sidebar and click 'Run Simulation' to begin.")
    st.markdown("""
    ### Interpretation Engine Guide
    1. **Spot vs Strike**: If Strike is much higher (for calls), Delta is 0. 
    2. **Ticker**: Use SPY, NVDA, or BTC-USD for real data.
    3. **Heston**: Use this to see how 'Vol-of-Vol' affects your tail risk (VaR).
    """)
