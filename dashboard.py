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
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4150;
    }
    .stAlert {
        background-color: #1e2130;
        border: 1px solid #3e4150;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Aegis Hedge Systems")
st.subheader("Institutional Derivatives Risk & Interpretation Engine")

# Cached Simulation logic
@st.cache_data
def get_sim_results(config: SimulationConfig):
    if config.model_type == "bsm":
        paths = generate_gbm_paths(S0=config.S0, mu=config.r, sigma=config.sigma, T=config.T, n_steps=config.n_steps, n_paths=config.n_paths)
        var_paths = None
        initial_premium = bsm_price(config.S0, config.K, config.T, config.r, config.sigma, config.option_type)
    else:
        paths, var_paths = generate_heston_paths(S0=config.S0, v0=config.v0, mu=config.r, kappa=config.kappa, theta=config.theta, sigma_v=config.sigma_v, rho=config.rho, T=config.T, n_steps=config.n_steps, n_paths=config.n_paths)
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

ticker_input = st.sidebar.text_input("Ticker (Optional)", value="", help="Enter a ticker like SPY or AAPL to fetch live data")

model_type = st.sidebar.selectbox("Pricing Model", ["bsm", "heston"])

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
        S0_val, sigma_val = fetch_ticker_data(ticker_input)
        st.sidebar.success(f"Live {ticker_input}: ${S0_val:.2f} | Vol: {sigma_val:.2%}")
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

with st.sidebar.expander("🛠️ Strategy & Costs", expanded=True):
    rebalance_val = st.selectbox("Frequency", ["daily", "weekly", "threshold", "gamma_scaled"])
    cost_model_val = st.selectbox("Cost Model", ["proportional", "fixed", "bps"])
    cost_param_val = st.number_input("Cost Value", value=0.001, format="%.4f")
    # Limit paths for better responsiveness
    n_paths_val = st.number_input("Num Paths", value=100 if model_type == "bsm" else 20, step=10)

# Build Config
config = SimulationConfig(
    S0=S0_val, K=K_val, T=T_val, r=r_val, sigma=sigma_val,
    n_paths=int(n_paths_val), n_steps=252,
    rebalance_freq=rebalance_val, model_type=model_type,
    v0=v0_val, kappa=kappa_val, theta=theta_val, sigma_v=sigma_v_val, rho=rho_val,
    cost_model=cost_model_val, cost_param=cost_param_val
)

# Run Simulation
if st.sidebar.button("🚀 Run Simulation"):
    with st.spinner("Generating paths and calculating hedges..."):
        paths, results, stats, initial_premium = get_sim_results(config)

    # 2. Display Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean PnL", f"${stats['mean_pnl']:.4f}")
    col2.metric("Sharpe-like Ratio", f"{stats['sharpe_ratio']:.2f}")
    col3.metric("Avg Trans. Cost", f"${stats['avg_costs']:.4f}")
    col4.metric("VaR (95%)", f"${stats['var_95']:.4f}")

    # 3. Visuals & Interpretation
    setup_plotting_style()
    
    tab1, tab2, tab3 = st.tabs(["📊 Price Paths", "📈 PnL Analysis", "🧠 Quant Interpretation"])
    
    with tab1:
        st.pyplot(plot_gbm_paths(paths))
        st.info("**Path Analysis:** This 'fan' chart shows the distribution of possible outcomes. The green/red dashed lines represent the 95th/5th percentiles. A wider fan indicates higher volatility regimes.")

    with tab2:
        st.pyplot(plot_pnl_distribution(results["final_pnl"]))
        st.warning("**PnL Skewness:** Notice the left-hand tail. This is caused by transaction costs and 'discrete' hedging—the reality that you cannot hedge continuously.")

    with tab3:
        st.markdown(f"""
        ### Executive Interpretation
        
        *   **Hedge Efficiency**: Your strategy required an average of **{stats['avg_trades']:.1f} trades**. 
        *   **Cost Drag**: Transaction costs removed **${stats['avg_costs']:.4f}** from each path on average.
        *   **Volatility Impact**: Using the **{config.model_type.upper()}** model, the standard deviation of your hedging error is **${stats['std_pnl']:.4f}**.
        
        #### Strategic Recommendation:
        {"Consider decreasing rebalance frequency to save on costs." if stats['avg_costs'] > abs(stats['mean_pnl']) else "Rebalancing frequency is well-optimized for this volatility regime."}
        """)

else:
    st.info("👈 Set your parameters in the sidebar and click 'Run Simulation' to begin.")
    
    # Placeholder/Intro Image/Text
    st.markdown("""
    ### Welcome to Aegis
    Aegis is built to simulate how professional derivatives desks manage risk. 
    
    **How to use:**
    1. Select a **Pricing Model** (Black-Scholes is standard, Heston is for stochastic volatility).
    2. Adjust the **Asset Parameters** or enter a **Ticker** to fetch live data.
    3. Choose a **Rebalancing Strategy** to see how it performs against transaction costs.
    """)
