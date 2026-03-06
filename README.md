# Aegis Hedge Systems 🛡️📈

A production-grade risk engine designed for hedge funds and quantitative trading desks to simulate and analyze the performance of delta-neutral hedging strategies.

## 🚀 Overview

This simulator bridges the gap between theoretical Black-Scholes pricing and real-world trading frictions. It simulates asset price paths using Geometric Brownian Motion (GBM) and manages a self-financing portfolio by dynamically rebalancing stock and cash positions to maintain delta-neutrality.

### Key Features
- **Full Greeks Suite**: $\Delta, \gamma, \nu, \theta, \rho$ plus cross-Greeks like Vanna and Charm.
- **Stochastic Volatility Models**: Integrated **Heston model** support for joint price/variance simulations and pricing.
- **Live Data Integration**: Fetch real-time spot prices and estimate volatility using Yahoo Finance.
- **Adaptive Rebalancing**: Daily, weekly, threshold-based, and Gamma-scaled strategies.
- **Institutional Friction Models**: Proportional costs, fixed fees, and Basis Points (bps) slippage.

- **PnL Attribution**: Detailed breakdown of hedging performance, costs, and risk metrics.
- **Vectorized Performance**: Built with NumPy for high-throughput Monte Carlo simulations.

## 📁 Architecture

```text
d:\quant_finance_proj_1\
├── config/             # Environment & strategy settings
├── models/             # BSM & Greeks mathematical engines
├── simulation/         # GBM path generators
├── engine/             # Hedging logic & cost models
├── visualisation/      # Professional plotting suite
├── reports/            # Auto-generated PNG/JSON analysis
└── tests/              # Comprehensive unit tests
```

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run a Standard Simulation
```bash
python main.py --n_paths 1000 --rebalance daily --cost_model bps --cost_param 10
```

### 3. Run Heston Simulation
```bash
python main.py --model heston --v0 0.04 --kappa 2.0 --theta 0.04
```

### 4. Fetch Live Data for SPY
```bash
python main.py --ticker SPY --model heston
```

### 5. Launch the Interpretation Dashboard
```bash
streamlit run dashboard.py
```

### 🚀 6. Deploy to Render (Cloud)
1.  Connect your GitHub repository to [Render.com](https://render.com).
2.  Choose **Blueprint** to use the included `render.yaml`.
3.  Render will automatically build and deploy your Aegis Command Center.


## 📊 Outputs

The simulator generates high-quality visual reports in the `reports/` directory:
- **GBM Fan Chart**: Visualizes price path distributions.
- **PnL Distribution**: Histogram showing the effectiveness of the hedge and variance of the strategy.
- **Summary JSON**: Machine-readable statistics for audit trails.

## 🧪 Mathematical Foundation

The engine implements the standard **Black-Scholes-Merton** model for European options:

$$ dS_t = \mu S_t dt + \sigma S_t dW_t $$

The Delta ($\Delta$) is computed analytically to determine the hedge ratio, while the portfolio is updated discretely, capturing the "Gamma bleed" and transaction costs that professional desks must manage.

---
*Developed for Institutional-grade Quantitative Research.*
