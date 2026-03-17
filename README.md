<p align="center">
  <strong>Raysteezy Learning</strong><br>
  Machine Learning &middot; Financial Data Pipelines &middot; Python Development
</p>

<p align="center">
  <a href="../../actions/workflows/update-planet-labs-data.yml">
    <img src="https://github.com/raysteezy/raysteezy-learning/actions/workflows/update-planet-labs-data.yml/badge.svg" alt="Planet Labs Data Status">
  </a>
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11">
  <img src="https://img.shields.io/badge/data-automated-green" alt="Automated Data">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT License">
</p>

---

## Overview

A personal learning repository for building automated data pipelines, financial analysis tools, and Python projects. All data is synced to [Airweave](https://airweave.ai) for AI-powered semantic search.

## Repository Structure

```
raysteezy-learning/
├── data/
│   └── planet-labs/                     # Planet Labs (NYSE: PL) financial data
│       ├── quote.json                   #   Current stock price & key metrics
│       ├── income_statement.csv         #   Quarterly income statements
│       ├── balance_sheet.csv            #   Quarterly balance sheets
│       ├── cash_flow.csv                #   Quarterly cash flow statements
│       ├── price_history.csv            #   1-year daily OHLCV prices
│       ├── README.md                    #   Data dictionary & field descriptions
│       └── predictions/
│           ├── README.md                #   Linear & polynomial regression models
│           ├── model_summary.json       #   Regression model results
│           ├── predicted_prices.csv     #   6-month regression forecasts
│           ├── pl_price_prediction.png  #   Price prediction chart
│           ├── pl_model_dashboard.png   #   4-panel model dashboard
│           └── monte-carlo/
│               ├── README.md            #   Monte Carlo simulation methodology
│               ├── monte_carlo_summary.json  #  Full MC/stress/robustness results
│               ├── mc_percentile_paths.csv   #  Percentile price paths (P5–P95)
│               ├── stress_test_paths.csv     #  5 stress scenario median paths
│               ├── sensitivity_analysis.csv  #  Parameter sensitivity grid
│               ├── mc_fan_chart.png          #  Monte Carlo fan chart
│               ├── stress_test_chart.png     #  Stress test comparison
│               └── robustness_dashboard.png  #  4-panel robustness dashboard
├── scripts/
│   ├── fetch_planet_labs_financials.py  # Data collection script
│   ├── pl_price_prediction_model.py    # Linear & polynomial regression
│   └── pl_monte_carlo_simulation.py    # Monte Carlo GBM + stress testing
├── .github/workflows/
│   └── update-planet-labs-data.yml     # Weekly automation
├── .gitignore
├── LICENSE
├── SECURITY.md
└── README.md                           # You are here
```

## Projects

### Planet Labs (NYSE: PL) — Financial Data Feed

An automated pipeline that collects Planet Labs financial data weekly and stores it as structured CSV/JSON files.

| Dataset | Format | Update Frequency | Description |
|---------|--------|-----------------|-------------|
| [Quote & Metrics](data/planet-labs/quote.json) | JSON | Weekly | Price, market cap, P/E, margins, growth rates |
| [Income Statement](data/planet-labs/income_statement.csv) | CSV | Weekly | Revenue, gross profit, operating income, EPS |
| [Balance Sheet](data/planet-labs/balance_sheet.csv) | CSV | Weekly | Assets, liabilities, equity, cash, debt |
| [Cash Flow](data/planet-labs/cash_flow.csv) | CSV | Weekly | Operating, investing, financing cash flows |
| [Price History](data/planet-labs/price_history.csv) | CSV | Weekly | 1-year daily open, high, low, close, volume |

**How it works:**

1. A GitHub Actions workflow runs **every Monday at 11:00 PM MST**
2. The Python script pulls fresh data from Yahoo Finance (no API key required)
3. Updated files are auto-committed to this repository
4. Airweave syncs the repo and indexes all data for semantic search

**Run manually:** [Actions tab](../../actions) → "Update Planet Labs Data" → "Run workflow"

---

### Price Prediction Models

#### Linear & Polynomial Regression

6-month price forecast using linear and degree-3 polynomial regression on historical close prices.

| Model | R² (Train) | RMSE | 6-Month Target |
|-------|-----------|------|----------------|
| Linear | 0.5568 | $4.47 | $25.64 |
| Polynomial (deg-3) | 0.6897 | $3.74 | $28.88 |

📊 **Charts:** [Price Prediction](data/planet-labs/predictions/pl_price_prediction.png) · [Model Dashboard](data/planet-labs/predictions/pl_model_dashboard.png)

#### Monte Carlo Simulation (GBM)

10,000-path Geometric Brownian Motion simulation with 5 stress-test scenarios and full robustness analysis.

| Horizon | Median | P5 (Downside) | P95 (Upside) |
|---------|--------|---------------|--------------|
| 6 Months | $23.38 | $9.65 | $54.65 |
| 1 Year | $22.29 | $6.48 | $76.08 |
| 2 Years | $20.10 | $3.52 | $117.12 |

**Key risk metrics:** VaR 95% = $3.52 · CVaR 95% = $2.39 · P(Profit 2yr) = 42.3%

**Stress tests:** Market Crash → $0.15 · Bear → $3.28 · Base → $20.14 · Bull → $49.26 · Extreme Bull → $31.44

**Robustness:** Walk-forward validation, bootstrap CIs, parameter sensitivity analysis

📊 **Charts:** [Fan Chart](data/planet-labs/predictions/monte-carlo/mc_fan_chart.png) · [Stress Tests](data/planet-labs/predictions/monte-carlo/stress_test_chart.png) · [Robustness Dashboard](data/planet-labs/predictions/monte-carlo/robustness_dashboard.png)

📖 **Full methodology:** [Monte Carlo README](data/planet-labs/predictions/monte-carlo/README.md)

> ⚠️ **Disclaimer:** All predictions are educational simulations — not financial advice.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Data collection, ML models, simulations |
| GitHub Actions | Weekly scheduled automation |
| Airweave | Semantic search over synced data |
| yfinance | Financial data from Yahoo Finance |
| NumPy / SciPy | Monte Carlo simulation & statistics |
| Matplotlib | Visualization & charting |
| scikit-learn | Regression models |

## Getting Started

```bash
# Clone the repository
git clone https://github.com/raysteezy/raysteezy-learning.git
cd raysteezy-learning

# Install dependencies
pip install yfinance pandas numpy scipy matplotlib scikit-learn

# Run the data collector
python scripts/fetch_planet_labs_financials.py

# Run regression models
python scripts/pl_price_prediction_model.py

# Run Monte Carlo simulation
python scripts/pl_monte_carlo_simulation.py
```

## License

This project is licensed under the [MIT License](LICENSE).

## Security

See [SECURITY.md](SECURITY.md) for security policies and responsible disclosure guidelines.

---

<p align="center">Built by <a href="https://github.com/raysteezy">@raysteezy</a></p>
