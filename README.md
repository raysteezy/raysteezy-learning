<p align="center">
  <strong>Raysteezy Learning</strong><br>
  Machine Learning · Financial Data Pipelines · Python Development
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

## What This Is

This is my personal learning repo. I'm a first-year college student teaching myself Python, data pipelines, and machine learning by working with real stock market data. Nothing fancy — just me figuring things out and documenting what I learn along the way.

The main project tracks Planet Labs (PL) stock data using ~5 years of daily price history (since their IPO in April 2021). I built a pipeline that grabs financial data every week, and I've been building prediction models in two rounds — V1 and V2. Both versions are in the repo so you can see the progression.

All the data in this repo also gets synced to [Airweave](https://airweave.ai) for AI-powered search.

---

## V1 — Baseline Models (Grade: C+)

My first attempt. I used basic regression with no proper validation. These models taught me a lot about what NOT to do, so I'm keeping them in the code for reference.

### V1 Price Prediction

| Model | R² (Training) | Problem |
|-------|--------------|---------|
| Linear Regression | 0.029 | Underfits — barely explains anything |
| Polynomial (deg-3) | 0.849 | Overfits — curves up forever past training data |

### V1 Monte Carlo

| Detail | Value |
|--------|-------|
| Model | Constant-vol Geometric Brownian Motion (GBM) |
| Stress tests | Made-up multipliers (not based on real data) |
| Validation | None |
| 2-year median | $20.09 |
| P(Profit) | 42.3% |

### Why V1 Got a C+

- R² was only computed on training data (no out-of-sample testing)
- Only used the date as input — no volume, no returns, no fundamentals
- No confidence intervals — just single-number predictions
- Polynomial extrapolation goes to infinity past the training data
- Monte Carlo used constant volatility (real stocks don't behave that way)
- Stress scenarios were arbitrary multipliers, not grounded in data

### V1 Files

| What | File |
|------|------|
| Prediction code (v1 section) | [pl_price_prediction_model.py](scripts/pl_price_prediction_model.py) |
| Monte Carlo code (v1 section) | [pl_monte_carlo_simulation.py](scripts/pl_monte_carlo_simulation.py) |

---

## V2 — Upgraded Models (Grade: Pending)

After getting a C+ on V1, I rebuilt everything to fix all 7 issues. This section has everything you need to grade V2.

### What I Fixed

| Problem from V1 | How V2 Fixes It |
|-----------------|----------------|
| R² only on training data | Walk-forward validation on 63 unseen trading days |
| Only used date as a feature | 10+ features (lagged returns, volatility, momentum, volume, SMA crossovers) |
| No overfitting control | Ridge regularization + ARIMA with AIC penalty |
| No confidence intervals | Bootstrap prediction intervals (90% and 50% bands) |
| Constant-vol Monte Carlo | Heston stochastic volatility + Merton jump diffusion |
| Made-up stress scenarios | HMM regime-switching (2 regimes detected from real PL data) |
| Single model, no comparison | 3 MC models side-by-side + v1 vs v2 comparison tables |

### V2 Price Prediction Results

| Model | R² (Out-of-Sample) | MAE | RMSE | Directional Accuracy |
|-------|-------------------|-----|------|---------------------|
| ARIMA | 0.835 | $0.97 | $1.19 | 50.0% |
| Ridge + features | 0.851 | $0.94 | $1.14 | 47.6% |

The key difference: V1's polynomial R² of 0.85 was on training data (cheating). V2's R² of 0.85 is on data the model never saw (honest).

**ARIMA 6-month forecast:** $24.60 — 90% CI: [$23.11, $28.51]

### V2 Monte Carlo Results

| Model | 2-Year Median | P(Profit) | P(Double) | VaR 95% |
|-------|--------------|-----------|-----------|---------|
| v1 GBM (baseline) | $20.09 | 42.3% | 20.1% | $3.52 |
| v2 Heston | $15.51 | 35.6% | 18.6% | $1.83 |
| v2 Jump Diffusion | $16.30 | 36.7% | 18.5% | $2.26 |

The V2 models are more pessimistic because they're more realistic about tail risk and volatility clustering.

**HMM Regime Detection:** Found 2 regimes in PL's history — Calm (85% of days, 46% annualized vol) and Volatile (15% of days, 131% annualized vol).

### V2 Charts

| Chart | What It Shows |
|-------|--------------|
| [v1 vs v2 Predictions](data/planet-labs/predictions/pl_price_prediction.png) | Full 5-year history + ARIMA forecast with confidence bands |
| [Walk-Forward Dashboard](data/planet-labs/predictions/pl_model_dashboard.png) | ARIMA + Ridge validation, residuals, and model comparison table |
| [Multi-Model Fan Chart](data/planet-labs/predictions/monte-carlo/mc_fan_chart.png) | GBM vs Heston vs Jump Diffusion with confidence intervals |
| [HMM Stress Tests](data/planet-labs/predictions/monte-carlo/stress_test_chart.png) | 5 scenarios from market crash to calm bull |
| [Robustness Dashboard](data/planet-labs/predictions/monte-carlo/robustness_dashboard.png) | Terminal distributions, bootstrap CIs, sensitivity heatmap, model comparison |

### V2 Files Quick Links

| What | File |
|------|------|
| Prediction code (v1 + v2) | [pl_price_prediction_model.py](scripts/pl_price_prediction_model.py) |
| Monte Carlo code (v1 + v2) | [pl_monte_carlo_simulation.py](scripts/pl_monte_carlo_simulation.py) |
| Prediction writeup | [predictions/README.md](data/planet-labs/predictions/README.md) |
| Monte Carlo writeup | [monte-carlo/README.md](data/planet-labs/predictions/monte-carlo/README.md) |
| Model results (JSON) | [model_summary.json](data/planet-labs/predictions/model_summary.json) |
| MC results (JSON) | [monte_carlo_summary.json](data/planet-labs/predictions/monte-carlo/monte_carlo_summary.json) |

---

## Repo Structure

```
raysteezy-learning/
├── data/
│   └── planet-labs/                     # Planet Labs (NYSE: PL) financial data
│       ├── quote.json                   #   Current stock price and key metrics
│       ├── income_statement.csv         #   Quarterly income statements
│       ├── balance_sheet.csv            #   Quarterly balance sheets
│       ├── cash_flow.csv                #   Quarterly cash flow statements
│       ├── price_history.csv            #   Full daily OHLCV prices (since IPO)
│       ├── README.md                    #   Data dictionary (what each field means)
│       └── predictions/
│           ├── README.md                #   v1 vs v2 model comparison
│           ├── model_summary.json       #   All model results (v1 + v2)
│           ├── predicted_prices.csv     #   Forecasts with confidence intervals
│           ├── pl_price_prediction.png  #   v1 vs v2 prediction chart
│           ├── pl_model_dashboard.png   #   Walk-forward validation dashboard
│           └── monte-carlo/
│               ├── README.md            #   MC v2 methodology (Heston + jumps)
│               ├── monte_carlo_summary.json  #  Full v1 vs v2 MC comparison
│               ├── mc_percentile_paths.csv   #  Heston percentile price paths
│               ├── stress_test_paths.csv     #  HMM regime-based stress tests
│               ├── sensitivity_analysis.csv  #  Parameter sensitivity grid
│               ├── mc_fan_chart.png          #  Multi-model fan chart
│               ├── stress_test_chart.png     #  HMM-based stress scenarios
│               └── robustness_dashboard.png  #  v1 vs v2 robustness comparison
├── scripts/
│   ├── fetch_planet_labs_financials.py  # Grabs financial data from Yahoo Finance
│   ├── pl_price_prediction_model.py    # v1 + v2 regression models
│   └── pl_monte_carlo_simulation.py    # v1 GBM + v2 Heston/jumps/HMM
├── .github/workflows/
│   └── update-planet-labs-data.yml     # Runs the data collector every week
├── .gitignore
├── LEGAL.md                            # All legal stuff in one place
├── LICENSE
├── SECURITY.md
└── README.md                           # You are here
```

## Planet Labs (NYSE: PL) — Weekly Data Feed

I picked Planet Labs because they're a space company that's publicly traded, which I think is cool. The pipeline grabs their financial data once a week and saves it here so I can use it for analysis later.

| Dataset | Format | How Often | What's In It |
|---------|--------|-----------|-------------|
| [Quote & Metrics](data/planet-labs/quote.json) | JSON | Weekly | Current price, market cap, P/E ratio, margins |
| [Income Statement](data/planet-labs/income_statement.csv) | CSV | Weekly | Revenue, gross profit, operating income, EPS |
| [Balance Sheet](data/planet-labs/balance_sheet.csv) | CSV | Weekly | Assets, liabilities, equity, cash, debt |
| [Cash Flow](data/planet-labs/cash_flow.csv) | CSV | Weekly | Operating, investing, financing cash flows |
| [Price History](data/planet-labs/price_history.csv) | CSV | Weekly | Full daily open/high/low/close/volume (since IPO) |

**How the pipeline works:**

1. A GitHub Actions workflow runs every **Monday at 11:00 PM MST**
2. The Python script uses `yfinance` to pull data from Yahoo Finance (free, no API key)
3. Updated files get auto-committed back to this repo
4. Airweave picks up the changes and indexes everything for search

**Want to run it yourself?** Go to the [Actions tab](../../actions) → "Update Planet Labs Data" → "Run workflow"

## Tools I Used

| Tool | What I Used It For |
|------|---------------------|
| Python 3.11 | Everything — data collection, ML models, charts |
| GitHub Actions | Automating the weekly data pulls |
| Airweave | Syncing this repo for AI-powered search |
| yfinance | Getting stock data from Yahoo Finance (free) |
| NumPy / SciPy | Math for simulations and statistics |
| Matplotlib | Making all the charts and dashboards |
| scikit-learn | Linear, polynomial, and Ridge regression |
| pmdarima | Auto-ARIMA model selection |
| hmmlearn | Hidden Markov Model for regime detection |

## How to Run This Yourself

```bash
# Clone the repo
git clone https://github.com/raysteezy/raysteezy-learning.git
cd raysteezy-learning

# Install the Python packages you need
pip install yfinance pandas numpy scipy matplotlib scikit-learn pmdarima hmmlearn

# Run the data collector
python scripts/fetch_planet_labs_financials.py

# Run the prediction models (v1 + v2)
python scripts/pl_price_prediction_model.py

# Run the Monte Carlo simulation (v1 + v2)
python scripts/pl_monte_carlo_simulation.py
```

## Legal

> **This repository is for educational and academic purposes only. Nothing herein constitutes financial advice, investment recommendations, or a solicitation to buy or sell any security.** All models, simulations, and predictions are statistical exercises based on historical data with significant limitations. Past performance does not guarantee future results.

Everything legal — disclaimer, data attribution, security policy, and the MIT License — is in one file: **[LEGAL.md](LEGAL.md)**.

---

<p align="center">Built by <a href="https://github.com/raysteezy">@raysteezy</a></p>
