<p align="center">
  <strong>Raysteezy Learning</strong><br>
  Machine Learning · Financial Data Pipelines · Python Development
</p>

<p align="center">
  <a href="../../actions/workflows/update-planet-labs-data.yml">
    <img src="https://github.com/raysteezy/raysteezy-learning/actions/workflows/update-planet-labs-data.yml/badge.svg" alt="Planet Labs Data Status">
  </a>
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Perplexity-Computer-Red" alt="Perplexity Computer">
  <img src="https://img.shields.io/badge/data-automated-green" alt="Automated Data">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT License">
</p>


## What this is and why

This is my personal learning expierence. I'm a first-year college student teaching myself Python, data pipelines, and machine learning by working with real stock market data. There is nothing fancy this is just me figuring things out and documenting what I learn along the way.

The main project tracks Planet Labs (PL) stock data using 5 years of daily price history (since their IPO in April 2021). I built a pipeline that grabs financial data every week, and I've been building prediction models in two rounds — V1 and V2. Both versions live in separate files so you can see the progression.

All the data in this repo also gets synced to [Airweave](https://airweave.ai) for AI-powered search.


## V1 — Baseline Models (Grade: C+)

My first attempt. I used basic regression with no proper validation. These models taught me a lot about what NOT to do, so I'm keeping them for reference.

### V1 Price Prediction

| Model | R² (Training) | Problem |
|-------|--------------:|---------|
| Linear Regression | 0.036 | Underfits — barely explains anything |
| Polynomial (deg-3) | 0.853 | Overfits — curves up forever past training data |

### V1 Monte Carlo

| Detail | Value |
|--------|-------|
| Model | Constant-vol Geometric Brownian Motion (GBM) |
| Stress tests | Made-up multipliers (not based on real data) |
| Validation | None |
| 2-year median | $31.02 |
| P(Profit) | 46.8% |

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
| Price prediction | [prediction_v1.py](scripts/prediction_v1.py) |
| Monte Carlo simulation | [monte_carlo_v1.py](scripts/monte_carlo_v1.py) |

---

## V2 — Upgraded Models (Grade: A-)

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
| Single model, no comparison | 3 MC models side-by-side + V1 vs V2 comparison tables |

### V2 Price Prediction Results

| Model | R² (Out-of-Sample) | MAE | RMSE | Directional Accuracy |
|-------|-------------------:|----:|-----:|---------------------:|
| ARIMA | 0.703 | $1.14 | $1.57 | 45.2% |
| Ridge + features | 0.737 | $1.11 | $1.47 | 50.8% |

The key difference: V1's polynomial R² of 0.85 was on training data (cheating). V2's R² of 0.74 is on data the model never saw (honest).

**ARIMA 6-month forecast:** $36.35 — 90% CI: [$34.22, $39.79]

### V2 Monte Carlo Results

| Model | 2-Year Median | P(Profit) | P(Double) | VaR 95% |
|-------|--------------:|----------:|----------:|--------:|
| V1 GBM (baseline) | $31.02 | 46.8% | 23.6% | $5.32 |
| V2 Heston | $23.89 | 39.3% | 21.2% | $2.76 |
| V2 Jump Diffusion | $24.84 | 40.4% | 22.0% | $3.22 |

The V2 models are more pessimistic because they're more realistic about tail risk and volatility clustering.

**HMM Regime Detection:** Found 2 regimes in PL's history — Calm (85% of days, 46% annualized vol) and Volatile (15% of days, 134% annualized vol).

### V2 Charts

![PL Price Demo](data/planet-labs/demo.gif)

### V2 Files

| What | File |
|------|------|
| Price prediction | [prediction_v2.py](scripts/prediction_v2.py) |
| Monte Carlo simulation | [monte_carlo_v2.py](scripts/monte_carlo_v2.py) |
| Prediction writeup | [predictions/README.md](data/planet-labs/predictions/README.md) |
| Monte Carlo writeup | [monte-carlo/README.md](data/planet-labs/predictions/monte-carlo/README.md) |
| Model results | [results.json](data/planet-labs/predictions/results.json) |
| MC results | [results.json](data/planet-labs/predictions/monte-carlo/results.json) |

---

## Repo Structure

```
raysteezy-learning/
├── data/
│   └── planet-labs/                     # Planet Labs (NYSE: PL) financial data
│       ├── quote.json                   #   Current stock price and key metrics
│       ├── income_statement.csv         #   Quarterly income statements
│       ├── balance_sheet.csv            #   Quarterly balance sheets
│       ├── cash_flow.csv               #   Quarterly cash flow statements
│       ├── price_history.csv            #   Full daily OHLCV prices (since IPO)
│       ├── README.md                    #   Data dictionary
│       └── predictions/
│           ├── README.md                #   V1 vs V2 model comparison
│           ├── results.json             #   V2 model results
│           ├── prices.csv               #   V2 forecasts with CIs
│           └── monte-carlo/
│               ├── README.md            #   MC methodology
│               ├── results.json         #   Full V1 vs V2 MC comparison
│               ├── paths.csv            #   Heston percentile paths
│               ├── stress_paths.csv     #   Stress scenario paths
│               └── sensitivity.csv      #   Parameter sensitivity grid
├── scripts/
│   ├── fetch_planet_labs_financials.py  # Grabs financial data from Yahoo Finance
│   ├── prediction_v1.py                # V1 baseline (linear + polynomial)
│   ├── prediction_v2.py                # V2 upgraded (ARIMA + Ridge + walk-forward)
│   ├── monte_carlo_v1.py               # V1 baseline (constant-vol GBM)
│   └── monte_carlo_v2.py               # V2 upgraded (Heston + jumps + HMM)
├── .github/workflows/
│   └── update-planet-labs-data.yml     # Runs the full pipeline every week
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
2. The Python script uses `yfinance` to pull data from Yahoo Finance (free, no API key needed)
3. Both V1 and V2 prediction + Monte Carlo scripts run automatically
4. Updated files get auto-committed back to this repo
5. Airweave picks up the changes and indexes everything for search

**Want to run it yourself?** Go to the [Actions tab](../../actions) → "Update Planet Labs Data" → "Run workflow"

## Tools I Used

| Tool | What I Used It For |
|------|---------------------|
| Python 3.11 | Everything — data collection, ML models, charts |
| GitHub Actions | Automating the weekly data pulls and model runs |
| Airweave | Syncing this repo for AI-powered search |
| yfinance | Getting stock data from Yahoo Finance (free) |
| NumPy / SciPy | Math for simulations and statistics |
| Matplotlib | Making all the charts and dashboards |
| scikit-learn | Linear, polynomial, and Ridge regression |
| pmdarima | Auto-ARIMA model selection |
| hmmlearn | Hidden Markov Model for regime detection |

## Legal

> **This repository is for educational and academic purposes only. Nothing herein constitutes financial advice, investment recommendations, or a solicitation to buy or sell any security.** All models, simulations, and predictions are statistical exercises based on historical data with significant limitations. Past performance does not guarantee future results.

Everything legal — disclaimer, data attribution, security policy, and the MIT License — is in one file: **[LEGAL.md](LEGAL.md)**.

---

<p align="center">Built by <a href="https://github.com/raysteezy">@raysteezy</a></p>
