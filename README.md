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

The main project tracks Planet Labs (PL) stock data. I built a pipeline that grabs financial data every week, and I've been iterating on prediction models — starting with basic regression (v1, grade: C+), then upgrading to ARIMA, Ridge regression with features, stochastic volatility Monte Carlo, and jump diffusion (v2). Both versions are in the code so you can see the progression.

All the data in this repo also gets synced to [Airweave](https://airweave.ai) for AI-powered search.

## Repo Structure

```
raysteezy-learning/
├── data/
│   └── planet-labs/                     # Planet Labs (NYSE: PL) financial data
│       ├── quote.json                   #   Current stock price and key metrics
│       ├── income_statement.csv         #   Quarterly income statements
│       ├── balance_sheet.csv            #   Quarterly balance sheets
│       ├── cash_flow.csv                #   Quarterly cash flow statements
│       ├── price_history.csv            #   1-year daily OHLCV prices
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
├── LICENSE
├── SECURITY.md
└── README.md                           # You are here
```

## Projects

### Planet Labs (NYSE: PL) — Weekly Data Feed

I picked Planet Labs because they're a space company that's publicly traded, which I think is cool. The pipeline grabs their financial data once a week and saves it here so I can use it for analysis later.

| Dataset | Format | How Often | What's In It |
|---------|--------|-----------|-------------|
| [Quote & Metrics](data/planet-labs/quote.json) | JSON | Weekly | Current price, market cap, P/E ratio, margins |
| [Income Statement](data/planet-labs/income_statement.csv) | CSV | Weekly | Revenue, gross profit, operating income, EPS |
| [Balance Sheet](data/planet-labs/balance_sheet.csv) | CSV | Weekly | Assets, liabilities, equity, cash, debt |
| [Cash Flow](data/planet-labs/cash_flow.csv) | CSV | Weekly | Operating, investing, financing cash flows |
| [Price History](data/planet-labs/price_history.csv) | CSV | Weekly | 1 year of daily open/high/low/close/volume |

**How the pipeline works:**

1. A GitHub Actions workflow runs every **Monday at 11:00 PM MST**
2. The Python script uses `yfinance` to pull data from Yahoo Finance (free, no API key)
3. Updated files get auto-committed back to this repo
4. Airweave picks up the changes and indexes everything for search

**Want to run it yourself?** Go to the [Actions tab](../../actions) → "Update Planet Labs Data" → "Run workflow"

---

### Price Prediction Models

#### V1 Baseline (Linear & Polynomial Regression)

I started with basic regression to learn the fundamentals. These models taught me a lot about overfitting and the importance of proper validation, even though the predictions themselves are bad.

| Model | R² (Training) | 6-Month Forecast | Problem |
|-------|--------------|-----------------|---------|
| Linear | 0.029 | ~$8 | Underfits — barely explains anything |
| Polynomial (deg-3) | 0.849 | ~$37 | Overfits — curves up forever past training data |

#### V2 Upgraded (ARIMA + Ridge)

After getting a C+ grade on v1, I rebuilt with models that actually handle time-series data properly.

| Model | R² (Out-of-Sample) | Validation | Key Improvement |
|-------|-------------------|------------|-----------------|
| ARIMA | ~0.84 | Walk-forward (63 days) | Handles autocorrelation, auto-selected parameters |
| Ridge + features | ~0.85 | Walk-forward (63 days) | Uses 10+ features (returns, vol, momentum, volume) |

The big difference: v1's R² was measured on training data (cheating). V2's R² is measured on data the model never saw (honest).

📊 **Charts:** [v1 vs v2 Predictions](data/planet-labs/predictions/pl_price_prediction.png) · [Validation Dashboard](data/planet-labs/predictions/pl_model_dashboard.png)

📖 **Full comparison:** [Predictions README](data/planet-labs/predictions/README.md)

#### Monte Carlo Simulation (V2: Heston + Jump Diffusion + HMM)

The Monte Carlo also got a major upgrade:

| Feature | V1 | V2 |
|---------|----|----|
| Price model | Constant-vol GBM | GBM + Heston stochastic vol + Merton jump diffusion |
| Stress tests | Made-up multipliers | HMM regime-switching (empirically detected) |
| Walk-forward | GBM only | GBM vs Heston comparison |
| Models compared | 1 | 3 (side-by-side) |

**2-Year Comparison:**

| Model | Median | P(Profit) |
|-------|--------|-----------|
| v1 GBM | ~$20 | ~42% |
| v2 Heston | ~$16 | ~36% |
| v2 Jump Diffusion | ~$16 | ~37% |

The v2 models are more pessimistic because they're more realistic about tail risk and volatility clustering.

📊 **Charts:** [Multi-Model Fan Chart](data/planet-labs/predictions/monte-carlo/mc_fan_chart.png) · [HMM Stress Tests](data/planet-labs/predictions/monte-carlo/stress_test_chart.png) · [Robustness Dashboard](data/planet-labs/predictions/monte-carlo/robustness_dashboard.png)

📖 **Full methodology:** [Monte Carlo v2 README](data/planet-labs/predictions/monte-carlo/README.md)

> ⚠️ **Disclaimer:** All predictions are educational simulations — not financial advice.

---

## Tools I Used

| Tool | What I Used It For |
|------|--------------------|
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

## Disclaimer

> **This repository is for educational and academic purposes only. Nothing herein constitutes financial advice, investment recommendations, or a solicitation to buy or sell any security.** All models, simulations, and predictions are statistical exercises based on historical data with significant limitations. Past performance does not guarantee future results. See [DISCLAIMER.md](DISCLAIMER.md) for full details.

## Data Attribution

Financial data sourced from [Yahoo Finance](https://finance.yahoo.com/) via [yfinance](https://github.com/ranaroussi/yfinance) (Apache 2.0). Yahoo!, Y!Finance, and Yahoo! Finance are registered trademarks of Yahoo, Inc. This repository is not affiliated with Yahoo. See [DATA_NOTICE.md](DATA_NOTICE.md) for usage terms and restrictions.

## License

Source code is licensed under the [MIT License](LICENSE). Data files are subject to [Yahoo's Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html). See [DATA_NOTICE.md](DATA_NOTICE.md) for details.

## Security

See [SECURITY.md](SECURITY.md) for security policies and responsible disclosure guidelines.

---

<p align="center">Built by <a href="https://github.com/raysteezy">@raysteezy</a></p>
