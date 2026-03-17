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

Right now the main project is tracking Planet Labs (PL) stock data. I built a pipeline that grabs financial data every week, and I've been experimenting with regression models and Monte Carlo simulations to see what I can learn about predicting stock prices (spoiler: it's really hard).

All the data in this repo also gets synced to [Airweave](https://airweave.ai) so I can search through it with AI.

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
│           ├── README.md                #   Explanation of the regression models
│           ├── model_summary.json       #   Regression results
│           ├── predicted_prices.csv     #   6-month forecast numbers
│           ├── pl_price_prediction.png  #   Price prediction chart
│           ├── pl_model_dashboard.png   #   4-panel model dashboard
│           └── monte-carlo/
│               ├── README.md            #   How the Monte Carlo simulation works
│               ├── monte_carlo_summary.json  #  All MC/stress/robustness results
│               ├── mc_percentile_paths.csv   #  Percentile price paths (P5–P95)
│               ├── stress_test_paths.csv     #  5 stress scenario median paths
│               ├── sensitivity_analysis.csv  #  Parameter sensitivity grid
│               ├── mc_fan_chart.png          #  Monte Carlo fan chart
│               ├── stress_test_chart.png     #  Stress test comparison
│               └── robustness_dashboard.png  #  4-panel robustness dashboard
├── scripts/
│   ├── fetch_planet_labs_financials.py  # Grabs financial data from Yahoo Finance
│   ├── pl_price_prediction_model.py    # Linear and polynomial regression
│   └── pl_monte_carlo_simulation.py    # Monte Carlo simulation + stress testing
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

#### Linear & Polynomial Regression

I tried fitting two basic regression models on the historical price data to forecast 6 months ahead. This was mostly a learning exercise to understand how regression works on time-series data.

| Model | R² (Training) | RMSE | 6-Month Forecast |
|-------|--------------|------|-----------------|
| Linear | 0.5568 | $4.47 | $25.64 |
| Polynomial (degree 3) | 0.6897 | $3.74 | $28.88 |

The polynomial model fits the training data better, but polynomial extrapolation can go off the rails pretty quickly — I learned that the hard way.

📊 **Charts:** [Price Prediction](data/planet-labs/predictions/pl_price_prediction.png) · [Model Dashboard](data/planet-labs/predictions/pl_model_dashboard.png)

#### Monte Carlo Simulation

After the regression models, I wanted to try something more advanced. I built a Monte Carlo simulation using Geometric Brownian Motion (GBM) — basically running 10,000 random price paths based on PL's historical drift and volatility. I also added stress tests and robustness checks because I wanted to see how the model holds up under different conditions.

| Horizon | Median Price | P5 (Worst Case) | P95 (Best Case) |
|---------|-------------|-----------------|-----------------|
| 6 Months | $23.38 | $9.65 | $54.65 |
| 1 Year | $22.29 | $6.48 | $76.08 |
| 2 Years | $20.10 | $3.52 | $117.12 |

**Risk numbers:** VaR 95% = $3.52 · CVaR 95% = $2.39 · P(Profit 2yr) = 42.3%

**Stress tests:** Market Crash → $0.15 · Bear → $3.28 · Base → $20.14 · Bull → $49.26 · Extreme Bull → $31.44

**Robustness checks:** Walk-forward validation, bootstrap confidence intervals, parameter sensitivity

📊 **Charts:** [Fan Chart](data/planet-labs/predictions/monte-carlo/mc_fan_chart.png) · [Stress Tests](data/planet-labs/predictions/monte-carlo/stress_test_chart.png) · [Robustness Dashboard](data/planet-labs/predictions/monte-carlo/robustness_dashboard.png)

📖 **Full write-up:** [Monte Carlo README](data/planet-labs/predictions/monte-carlo/README.md)

> ⚠️ **Disclaimer:** All predictions are educational simulations — not financial advice.

---

## Tools I Used

| Tool | What I Used It For |
|------|--------------------|
| Python 3.11 | Everything — data collection, ML models, charts |
| GitHub Actions | Automating the weekly data pulls |
| Airweave | Syncing this repo for AI-powered search |
| yfinance | Getting stock data from Yahoo Finance (free) |
| NumPy / SciPy | Math for the Monte Carlo simulation |
| Matplotlib | Making all the charts and dashboards |
| scikit-learn | Linear and polynomial regression |

## How to Run This Yourself

```bash
# Clone the repo
git clone https://github.com/raysteezy/raysteezy-learning.git
cd raysteezy-learning

# Install the Python packages you need
pip install yfinance pandas numpy scipy matplotlib scikit-learn

# Run the data collector
python scripts/fetch_planet_labs_financials.py

# Run the regression models
python scripts/pl_price_prediction_model.py

# Run the Monte Carlo simulation
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
