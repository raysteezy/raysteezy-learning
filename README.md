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
│   └── planet-labs/           # Planet Labs (NYSE: PL) financial data
│       ├── quote.json         #   Current stock price & key metrics
│       ├── income_statement.csv   #   Quarterly income statements
│       ├── balance_sheet.csv  #   Quarterly balance sheets
│       ├── cash_flow.csv      #   Quarterly cash flow statements
│       ├── price_history.csv  #   1-year daily OHLCV prices
│       └── README.md          #   Data dictionary & field descriptions
├── scripts/
│   └── fetch_planet_labs_financials.py  # Data collection script
├── .github/workflows/
│   └── update-planet-labs-data.yml     # Weekly automation
├── .gitignore
├── LICENSE
├── SECURITY.md
└── README.md                  # You are here
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

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Data collection & processing |
| GitHub Actions | Weekly scheduled automation |
| Airweave | Semantic search over synced data |
| yfinance | Financial data from Yahoo Finance |

## Getting Started

```bash
# Clone the repository
git clone https://github.com/raysteezy/raysteezy-learning.git
cd raysteezy-learning

# Install dependencies
pip install yfinance pandas

# Run the data collector
python scripts/fetch_planet_labs_financials.py
```

## License

This project is licensed under the [MIT License](LICENSE).

## Security

See [SECURITY.md](SECURITY.md) for security policies and responsible disclosure guidelines.

---

<p align="center">Built by <a href="https://github.com/raysteezy">@raysteezy</a></p>
