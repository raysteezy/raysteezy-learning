# Raysteezy Learning

A repository for learning projects — machine learning, financial data analysis, and Python development.

## Projects

### Planet Labs (NYSE: PL) — Financial Data Feed

Automated weekly pipeline that collects and stores Planet Labs financial data.

| File | Description |
|------|-------------|
| [`data/planet-labs/quote.json`](data/planet-labs/quote.json) | Current stock price, market cap, margins, and key ratios |
| [`data/planet-labs/income_statement.csv`](data/planet-labs/income_statement.csv) | Quarterly income statements |
| [`data/planet-labs/balance_sheet.csv`](data/planet-labs/balance_sheet.csv) | Quarterly balance sheets |
| [`data/planet-labs/cash_flow.csv`](data/planet-labs/cash_flow.csv) | Quarterly cash flow statements |
| [`data/planet-labs/price_history.csv`](data/planet-labs/price_history.csv) | 1-year daily stock price history (OHLCV) |

**How it works:**
- A GitHub Actions workflow runs every Monday at 11:00 PM MST
- The script [`scripts/fetch_planet_labs_financials.py`](scripts/fetch_planet_labs_financials.py) pulls data from Yahoo Finance
- Updated files are committed automatically
- Data is synced to [Airweave](https://airweave.ai) for semantic search

**Manual update:** Go to the [Actions tab](../../actions) → select "Update Planet Labs Data" → click "Run workflow"

## Tech Stack

- **Python** — data collection scripts
- **GitHub Actions** — weekly automation
- **Airweave** — semantic search over synced data
- **yfinance** — free financial data via Yahoo Finance

## Author

[@raysteezy](https://github.com/raysteezy)
