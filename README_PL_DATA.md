# Planet Labs (PL) Live Financial Data Feed

This repository contains an automated weekly data feed for **Planet Labs PBC** (NYSE: PL).

## What's Included

| File | Description |
|------|-------------|
| `data/pl_quote.json` | Current stock price, market cap, margins, ratios, and key metrics |
| `data/pl_income_statement.csv` | Quarterly income statement (revenue, gross profit, operating expenses, net income, EPS) |
| `data/pl_balance_sheet.csv` | Quarterly balance sheet (assets, liabilities, equity, cash, debt) |
| `data/pl_cash_flow.csv` | Quarterly cash flow (operating, investing, financing, free cash flow) |
| `data/pl_price_history.csv` | 1-year daily stock price history (OHLCV) |

## How It Works

- A **GitHub Actions workflow** runs every Monday at 6:00 AM UTC (11:00 PM Sunday MST)
- The Python script `fetch_planet_labs_data.py` uses `yfinance` to pull the latest data from Yahoo Finance
- Updated data files are automatically committed to this repo
- You can also trigger an update manually from the **Actions** tab in GitHub

## Manual Update

To run the data fetch manually:
1. Go to the **Actions** tab in this repository
2. Select **"Update Planet Labs Financial Data"** workflow
3. Click **"Run workflow"**

Or run locally:
```bash
pip install yfinance pandas
python fetch_planet_labs_data.py
```

## Data Source

All financial data is sourced from [Yahoo Finance](https://finance.yahoo.com/quote/PL/) via the `yfinance` Python library. No API key required.
