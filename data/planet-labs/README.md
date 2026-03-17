# Planet Labs (NYSE: PL) — Data Dictionary

Last updated automatically via [GitHub Actions](../../../actions).

## Datasets

### `quote.json` — Current Quote & Key Metrics

Real-time snapshot of Planet Labs stock data and fundamental metrics.

| Field | Type | Description |
|-------|------|-------------|
| `ticker` | string | Stock ticker symbol (PL) |
| `company_name` | string | Full company name |
| `last_updated` | datetime | Timestamp of last data fetch (UTC) |
| `price` | float | Current stock price (USD) |
| `previous_close` | float | Previous trading day close price |
| `market_cap` | int | Market capitalization (USD) |
| `enterprise_value` | int | Enterprise value (USD) |
| `pe_ratio_trailing` | float | Trailing 12-month P/E ratio |
| `pe_ratio_forward` | float | Forward P/E ratio |
| `eps_trailing` | float | Trailing earnings per share |
| `eps_forward` | float | Forward earnings per share |
| `price_to_sales` | float | Price-to-sales ratio (TTM) |
| `price_to_book` | float | Price-to-book ratio |
| `52_week_high` | float | 52-week high price |
| `52_week_low` | float | 52-week low price |
| `50_day_avg` | float | 50-day moving average |
| `200_day_avg` | float | 200-day moving average |
| `volume` | int | Current day trading volume |
| `avg_volume` | int | Average daily trading volume |
| `shares_outstanding` | int | Total shares outstanding |
| `revenue_ttm` | int | Trailing 12-month revenue (USD) |
| `gross_profit_ttm` | int | Trailing 12-month gross profit (USD) |
| `ebitda_ttm` | int | Trailing 12-month EBITDA (USD) |
| `free_cash_flow_ttm` | int | Trailing 12-month free cash flow (USD) |
| `total_cash` | int | Total cash on hand (USD) |
| `total_debt` | int | Total debt (USD) |
| `debt_to_equity` | float | Debt-to-equity ratio |
| `current_ratio` | float | Current ratio |
| `gross_margin` | float | Gross margin (decimal, e.g., 0.55 = 55%) |
| `operating_margin` | float | Operating margin |
| `profit_margin` | float | Net profit margin |
| `revenue_growth_yoy` | float | Year-over-year revenue growth |
| `sector` | string | Market sector |
| `industry` | string | Industry classification |

### `income_statement.csv` — Quarterly Income Statements

| Column | Description |
|--------|-------------|
| Date | Quarter end date |
| Total Revenue | Total revenue for the quarter |
| Cost Of Revenue | Direct costs of goods/services |
| Gross Profit | Revenue minus cost of revenue |
| Operating Expense | Total operating expenses |
| Operating Income | Gross profit minus operating expenses |
| Net Income | Bottom-line profit/loss |
| Basic EPS | Earnings per share (basic) |
| Diluted EPS | Earnings per share (diluted) |
| EBITDA | Earnings before interest, taxes, depreciation, amortization |
| Research Development | R&D spending |

### `balance_sheet.csv` — Quarterly Balance Sheets

| Column | Description |
|--------|-------------|
| Date | Quarter end date |
| Total Assets | Sum of all company assets |
| Total Liabilities Net Minority Interest | Total liabilities |
| Stockholders Equity | Total shareholders' equity |
| Cash And Cash Equivalents | Cash and liquid assets |
| Total Debt | Short-term + long-term debt |
| Current Assets | Assets convertible to cash within 1 year |
| Current Liabilities | Obligations due within 1 year |

### `cash_flow.csv` — Quarterly Cash Flow Statements

| Column | Description |
|--------|-------------|
| Date | Quarter end date |
| Operating Cash Flow | Cash from core business operations |
| Capital Expenditure | Spending on property, equipment, satellites |
| Free Cash Flow | Operating cash flow minus capex |

### `price_history.csv` — Daily Stock Prices (1 Year)

| Column | Description |
|--------|-------------|
| Date | Trading date |
| Open | Opening price |
| High | Highest price of the day |
| Low | Lowest price of the day |
| Close | Closing price |
| Volume | Number of shares traded |
| Dividends | Dividend payments |
| Stock Splits | Stock split events |

## Data Source

All data is sourced from [Yahoo Finance](https://finance.yahoo.com/quote/PL/) via the open-source `yfinance` Python library. No API key is required.

## Update Schedule

Data refreshes automatically every **Monday at 11:00 PM MST** via GitHub Actions.
