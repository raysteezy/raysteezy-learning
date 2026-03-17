# Planet Labs (NYSE: PL) — Data Dictionary

This file explains what each column/field means in the data files. The data updates automatically every week through [GitHub Actions](../../../actions).

## Datasets

### `quote.json` — Current Quote & Key Metrics

A snapshot of Planet Labs stock data and fundamental metrics. This gets overwritten each time the script runs, so it always has the latest numbers.

| Field | Type | What It Means |
|-------|------|--------------|
| `ticker` | string | Stock ticker symbol (PL) |
| `company_name` | string | Full company name |
| `last_updated` | datetime | When the data was last fetched (UTC) |
| `price` | float | Current stock price in USD |
| `previous_close` | float | What the stock closed at yesterday |
| `market_cap` | int | Market capitalization — basically how much the whole company is worth on paper |
| `enterprise_value` | int | Market cap + debt - cash — another way to value a company |
| `pe_ratio_trailing` | float | Price-to-earnings ratio using last 12 months of earnings |
| `pe_ratio_forward` | float | P/E ratio using estimated future earnings |
| `eps_trailing` | float | Earnings per share (last 12 months) |
| `eps_forward` | float | Earnings per share (estimated) |
| `price_to_sales` | float | Price-to-sales ratio (trailing 12 months) |
| `price_to_book` | float | Price-to-book ratio |
| `52_week_high` | float | Highest price in the past year |
| `52_week_low` | float | Lowest price in the past year |
| `50_day_avg` | float | Average price over the last 50 trading days |
| `200_day_avg` | float | Average price over the last 200 trading days |
| `volume` | int | How many shares traded today |
| `avg_volume` | int | Average daily trading volume |
| `shares_outstanding` | int | Total number of shares that exist |
| `revenue_ttm` | int | Total revenue over the last 12 months (USD) |
| `gross_profit_ttm` | int | Revenue minus cost of goods over the last 12 months |
| `ebitda_ttm` | int | Earnings before interest, taxes, depreciation, and amortization (12 months) |
| `free_cash_flow_ttm` | int | Cash left over after operating expenses and capital spending (12 months) |
| `total_cash` | int | Cash the company has on hand |
| `total_debt` | int | Total debt the company owes |
| `debt_to_equity` | float | Debt compared to shareholder equity — higher means more leveraged |
| `current_ratio` | float | Current assets / current liabilities — above 1.0 means they can pay short-term bills |
| `gross_margin` | float | Gross profit / revenue (as a decimal, so 0.55 = 55%) |
| `operating_margin` | float | Operating income / revenue |
| `profit_margin` | float | Net income / revenue |
| `revenue_growth_yoy` | float | How much revenue grew compared to last year |
| `sector` | string | Market sector (e.g., Technology) |
| `industry` | string | Industry classification (e.g., Aerospace & Defense) |

### `income_statement.csv` — Quarterly Income Statements

| Column | What It Means |
|--------|--------------|
| Date | End of the quarter |
| Total Revenue | Total money the company brought in that quarter |
| Cost Of Revenue | What it cost to deliver their product/service |
| Gross Profit | Revenue minus cost of revenue |
| Operating Expense | All the costs of running the business |
| Operating Income | Gross profit minus operating expenses |
| Net Income | The bottom line — profit or loss after everything |
| Basic EPS | Earnings per share (basic) |
| Diluted EPS | Earnings per share (assuming all options/warrants convert) |
| EBITDA | Earnings before interest, taxes, depreciation, amortization |
| Research Development | How much they spent on R&D |

### `balance_sheet.csv` — Quarterly Balance Sheets

| Column | What It Means |
|--------|--------------|
| Date | End of the quarter |
| Total Assets | Everything the company owns |
| Total Liabilities Net Minority Interest | Everything the company owes |
| Stockholders Equity | Assets minus liabilities — what shareholders actually "own" |
| Cash And Cash Equivalents | Cash and things that are basically cash |
| Total Debt | Short-term + long-term debt combined |
| Current Assets | Assets that can be turned into cash within a year |
| Current Liabilities | Bills due within a year |

### `cash_flow.csv` — Quarterly Cash Flow Statements

| Column | What It Means |
|--------|--------------|
| Date | End of the quarter |
| Operating Cash Flow | Cash generated from the actual business |
| Capital Expenditure | Money spent on equipment, satellites, infrastructure |
| Free Cash Flow | Operating cash flow minus capex — cash the company has "left over" |

### `price_history.csv` — Daily Stock Prices (Full History Since IPO)

| Column | What It Means |
|--------|--------------|
| Date | Trading date |
| Open | Price when the market opened |
| High | Highest price during the day |
| Low | Lowest price during the day |
| Close | Price when the market closed |
| Volume | Number of shares traded that day |
| Dividends | Any dividend payments |
| Stock Splits | Any stock split events |

## Where the Data Comes From

All data comes from [Yahoo Finance](https://finance.yahoo.com/quote/PL/) through the `yfinance` Python library. It's free and doesn't need an API key.

## Update Schedule

Data refreshes automatically every **Monday at 11:00 PM MST** through GitHub Actions.
