"""
Planet Labs (PL) — Financial Data Collector
============================================
Grabs financial data for Planet Labs (NYSE: PL) from Yahoo Finance
and saves it as CSV and JSON files in data/planet-labs/.

This runs automatically every week through GitHub Actions.
You can also run it manually whenever you want.

No API key needed — yfinance pulls from Yahoo Finance for free.
"""

import json
import os
from datetime import datetime, timezone

# Try to import yfinance. If it's not installed, install it first.
try:
    import yfinance as yf
except ImportError:
    os.system("pip install yfinance")
    import yfinance as yf


# ── Settings ─────────────────────────────────────────────────────────
TICKER = "PL"
OUTPUT_DIR = os.path.join("data", "planet-labs")


def ensure_output_dir():
    """Create the output folder if it doesn't exist yet."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_quote(stock):
    """
    Pull the current stock quote and key financial metrics.

    Returns a dictionary with things like price, market cap, margins,
    ratios, etc. We use .get() on everything so the script doesn't
    crash if Yahoo Finance is missing a field.
    """
    info = stock.info

    quote = {
        # Basic info
        "ticker": TICKER,
        "company_name": info.get("longName", "Planet Labs PBC"),
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),

        # Price data
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "previous_close": info.get("previousClose"),

        # Valuation
        "market_cap": info.get("marketCap"),
        "enterprise_value": info.get("enterpriseValue"),
        "pe_ratio_trailing": info.get("trailingPE"),
        "pe_ratio_forward": info.get("forwardPE"),
        "eps_trailing": info.get("trailingEps"),
        "eps_forward": info.get("forwardEps"),
        "price_to_sales": info.get("priceToSalesTrailing12Months"),
        "price_to_book": info.get("priceToBook"),

        # Price range
        "52_week_high": info.get("fiftyTwoWeekHigh"),
        "52_week_low": info.get("fiftyTwoWeekLow"),
        "50_day_avg": info.get("fiftyDayAverage"),
        "200_day_avg": info.get("twoHundredDayAverage"),

        # Volume
        "volume": info.get("volume"),
        "avg_volume": info.get("averageVolume"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "float_shares": info.get("floatShares"),

        # Financials
        "revenue_ttm": info.get("totalRevenue"),
        "gross_profit_ttm": info.get("grossProfits"),
        "ebitda_ttm": info.get("ebitda"),
        "free_cash_flow_ttm": info.get("freeCashflow"),
        "operating_cash_flow_ttm": info.get("operatingCashflow"),
        "total_cash": info.get("totalCash"),
        "total_debt": info.get("totalDebt"),

        # Ratios and margins
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "gross_margin": info.get("grossMargins"),
        "operating_margin": info.get("operatingMargins"),
        "profit_margin": info.get("profitMargins"),
        "revenue_growth_yoy": info.get("revenueGrowth"),
        "earnings_growth_yoy": info.get("earningsGrowth"),

        # Classification
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }
    return quote


def fetch_dataframe(df_raw, label):
    """
    Clean up a financial statement from yfinance.

    yfinance returns financial statements as DataFrames where columns
    are dates and rows are line items. We flip (transpose) them so
    each row is a quarter and each column is a metric — much easier
    to read as a CSV.
    """
    if df_raw is not None and not df_raw.empty:
        df = df_raw.T                    # flip rows and columns
        df.index.name = "date"
        df = df.reset_index()            # make date a regular column
        df["date"] = df["date"].astype(str)
        return df

    print(f"  No {label} data available")
    return None


def save_json(data, filename):
    """Save a dictionary as a pretty-printed JSON file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {filepath}")


def save_csv(df, filename):
    """Save a DataFrame as a CSV file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")


def main():
    """
    Main function — runs all 5 data collection steps in order.
    Each step prints what it's doing so you can follow along.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"{'=' * 60}")
    print(f"  Planet Labs (PL) — Financial Data Collector")
    print(f"  {timestamp}")
    print(f"{'=' * 60}")

    ensure_output_dir()

    # Create a yfinance Ticker object — this is how we talk to Yahoo Finance
    stock = yf.Ticker(TICKER)

    # Step 1: Get the current stock quote
    print("\n[1/5] Quote & key metrics...")
    quote = fetch_quote(stock)
    save_json(quote, "quote.json")

    # Step 2: Quarterly income statements
    print("\n[2/5] Quarterly income statement...")
    income = fetch_dataframe(stock.quarterly_financials, "income statement")
    if income is not None:
        save_csv(income, "income_statement.csv")

    # Step 3: Quarterly balance sheets
    print("\n[3/5] Quarterly balance sheet...")
    balance = fetch_dataframe(stock.quarterly_balance_sheet, "balance sheet")
    if balance is not None:
        save_csv(balance, "balance_sheet.csv")

    # Step 4: Quarterly cash flow
    print("\n[4/5] Quarterly cash flow...")
    cashflow = fetch_dataframe(stock.quarterly_cashflow, "cash flow")
    if cashflow is not None:
        save_csv(cashflow, "cash_flow.csv")

    # Step 5: Full price history (all available data since IPO)
    print("\n[5/5] Full price history (since IPO)...")
    prices = stock.history(period="max")
    if prices is not None and not prices.empty:
        prices.index.name = "date"
        prices = prices.reset_index()
        prices["date"] = prices["date"].astype(str)
        save_csv(prices, "price_history.csv")
    else:
        print("  No price history data available")

    # Print a summary at the end
    print(f"\n{'=' * 60}")
    print(f"  Complete. Files saved to {OUTPUT_DIR}/")
    if quote.get("price"):
        print(f"  PL Stock Price: ${quote['price']}")
    if quote.get("market_cap"):
        print(f"  Market Cap: ${quote['market_cap']:,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
