"""
Planet Labs (PL) Financial Data Feed
Fetches live financial data for Planet Labs PBC (NYSE: PL)
and saves it as CSV/JSON files in this repository.

Runs weekly via GitHub Actions — no API key required.
Uses yfinance (Yahoo Finance) for free, reliable data.
"""

import json
import csv
import os
from datetime import datetime, timezone

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    os.system("pip install yfinance")
    import yfinance as yf


TICKER = "PL"
OUTPUT_DIR = "data"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_quote(stock):
    """Fetch current stock quote and key metrics."""
    info = stock.info
    quote = {
        "ticker": TICKER,
        "company_name": info.get("longName", "Planet Labs PBC"),
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "previous_close": info.get("previousClose"),
        "market_cap": info.get("marketCap"),
        "enterprise_value": info.get("enterpriseValue"),
        "pe_ratio_trailing": info.get("trailingPE"),
        "pe_ratio_forward": info.get("forwardPE"),
        "eps_trailing": info.get("trailingEps"),
        "eps_forward": info.get("forwardEps"),
        "price_to_sales": info.get("priceToSalesTrailing12Months"),
        "price_to_book": info.get("priceToBook"),
        "52_week_high": info.get("fiftyTwoWeekHigh"),
        "52_week_low": info.get("fiftyTwoWeekLow"),
        "50_day_avg": info.get("fiftyDayAverage"),
        "200_day_avg": info.get("twoHundredDayAverage"),
        "volume": info.get("volume"),
        "avg_volume": info.get("averageVolume"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "float_shares": info.get("floatShares"),
        "revenue_ttm": info.get("totalRevenue"),
        "gross_profit_ttm": info.get("grossProfits"),
        "ebitda_ttm": info.get("ebitda"),
        "free_cash_flow_ttm": info.get("freeCashflow"),
        "operating_cash_flow_ttm": info.get("operatingCashflow"),
        "total_cash": info.get("totalCash"),
        "total_debt": info.get("totalDebt"),
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "gross_margin": info.get("grossMargins"),
        "operating_margin": info.get("operatingMargins"),
        "profit_margin": info.get("profitMargins"),
        "revenue_growth_yoy": info.get("revenueGrowth"),
        "earnings_growth_yoy": info.get("earningsGrowth"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }
    return quote


def fetch_income_statement(stock):
    """Fetch quarterly income statement data."""
    df = stock.quarterly_financials
    if df is not None and not df.empty:
        df = df.T  # Transpose: dates as rows
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = df["date"].astype(str)
        return df
    return None


def fetch_balance_sheet(stock):
    """Fetch quarterly balance sheet data."""
    df = stock.quarterly_balance_sheet
    if df is not None and not df.empty:
        df = df.T
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = df["date"].astype(str)
        return df
    return None


def fetch_cash_flow(stock):
    """Fetch quarterly cash flow data."""
    df = stock.quarterly_cashflow
    if df is not None and not df.empty:
        df = df.T
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = df["date"].astype(str)
        return df
    return None


def fetch_price_history(stock, period="1y"):
    """Fetch historical price data."""
    df = stock.history(period=period)
    if df is not None and not df.empty:
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = df["date"].astype(str)
        return df
    return None


def save_json(data, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {filepath}")


def save_csv(df, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")


def main():
    print(f"=" * 60)
    print(f"Planet Labs (PL) Financial Data Feed")
    print(f"Fetching data at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"=" * 60)

    stock = yf.Ticker(TICKER)

    # 1. Current quote & key metrics
    print("\n[1/5] Fetching current quote & key metrics...")
    quote = fetch_quote(stock)
    save_json(quote, "pl_quote.json")

    # 2. Income statement
    print("\n[2/5] Fetching quarterly income statement...")
    income = fetch_income_statement(stock)
    if income is not None:
        save_csv(income, "pl_income_statement.csv")
    else:
        print("  No income statement data available")

    # 3. Balance sheet
    print("\n[3/5] Fetching quarterly balance sheet...")
    balance = fetch_balance_sheet(stock)
    if balance is not None:
        save_csv(balance, "pl_balance_sheet.csv")
    else:
        print("  No balance sheet data available")

    # 4. Cash flow statement
    print("\n[4/5] Fetching quarterly cash flow statement...")
    cashflow = fetch_cash_flow(stock)
    if cashflow is not None:
        save_csv(cashflow, "pl_cash_flow.csv")
    else:
        print("  No cash flow data available")

    # 5. Price history (1 year)
    print("\n[5/5] Fetching 1-year price history...")
    prices = fetch_price_history(stock, period="1y")
    if prices is not None:
        save_csv(prices, "pl_price_history.csv")
    else:
        print("  No price history data available")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Done! All files saved to '{OUTPUT_DIR}/' directory.")
    print(f"Stock Price: ${quote.get('price', 'N/A')}")
    print(f"Market Cap: ${quote.get('market_cap', 'N/A'):,}" if quote.get('market_cap') else "Market Cap: N/A")
    print(f"Revenue (TTM): ${quote.get('revenue_ttm', 'N/A'):,}" if quote.get('revenue_ttm') else "Revenue TTM: N/A")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
