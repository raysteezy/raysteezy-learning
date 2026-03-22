"""
Planet Labs (PL) — Financial Data Collector

Grabs financial data for Planet Labs (NYSE: PL) from Yahoo Finance
and saves it as CSV and JSON files in data/planet-labs/.

This runs automatically every week through GitHub Actions.
You can also run it manually whenever you want.

No API key needed — yfinance pulls from Yahoo Finance for free.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

TICKER = "PL"
OUTPUT_DIR = os.path.join("data", "planet-labs")


def fetch_quote(stock: yf.Ticker) -> dict:
    """Pull current stock quote and key financial metrics."""
    info = stock.info
    return {
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


def fetch_dataframe(df_raw: pd.DataFrame | None, label: str) -> pd.DataFrame | None:
    """Transpose a yfinance financial statement so rows = quarters."""
    if df_raw is None or df_raw.empty:
        print(f"  No {label} data available")
        return None
    df = df_raw.T
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = df["date"].astype(str)
    return df


def save_json(data: dict, filename: str) -> None:
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {filepath}")


def save_csv(df: pd.DataFrame, filename: str) -> None:
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"{'=' * 60}")
    print(f"  Planet Labs (PL) — Financial Data Collector")
    print(f"  {timestamp}")
    print(f"{'=' * 60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stock = yf.Ticker(TICKER)

    print("\n[1/5] Quote & key metrics...")
    quote = fetch_quote(stock)
    save_json(quote, "quote.json")

    print("\n[2/5] Quarterly income statement...")
    income = fetch_dataframe(stock.quarterly_financials, "income statement")
    if income is not None:
        save_csv(income, "income_statement.csv")

    print("\n[3/5] Quarterly balance sheet...")
    balance = fetch_dataframe(stock.quarterly_balance_sheet, "balance sheet")
    if balance is not None:
        save_csv(balance, "balance_sheet.csv")

    print("\n[4/5] Quarterly cash flow...")
    cashflow = fetch_dataframe(stock.quarterly_cashflow, "cash flow")
    if cashflow is not None:
        save_csv(cashflow, "cash_flow.csv")

    print("\n[5/5] Full price history (since IPO)...")
    prices = stock.history(period="max")
    if prices is not None and not prices.empty:
        prices.index.name = "date"
        prices = prices.reset_index()
        prices["date"] = prices["date"].astype(str)
        save_csv(prices, "price_history.csv")
    else:
        print("  No price history data available")

    print(f"\n{'=' * 60}")
    print(f"  Complete. Files saved to {OUTPUT_DIR}/")
    if quote.get("price"):
        print(f"  PL Stock Price: ${quote['price']}")
    if quote.get("market_cap"):
        print(f"  Market Cap: ${quote['market_cap']:,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
