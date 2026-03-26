#!/usr/bin/env python3
"""
(PL) — Monte Carlo Simulation V1 (Baseline)
My first Monte Carlo simulation. Uses standard Geometric Brownian
Motion (GBM) with constant drift and volatility. The stress tests
are just made-up multipliers — not based on real market data.

This earned a C+ because:
  - Constant volatility is unrealistic
  - No jump modeling
  - Stress scenarios are arbitrary
  - No walk-forward validation

See monte_carlo_v2.py for the upgraded version.
Outputs: v1_mc_summary.json, v1_mc_percentile_paths.csv
Disclaimer: Educational simulation only — not financial advice.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "PL"
N_SIMS = 10_000
FORECAST_DAYS = 504
RANDOM_SEED = 42


def fetch_data():
    stock = yf.Ticker(TICKER)
    hist = stock.history(period="max")
    hist = hist[hist.index >= "2021-01-01"]
    close = hist["Close"].dropna()
    log_ret = np.log(close / close.shift(1)).dropna()
    return stock, close, log_ret


def run_gbm(price, mu, sigma):
    np.random.seed(RANDOM_SEED)
    Z = np.random.standard_normal((FORECAST_DAYS, N_SIMS))
    log_paths = np.cumsum((mu - 0.5 * sigma ** 2) + sigma * Z, axis=0)
    log_paths = np.vstack([np.zeros(N_SIMS), log_paths])
    return price * np.exp(log_paths)


def run_stress_tests(price, mu, sigma):
    scenarios = {
        "market_crash": ("Market Crash (2008-style)", -3.0, 2.5),
        "bear_market":  ("Prolonged Bear Market",     -1.5, 1.5),
        "base_case":    ("Base Case (Historical)",     1.0, 1.0),
        "bull_market":  ("Strong Bull Market",         3.0, 0.8),
        "extreme_bull": ("Extreme Bull (Sector Boom)", 5.0, 2.0),
    }
    results = {}
    for key, (name, mu_mult, sig_mult) in scenarios.items():
        sc_mu, sc_sig = mu * mu_mult, sigma * sig_mult
        Z = np.random.standard_normal((FORECAST_DAYS, N_SIMS))
        terminal = price * np.exp(np.sum((sc_mu - 0.5 * sc_sig ** 2) + sc_sig * Z, axis=0))
        results[key] = {
            "name": name,
            "median_2yr": round(float(np.median(terminal)), 2),
            "p10_2yr": round(float(np.percentile(terminal, 10)), 2),
            "p90_2yr": round(float(np.percentile(terminal, 90)), 2),
            "prob_profit": round(float(np.mean(terminal > price) * 100), 1),
        }
        print(f"  {name:35s} -> Median: ${results[key]['median_2yr']:.2f}")
    return results


def save_results(stock, price, log_ret, mu, sigma, paths, stress, close):
    terminal = paths[FORECAST_DAYS]
    var_95 = round(float(np.percentile(terminal, 5)), 2)
    var_99 = round(float(np.percentile(terminal, 1)), 2)

    summary = {
        "ticker": TICKER,
        "company": stock.info.get("longName", "Planet Labs PBC"),
        "version": "v1 — constant-volatility GBM", "grade": "C+",
        "current_price": round(float(price), 2),
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "parameters": {
            "trading_days": len(log_ret),
            "daily_drift": round(float(mu), 6), "daily_volatility": round(float(sigma), 6),
            "annual_drift": round(float(mu * 252), 4), "annual_volatility": round(float(sigma * np.sqrt(252)), 4),
        },
        "results": {h: {"median": round(float(np.median(paths[d])), 2),
                        "p5": round(float(np.percentile(paths[d], 5)), 2),
                        "p95": round(float(np.percentile(paths[d], 95)), 2)}
                   for h, d in [("6_months", 126), ("1_year", 252), ("2_years", FORECAST_DAYS)]},
        "risk_metrics": {
            "var_95": var_95, "var_99": var_99,
            "cvar_95": round(float(np.mean(terminal[terminal <= var_95])), 2),
            "prob_profit": round(float(np.mean(terminal > price) * 100), 1),
            "prob_double": round(float(np.mean(terminal > 2 * price) * 100), 1),
        },
        "stress_tests": stress,
        "problems": [
            "Constant volatility assumption", "No jump modeling",
            "Stress scenarios use arbitrary multipliers", "No walk-forward validation",
        ],
        "disclaimer": "Educational Monte Carlo simulation. Not financial advice.",
    }
    with open("v1_mc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  Saved: v1_mc_summary.json")

    dates = pd.bdate_range(start=close.index[-1], periods=FORECAST_DAYS + 1)
    pcts = [5, 10, 25, 50, 75, 90, 95]
    pct_vals = np.percentile(paths, pcts, axis=1)
    pct_df = pd.DataFrame({f"P{p}": pct_vals[i] for i, p in enumerate(pcts)}, index=dates)
    pct_df.index.name = "date"
    pct_df.to_csv("v1_mc_percentile_paths.csv")
    print("  Saved: v1_mc_percentile_paths.csv")


def main():
    print(f"[1/4] Fetching {TICKER} price history ...")
    stock, close, log_ret = fetch_data()
    price, mu, sigma = close.iloc[-1], log_ret.mean(), log_ret.std()
    print(f"  ${price:.2f}, {len(log_ret)} days")

    print(f"\n[2/4] Running GBM ({N_SIMS:,} paths) ...")
    paths = run_gbm(price, mu, sigma)
    print(f"  2yr median: ${np.median(paths[FORECAST_DAYS]):.2f}")

    print("\n[3/4] Stress tests ...")
    stress = run_stress_tests(price, mu, sigma)

    print("\n[4/4] Saving ...")
    save_results(stock, price, log_ret, mu, sigma, paths, stress, close)


if __name__ == "__main__":
    main()
